import logging
import re
from typing import List, Dict, Any, Literal

import modal

# Образ с mystem и razdel
image = (
    modal.Image.debian_slim()
    .pip_install("pymystem3", "razdel")
    .run_commands("python -c 'from pymystem3 import Mystem; Mystem()'")
)

app = modal.App("booknlp-ru-mystem")

# Маппинг POS mystem -> UD POS
MYSTEM_TO_UPOS = {
    "S": "NOUN",
    "A": "ADJ",
    "V": "VERB",
    "ADV": "ADV",
    "SPRO": "PRON",
    "PR": "ADP",
    "CONJ": "CCONJ",
    "PART": "PART",
    "INTJ": "INTJ",
    "NUM": "NUM",
    "COM": "X",
    "APRO": "DET",
    "ANUM": "ADJ",
    "ADVPRO": "ADV",
}

PUNCT_CHARS = ".!?,;:—–-\"«»()[]{}"

OutputFormat = Literal["native", "conllu"]


@app.cls(image=image, timeout=600)
class MystemService:
    @modal.enter()
    def setup(self):
        from pymystem3 import Mystem
        from razdel import tokenize

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MystemService")
        self.mystem = Mystem(entire_input=False, disambiguation=True)
        self._tokenize = tokenize
        self.logger.info("Mystem initialized!")

    # ========= INTERNAL: mystem сам токенизирует предложение =========

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: OutputFormat = "native",
    ) -> List[List[Dict[str, Any]]]:
        """
        INTERNAL режим.
        Вход: список предложений (строки), уже razdel.sentenize во wrapper.
        Для каждого предложения вызываем mystem отдельно.
        Выход: список предложений; каждое предложение — список токенов.
        """
        if output_format not in {"native", "conllu"}:
            raise ValueError(f"Unknown output_format: {output_format!r}. Use 'native' or 'conllu'.")

        results: List[List[Dict[str, Any]]] = []

        for sent in sentences:
            sent = (sent or "").strip()
            if not sent:
                results.append([])
                continue
            try:
                analysis = self.mystem.analyze(sent)
                if output_format == "native":
                    tokens = self._process_native(analysis)
                else:
                    tokens = self._process_simplified(analysis)
                results.append(tokens)
            except Exception as e:
                self.logger.error(f"mystem.analyze error (sent='{sent[:40]}'): {e}")
                results.append([])
                continue

        return results

    # ========= EXTERNAL: токены фиксирует razdel в modal =========

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences: List[str],
        output_format: OutputFormat = "native",
    ) -> List[List[Dict[str, Any]]]:
        """
        EXTERNAL режим.
        Вход: список предложений (строки) от wrapper (razdel.sentenize).
        Для каждого предложения:
          - токенизация razdel.tokenize в modal,
          - сбор строки из токенов,
          - вызов mystem.analyze,
          - получение морфоразбора для последовательности токенов.
        Выход: список предложений; каждое предложение — список токенов mystem.
        """
        if output_format not in {"native", "conllu"}:
            raise ValueError(f"Unknown output_format: {output_format!r}. Use 'native' or 'conllu'.")

        results: List[List[Dict[str, Any]]] = []

        for sent in sentences:
            sent = (sent or "").strip()
            if not sent:
                results.append([])
                continue

            # 1. razdel-токенизация предложения
            tokens_text = [t.text for t in self._tokenize(sent)]
            if not tokens_text:
                results.append([])
                continue

            # 2. Собираем строку, где каждый токен отделён пробелом
            text_for_mystem = "\n".join(tokens_text)

            # 3. Анализ mystem
            try:
                analysis = self.mystem.analyze(text_for_mystem)

                base_tokens = (
                    self._process_native(analysis)
                    if output_format == "native"
                    else self._process_simplified(analysis)
                )

                # ВАЖНО: не требуем равенства количества токенов.
                # Принимаем токенизацию mystem как есть, в порядке следования.
                results.append(base_tokens)
            except Exception as e:
                self.logger.error(f"mystem.analyze error (sent='{sent[:40]}'): {e}")
                results.append([])
                continue

        return results

    # ========= УТИЛИТЫ ДЛЯ ОБРАБОТКИ РЕЗУЛЬТАТА MYSTEM =========

    def _process_native(self, analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Полный нативный формат: один список токенов для одного предложения.
        analysis — результат mystem.analyze(строка).
        """
        tokens: List[Dict[str, Any]] = []
        for item in analysis:
            token_text = item.get("text", "")
            if not token_text:
                continue
            token_text = token_text.strip()
            if not token_text:
                continue

            native_token = {
                "id": len(tokens) + 1,
                "text": token_text,
                # analysis: полный список вариантов, как отдаёт mystem
                "analysis": item.get("analysis", []),
            }
            tokens.append(native_token)
        return tokens

    def _process_simplified(self, analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Упрощённый формат для дальнейшего преобразования в CoNLL-U.
        Здесь сразу формируем 10 полей CoNLL-U, но интерпретируемыми
        реально являются только ID, FORM, LEMMA, UPOS.
        Остальное заполняется заглушками, а сырые поля mystem
        складываются в MISC.
        """
        tokens: List[Dict[str, Any]] = []
        for item in analysis:
            token_text = item.get("text", "")
            if not token_text:
                continue
            token_text = token_text.strip()
            if not token_text:
                continue

            lemma = token_text.lower()
            upos = "X"
            misc_parts = []

            analyses = item.get("analysis") or []
            if analyses:
                best = analyses[0]
                lemma = best.get("lex", token_text.lower())
                gr_full = best.get("gr", "")
                gr_pos = re.split(r"[\[,=]", gr_full)[0]
                upos = MYSTEM_TO_UPOS.get(gr_pos, "X")

                # Всё, что не пошло в LEMMA/UPOS, кодируем в MISC:
                if gr_full:
                    misc_parts.append(f"Gr={gr_full}")
                if "wt" in best:
                    misc_parts.append(f"Wt={best['wt']}")
                if "qual" in best:
                    misc_parts.append(f"Qual={best['qual']}")
                misc_parts.append(f"Analyses={len(analyses)}")
                misc_parts.append("Best=0")

            if all(ch in PUNCT_CHARS for ch in token_text) and not analyses:
                upos = "PUNCT"

            misc = "|".join(misc_parts) if misc_parts else "_"

            tokens.append(
                {
                    "id": len(tokens) + 1,   # ID
                    "form": token_text,      # FORM
                    "lemma": lemma,          # LEMMA
                    "upos": upos,            # UPOS
                    "xpos": "_",             # XPOS (пока не используем)
                    "feats": "_",            # FEATS
                    "head": "_",             # HEAD
                    "deprel": "_",           # DEPREL
                    "deps": "_",             # DEPS
                    "misc": misc,            # MISC (сырые поля mystem)
                }
            )
        return tokens


@app.local_entrypoint()
def main():
    service = MystemService()

    sentences = [
        "Это тестовое предложение.",
        "Мама мыла раму.",
    ]

    # ========== 1. EXTERNAL + NATIVE ==========
    print("=" * 60)
    print("Mystem: EXTERNAL tokenizer (razdel в modal) + NATIVE формат")
    print("=" * 60)
    ext_native = service.parse_sentence_chunk.remote(
        sentences,
        output_format="native",
    )
    for i, sent in enumerate(ext_native, 1):
        # print(f"\n# sent_id = {i}")
        print(f"# text = {sentences[i - 1]}")
        for tok in sent:
            print(f"  Token: {tok['text']}")
            variants = tok.get("analysis") or []
            print(f"    Analysis variants: {len(variants)}")
            for j, var in enumerate(variants, 1):
                lex = var.get("lex", "")
                gr = var.get("gr", "")
                wt = var.get("wt", "")
                qual = var.get("qual", "")
                extra = []
                if wt != "":
                    extra.append(f"wt={wt}")
                if qual != "":
                    extra.append(f"qual={qual}")
                extra_str = ", ".join(extra) if extra else ""
                print(f"      [{j}] lex={lex}, gr={gr}{(', ' + extra_str) if extra_str else ''}")

    # ========== 2. EXTERNAL + CONLLU ==========
    print("\n" + "=" * 60)
    print("Mystem: EXTERNAL tokenizer (razdel в modal) + CONLLU формат")
    print("=" * 60)
    ext_conllu = service.parse_sentence_chunk.remote(
        sentences,
        output_format="conllu",
    )
    # print(f"\n  {'ID':<4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
    #       f"{'XPOS':<5} {'FEATS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5} MISC")
    print("  " + "-" * 110)
    for i, sent in enumerate(ext_conllu, 1):  # или int_conllu
        # print(f"\n# sent_id = {i}")
        print(f"# text = {sentences[i - 1]}")
        print(
            f"\n  {'ID':<4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
            f"{'XPOS':<5} {'FEATS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5}"
        )
        print("  " + "-" * 90)
        for tok in sent:
            print(
                f"  {tok['id']:<4} {tok['form']:<16} {tok['lemma']:<16} {tok['upos']:<7}"
                f" {tok['xpos']:<5} {tok['feats']:<5}"
                f" {tok['head']:<5} {tok['deprel']:<10} {tok['deps']:<5}"
            )
            if tok.get("misc", "_") != "_":
                print(f"        misc: {tok['misc']}")

    # ========== 3. INTERNAL + NATIVE ==========
    print("\n" + "=" * 60)
    print("Mystem: INTERNAL tokenizer (mystem) + NATIVE формат")
    print("=" * 60)
    int_native = service.parse_sentence_chunk_native.remote(
        sentences,
        output_format="native",
    )
    for i, sent in enumerate(int_native, 1):
        # print(f"\n# sent_id = {i}")
        print(f"# text = {sentences[i - 1]}")
        for tok in sent:
            print(f"  Token: {tok['text']}")
            variants = tok.get("analysis") or []
            print(f"    Analysis variants: {len(variants)}")
            for j, var in enumerate(variants, 1):
                lex = var.get("lex", "")
                gr = var.get("gr", "")
                wt = var.get("wt", "")
                qual = var.get("qual", "")
                extra = []
                if wt != "":
                    extra.append(f"wt={wt}")
                if qual != "":
                    extra.append(f"qual={qual}")
                extra_str = ", ".join(extra) if extra else ""
                print(f"      [{j}] lex={lex}, gr={gr}{(', ' + extra_str) if extra_str else ''}")

    # ========== 4. INTERNAL + CONLLU ==========
    print("\n" + "=" * 60)
    print("Mystem: INTERNAL tokenizer (mystem) + CONLLU формат")
    print("=" * 60)
    int_conllu = service.parse_sentence_chunk_native.remote(
        sentences,
        output_format="conllu",
    )
    # print(f"\n  {'ID':<4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
    #       f"{'XPOS':<5} {'FEATS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5} MISC")
    print("  " + "-" * 110)
    for i, sent in enumerate(ext_conllu, 1):  # или int_conllu
        # print(f"\n# sent_id = {i}")
        print(f"# text = {sentences[i - 1]}")
        print(
            f"\n  {'ID':<4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
            f"{'XPOS':<5} {'FEATS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5}"
        )
        print("  " + "-" * 90)
        for tok in sent:
            print(
                f"  {tok['id']:<4} {tok['form']:<16} {tok['lemma']:<16} {tok['upos']:<7}"
                f" {tok['xpos']:<5} {tok['feats']:<5}"
                f" {tok['head']:<5} {tok['deprel']:<10} {tok['deps']:<5}"
            )
            if tok.get("misc", "_") != "_":
                print(f"        misc: {tok['misc']}")

        print("=" * 60)
        print("СРАВНЕНИЕ ТОКЕНИЗАЦИЙ: external vs internal")
        print("=" * 60)
        sent_compare = ["Зло, которым ты меня пугаешь, вовсе не так зло."]
        ext = service.parse_sentence_chunk.remote(sent_compare, output_format="conllu")
        int_ = service.parse_sentence_chunk_native.remote(sent_compare, output_format="conllu")
        for s_idx, (s_e, s_i) in enumerate(zip(ext, int_), 1):
            print(f"\n# Sentence {s_idx}")
            print(f"  {'#':>3}  {'external':<20} {'internal':<20} {'UPOS ext':<10} {'UPOS int':<10} match")
            print("  " + "─" * 70)
            max_len = max(len(s_e), len(s_i))
            for t_idx in range(max_len):
                te = s_e[t_idx] if t_idx < len(s_e) else None
                ti = s_i[t_idx] if t_idx < len(s_i) else None
                fe = te["form"] if te else "—"
                fi = ti["form"] if ti else "—"
                ue = te["upos"] if te else "—"
                ui = ti["upos"] if ti else "—"
                match = "✅" if fe == fi and ue == ui else "❌"
                print(f"  {t_idx + 1:>3}  {fe:<20} {fi:<20} {ue:<10} {ui:<10} {match}")

