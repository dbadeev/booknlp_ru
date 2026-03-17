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
    "S":      "NOUN",
    "A":      "ADJ",
    "V":      "VERB",
    "ADV":    "ADV",
    "SPRO":   "PRON",
    "PR":     "ADP",
    "CONJ":   "CCONJ",
    "PART":   "PART",
    "INTJ":   "INTJ",
    "NUM":    "NUM",
    "COM":    "X",
    "APRO":   "DET",
    "ANUM":   "ADJ",
    "ADVPRO": "ADV",
}

PUNCT_CHARS = ".!?,;:—–‒―\"\\'`«»„‹›()[]{}…\u2012\u2013\u2014\u2015"


OutputFormat = Literal["native", "conllu"]


@app.cls(image=image, timeout=600, memory=1024)
class MystemService:

    @modal.enter()
    def setup(self):
        from pymystem3 import Mystem
        from razdel import tokenize

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MystemService")

        # ✅ entire_input=True (-c): пробелы и пунктуация попадают в вывод,
        #    что необходимо для внутренней токенизации mystem
        self.mystem = Mystem(entire_input=True, disambiguation=True)
        self.mystem.start()  # запускаем subprocess заранее

        self.tokenize = tokenize
        self.logger.info("Mystem initialized!")

    def _debug_analysis(self, sent: str, analysis: list, mode: str = "") -> None:
        """Логирует сырой вывод mystem.analyze."""
        self.logger.debug("-" * 40)
        if mode:
            self.logger.debug(f"Mode: {mode}")
        self.logger.debug(f"Input: {repr(sent)}")
        self.logger.debug(f"Raw analysis: {len(analysis)} items")
        for i, item in enumerate(analysis):
            text = item.get("text", "")
            ans = item.get("analysis", [])
            self.logger.debug(
                f"  {i}: text={repr(text)} "
                f"stripped={repr(text.strip())} "
                f"analysis[{len(ans)}]"
            )

    # ------------------------------------------------------------------ #
    #  INTERNAL — токенизатор mystem (entire_input=True)                  #
    # ------------------------------------------------------------------ #

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: OutputFormat = "native",
    ) -> List[List[Dict[str, Any]]]:
        """
        INTERNAL токенизатор: mystem сам сегментирует текст.
        entire_input=True гарантирует, что пунктуация попадает в вывод
        как токены без analysis — они корректно обрабатываются далее.
        """
        if output_format not in ("native", "conllu"):
            raise ValueError(f"Unknown output_format {output_format!r}.")

        results: List[List[Dict[str, Any]]] = []
        for sent in sentences:
            sent = (sent or "").strip()
            if not sent:
                results.append([])
                continue
            try:
                analysis = self.mystem.analyze(sent)
                self._debug_analysis(sent, analysis, mode="INTERNAL (mystem tokenizer)")
                if output_format == "native":
                    tokens = self._process_native(analysis)
                else:
                    tokens = self._process_simplified(analysis)
                results.append(tokens)
            except Exception as e:
                self.logger.error(f"mystem.analyze error sent={sent[:40]!r}: {e}")
                results.append([])
        return results

    # ------------------------------------------------------------------ #
    #  EXTERNAL — токенизатор razdel → mystem                             #
    # ------------------------------------------------------------------ #

    @modal.method()
    def parse_sentence_chunk(self, sentences, output_format="native"):
        """
        EXTERNAL: razdel токенизирует → mystem анализирует полную строку,
        включая знаки препинания (они вернутся без analysis).
        """
        results = []
        for sent in sentences:
            sent = (sent or "").strip()
            if not sent:
                results.append([])
                continue

            # 1. razdel даёт позиции токенов
            razdel_tokens = list(self.tokenize(sent))
            if not razdel_tokens:
                results.append([])
                continue

            # 2. Собираем строку для mystem — ВСЕ токены включая пунктуацию,
            #    разделённые пробелами (чтобы mystem не слипал их со словами)
            text_for_mystem = " ".join(t.text for t in razdel_tokens if t.text.strip())

            try:
                analysis = self.mystem.analyze(text_for_mystem)
                self._debug_analysis(text_for_mystem, analysis, mode="EXTERNAL (razdel tokenizer)")
                if output_format == "native":
                    tokens = self._process_native(analysis)
                else:
                    tokens = self._process_simplified(analysis)
                results.append(tokens)
            except Exception as e:
                self.logger.error(f"mystem.analyze error: {e}")
                results.append([])

        return results

    def _merge_punct(self, razdel_tokens, mystem_tokens, output_format: str):
        """
        Вставляет пунктуационные токены (от razdel) обратно в список
        токенов mystem на правильные позиции, перенумеровывая ID.
        """
        merged = []
        m_iter = iter(mystem_tokens)
        token_id = 1

        for rt in razdel_tokens:
            text = rt.text
            if not text.strip():
                continue  # пробелы пропускаем

            is_punct = all(ch in PUNCT_CHARS for ch in text)

            if is_punct:
                # Пунктуационный токен — все поля прочерк
                if output_format == "conllu":
                    merged.append({
                        "id":     token_id,
                        "form":   text,
                        "lemma":  text,        # лемма = сам знак (CoNLL-U конвенция)
                        "upos":   "PUNCT",
                        "xpos":   "_",
                        "feats":  "_",
                        "head":   "_",
                        "deprel": "_",
                        "deps":   "_",
                        "misc":   "_",
                    })
                else:  # native
                    merged.append({
                        "id":       token_id,
                        "text":     text,
                        "upos":     "PUNCT",
                        "analysis": [],
                        "is_punct": True,
                    })
            else:
                # Словный токен — берём из mystem
                mt = next(m_iter, None)
                if mt is None:
                    break
                mt["id"] = token_id
                merged.append(mt)

            token_id += 1

        return merged

    # ------------------------------------------------------------------ #
    #  Обработка сырого вывода mystem.analyze                             #
    # ------------------------------------------------------------------ #

    def _process_native(self, analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Обрабатывает analysis из mystem.analyze (entire_input=True).
        Пробелы пропускаются.
        Пунктуация (нет analysis, непустой text) — токен с PUNCT и пустым analysis[].
        """
        tokens = []
        for item in analysis:
            token_text = (item.get("text") or "").strip()
            if not token_text:
                continue  # пробел / \n — пропускаем

            ana = item.get("analysis") or []

            if ana:
                # Словный токен с морфологией
                gr_first = ana[0].get("gr", "").strip()
                gr_pos = re.split(r"[,=]", gr_first)[0].strip() if gr_first else ""
                # is_punct = bool(token_text.strip()) and all(ch in PUNCT_CHARS for ch in token_text.strip())
                upos = MYSTEM_TO_UPOS.get(gr_pos, "X")
            else:
                # Нет analysis — пунктуация или неизвестный символ
                upos = "PUNCT" if all(ch in PUNCT_CHARS for ch in token_text) else "X"

            tokens.append({
                "id":       len(tokens) + 1,
                "text":     token_text,
                "upos":     upos,
                "analysis": ana,
                "is_punct": not bool(ana),
            })
        return tokens

    def _process_simplified(self, analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CoNLL-U вывод из mystem.analyze (entire_input=True).
        Для пунктуации все поля кроме ID/FORM/LEMMA/UPOS = "_".
        """
        tokens: List[Dict[str, Any]] = []
        for item in analysis:
            token_text = item.get("text", "")
            if not token_text:
                continue
            token_text = token_text.strip()
            if not token_text:
                continue  # пробел / \n

            analyses = item.get("analysis") or []

            if analyses:
                best = analyses[0]
                lemma  = best.get("lex", token_text.lower())
                gr_full = best.get("gr", "")
                gr_pos  = re.split(r"[,=]", gr_full)[0].strip() if gr_full else ""
                upos    = MYSTEM_TO_UPOS.get(gr_pos, "X")

                misc_parts = []
                if gr_full:
                    misc_parts.append(f"Gr={gr_full}")
                if "wt" in best:
                    misc_parts.append(f"Wt={best['wt']}")
                if "qual" in best:
                    misc_parts.append(f"Qual={best['qual']}")
                misc_parts.append(f"Analyses={len(analyses)}")
                misc_parts.append("Best=0")
                misc = "|".join(misc_parts) if misc_parts else "_"

                tokens.append({
                    "id":     len(tokens) + 1,   # ID
                    "form":   token_text,          # FORM
                    "lemma":  lemma,               # LEMMA
                    "upos":   upos,                # UPOS
                    "xpos":   gr_pos or "_",       # XPOS
                    "feats":  "_",                 # FEATS (не реализовано)
                    "head":   "_",                 # HEAD
                    "deprel": "_",                 # DEPREL
                    "deps":   "_",                 # DEPS
                    "misc":   misc,                # MISC
                })
            else:
                # ✅ Пунктуация / символ без морфологии — все поля "_"
                is_punct = bool(token_text.strip()) and all(ch in PUNCT_CHARS for ch in token_text.strip())
                upos = "PUNCT" if is_punct else "X"

                tokens.append({
                    "id":     len(tokens) + 1,
                    "form":   token_text,
                    "lemma":  token_text,   # лемма = сам знак (CoNLL-U конвенция)
                    "upos":   upos,
                    "xpos":   "_",
                    "feats":  "_",
                    "head":   "_",
                    "deprel": "_",
                    "deps":   "_",
                    "misc":   "_",
                })
        return tokens

# ------------------------------------------------------------------ #
#  Local entrypoint                                                    #
# ------------------------------------------------------------------ #

@app.local_entrypoint()
def main():
    service = MystemService()

    sentences = [
        "Мама мыла раму без мыла.",
        "Привет, как дела?",
        "Стоимость кресла-качалки — 500 рублей.",
        "Он сказал: «Не беспокойтесь».",
    ]

    # Тестовое предложение для сравнения EXTERNAL vs INTERNAL
    sent_compare = ["Мама, которую я видел вчера, пошла домой."]

    # ------------------------------------------------------------------ #
    #  1. EXTERNAL (razdel) → NATIVE                                      #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("Mystem EXTERNAL (tokenizer: razdel) → NATIVE")
    print("=" * 70)

    ext_native = service.parse_sentence_chunk.remote(sentences, output_format="native")

    for i, (sent_tokens, input_text) in enumerate(ext_native, 1):
        print(f"\n  text (sent to mystem): {input_text!r}")
        for tok in sent_tokens:
            variants = tok.get("analysis") or []
            is_punct = tok.get("is_punct", False)
            if is_punct:
                print(f"    [{tok['id']:>2}] {tok['text']!r:12}  PUNCT  (no analysis)")
            else:
                print(f"    [{tok['id']:>2}] {tok['text']!r:12}  upos={tok['upos']}")
                for j, var in enumerate(variants, 1):
                    lex  = var.get("lex", "")
                    gr   = var.get("gr", "")
                    wt   = var.get("wt", "")
                    qual = var.get("qual", "")
                    extra = []
                    if wt != "":   extra.append(f"wt={wt}")
                    if qual != "": extra.append(f"qual={qual}")
                    extra_str = ", ".join(extra)
                    print(f"           {j}: lex={lex!r}  gr={gr!r}"
                          + (f"  [{extra_str}]" if extra_str else ""))

    # ------------------------------------------------------------------ #
    #  2. EXTERNAL (razdel) → CONLLU                                      #
    # ------------------------------------------------------------------ #

    print("\n" + "=" * 70)
    print("Mystem EXTERNAL (tokenizer: razdel) → CONLLU")
    print("=" * 70)

    ext_conllu = service.parse_sentence_chunk.remote(sentences, output_format="conllu")
    ext_input_texts = [text for _, text in ext_conllu]
    ext_tokens = [toks for toks, _ in ext_conllu]
    _print_conllu(ext_input_texts, ext_tokens)

    # ------------------------------------------------------------------ #
    #  3. INTERNAL (mystem) → NATIVE                                      #
    # ------------------------------------------------------------------ #

    print("\n" + "=" * 70)
    print("Mystem INTERNAL (tokenizer: mystem) → NATIVE")
    print("=" * 70)

    int_native = service.parse_sentence_chunk_native.remote(sentences, output_format="native")

    for i, sent_tokens in enumerate(int_native, 1):
        print(f"\n  text: {sentences[i - 1]!r}")
        for tok in sent_tokens:
            variants = tok.get("analysis") or []
            is_punct = tok.get("is_punct", False)
            if is_punct:
                print(f"    [{tok['id']:>2}] {tok['text']!r:12}  PUNCT  (no analysis)")
            else:
                print(f"    [{tok['id']:>2}] {tok['text']!r:12}  upos={tok['upos']}")
                for j, var in enumerate(variants, 1):
                    lex  = var.get("lex", "")
                    gr   = var.get("gr", "")
                    wt   = var.get("wt", "")
                    qual = var.get("qual", "")
                    extra = []
                    if wt != "":   extra.append(f"wt={wt}")
                    if qual != "": extra.append(f"qual={qual}")
                    extra_str = ", ".join(extra)
                    print(f"           {j}: lex={lex!r}  gr={gr!r}"
                          + (f"  [{extra_str}]" if extra_str else ""))

    # ------------------------------------------------------------------ #
    #  4. INTERNAL (mystem) → CONLLU                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Mystem INTERNAL (tokenizer: mystem) → CONLLU")
    print("=" * 70)

    int_conllu = service.parse_sentence_chunk_native.remote(sentences, output_format="conllu")

    _print_conllu(sentences, int_conllu)

    # ------------------------------------------------------------------ #
    #  5. EXTERNAL vs INTERNAL — сравнение токенизации и UPOS             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("EXTERNAL vs INTERNAL — сравнение")
    print("=" * 70)

    ext_cmp_raw = service.parse_sentence_chunk.remote(sent_compare, output_format="conllu")
    ext_cmp = [toks for toks, _ in ext_cmp_raw]
    int_cmp = service.parse_sentence_chunk_native.remote(sent_compare, output_format="conllu")

    # ext_cmp = service.parse_sentence_chunk.remote(sent_compare, output_format="conllu")
    # int_cmp = service.parse_sentence_chunk_native.remote(sent_compare, output_format="conllu")

    for s_idx, (se, si) in enumerate(zip(ext_cmp, int_cmp), 1):
        print(f"\n  Sentence {s_idx}: {sent_compare[s_idx - 1]!r}")
        match_icon = "✓" if len(se) == len(si) else "✗"
        print(f"  Tokens: external={len(se)}, internal={len(si)}  {match_icon}")
        print(f"\n  {'':>3}  {'EXTERNAL':^20}  {'INTERNAL':^20}  "
              f"{'UPOS_EXT':^10}  {'UPOS_INT':^10}  MATCH")
        print("  " + "-" * 75)

        if len(se) != len(si):
            print("  ! Количество токенов отличается — построчное сравнение невозможно")
            print(f"\n  external: {[t['form'] for t in se]}")
            print(f"  internal: {[t['form'] for t in si]}")
            continue

        for t_idx, (te, ti) in enumerate(zip(se, si), 1):
            form_match = "✓" if te["form"] == ti["form"] else "✗"
            upos_match = "✓" if te["upos"] == ti["upos"] else "✗"
            print(f"  {t_idx:>3}  {te['form']:^20}  {ti['form']:^20}  "
                  f"{te['upos']:^10}  {ti['upos']:^10}  "
                  f"form={form_match} upos={upos_match}")

def _print_conllu(input_texts: list, results: list) -> None:
    """
    Выводит CoNLL-U таблицу для списка предложений.
    Поле MISC выводится отдельной строкой под токеном, если не '_'.
    """
    for i, sent_tokens in enumerate(results, 1):
        print(f"\n  text: {input_texts[i - 1]!r}")
        header = (
            f"  {'ID':>4}  {'FORM':<16}  {'LEMMA':<16}  {'UPOS':<7}  "
            f"{'XPOS':<6}  {'FEATS':<5}  {'HEAD':<5}  {'DEPREL':<10}  {'DEPS':<5}"
        )
        print(header)
        print("  " + "-" * 75)
        for tok in sent_tokens:
            misc = tok.get("misc", "_")
            print(
                f"  {tok['id']:>4}  {tok['form']:<16}  {tok['lemma']:<16}  "
                f"{tok['upos']:<7}  {tok.get('xpos', '_'):<6}  "
                f"{tok.get('feats', '_'):<5}  {str(tok.get('head', '_')):<5}  "
                f"{tok.get('deprel', '_'):<10}  {tok.get('deps', '_'):<5}"
            )
            if misc != "_":
                print(f"          misc: {misc}")
