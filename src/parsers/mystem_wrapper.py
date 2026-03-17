#!/usr/bin/env python3
"""
mystem_wrapper.py — тонкий клиент для Mystem (Modal).

Wrapper содержит ровно три обязанности:
    1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
    2. Маршрутизация чанков в нужный Modal-метод (external / internal ветка).
    3. Сборка результатов чанков в единый ответ.

Вся NLP-логика (токенизация, морфоанализ, форматирование) — в mystem_modal.py.

Два пути (оба используют razdel.sentenize для сентенизации):
    external (внешняя сентенизация, razdel.tokenize в modal):
        sentenize → List[List[str]]
        → service.parse_sentence_chunk.map(chunks)
        Mystem получает предложения, токенизированные razdel внутри modal.

    internal (внешняя сентенизация, mystem токенизирует сам):
        sentenize → List[List[str]]
        → service.parse_sentence_chunk_native.map(chunks)
        Mystem сам режет предложение на токены.

Два формата вывода:
    conllu — CoNLL-U совместимые поля:
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
    native — полный нативный формат Mystem:
        id, text, analysis (список вариантов: lex, gr, wt, qual)

Использование:
    from mystem_wrapper import MystemParser
    parser = MystemParser()
    # 4 варианта (2 формата × 2 пути):
    result = parser.parse_text(text, output_format="conllu",  tokenizer="external")
    result = parser.parse_text(text, output_format="native",  tokenizer="external")
    result = parser.parse_text(text, output_format="conllu",  tokenizer="internal")
    result = parser.parse_text(text, output_format="native",  tokenizer="internal")
    # batch_size подбирается под нагрузку (по умолчанию 32):
    result = parser.parse_text(text, tokenizer="external", batch_size=16)
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Literal, TypedDict, TypeVar, Union, overload

import modal
from razdel import sentenize

# ─── Типы ─────────────────────────────────────────────────────────────────────
_T = TypeVar("_T")

OutputFormat = Literal["conllu", "native"]
TokenizerType = Literal["external", "internal"]

default_batch_size: int = 32


# ─── TypedDicts ───────────────────────────────────────────────────────────────
class TokenDictCoNLLU(TypedDict, total=False):
    """
    CoNLL-U совместимый формат токена.
    Mystem заполняет: id, form, lemma, upos, misc.
    Остальные поля — заглушки: mystem не предсказывает синтаксис.
    """
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str     # всегда «_»
    feats: str    # всегда «_»
    head: str     # всегда «_»
    deprel: str   # всегда «_»
    deps: str     # всегда «_»
    misc: str     # Gr=...|Wt=...|Qual=...|Analyses=N|Best=0


class TokenDictNative(TypedDict, total=False):
    """
    Полный нативный формат токена Mystem.
    analysis — список всех гипотез разбора с полями:
        lex, gr, wt, qual
    """
    id: int
    text: str
    analysis: List[Dict[str, Any]]


# ─── MystemParser ─────────────────────────────────────────────────────────────
class MystemParser:
    """
    Тонкий клиент для Mystem (Modal).
    - Сентенизация через razdel.sentenize (всегда, до отправки в Modal).
    - Разбивка предложений на чанки по batch_size.
    - Маршрутизация: external → parse_sentence_chunk,
                     internal → parse_sentence_chunk_native.
    - Сборка результатов через _merge_chunks.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-mystem", "MystemService"
            )()
            self.logger.info("✓ Connected to Mystem via Modal.")
        except Exception as exc:
            self.logger.error(f"❌ Failed to connect to Modal: {exc}")
            raise

    # ─── Chunking ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_to_sentence_chunks(
        text: str,
        batch_size: int,
    ) -> List[List[str]]:
        """
        Разбивает текст на чанки предложений (только тексты, без офсетов).
        Используется для обоих путей (external / internal).

        Args:
            text:       входной текст
            batch_size: количество предложений на чанк
        Returns:
            List[List[str]]
        Raises:
            ValueError: если batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        sentences = list(sentenize(text))
        if not sentences:
            return []
        return [
            [s.text for s in sentences[i:i + batch_size]]
            for i in range(0, len(sentences), batch_size)
        ]

    @staticmethod
    def _merge_chunks(
        chunk_results: List[List[List[_T]]],
    ) -> List[List[_T]]:
        """
        Склеивает результаты чанков в единый список предложений.
        Args:
            chunk_results: каждый элемент — List[List[Dict]] одного чанка
        Returns:
            List[List[Dict]] — все предложения в порядке следования
        """
        return [sent for chunk in chunk_results for sent in chunk]

    # ─── parse_text ───────────────────────────────────────────────────────────

    @overload
    def parse_text(
        self,
        text: str,
        tokenizer: TokenizerType = ...,
        output_format: Literal["conllu"] = ...,
        batch_size: int = ...,
    ) -> List[List[TokenDictCoNLLU]]: ...

    @overload
    def parse_text(
        self,
        text: str,
        tokenizer: TokenizerType = ...,
        output_format: Literal["native"] = ...,
        batch_size: int = ...,
    ) -> List[List[TokenDictNative]]: ...

    def parse_text(
        self,
        text: str,
        tokenizer: str = "external",
        output_format: str = "conllu",
        batch_size: int = default_batch_size,
    ) -> List[List[Union[TokenDictCoNLLU, TokenDictNative]]]:
        """
        Парсит текст через Mystem в Modal.

        Алгоритм:
            1. razdel.sentenize → чанки по batch_size.
            2. Один чанк → .remote(); несколько → .map().
            3. _merge_chunks → единый список предложений.

        Args:
            text:          входной текст
            output_format: «conllu» | «native»
            tokenizer:     «external» (razdel в modal) | «internal» (mystem)
            batch_size:    предложений на чанк (default: 32)
        Returns:
            List[List[Dict]]
        """
        try:
            if tokenizer not in ("external", "internal"):
                raise ValueError(
                    f"Unknown tokenizer '{tokenizer}'. "
                    f"Expected 'external' or 'internal'."
                )
            if output_format not in ("conllu", "native"):
                raise ValueError(
                    f"Unknown output_format '{output_format}'. "
                    f"Expected 'conllu' or 'native'."
                )

            chunks = self._split_to_sentence_chunks(text, batch_size)
            if not chunks:
                return []

            if tokenizer == "external":
                if len(chunks) == 1:
                    raw = self.service.parse_sentence_chunk.remote(chunks[0], output_format=output_format)
                    return self._merge_chunks([raw])
                chunk_results = list(self.service.parse_sentence_chunk.map(
                    chunks, kwargs={"output_format": output_format}
                ))
                chunk_results = [[toks for toks, _ in chunk] for chunk in chunk_results]
            else:  # internal
                if len(chunks) == 1:
                    raw = self.service.parse_sentence_chunk_native.remote(chunks[0], output_format=output_format)
                    return self._merge_chunks([raw])
                chunk_results = list(self.service.parse_sentence_chunk_native.map(
                    chunks, kwargs={"output_format": output_format}
                ))

            return self._merge_chunks(chunk_results)

        except Exception as exc:
            self.logger.error(f"❌ parse_text error: {exc}")
            raise

    # ─── parse_batch ──────────────────────────────────────────────────────────

    @overload
    def parse_batch(
            self,
            texts: List[str],
            tokenizer: TokenizerType = ...,
            output_format: Literal["conllu"] = ...,
            batch_size: int = ...,
    ) -> List[List[List[TokenDictCoNLLU]]]:
        ...

    @overload
    def parse_batch(
            self,
            texts: List[str],
            tokenizer: TokenizerType = ...,
            output_format: Literal["native"] = ...,
            batch_size: int = ...,
    ) -> List[List[List[TokenDictNative]]]:
        ...

    def parse_batch(
            self,
            texts: List[str],
            tokenizer: str = "external",
            output_format: str = "conllu",
            batch_size: int = default_batch_size,
    ) -> List[List[List[Union[TokenDictCoNLLU, TokenDictNative]]]]:
        """
        Пакетная обработка нескольких текстов единым .map().

        Алгоритм:
            1. Разбить каждый текст на чанки, запомнить кол-во (chunks_per_text).
            2. Объединить все чанки в один список all_chunks.
            3. Один .map() — Modal распределяет по воркерам.
            4. Восстановить результаты по текстам через chunks_per_text.

        Args:
            texts:         список входных текстов
            output_format: «conllu» | «native»
            tokenizer:     «external» | «internal»
            batch_size:    предложений на чанк
        Returns:
            List[List[List[Dict]]] — результат для каждого текста
        """
        try:
            if tokenizer not in ("external", "internal"):
                raise ValueError(
                    f"Unknown tokenizer '{tokenizer}'. "
                    f"Expected 'external' or 'internal'."
                )
            if output_format not in ("conllu", "native"):
                raise ValueError(
                    f"Unknown output_format '{output_format}'. "
                    f"Expected 'conllu' or 'native'."
                )

            chunks_per_text: List[int] = []
            all_chunks: List[List[str]] = []

            for text in texts:
                text_chunks = self._split_to_sentence_chunks(text, batch_size)
                chunks_per_text.append(len(text_chunks))
                all_chunks.extend(text_chunks)

            if not all_chunks:
                return [[] for _ in texts]

            if tokenizer == "external":
                if len(all_chunks) == 1:
                    all_results = [
                        self.service.parse_sentence_chunk.remote(
                            all_chunks[0], output_format=output_format
                        )
                    ]
                else:
                    all_results = list(self.service.parse_sentence_chunk.map(
                        all_chunks, kwargs={"output_format": output_format}
                    ))
                all_results = [[toks for toks, _ in chunk] for chunk in all_results]
            else:  # internal
                if len(all_chunks) == 1:
                    all_results = [
                        self.service.parse_sentence_chunk_native.remote(
                            all_chunks[0], output_format=output_format
                        )
                    ]
                else:
                    all_results = list(self.service.parse_sentence_chunk_native.map(
                        all_chunks, kwargs={"output_format": output_format}
                    ))

            # Восстанавливаем результаты по текстам
            results: List[List[List[Any]]] = []
            offset = 0
            for n_chunks in chunks_per_text:
                results.append(
                    self._merge_chunks(all_results[offset:offset + n_chunks])
                )
                offset += n_chunks
            return results

        except Exception as exc:
            self.logger.error(f"❌ parse_batch error: {exc}")
            raise

# ─── Вспомогательные функции вывода ──────────────────────────────────────────
def _print_conllu(input_texts: list, results: list) -> None:
    """
    Выводит результат в conllu-формате по образцу trankit_wrapper.
    """
    for i, sent_tokens in enumerate(results, 1):
        print(f"\n  text: {input_texts[i - 1]!r}")
        header = (
            f"    {'ID':>4}  {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
            f"{'XPOS':<6} {'FEATS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5}"
        )
        print(header)
        print("  " + "-" * 75)
        for tok in sent_tokens:
            misc = tok.get("misc", "_")
            print(
                f"    {tok['id']:>4}  {tok['form']:<16} {tok['lemma']:<16} "
                f"{tok['upos']:<7} {tok.get('xpos', '_'):<6} "
                f"{tok.get('feats', '_'):<5} {str(tok.get('head', '_')):<5} "
                f"{tok.get('deprel', '_'):<10} {tok.get('deps', '_'):<5}"
            )
            if misc != "_":
                print(f"         misc: {misc}")

def _print_native(input_texts: list, results: list) -> None:
    """Выводит результат в native-формате (все поля Mystem)."""
    for i, sent_tokens in enumerate(results, 1):
        print(f"\n  text (sent to mystem): {input_texts[i - 1]!r}")
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
                    if wt   != "": extra.append(f"wt={wt}")
                    if qual != "": extra.append(f"qual={qual}")
                    extra_str = ", ".join(extra)
                    print(f"           {j}: lex={lex!r}  gr={gr!r}"
                          + (f"  [{extra_str}]" if extra_str else ""))


# ─── __main__: тест через wrapper (с chunking) ────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    ap = argparse.ArgumentParser(description="MystemParser wrapper тест")
    ap.add_argument(
        "--tokenizer",
        choices=["external", "internal"],
        default="external",
        help="Путь токенизации (default: external)",
    )
    ap.add_argument(
        "--output-format",
        choices=["conllu", "native"],
        default="conllu",
        dest="output_format",
        help="Формат вывода (default: conllu)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        dest="batch_size",
        help=f"Предложений на чанк (default: {default_batch_size})",
    )
    args = ap.parse_args()

    sep = "=" * 70

    # ── Проверка доступности Modal ───────────────────────────────────────────
    print(sep)
    print("ПРОВЕРКА ДОСТУПНОСТИ MODAL-СЕРВИСА")
    print(sep)
    try:
        parser = MystemParser()
    except Exception as e:
        print(f"⚠️  Modal-сервис недоступен: {e}")
        print("\nЗапустите сервис командой:")
        print("  modal deploy src/parsers/mystem_modal.py")
        sys.exit(1)

    # Те же предложения, что и в modal local_entrypoint
    sentences = [
        "Мама мыла раму без мыла.",
        "Привет, как дела?",
        "Кружка-термос стоит — 500 рублей.",
        "Он сказал: «Не беспокойтесь».",
    ]
    # Тексты для вывода (что именно ушло в mystem)
    # external: razdel токенизирует → склеивает через пробел
    # internal: оригинальное предложение
    sent_compare = ["Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."]

    # ── 1. EXTERNAL (razdel) → NATIVE ────────────────────────────────────────
    from razdel import tokenize as razdel_tokenize

    print(sep)
    print("Mystem EXTERNAL (tokenizer: razdel) → NATIVE")
    print(sep)
    input_texts_ext = []
    for s in sentences:
        toks = list(razdel_tokenize(s))
        input_texts_ext.append(" ".join(t.text for t in toks if t.text.strip()))

    ext_native = parser.parse_text(
        "\n".join(sentences),
        output_format="native",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    # parse_text (external) возвращает List[Tuple[tokens, input_text]]
    for sent_tokens, input_text in zip(ext_native, input_texts_ext):
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
                    if wt   != "": extra.append(f"wt={wt}")
                    if qual != "": extra.append(f"qual={qual}")
                    extra_str = ", ".join(extra)
                    print(
                        f"           {j}: lex={lex!r}  gr={gr!r}"
                        + (f"  [{extra_str}]" if extra_str else "")
                    )

    # ── 2. EXTERNAL (razdel) → CONLLU ────────────────────────────────────────
    print("\n" + sep)
    print("Mystem EXTERNAL (tokenizer: razdel) → CONLLU")
    print(sep)
    ext_conllu = parser.parse_text(
        "\n".join(sentences),
        output_format="conllu",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    ext_input_texts = [text for _, text in ext_conllu]
    ext_tokens      = [toks for toks, _ in ext_conllu]
    _print_conllu(input_texts_ext, ext_conllu)

    # ── 3. INTERNAL (mystem) → NATIVE ────────────────────────────────────────
    print("\n" + sep)
    print("Mystem INTERNAL (tokenizer: mystem) → NATIVE")
    print(sep)
    int_native = parser.parse_text(
        "\n".join(sentences),
        output_format="native",
        tokenizer="internal",
        batch_size=args.batch_size,
    )
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
                    if wt   != "": extra.append(f"wt={wt}")
                    if qual != "": extra.append(f"qual={qual}")
                    extra_str = ", ".join(extra)
                    print(
                        f"           {j}: lex={lex!r}  gr={gr!r}"
                        + (f"  [{extra_str}]" if extra_str else "")
                    )

    # ── 4. INTERNAL (mystem) → CONLLU ────────────────────────────────────────
    print("\n" + sep)
    print("Mystem INTERNAL (tokenizer: mystem) → CONLLU")
    print(sep)
    int_conllu = parser.parse_text(
        "\n".join(sentences),
        output_format="conllu",
        tokenizer="internal",
        batch_size=args.batch_size,
    )
    _print_conllu(sentences, int_conllu)

    # ── 5. EXTERNAL vs INTERNAL — сравнение ──────────────────────────────────
    print("\n" + sep)
    print("EXTERNAL vs INTERNAL — сравнение")
    print(sep)
    ext_cmp_raw = parser.parse_text(
        sent_compare[0],
        output_format="conllu",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    ext_cmp = [toks for toks, _ in ext_cmp_raw]
    int_cmp = parser.parse_text(
        sent_compare[0],
        output_format="conllu",
        tokenizer="internal",
        batch_size=args.batch_size,
    )

    for s_idx, (se, si) in enumerate(zip(ext_cmp, int_cmp), 1):
        print(f"\n  Sentence {s_idx}: {sent_compare[s_idx - 1]!r}")
        match_icon = "✓" if len(se) == len(si) else "✗"
        print(f"  Tokens: external={len(se)}, internal={len(si)}  {match_icon}")
        print(
            f"\n  {'':>3} {'EXTERNAL':^20} {'INTERNAL':^20} "
            f"{'UPOS_EXT':^10} {'UPOS_INT':^10}  MATCH"
        )
        print("  " + "-" * 75)
        if len(se) != len(si):
            print("  ! Количество токенов отличается — построчное сравнение невозможно")
            print(f"\n  external: {[t['form'] for t in se]}")
            print(f"  internal: {[t['form'] for t in si]}")
            continue
        for t_idx, (te, ti) in enumerate(zip(se, si), 1):
            form_match = "✓" if te["form"] == ti["form"] else "✗"
            upos_match = "✓" if te["upos"] == ti["upos"] else "✗"
            print(
                f"  {t_idx:>3}  {te['form']:^20} {ti['form']:^20} "
                f"{te['upos']:^10} {ti['upos']:^10} "
                f"form={form_match} upos={upos_match}"
            )
