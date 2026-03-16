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
                    return self.service.parse_sentence_chunk.remote(
                        chunks[0], output_format=output_format
                    )
                chunk_results = list(self.service.parse_sentence_chunk.map(
                    chunks, kwargs={"output_format": output_format}
                ))
            else:  # internal
                if len(chunks) == 1:
                    return self.service.parse_sentence_chunk_native.remote(
                        chunks[0], output_format=output_format
                    )
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

def _print_conllu(result: List[List[Any]], title: str = "") -> None:
    """Выводит результат в conllu-формате по образцу trankit_wrapper."""
    if title:
        print(f"\n{title}")
    for sent_idx, sent in enumerate(result, 1):
        if not sent:
            continue
        print(f"\n# text = {' '.join(t['form'] for t in sent)}")
        print(
            f"  {'ID':<4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<7} "
            f"{'XPOS':<5} {'HEAD':<5} {'DEPREL':<10} {'DEPS':<5}"
        )
        print("  " + "─" * 90)
        for t in sent:
            print(
                f"  {t['id']:<4} {t['form']:<16} {t['lemma']:<16} "
                f"{t['upos']:<7} {t.get('xpos', '_'):<5} "
                f"{t.get('head', '_'):<5} {t.get('deprel', '_'):<10} "
                f"{t.get('deps', '_'):<5}"
            )
            if t.get("misc", "_") != "_":
                print(f"    misc: {t['misc']}")

def _print_native(result: List[List[Any]], title: str = "") -> None:
    """Выводит результат в native-формате (все поля Mystem)."""
    if title:
        print(f"\n{title}")
    for sent_idx, sent in enumerate(result, 1):
        if not sent:
            continue
        print(f"\n# text = {' '.join(t['text'] for t in sent)}")
        for t in sent:
            print(f"  Token: {t['text']}")
            variants = t.get("analysis") or []
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
                extra_str = (", " + ", ".join(extra)) if extra else ""
                print(f"      [{j}] lex={lex}, gr={gr}{extra_str}")

# ─── __main__: тест через wrapper (с chunking) ───────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    ap = argparse.ArgumentParser(description="MystemParser wrapper тест")
    ap.add_argument(
        "--tokenizer", choices=["external", "internal"], default="external",
        help="Путь токенизации (default: external)"
    )
    ap.add_argument(
        "--output-format", choices=["conllu", "native"], default="conllu",
        dest="output_format", help="Формат вывода (default: conllu)"
    )
    ap.add_argument(
        "--batch-size", type=int, default=default_batch_size, dest="batch_size",
        help=f"Предложений на чанк (default: {default_batch_size})"
    )
    args = ap.parse_args()

    sep = "=" * 72

    # ── Проверка доступности Modal ────────────────────────────────────────────
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

    text_single = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    text_multi = (
        "Зло, которым ты меня пугаешь, вовсе не так зло. "
        "Москва — столица России. "
        "Кружка-термос стоит 500р."
    )

    # ── Вариант 1: conllu + external ─────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 1: conllu + external (razdel.tokenize в modal)")
    print(sep)
    res_1 = parser.parse_text(
        text_single,
        output_format="conllu",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    _print_conllu(res_1)
    print(f"\nКлючи токена: {list(res_1[0][0].keys()) if res_1 else '—'}")

    # ── Вариант 2: native + external ─────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 2: native + external (razdel.tokenize в modal)")
    print(sep)
    res_2 = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    _print_native(res_2)
    print(f"\nКлючи токена: {list(res_2[0][0].keys()) if res_2 else '—'}")

    # ── Вариант 3: conllu + internal ─────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 3: conllu + internal (mystem токенизирует сам)")
    print(sep)
    res_3 = parser.parse_text(
        text_single,
        output_format="conllu",
        tokenizer="internal",
        batch_size=args.batch_size,
    )
    _print_conllu(res_3)

    # ── Вариант 4: native + internal ─────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 4: native + internal (mystem токенизирует сам)")
    print(sep)
    res_4 = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="internal",
        batch_size=args.batch_size,
    )
    _print_native(res_4)

    # ── parse_batch ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("BATCH: conllu + external (2 текста)")
    print(sep)
    batch_texts = [
        "Зло, которым ты меня пугаешь, вовсе не так зло.",
        "Москва — столица России. Петербург — культурная столица.",
    ]
    batch_results = parser.parse_batch(
        batch_texts,
        output_format="conllu",
        tokenizer="external",
        batch_size=args.batch_size,
    )
    for idx, (bt, br) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"\n── Текст {idx}: '{bt}'")
        _print_conllu(br)

    print(f"\n{'✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ':^72}")

    # ── Сравнение токенизаций: external vs internal ───────────────────────────
    print(f"\n{sep}")
    print("СРАВНЕНИЕ ТОКЕНИЗАЦИЙ: external vs internal (conllu)")
    print(sep)

    res_ext = parser.parse_text(
        text_multi, output_format="conllu", tokenizer="external",
        batch_size=args.batch_size,
    )
    res_int = parser.parse_text(
        text_multi, output_format="conllu", tokenizer="internal",
        batch_size=args.batch_size,
    )

    print(
        f"\n  {'Предл.':<8} {'#':>3}  "
        f"{'external form':<20} {'internal form':<20} "
        f"{'UPOS ext':<10} {'UPOS int':<10} match"
    )
    print("  " + "─" * 90)

    for s_idx, (s_e, s_i) in enumerate(zip(res_ext, res_int), 1):
        if len(s_e) != len(s_i):
            print(
                f"\n  ⚠️  Предложение {s_idx}: разное кол-во токенов "
                f"(external={len(s_e)}, internal={len(s_i)}) — "
                f"токенизации различаются:"
            )
            print(f"    external: {[t['form'] for t in s_e]}")
            print(f"    internal: {[t['form'] for t in s_i]}")
            continue
        for t_idx, (te, ti) in enumerate(zip(s_e, s_i), 1):
            form_match = "✅" if te["form"] == ti["form"] else "⚠️ "
            upos_match = "✅" if te["upos"] == ti["upos"] else "❌"
            print(
                f"  {s_idx:<8} {t_idx:>3}  "
                f"{te['form']:<20} {ti['form']:<20} "
                f"{te['upos']:<10} {ti['upos']:<10} "
                f"upos:{upos_match} form:{form_match}"
            )