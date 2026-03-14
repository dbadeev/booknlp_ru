#!/usr/bin/env python3
"""
trankit_wrapper.py — тонкий клиент для Trankit (Modal).

Wrapper содержит ровно три обязанности:
  1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
  2. Маршрутизация чанков в нужный Modal-метод (razdel / native ветка).
  3. Сборка результатов чанков в единый ответ.

Вся NLP-логика (токенизация, pipeline, форматирование) — в trankit_modal.py.

Два пути (оба используют razdel.sentenize для сентенизации):

  razdel (внешняя сентенизация + символьные офсеты):
      sentenize → List[List[(text, start_char)]]
      → service.parse_sentence_chunk.map(chunks)
      Trankit получает is_sent=True: пропускает внутреннюю сентенизацию.
      Токены содержат глобальные офсеты (start_char / dspan) относительно
      исходного документа.

  native (внешняя сентенизация, без офсетов):
      sentenize → List[List[str]]
      → service.parse_sentence_chunk_native.map(chunks)
      Trankit получает is_sent=True: пропускает внутреннюю сентенизацию.
      Офсеты токенов — относительны начала каждого предложения.

Два формата вывода:
  simplified — CoNLL-U совместимые поля:
               id, form, lemma, upos, xpos, feats, head, deprel,
               deps, misc, start_char, end_char
  native     — полный нативный формат Trankit:
               id, text, lemma, upos, xpos, feats, head, deprel,
               span, dspan, ner, expanded, lang

Использование:
    from trankit_wrapper import TrankitParser

    parser = TrankitParser()

    # 4 варианта (2 формата × 2 пути):
    result = parser.parse_text(text, output_format="simplified", tokenizer="razdel")
    result = parser.parse_text(text, output_format="native",     tokenizer="razdel")
    result = parser.parse_text(text, output_format="simplified", tokenizer="native")
    result = parser.parse_text(text, output_format="native",     tokenizer="native")

    # chunk_size подбирается под GPU и тип текста (по умолчанию 32):
    result = parser.parse_text(text, tokenizer="razdel", chunk_size=16)
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Literal, Tuple, TypedDict, TypeVar, Union, overload

import modal
from razdel import sentenize

# ─── Типы ─────────────────────────────────────────────────────────────────────
_T = TypeVar("_T")

OutputFormat  = Literal["simplified", "native"]
TokenizerType = Literal["razdel", "native"]

# Предложений на один чанк: подбирается под GPU и средний размер предложений.
# T4 (16 ГБ): для коротких предложений можно увеличить до 64,
# для длинных (>50 слов) уменьшить до 8–16.
default_chunk_size: int = 32


# ─── TypedDicts для аннотаций ──────────────────────────────────────────────────

class TokenDictSimplified(TypedDict, total=False):
    """
    CoNLL-U совместимый формат токена (simplified).

    Все 10 стандартных CoNLL-U колонок + start_char / end_char.
    Поля deps и misc Trankit не предсказывает — заполняются "_".
    xpos для русского языка Trankit не предсказывает — всегда "_".
    """
    id:         int     # Номер токена в предложении (1-based)
    form:       str     # Текстовая форма токена
    lemma:      str     # Лемма
    upos:       str     # Universal POS tag
    xpos:       str     # Language-specific POS (для русского — "_")
    feats:      str     # Морфологические признаки (или "_")
    head:       int     # Индекс головы (0 = root)
    deprel:     str     # Тип синтаксической связи
    deps:       str     # Enhanced Dependencies ("_" — Trankit не поддерживает)
    misc:       str     # SpaceAfter и пр. ("_" — Trankit не поддерживает)
    start_char: int     # Начало токена в исходном документе (razdel path)
    end_char:   int     # Конец токена в исходном документе (razdel path)


class TokenDictNative(TypedDict, total=False):
    """
    Полный нативный формат токена Trankit.

    span  — (start, end) sentence-local: позиция относительно начала предложения.
            Корректен при обоих путях.
    dspan — (start, end) зависит от пути:
            razdel path (tokenizer="razdel"): глобальные офсеты = span + char_offset.
            native path (tokenizer="native"): sentence-local = span (char_offset=0).
            Для глобальных позиций используйте только tokenizer="razdel".
    """
    id:       Union[int, List[int]]  # int или [start, end] для MWT
    text:     str                    # Текстовая форма токена
    lemma:    str                    # Лемма
    upos:     str                    # Universal POS tag
    xpos:     str                    # Language-specific POS (для русского — "_")
    feats:    str                    # Морфологические признаки
    head:     int                    # Индекс головы
    deprel:   str                    # Тип синтаксической связи
    span:     Tuple[int, int]        # Sentence-local офсеты
    dspan:    Tuple[int, int]        # razdel path: глобальные; native path: sentence-local
    ner:      str                    # NER-тег (BIO/BIOES: B-PER, O и т.д.)
    expanded: List[Dict[str, Any]]   # MWT: список словарей под-токенов
    lang:     str                    # Язык предложения (например, "russian")


# ─── TrankitParser ────────────────────────────────────────────────────────────

class TrankitParser:
    """
    Тонкий клиент для Trankit (Modal).

    Обязанности:
    - Сентенизация через razdel.sentenize (всегда, до отправки в Modal).
    - Разбивка предложений на чанки по chunk_size.
    - Маршрутизация чанков в parse_sentence_chunk / parse_sentence_chunk_native.
    - Сборка результатов.

    Никакого форматирования, никакой морфологии, никакого вывода токенов.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-trankit", "TrankitService"
            )()
            self.logger.info("✓ Connected to Trankit via Modal.")
        except Exception as exc:
            self.logger.error(f"❌ Failed to connect to Modal: {exc}")
            raise

    # ─── Chunking ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_to_chunks(
        text: str,
        chunk_size: int,
        base_offset: int = 0,
    ) -> List[List[Tuple[str, int]]]:
        """
        Razdel path: разбивает текст на чанки с символьными офсетами.

        Каждый чанк — список пар (sentence_text, start_char_in_original).
        start_char передаётся в Modal для вычисления глобальных позиций токенов.

        base_offset — глобальное смещение начала text в документе.
                      В parse_batch всегда 0: офсеты токенов относительны
                      начала каждого текста, а не позиции текста в батче.

        Args:
            text: входной текст
            chunk_size: количество предложений на чанк
            base_offset: смещение начала text в исходном документе
        Returns:
            List[List[Tuple[str, int]]] — чанки с офсетами
        Raises:
            ValueError: если chunk_size <= 0
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        if not sentences:
            return []
        return [
            [
                (s.text, base_offset + s.start)
                for s in sentences[i:i + chunk_size]
            ]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _split_to_sentence_chunks(
        text: str,
        chunk_size: int,
    ) -> List[List[str]]:
        """
        Native path: разбивает текст на чанки предложений (только тексты).

        Офсеты не передаются: токены будут иметь позиции относительно
        начала каждого предложения (char_offset=0 на стороне Modal).

        Args:
            text: входной текст
            chunk_size: количество предложений на чанк
        Returns:
            List[List[str]] — чанки текстов предложений
        Raises:
            ValueError: если chunk_size <= 0
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        if not sentences:
            return []
        return [
            [s.text for s in sentences[i:i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(
            chunk_results: List[List[List[_T]]],
    ) -> List[List[_T]]:
        """
        Склеивает результаты чанков в единый список предложений.

        Каждый чанк возвращает List[List[Dict]] (список предложений).
        Merge = плоский список всех предложений из всех чанков.

        Args:
            chunk_results: список результатов чанков (каждый — список предложений)
        Returns:
            List[List[Dict]] — все предложения в порядке следования
        """
        return [sent for chunk in chunk_results for sent in chunk]

    # ─── Public API ───────────────────────────────────────────────────────────
    @overload
    def parse_text(
            self,
            text: str,
            tokenizer: TokenizerType = ...,
            output_format: Literal["simplified"] = ...,
            chunk_size: int = ...,
    ) -> List[List[TokenDictSimplified]]: ...

    @overload
    def parse_text(
            self,
            text: str,
            tokenizer: TokenizerType = ...,
            output_format: Literal["native"] = ...,
            chunk_size: int = ...,
    ) -> List[List[TokenDictNative]]: ...

    def parse_text(
            self,
            text: str,
            tokenizer: str = "razdel",
            output_format: str = "simplified",
            chunk_size: int = default_chunk_size,
    ) -> List[List[Union[TokenDictSimplified, TokenDictNative]]]:
        """
        Парсит текст через Trankit в Modal.

        Алгоритм:
        1. Разбить текст на предложения (razdel.sentenize).
        2. Сгруппировать в чанки по chunk_size.
        3. Один чанк → .remote(); несколько → .map() (параллельно).
        4. Склеить результаты через _merge_chunks.

        Args:
            text:          входной текст
            output_format: "simplified" (CoNLL-U поля) | "native" (полный Trankit)
            tokenizer:     "razdel" — с глобальными офсетами |
                           "native" — офсеты sentence-local
            chunk_size:    предложений на чанк (подбирается под GPU).
                           По умолчанию default_chunk_size = 32.
        Returns:
            List[List[Dict]] — список предложений, каждое — список токенов
        """
        try:
            if tokenizer not in ("razdel", "native"):
                raise ValueError(
                    f"Unknown tokenizer '{tokenizer}'. Expected 'razdel' or 'native'."
                )
            if output_format not in ("simplified", "native"):
                raise ValueError(
                    f"Unknown output_format '{output_format}'. "
                    f"Expected 'simplified' or 'native'."
                )
            if tokenizer == "razdel":
                # Каждый текст обрабатывается независимо (base_offset=0):
                # start_char токенов относителен начала своего текста, не батча.
                chunks = self._split_to_chunks(text, chunk_size, base_offset=0)
                if not chunks:
                    return []
                # Один чанк: .remote() дешевле (нет накладных расходов .map())
                if len(chunks) == 1:
                    return self.service.parse_sentence_chunk.remote(
                        chunks[0], output_format=output_format
                    )
                # Несколько чанков: .map() — параллельная обработка в Modal
                chunk_results = list(self.service.parse_sentence_chunk.map(
                    chunks, kwargs={"output_format": output_format}
                ))
                return self._merge_chunks(chunk_results)

            else:  # tokenizer == "native"
                chunks_native = self._split_to_sentence_chunks(text, chunk_size)
                if not chunks_native:
                    return []
                if len(chunks_native) == 1:
                    return self.service.parse_sentence_chunk_native.remote(
                        chunks_native[0], output_format=output_format
                    )
                chunk_results = list(self.service.parse_sentence_chunk_native.map(
                    chunks_native, kwargs={"output_format": output_format}
                ))
                return self._merge_chunks(chunk_results)

        except Exception as exc:
            self.logger.error(f"❌ parse_text error: {exc}")
            raise

    @overload
    def parse_batch(
            self,
            texts: List[str],
            tokenizer: TokenizerType = ...,  # было: str
            output_format: Literal["simplified"] = ...,
            chunk_size: int = ...,
    ) -> List[List[List[TokenDictSimplified]]]:
        ...

    @overload
    def parse_batch(
            self,
            texts: List[str],
            tokenizer: TokenizerType = ...,  # было: str
            output_format: Literal["native"] = ...,
            chunk_size: int = ...,
    ) -> List[List[List[TokenDictNative]]]:
        ...

    def parse_batch(
            self,
            texts: List[str],
            tokenizer: str = "razdel",
            output_format: str = "simplified",
            chunk_size: int = default_chunk_size,
    ) -> List[List[List[Union[TokenDictSimplified, TokenDictNative]]]]:
        """
        Пакетная обработка нескольких текстов единым .map().

        Алгоритм:
        1. Разбить каждый текст на чанки, запомнить кол-во чанков (chunks_per_text).
        2. Объединить все чанки всех текстов в один список all_chunks.
        3. Один .map() по all_chunks — Modal распределяет по контейнерам.
        4. Восстановить результаты по текстам через chunks_per_text.

        Args:
            texts:         список входных текстов
            output_format: "simplified" | "native"
            tokenizer:     "razdel" | "native"
            chunk_size:    предложений на чанк
        Returns:
            List[результат для каждого текста] —
            каждый результат: List[List[Dict]] (список предложений)
        """
        try:
            # chunks_per_text[i] — количество чанков для texts[i];
            # нужно для восстановления границ текстов после единого .map()
            chunks_per_text: List[int] = []

            if tokenizer not in ("razdel", "native"):
                raise ValueError(
                    f"Unknown tokenizer '{tokenizer}'. Expected 'razdel' or 'native'."
                )
            if output_format not in ("simplified", "native"):
                raise ValueError(
                    f"Unknown output_format '{output_format}'. "
                    f"Expected 'simplified' or 'native'."
                )
            if tokenizer == "razdel":
                all_chunks: List[List[Tuple[str, int]]] = []
                for text in texts:
                    # Каждый текст обрабатывается независимо (base_offset=0):
                    # start_char токенов относителен начала своего текста, не батча.
                    text_chunks = self._split_to_chunks(text, chunk_size, base_offset=0)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks.extend(text_chunks)

                if not all_chunks:
                    return [[] for _ in texts]

                # ── оптимизация: один чанк не требует .map() ──
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

            else:  # native
                all_chunks_native: List[List[str]] = []
                for text in texts:
                    text_chunks = self._split_to_sentence_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks_native.extend(text_chunks)

                if not all_chunks_native:
                    return [[] for _ in texts]

                if len(all_chunks_native) == 1:
                    all_results = [
                        self.service.parse_sentence_chunk_native.remote(
                            all_chunks_native[0], output_format=output_format
                        )
                    ]
                else:
                    all_results = list(self.service.parse_sentence_chunk_native.map(
                        all_chunks_native, kwargs={"output_format": output_format}
                    ))

            # Восстанавливаем результаты по текстам:
            # all_results[offset : offset + n_chunks] — чанки текста i
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

def _print_simplified(result: List[List[Any]], title: str = "") -> None:
    """Выводит результат в simplified (CoNLL-U) формате."""
    if title:
        print(f"\n{title}")
    for sent_idx, sent in enumerate(result, 1):
        if not sent:
            continue
        print(f"\n  Предложение {sent_idx}:")
        print(
            f"  {'ID':<4} {'FORM':<14} {'LEMMA':<14} {'UPOS':<7} {'XPOS':<5} "
            f"{'HEAD':<5} {'DEPREL':<12} {'DEPS':<5} {'MISC':<5} START END"
        )
        print("  " + "─" * 110)
        for t in sent:
            print(
                f"  {t['id']:<4} {t['form']:<14} {t['lemma']:<14} "
                f"{t['upos']:<7} {t.get('xpos', '_'):<5} "
                f"{t['head']:<5} {t['deprel']:<12} "
                f"{t.get('deps', '_'):<5} {t.get('misc', '_'):<5} "
                f"{t.get('start_char', 0)} {t.get('end_char', 0)}"
            )
            if t.get("feats", "_") not in ("_", "", None):
                print(f"       feats: {t['feats']}")


def _print_native(result: List[List[Any]], title: str = "") -> None:
    """Выводит результат в native-формате (все поля Trankit)."""
    if title:
        print(f"\n{title}")
    for sent_idx, sent in enumerate(result, 1):
        if not sent:
            continue
        print(f"\n  Предложение {sent_idx}:")
        print(
            f"  {'ID':<4} {'TEXT':<14} {'LEMMA':<14} {'UPOS':<7} "
            f"{'HEAD':<5} {'DEPREL':<12} span           dspan          "
            f"{'NER':<8} LANG"
        )
        print("  " + "─" * 120)
        for t in sent:
            print(
                f"  {str(t.get('id', '')):<4} {t.get('text', ''):<14} "
                f"{t.get('lemma', ''):<14} {t.get('upos', ''):<7} "
                f"{t.get('head', ''):<5} {t.get('deprel', ''):<12} "
                f"{str(t.get('span', '')):<15} {str(t.get('dspan', '')):<15} "
                f"{t.get('ner', 'O'):<8} {t.get('lang', '')}"
            )
            if t.get("feats", "_") not in ("_", "", None):
                print(f"       feats: {t['feats']}")
            if t.get("expanded"):
                print(f"       expanded (MWT): {t['expanded']}")


# ─── __main__: тест через wrapper (с chunking) ───────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    ap = argparse.ArgumentParser(description="TrankitParser wrapper тест")
    ap.add_argument(
        "--tokenizer", choices=["razdel", "native"], default="razdel",
        help="Путь сентенизации (default: razdel)"
    )
    ap.add_argument(
        "--output-format", choices=["simplified", "native"], default="simplified",
        dest="output_format", help="Формат вывода (default: simplified)"
    )
    ap.add_argument(
        "--chunk-size", type=int, default=default_chunk_size, dest="chunk_size",
        help=f"Предложений на чанк (default: {default_chunk_size})"
    )
    args = ap.parse_args()

    sep = "=" * 72

    # ── Проверка доступности Modal ────────────────────────────────────────────
    print(sep)
    print("ПРОВЕРКА ДОСТУПНОСТИ MODAL-СЕРВИСА")
    print(sep)
    try:
        parser = TrankitParser()
    except Exception as e:
        print(f"⚠️  Modal-сервис недоступен: {e}")
        print("\nЗапустите сервис командой:")
        print("  modal deploy src/parsers/trankit_modal.py")
        sys.exit(1)

    text_single = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    text_multi = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься. "
        "Москва — столица России. "
        "Кружка-термос стоит 500р."
    )

    # ── Вариант 1: simplified + razdel ───────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 1: simplified + razdel (глобальные офсеты)")
    print(sep)
    res_1 = parser.parse_text(
        text_single,
        output_format="simplified",
        tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    _print_simplified(res_1)
    print(f"\nКлючи токена: {list(res_1[0][0].keys()) if res_1 else '—'}")

    # ── Вариант 2: native + razdel ────────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 2: native + razdel (глобальные офсеты)")
    print(sep)
    res_2 = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    _print_native(res_2)
    print(f"\nКлючи токена: {list(res_2[0][0].keys()) if res_2 else '—'}")

    # ── Вариант 3: simplified + native ───────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 3: simplified + native (sentence-local офсеты)")
    print(sep)
    res_3 = parser.parse_text(
        text_single,
        output_format="simplified",
        tokenizer="native",
        chunk_size=args.chunk_size,
    )
    _print_simplified(res_3)

    # ── Вариант 4: native + native ────────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 4: native + native (sentence-local офсеты)")
    print(sep)
    res_4 = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="native",
        chunk_size=args.chunk_size,
    )
    _print_native(res_4)

    # ── Сравнение офсетов: razdel vs native ──────────────────────────────────
    print(f"\n{sep}")
    print("СРАВНЕНИЕ ОФСЕТОВ: razdel vs native (simplified, многопредложенный текст)")
    print(sep)
    res_razdel = parser.parse_text(
        text_multi, output_format="simplified", tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    res_native_tok = parser.parse_text(
        text_multi, output_format="simplified", tokenizer="native",
        chunk_size=args.chunk_size,
    )
    print(f"\n  {'Предл.':<8} {'Токен':<14} {'razdel start':<14} {'native start':<14}")
    print("  " + "─" * 54)
    for s_idx, (s_r, s_n) in enumerate(zip(res_razdel, res_native_tok), 1):
        for t_r, t_n in zip(s_r, s_n):
            match = "✅" if t_r["start_char"] == t_n["start_char"] else "⚠️"
            print(
                f"  {s_idx:<8} {t_r['form']:<14} "
                f"{t_r['start_char']:<14} {t_n['start_char']:<14} {match}"
            )

    # ── parse_batch ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("BATCH: simplified + razdel (2 текста)")
    print(sep)
    batch_texts = [
        "Зло, которым ты меня пугаешь, вовсе не так зло.",
        "Москва — столица России. Петербург — культурная столица.",
    ]
    batch_results = parser.parse_batch(
        batch_texts,
        output_format="simplified",
        tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    for idx, (bt, br) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"\n── Текст {idx}: '{bt}'")
        _print_simplified(br)

    print(f"\n{'✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ':^72}")

