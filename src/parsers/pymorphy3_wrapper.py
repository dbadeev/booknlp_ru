#!/usr/bin/env python3
"""
Локальный wrapper для Pymorphy3 (Modal).

Wrapper — тонкий клиент. Три обязанности:
  1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
  2. Маршрутизация чанков в Modal-сервис (.remote() / .map()).
  3. Склейка результатов чанков (_merge_chunks).

Вся морфология, форматирование, вывод — в pymorphy3_modal.py.

Два пути сентенизации (оба через razdel.sentenize):
  razdel (внешний, с офсетами):   → List[List[(text, start_char)]]
  native (внутренний, без офсетов): → List[List[str]]

Использование:
    from pymorphy3_wrapper import Pymorphy3Parser

    parser = Pymorphy3Parser()
    result = parser.parse_text(text, output_format="simplified", tokenizer="razdel")
    result = parser.parse_text(text, output_format="native",     tokenizer="razdel")
    result = parser.parse_text(text, output_format="simplified", tokenizer="native")
    result = parser.parse_text(text, output_format="native",     tokenizer="native")
    # chunk_size подбирается под GPU и тип текстов (по умолчанию 32):
    result = parser.parse_text(text, tokenizer="razdel", chunk_size=16)

Запуск тестов:
    python src/parsers/pymorphy3_wrapper.py
    python src/parsers/pymorphy3_wrapper.py --tokenizer native --output-format native --chunk-size 2
"""

import argparse
import logging
import sys
import modal
from razdel import sentenize
from typing import Any, Dict, List, Literal, Tuple

OutputFormat  = Literal["simplified", "native"]
TokenizerType = Literal["razdel", "native"]

default_chunk_size: int = 32  # предложений на чанк; подбирается под GPU и тип текстов

# ─── Pymorphy3Parser ──────────────────────────────────────────────────────────

class Pymorphy3Parser:
    """
    Клиент для Pymorphy3-сервиса (Modal).

    Поддерживает:
      - 2 формата вывода: simplified (CoNLL-подобный), native (pymorphy3)
      - 2 пути токенизации: razdel (с офсетами), native (без офсетов)

    Сентенизация выполняется локально (razdel.sentenize) до отправки в Modal.
    Чанкинг управляет памятью GPU (OOM prevention).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-pymorphy3", "Pymorphy3Service"
            )()
            self.logger.info("Pymorphy3Parser initialized (modal).")
        except Exception as exc:
            self.logger.error(f"Failed to connect to Modal: {exc}")
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

        Returns:
            List[List[(sentence_text, start_char_in_original)]]

        base_offset — смещение text внутри большего документа
                      (используется в parse_batch).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [(s.text, base_offset + s.start) for s in sentences[i:i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _split_to_sentence_chunks(
        text: str,
        chunk_size: int,
    ) -> List[List[str]]:
        """
        Native path: разбивает текст на чанки (только тексты предложений).

        Returns:
            List[List[str]]
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [s.text for s in sentences[i:i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(
        chunk_results: List[Any],
    ) -> List[List[Dict[str, Any]]]:
        """
        Склеивает результаты чанков в единый список предложений.
        Каждый chunk_result — List[List[Dict]] (список предложений).
        """
        return [sent for cr in chunk_results for sent in cr]

    # ─── Public API ───────────────────────────────────────────────────────────

    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "simplified",
        tokenizer: TokenizerType = "razdel",
        chunk_size: int = default_chunk_size,
    ) -> List[List[Dict[str, Any]]]:
        """
        Парсит текст через Pymorphy3Service в Modal.

        Алгоритм:
          1. Разбить текст на предложения (razdel.sentenize).
          2. Сгруппировать в чанки по chunk_size.
          3. Один чанк → .remote(); несколько → .map() (параллельно).
          4. Склеить результаты.

        Args:
            text:          Входной текст
            output_format: 'simplified' | 'native'
            tokenizer:     'razdel' (с офсетами) | 'native' (без офсетов)
            chunk_size:    Предложений на чанк (подбирается под GPU).
                           По умолчанию default_chunk_size = 32.
        Returns:
            List[List[Dict]]  — список предложений, каждое — список токенов
        """
        if output_format not in ("simplified", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")
        if tokenizer not in ("razdel", "native"):
            raise ValueError(f"Unknown tokenizer: {tokenizer!r}")

        try:
            if tokenizer == "razdel":
                chunks = self._split_to_chunks(text, chunk_size)
                if not chunks:
                    return []
                chunk_results = list(self.service.parse_sentence_chunk.map(
                    chunks, kwargs={"output_format": output_format}
                ))
            else:  # native
                chunks = self._split_to_sentence_chunks(text, chunk_size)
                if not chunks:
                    return []
                chunk_results = list(self.service.parse_sentence_chunk_native.map(
                    chunks, kwargs={"output_format": output_format}
                ))
            return self._merge_chunks(chunk_results)

        except Exception as exc:
            self.logger.error(f"Error during pymorphy3 parsing: {exc}")
            raise

    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "simplified",
        tokenizer: TokenizerType = "razdel",
        chunk_size: int = default_chunk_size,
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Разбивает все тексты на чанки и отправляет их единым .map() —
        Modal распределяет по доступным контейнерам.

        Args:
            texts:         Список текстов
            output_format: 'simplified' | 'native'
            tokenizer:     'razdel' | 'native'
            chunk_size:    Предложений на чанк
        Returns:
            List[результаты для каждого текста]
        """
        if output_format not in ("simplified", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")
        if tokenizer not in ("razdel", "native"):
            raise ValueError(f"Unknown tokenizer: {tokenizer!r}")

        try:
            chunks_per_text: List[int] = []

            if tokenizer == "razdel":
                all_chunks: List[List[Tuple[str, int]]] = []
                for text in texts:
                    text_chunks = self._split_to_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks.extend(text_chunks)
                if not all_chunks:
                    return [[] for _ in texts]
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
                all_results = list(self.service.parse_sentence_chunk_native.map(
                    all_chunks_native, kwargs={"output_format": output_format}
                ))

            results, offset = [], 0
            for n_chunks in chunks_per_text:
                results.append(self._merge_chunks(
                    all_results[offset:offset + n_chunks]
                ))
                offset += n_chunks
            return results

        except Exception as exc:
            self.logger.error(f"Error during batch parsing: {exc}")
            raise

# ─── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Тестирует Pymorphy3Parser (wrapper + Modal).

    Тест-секции:
      [1] Chunking (локально, без Modal)
          [1.1] _split_to_chunks: офсеты корректны
          [1.2] _split_to_sentence_chunks: только строки, без офсетов
          [1.3] Оба пути дают одинаковое число предложений
          [1.4] _merge_chunks: склейка корректна
          [1.5] Невалидный chunk_size → ValueError
          [1.6] Невалидный output_format → ValueError
          [1.7] Невалидный tokenizer → ValueError
      [2] parse_text — razdel path, simplified
      [3] parse_text — razdel path, native
      [4] parse_text — native path, simplified
      [5] parse_text — native path, native
      [6] parse_text — chunk_size=1 (каждое предложение отдельный чанк)
      [7] parse_text — пустой текст → []
      [8] parse_batch — razdel path
      [9] parse_batch — native path
      [10] parse_batch — результат совпадает с parse_text по одному
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    ap = argparse.ArgumentParser(description="Pymorphy3 wrapper тест")
    ap.add_argument("--tokenizer",     choices=["razdel", "native"],     default="razdel")
    ap.add_argument("--output-format", choices=["simplified", "native"], default="simplified",
                    dest="output_format")
    ap.add_argument("--chunk-size",    type=int, default=default_chunk_size, dest="chunk_size")
    args = ap.parse_args()

    sep = "=" * 72
    passed = 0
    failed = 0

    def ok(name: str):
        global passed
        passed += 1
        print(f"  ✅  {name}")

    def fail(name: str, err):
        global failed
        failed += 1
        print(f"  ❌  {name}: {err}")

    TEXT   = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    MULTI  = "Зло пугает. Москва — столица России. Крупнейший город страны."
    BATCH  = [TEXT, "Москва — столица.", "Лиса прыгает через забор."]

    # ── [1] Chunking (без Modal) ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("[1] Chunking (локально, без Modal)")
    print(sep)

    # [1.1] _split_to_chunks офсеты
    try:
        chunks = Pymorphy3Parser._split_to_chunks(MULTI, chunk_size=10)
        for chunk in chunks:
            for sent_text, start in chunk:
                assert MULTI[start:start + len(sent_text)] == sent_text, \
                    f"offset={start} → {MULTI[start:start+len(sent_text)]!r} ≠ {sent_text!r}"
        ok("[1.1] _split_to_chunks — офсеты корректны")
    except Exception as e:
        fail("[1.1] _split_to_chunks офсеты", e)

    # [1.2] _split_to_sentence_chunks — только строки
    try:
        sc = Pymorphy3Parser._split_to_sentence_chunks(MULTI, chunk_size=10)
        for chunk in sc:
            for item in chunk:
                assert isinstance(item, str), f"ожидался str, получено {type(item)}"
        ok("[1.2] _split_to_sentence_chunks — только строки, без офсетов")
    except Exception as e:
        fail("[1.2] _split_to_sentence_chunks", e)

    # [1.3] одинаковое число предложений в обоих путях
    try:
        rc = Pymorphy3Parser._split_to_chunks(MULTI, chunk_size=10)
        nc = Pymorphy3Parser._split_to_sentence_chunks(MULTI, chunk_size=10)
        r_total = sum(len(c) for c in rc)
        n_total = sum(len(c) for c in nc)
        assert r_total == n_total, f"razdel={r_total}, native={n_total}"
        ok(f"[1.3] Оба пути: {r_total} предложений — совпадают")
    except Exception as e:
        fail("[1.3] Число предложений", e)

    # [1.4] _merge_chunks
    try:
        fake_sent = [{"id": 1, "form": "слово"}]
        chunk_results = [[fake_sent, fake_sent], [fake_sent]]
        merged = Pymorphy3Parser._merge_chunks(chunk_results)
        assert len(merged) == 3, f"ожидалось 3, получено {len(merged)}"
        ok("[1.4] _merge_chunks — склейка корректна")
    except Exception as e:
        fail("[1.4] _merge_chunks", e)

    # [1.5] chunk_size=0 → ValueError
    try:
        try:
            Pymorphy3Parser._split_to_chunks("Текст.", chunk_size=0)
            fail("[1.5] chunk_size=0", "ValueError не выброшен")
        except ValueError as exc:
            print(f"  Поймано: {exc!r}")
            ok("[1.5] chunk_size=0 → ValueError")
    except Exception as e:
        fail("[1.5] chunk_size ValueError", e)

    # [1.6] невалидный output_format
    try:
        parser_local = object.__new__(Pymorphy3Parser)
        parser_local.logger = logging.getLogger("test")
        parser_local.service = None
        try:
            Pymorphy3Parser.parse_text(parser_local, "Текст.", output_format="conllu")
            fail("[1.6] output_format=conllu", "ValueError не выброшен")
        except ValueError as exc:
            print(f"  Поймано: {exc!r}")
            ok("[1.6] output_format неверный → ValueError")
    except Exception as e:
        fail("[1.6] output_format", e)

    # [1.7] невалидный tokenizer
    try:
        try:
            Pymorphy3Parser.parse_text(parser_local, "Текст.", tokenizer="spacy")
            fail("[1.7] tokenizer=spacy", "ValueError не выброшен")
        except ValueError as exc:
            print(f"  Поймано: {exc!r}")
            ok("[1.7] tokenizer неверный → ValueError")
    except Exception as e:
        fail("[1.7] tokenizer", e)

    # ── Инициализация parser (Modal) ──────────────────────────────────────────
    print(f"\n{sep}")
    print("Подключение к Modal...")
    print(sep)
    try:
        parser = Pymorphy3Parser()
    except Exception as e:
        print(f"\n⚠️  Modal-сервис недоступен: {e}")
        print("Запустите сервис: modal deploy src/parsers/pymorphy3_modal.py")
        print(f"\n── Локальные тесты: {passed} ✅  Modal-тесты: пропущены")
        sys.exit(1)

    # ── [2] parse_text — razdel, simplified ───────────────────────────────────
    print(f"\n{sep}")
    print(f"[2] parse_text  (razdel + simplified, chunk_size={args.chunk_size})")
    print(sep)
    try:
        result = parser.parse_text(TEXT, output_format="simplified",
                                   tokenizer="razdel", chunk_size=args.chunk_size)
        assert isinstance(result, list) and len(result) > 0
        for sent in result:
            assert all(k in t for t in sent for k in ("id", "form", "lemma", "upos", "head", "deprel"))
            roots = [t for t in sent if t["deprel"] == "root"]
            if roots:  # проверяем только если глагол найден
                bad = [t for t in sent if t["head"] == 0 and t["deprel"] == "dep"]
                assert bad == [], f"head=0 deprel=dep при наличии root: ..."
        sentences = list(sentenize(TEXT))
        for sent, s in zip(result, sentences):
            print_simplified(sent, s.text)
            print()
        ok("[2] parse_text razdel/simplified")
    except Exception as e:
        fail("[2] parse_text razdel/simplified", e)

    # ── [3] parse_text — razdel, native ───────────────────────────────────────
    print(f"\n{sep}")
    print(f"[3] parse_text  (razdel + native, chunk_size={args.chunk_size})")
    print(sep)
    try:
        result = parser.parse_text(TEXT, output_format="native",
                                   tokenizer="razdel", chunk_size=args.chunk_size)
        assert isinstance(result, list) and len(result) > 0
        for sent in result:
            for tok in sent:
                assert "word" in tok and "normal_form" in tok and "tag" in tok
                assert 0.0 <= tok["score"] <= 1.0
        for sent in result:
            _print_native(sent)
        ok("[3] parse_text razdel/native")
    except Exception as e:
        fail("[3] parse_text razdel/native", e)

    # ── [4] parse_text — native, simplified ───────────────────────────────────
    print(f"\n{sep}")
    print(f"[4] parse_text  (native + simplified, chunk_size={args.chunk_size})")
    print(sep)
    try:
        result = parser.parse_text(TEXT, output_format="simplified",
                                   tokenizer="native", chunk_size=args.chunk_size)
        assert isinstance(result, list) and len(result) > 0
        sentences = list(sentenize(TEXT))
        for sent, s in zip(result, sentences):
            print_simplified(sent, s.text)
            print()
        ok("[4] parse_text native/simplified")
    except Exception as e:
        fail("[4] parse_text native/simplified", e)

    # ── [5] parse_text — native, native ───────────────────────────────────────
    print(f"\n{sep}")
    print(f"[5] parse_text  (native + native, chunk_size={args.chunk_size})")
    print(sep)
    try:
        result = parser.parse_text(TEXT, output_format="native",
                                   tokenizer="native", chunk_size=args.chunk_size)
        assert isinstance(result, list) and len(result) > 0
        for sent in result:
            _print_native(sent)
        ok("[5] parse_text native/native")
    except Exception as e:
        fail("[5] parse_text native/native", e)

    # ── [6] chunk_size=1 — результат совпадает с chunk_size=32 ────────────────
    print(f"\n{sep}")
    print("[6] parse_text — chunk_size=1 совпадает с chunk_size=32")
    print(sep)
    try:
        r1 = parser.parse_text(MULTI, output_format="simplified",
                                tokenizer="razdel", chunk_size=1)
        r32 = parser.parse_text(MULTI, output_format="simplified",
                                 tokenizer="razdel", chunk_size=32)
        assert len(r1) == len(r32), f"len: chunk=1 → {len(r1)}, chunk=32 → {len(r32)}"
        for s1, s32 in zip(r1, r32):
            f1  = [t["form"]  for t in s1]
            f32 = [t["form"]  for t in s32]
            assert f1 == f32, f"forms differ: {f1} vs {f32}"
        ok(f"[6] chunk_size=1 ({len(r1)} предл.) ≡ chunk_size=32")
    except Exception as e:
        fail("[6] chunk_size совместимость", e)

    # ── [7] Пустой текст → [] ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[7] parse_text — пустой текст → []")
    print(sep)
    try:
        r_r = parser.parse_text("", tokenizer="razdel")
        r_n = parser.parse_text("", tokenizer="native")
        assert r_r == [], f"razdel: ожидался [], получено {r_r!r}"
        assert r_n == [], f"native: ожидался [], получено {r_n!r}"
        ok("[7] Пустой текст → []")
    except Exception as e:
        fail("[7] Пустой текст", e)

    # ── [8] parse_batch — razdel ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[8] parse_batch  (razdel + simplified, {len(BATCH)} текста)")
    print(sep)
    try:
        results = parser.parse_batch(BATCH, output_format="simplified",
                                     tokenizer="razdel", chunk_size=args.chunk_size)
        assert len(results) == len(BATCH), f"ожидалось {len(BATCH)}, получено {len(results)}"
        for idx, (text, res) in enumerate(zip(BATCH, results), 1):
            assert isinstance(res, list), f"текст {idx}: результат не list"
            print(f"  Текст {idx}: {len(res)} предл.")
            sentences = list(sentenize(text))  # ← text из цикла
            for sent, s in zip(res, sentences):  # ← res вместо result
                print_simplified(sent, s.text)
            print()
        ok(f"[8] parse_batch razdel/simplified — {len(BATCH)} текста")
    except Exception as e:
        fail("[8] parse_batch razdel", e)

    # ── [9] parse_batch — native ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[9] parse_batch  (native + simplified, {len(BATCH)} текста)")
    print(sep)
    try:
        results = parser.parse_batch(BATCH, output_format="simplified",
                                     tokenizer="native", chunk_size=args.chunk_size)
        assert len(results) == len(BATCH)
        for idx, res in enumerate(results, 1):
            print(f"  Текст {idx}: {len(res)} предл.")
        ok(f"[9] parse_batch native/simplified — {len(BATCH)} текста")
    except Exception as e:
        fail("[9] parse_batch native", e)

    # ── [10] parse_batch == parse_text × N ────────────────────────────────────
    print(f"\n{sep}")
    print("[10] parse_batch ≡ parse_text × N  (razdel, chunk_size=1)")
    print(sep)
    try:
        batch = parser.parse_batch(BATCH, output_format="simplified",
                                   tokenizer="razdel", chunk_size=1)
        for i, text in enumerate(BATCH):
            single = parser.parse_text(text, output_format="simplified",
                                       tokenizer="razdel", chunk_size=1)
            assert len(batch[i]) == len(single), \
                f"текст {i+1}: batch={len(batch[i])} vs single={len(single)}"
            for sb, ss in zip(batch[i], single):
                fb = [t["form"] for t in sb]
                fs = [t["form"] for t in ss]
                assert fb == fs, f"текст {i+1}: forms differ: {fb} vs {fs}"
        ok(f"[10] parse_batch ≡ parse_text × {len(BATCH)}")
    except Exception as e:
        fail("[10] parse_batch vs parse_text", e)

    # ── Итог ──────────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{sep}")
    print(f"ИТОГ: {passed}/{total} тестов прошло" + (" ✅" if failed == 0 else f"  ❌ {failed} упало"))
    print(sep)
    sys.exit(0 if failed == 0 else 1)
