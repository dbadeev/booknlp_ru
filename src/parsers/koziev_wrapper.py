#!/usr/bin/env python3
"""
Локальная обёртка для Koziev парсера (Modal).

Wrapper — тонкий клиент. Три обязанности:
  1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
  2. Маршрутизация чанков в нужный Modal-метод (razdel / native ветка).
  3. Сборка результатов чанков в единый ответ.

Вся NLP-логика (rutokenizer, rupostagger, rulemma) — в koziev_modal.py.
Сентенизация ВСЕГДА выполняется здесь, через razdel.sentenize,
ДО отправки в Modal.

Два пути (оба используют razdel.sentenize для сентенизации):

  razdel (с символьными офсетами):
      sentenize → List[List[(text, start_char)]]
      → service.parse_sentence_chunk.map(chunks)

  native (без символьных офсетов):
      sentenize → List[List[str]]
      → service.parse_sentence_chunk_native.map(chunks)

Использование:
    from koziev_wrapper import KozievWrapper

    wrapper = KozievWrapper()
    # 4 варианта (2 формата × 2 пути):
    result = wrapper.parse_text(text, output_format="native",  tokenizer="razdel")
    result = wrapper.parse_text(text, output_format="native",  tokenizer="native")
    result = wrapper.parse_text(text, output_format="conllu",  tokenizer="razdel")
    result = wrapper.parse_text(text, output_format="conllu",  tokenizer="native")

    # chunk_size подбирается под GPU и тип текста (по умолчанию 10):
    result = wrapper.parse_text(text, tokenizer="razdel", chunk_size=5)
"""

import argparse
import logging
import json
import sys
import modal
from razdel import sentenize
from typing import Any, Dict, List, Literal, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

OutputFormat = Literal["native", "conllu"]
TokenizerType = Literal["native", "razdel"]

default_chunk_size: int = 10  # предложений на чанк; подбирается под GPU и тип текста

CONLLU_HEADER = "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"


class KozievWrapper:
    """
    Клиент для Koziev парсера (Modal).

    Поддерживает:
      - 2 формата: native (словари токенов), conllu (стандарт UD)
      - 2 пути: razdel (с символьными офсетами), native (только тексты)

    Сентенизация выполняется локально (razdel.sentenize) до отправки в Modal.
    Чанки распределяются по контейнерам через .map().
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-koziev-service", "KozievService"
            )()
            self.logger.info("✓ Connected to KozievService via Modal.")
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

        Returns:
            List[List[(sentence_text, start_char_in_original)]]
        base_offset — смещение text в более крупном документе
                      (используется в parse_batch).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [(s.text, base_offset + s.start) for s in sentences[i : i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _split_to_sentence_chunks(
        text: str,
        chunk_size: int,
    ) -> List[List[str]]:
        """
        Native path: разбивает текст на чанки предложений (только тексты).

        Returns:
            List[List[str]]
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [s.text for s in sentences[i : i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(
        chunk_results: List[Any],
        output_format: str,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Склеивает результаты чанков в единый ответ.
          native → List[Dict] (предложения из всех чанков, плоский список)
          conllu → str (CoNLL-U блоки через двойной перенос строки)
        """
        if output_format == "conllu":
            parts = [cr.strip() for cr in chunk_results if cr.strip()]
            return "\n\n".join(parts) + "\n" if parts else ""
        return [sent for cr in chunk_results for sent in cr]

    # ─── Public API ───────────────────────────────────────────────────────────
    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "conllu",
        tokenizer: TokenizerType = "native",
        chunk_size: int = default_chunk_size,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Парсит текст через Koziev в Modal.

        Алгоритм:
          1. Разбить текст на предложения (razdel.sentenize).
          2. Сгруппировать в чанки по chunk_size.
          3. Один чанк → .remote(); несколько → .map() (параллельно).
          4. Склеить результаты.

        Args:
            text:          Входной текст
            output_format: 'native' | 'conllu'
            tokenizer:     'razdel' (с символьными офсетами)
                         | 'native' (без офсетов, только тексты)
            chunk_size:    Предложений на чанк (подбирается под GPU и тип текста).
                           По умолчанию default_chunk_size = 10.

        Returns:
            native → List[Dict]
            conllu → str
        """
        try:
            if tokenizer == "razdel":
                chunks = self._split_to_chunks(text, chunk_size)
                if not chunks:
                    return [] if output_format == "native" else ""
                if len(chunks) == 1:
                    return self.service.parse_sentence_chunk.remote(
                        chunks[0], output_format=output_format
                    )
                chunk_results = list(
                    self.service.parse_sentence_chunk.map(
                        chunks, kwargs={"output_format": output_format}
                    )
                )
            else:  # native
                chunks = self._split_to_sentence_chunks(text, chunk_size)
                if not chunks:
                    return [] if output_format == "native" else ""
                if len(chunks) == 1:
                    return self.service.parse_sentence_chunk_native.remote(
                        chunks[0], output_format=output_format
                    )
                chunk_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        chunks, kwargs={"output_format": output_format}
                    )
                )
            return self._merge_chunks(chunk_results, output_format)
        except Exception as exc:
            self.logger.error(f"❌ Error during Koziev parsing: {exc}")
            raise

    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "conllu",
        tokenizer: TokenizerType = "native",
        chunk_size: int = default_chunk_size,
    ) -> List[Union[List[Dict[str, Any]], str]]:
        """
        Разбивает все тексты на чанки и отправляет единым .map() —
        Modal распределяет по доступным контейнерам.

        Args:
            texts:         Список текстов
            output_format: 'native' | 'conllu'
            tokenizer:     'razdel' | 'native'
            chunk_size:    Предложений на чанк

        Returns:
            List[результат для каждого текста]
        """
        try:
            chunks_per_text: List[int] = []
            if tokenizer == "razdel":
                all_chunks: List[List[Tuple[str, int]]] = []
                for text in texts:
                    text_chunks = self._split_to_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks.extend(text_chunks)
                if not all_chunks:
                    return [[] if output_format == "native" else "" for _ in texts]
                all_results = list(
                    self.service.parse_sentence_chunk.map(
                        all_chunks, kwargs={"output_format": output_format}
                    )
                )
            else:  # native
                all_chunks_native: List[List[str]] = []
                for text in texts:
                    text_chunks = self._split_to_sentence_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks_native.extend(text_chunks)
                if not all_chunks_native:
                    return [[] if output_format == "native" else "" for _ in texts]
                all_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        all_chunks_native, kwargs={"output_format": output_format}
                    )
                )

            # Восстанавливаем результаты по текстам
            results, offset = [], 0
            for n_chunks in chunks_per_text:
                results.append(
                    self._merge_chunks(
                        all_results[offset : offset + n_chunks], output_format
                    )
                )
                offset += n_chunks
            return results
        except Exception as exc:
            self.logger.error(f"❌ Error during batch Koziev parsing: {exc}")
            raise


# ─── Вспомогательные функции вывода ───────────────────────────────────────────
def _print_token(tok: Dict[str, Any]) -> None:
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"     form:  {tok['form']}")
    print(f"     lemma: {tok['lemma']}")
    print(f"     upos:  {tok['upos']}")
    print(f"     feats: {tok['feats']}")


def _print_conllu(text: str, conllu: str) -> None:
    print(f"\n# text = {text}")
    print(CONLLU_HEADER)
    print(conllu)


# ─── __main__: тест через wrapper (с chunking) ───────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Koziev wrapper тест")
    ap.add_argument(
        "--tokenizer", choices=["native", "razdel"], default="native",
        help="Путь сентенизации: native (без офсетов) или razdel (с офсетами)",
    )
    ap.add_argument(
        "--output-format", choices=["native", "conllu"], default="conllu",
        dest="output_format", help="Формат вывода (default: conllu)",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=default_chunk_size, dest="chunk_size",
        help=f"Предложений на чанк (default: {default_chunk_size})",
    )
    args = ap.parse_args()

    sep = "=" * 72

    print(sep)
    print("ПРОВЕРКА ДОСТУПНОСТИ MODAL-СЕРВИСА")
    print(sep)
    try:
        wrapper = KozievWrapper()
    except Exception as e:
        print(f"⚠️  Modal-сервис недоступен: {e}")
        print("\nЗапустите сервис командой:")
        print("  modal deploy src/parsers/koziev_modal.py")
        sys.exit(1)

    text_single = "Кружка-термос стоит 500р."
    text_multi = (
        "Зло, которым пугаешь, не так зло. "
        "Москва — столица России. "
        "Кружка-термос стоит 500р."
    )

    # ── Вариант 1: NATIVE + RAZDEL PATH ──────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 1: NATIVE + RAZDEL PATH (с символьными офсетами)")
    print(sep)
    result_nr = wrapper.parse_text(
        text_single, output_format="native", tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    print(f"\nТекст: '{text_single}'")
    for sentence in result_nr:
        print(
            f"\nПредложение: '{sentence['text']}' "
            f"(start_char={sentence['start_char']})"
        )
        for token in sentence["words"]:
            _print_token(token)

    # ── Вариант 2: NATIVE + NATIVE PATH ──────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 2: NATIVE + NATIVE PATH (без символьных офсетов)")
    print(sep)
    result_nn = wrapper.parse_text(
        text_single, output_format="native", tokenizer="native",
        chunk_size=args.chunk_size,
    )
    print(f"\n⚡ Сравнение путей для: '{text_single}'")
    print(f"  razdel: {[w['form'] for s in result_nr for w in s['words']]}")
    print(f"  native: {[w['form'] for s in result_nn for w in s['words']]}")
    for sentence in result_nn:
        print(f"\nПредложение: '{sentence['text']}'")
        for token in sentence["words"]:
            _print_token(token)

    # ── Вариант 3: CONLL-U + RAZDEL ──────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 3: CONLL-U + RAZDEL PATH")
    print(sep)
    _print_conllu(
        text_multi,
        wrapper.parse_text(
            text_multi, output_format="conllu", tokenizer="razdel",
            chunk_size=args.chunk_size,
        ),
    )

    # ── Вариант 4: CONLL-U + NATIVE ──────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 4: CONLL-U + NATIVE PATH")
    print(sep)
    _print_conllu(
        text_multi,
        wrapper.parse_text(
            text_multi, output_format="conllu", tokenizer="native",
            chunk_size=args.chunk_size,
        ),
    )

    # ── parse_batch ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("BATCH: CONLL-U + RAZDEL (2 текста)")
    print(sep)
    batch_texts = [text_single, "Зло, которым пугаешь, не так зло."]
    batch_results = wrapper.parse_batch(
        batch_texts, output_format="conllu", tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    for idx, (batch_text, batch_res) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"\n── Текст {idx}: '{batch_text}'")
        _print_conllu(batch_text, batch_res)

    print(f"\n{'✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ':^72}")