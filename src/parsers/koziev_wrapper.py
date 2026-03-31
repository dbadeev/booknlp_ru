#!/usr/bin/env python3
"""
Локальная обёртка для Koziev парсера (Modal).

Wrapper — тонкий клиент. Три обязанности:
  1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
  2. При tokenizer="razdel": токенизация слов (razdel.tokenize) локально.
  3. Маршрутизация чанков в нужный Modal-метод и сборка результатов.

Вся NLP-логика (rupostagger, rulemma) — в koziev_modal.py.
Сентенизация ВСЕГДА выполняется здесь, через razdel.sentenize, ДО отправки.

Два пути (оба используют razdel.sentenize):

  tokenizer="native" — слова токенизирует rutokenizer ВНУТРИ Modal:
#   sentenize → List[List[(text, start_char)]]
#   → service.parse_sentence_chunk.map(chunks)
#   Офсеты: ДА

  tokenizer="razdel" — слова токенизирует razdel.tokenize ЗДЕСЬ, в wrapper:
      sentenize + tokenize → List[List[(text, tokens, start_char)]]
      → service.parse_pretokenized_chunk.map(chunks)
      Офсеты: ДА (из razdel.sentenize .start)

Использование:
    from koziev_wrapper import KozievWrapper

    wrapper = KozievWrapper()
    # tokenizer="native"  — rutokenizer в Modal, без офсетов
    result = wrapper.parse_text(text, output_format="native",  tokenizer="native")
    result = wrapper.parse_text(text, output_format="conllu",  tokenizer="native")

    # tokenizer="razdel" — razdel.tokenize локально, с офсетами
    result = wrapper.parse_text(text, output_format="native",  tokenizer="razdel")
    result = wrapper.parse_text(text, output_format="conllu",  tokenizer="razdel")

    # chunk_size подбирается под GPU и тип текста:
    result = wrapper.parse_text(text, tokenizer="razdel", chunk_size=5)
"""

import argparse
import logging
import sys
import modal
from razdel import sentenize
from razdel import tokenize as razdel_tokenize
from typing import Any, Dict, List, Literal, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

OutputFormat = Literal["native", "conllu"]
TokenizerType = Literal["native", "razdel"]

default_chunk_size: int = 10  # предложений на чанк; подбирается под GPU и тип текста


class KozievWrapper:
    """
    Клиент для Koziev парсера (Modal).

    Поддерживает:
      - 2 токенизатора слов: native (rutokenizer внутри Modal),
                             razdel (razdel.tokenize локально)
      - 2 формата вывода:   native (словари), conllu (стандарт UD)

    Сентенизация — всегда razdel.sentenize, локально, ДО отправки в Modal.
    Чанки распределяются по контейнерам через .map().

    Известные ограничения rupostagger / rulemma:
        - PROPN не выделяется: имена собственные получают NOUN (реже ADJ).
        - Все леммы возвращаются в нижнем регистре, включая имена собственных.
          Постобработка по upos == "PROPN" невозможна до исправления теггера.
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
    # noinspection DuplicatedCode
    @staticmethod
    def _split_to_chunks(
            text: str,
            chunk_size: int,
            base_offset: int = 0,
    ) -> List[List[Tuple[str, int]]]:
        """
        Native path: разбивает текст на чанки предложений с символьными офсетами.
        Офсеты нужны для корректной склейки предложений при обработке большого текста.

        Returns:
            List[List[(sentence_text, start_char)]]
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [(s.text, base_offset + s.start) for s in sentences[i: i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _split_to_razdel_chunks(
        text: str,
        chunk_size: int,
        base_offset: int = 0,
    ) -> List[List[Tuple[str, List[str], int]]]:
        """
        Razdel path: разбивает текст на чанки с локальной токенизацией слов.
        Каждое предложение = (текст, список_токенов, символьный_офсет_в_документе).

        razdel.tokenize выполняется ЗДЕСЬ, в wrapper, ДО отправки в Modal —
        это и есть «внешний токенизатор».

        Returns:
            List[List[(sentence_text, tokens, start_char)]]
        base_offset используется в parse_batch для документов с ненулевым офсетом.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [
                (
                    s.text,
                    [t.text for t in razdel_tokenize(s.text)],
                    base_offset + s.start,
                )
                for s in sentences[i: i + chunk_size]
            ]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(
        chunk_results: List[Any],
        output_format: str,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Склеивает результаты чанков в единый ответ.
          native → плоский List[Dict] (предложения из всех чанков)
          conllu → str (CoNLL-U блоки через двойной перенос строки)
                   с последующей перенумерацией sent_id.
        """
        if output_format == "conllu":
            parts = [cr.strip() for cr in chunk_results if cr.strip()]
            if not parts:  # ← явный ранний выход
                return ""
            merged = "\n\n".join(parts) + "\n"
            if len(parts) == 1:
                return merged
            return KozievWrapper._renumber_sent_ids(merged)
        return [sent for cr in chunk_results for sent in cr]

    @staticmethod
    def _renumber_sent_ids(conllu: str) -> str:
        """Перенумеровывает # sent_id глобально в объединённом CoNLL-U блоке."""
        if not conllu.strip():  # ← ранний выход для пустого ввода
            return ""
        counter = 0
        lines = []
        for line in conllu.splitlines():
            if line.startswith("# sent_id"):
                counter += 1
                lines.append(f"# sent_id = {counter}")
            else:
                lines.append(line)
        return "\n".join(lines) + "\n"

    # ─── Public API ───────────────────────────────────────────────────────────
    # noinspection DuplicatedCode
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
          3. tokenizer="native": тексты + офсеты → parse_sentence_chunk
             tokenizer="razdel": тексты + razdel-токены + офсеты → parse_pretokenized_chunk
          4. Один чанк → .remote(); несколько → .map() (параллельно).
          5. Склеить результаты, перенумеровать sent_id.

        Args:
            text:          Входной текст
            output_format: 'native' | 'conllu'
            tokenizer:     'native' — rutokenizer токенизирует слова в Modal
                           'razdel' — razdel.tokenize токенизирует здесь, результаты
                                      отправляются предтокенизированными
            chunk_size:    Предложений на чанк. По умолчанию = 10.

        Returns:
            native → List[Dict]
            conllu → str
        """
        try:
            if tokenizer == "native":
                # Продакшн-путь: офсеты нужны для корректной склейки в документ
                chunks = self._split_to_chunks(text, chunk_size)
                if not chunks:
                    return [] if output_format == "native" else ""
                if len(chunks) == 1:
                    chunk_results = [self.service.parse_sentence_chunk.remote(
                        chunks[0], output_format=output_format
                    )]
                else:
                    chunk_results = list(self.service.parse_sentence_chunk.map(
                        chunks, kwargs={"output_format": output_format}
                    ))
                return self._merge_chunks(chunk_results, output_format)

            else:  # tokenizer == "razdel"
                chunks = self._split_to_razdel_chunks(text, chunk_size)
                if not chunks:
                    return [] if output_format == "native" else ""
                if len(chunks) == 1:
                    chunk_results = [self.service.parse_pretokenized_chunk.remote(
                        chunks[0], output_format=output_format
                    )]
                else:
                    chunk_results = list(self.service.parse_pretokenized_chunk.map(
                        chunks, kwargs={"output_format": output_format}
                    ))
                return self._merge_chunks(chunk_results, output_format)

        except Exception as exc:
            self.logger.error(f"❌ Error during Koziev parsing: {exc}")
            raise

    # noinspection DuplicatedCode
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
            tokenizer:     'native' | 'razdel'
            chunk_size:    Предложений на чанк

        Returns:
            List[результат для каждого текста]
        """
        try:
            chunks_per_text: List[int] = []

            if tokenizer == "native":
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

            else:  # tokenizer == "razdel"
                all_chunks_r: List[List[Tuple[str, List[str], int]]] = []
                for text in texts:
                    text_chunks = self._split_to_razdel_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks_r.extend(text_chunks)
                if not all_chunks_r:
                    return [[] if output_format == "native" else "" for _ in texts]
                all_results = list(
                    self.service.parse_pretokenized_chunk.map(
                        all_chunks_r, kwargs={"output_format": output_format}
                    )
                )

            results, offset = [], 0
            for n_chunks in chunks_per_text:
                results.append(
                    self._merge_chunks(
                        all_results[offset: offset + n_chunks], output_format
                    )
                )
                offset += n_chunks
            return results

        except Exception as exc:
            self.logger.error(f"❌ Error during batch Koziev parsing: {exc}")
            raise


# ─── Вспомогательные функции вывода ───────────────────────────────────────────
# noinspection DuplicatedCode
def _print_token(tok: Dict[str, Any]) -> None:
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"     form:  {tok['form']}")
    print(f"     lemma: {tok['lemma']}")
    print(f"     upos:  {tok['upos']}")
    print(f"     feats: {tok['feats']}")


def _print_conllu(conllu: str) -> None:
    """Выводит CoNLL-U — CONLLU_HEADER вшит внутри каждого блока предложения."""
    print(conllu)


# ─── __main__: тесты через wrapper (с chunking) ──────────────────────────────
# noinspection DuplicatedCode
def main() -> None:
    """CLI entry point для тестирования KozievWrapper."""
    ap = argparse.ArgumentParser(description="Koziev wrapper тест")
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
    chunk_size = args.chunk_size

    # ── 1. NATIVE tokenizer, NATIVE format ───────────────────────────────────
    print(f"\n{sep}")
    print("1. NATIVE tokenizer — NATIVE format (rutokenizer в Modal, без офсетов)")
    print(sep)
    result = wrapper.parse_text(
        text_single, output_format="native", tokenizer="native",
        chunk_size=chunk_size,
    )
    print(f"\nТекст: '{text_single}'")
    for sentence in result:
        print(f"\nПредложение: '{sentence['text']}' (start_char={sentence['start_char']})")
        for token in sentence["words"]:
            _print_token(token)

    # ── 2. NATIVE tokenizer, CONLL-U format ──────────────────────────────────
    print(f"\n{sep}")
    print("2. NATIVE tokenizer — CoNLL-U format")
    print(sep)
    _print_conllu(
        wrapper.parse_text(
            text_multi, output_format="conllu", tokenizer="native",
            chunk_size=chunk_size,
        )
    )

    # ── 3. RAZDEL tokenizer, NATIVE format ───────────────────────────────────
    print(f"\n{sep}")
    print("3. RAZDEL tokenizer — NATIVE format (razdel.tokenize локально, с офсетами)")
    print(sep)
    result_r = wrapper.parse_text(
        text_single, output_format="native", tokenizer="razdel",
        chunk_size=chunk_size,
    )
    print(f"\nТекст: '{text_single}'")
    for sentence in result_r:
        print(
            f"\nПредложение: '{sentence['text']}' "
            f"(start_char={sentence['start_char']})"
        )
        for token in sentence["words"]:
            _print_token(token)

    # ── 4. RAZDEL tokenizer, CONLL-U format ──────────────────────────────────
    print(f"\n{sep}")
    print("4. RAZDEL tokenizer — CoNLL-U format")
    print(sep)
    _print_conllu(
        wrapper.parse_text(
            text_multi, output_format="conllu", tokenizer="razdel",
            chunk_size=chunk_size,
        )
    )

    # ── 5. Сравнение: native vs razdel tokenizer ──────────────────────────────
    print(f"\n{sep}")
    print("5. СРАВНЕНИЕ токенизаторов (native vs razdel) — NATIVE format")
    print(sep)
    result_n = wrapper.parse_text(
        text_multi, output_format="native", tokenizer="native",
        chunk_size=chunk_size,
    )
    result_r = wrapper.parse_text(
        text_multi, output_format="native", tokenizer="razdel",
        chunk_size=chunk_size,
    )
    print(f"\nТекст: '{text_multi}'")
    for sn, sr in zip(result_n, result_r):
        n_forms = [w["form"] for w in sn["words"]]
        r_forms = [w["form"] for w in sr["words"]]
        print(f"\n  Предложение: '{sn['text']}'")
        print(f"    native (rutokenizer): {n_forms}")
        print(f"    razdel (razdel.tok):  {r_forms}")
        if n_forms != r_forms:
            print(f"    ⚠️  Токенизация различается!")
        else:
            print(f"    ✓  Токенизация совпадает")

    # ── 6. parse_batch, NATIVE tokenizer ──────────────────────────────────────
    print(f"\n{sep}")
    print("6. BATCH (2 текста) — NATIVE tokenizer, CoNLL-U")
    print(sep)
    batch_texts = [
        text_single,
        "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься.",
    ]
    batch_results = wrapper.parse_batch(
        batch_texts, output_format="conllu", tokenizer="native",
        chunk_size=chunk_size,
    )
    for idx, (bt, br) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"\n── Текст {idx}: '{bt}'")
        _print_conllu(br)

    # ── 7. parse_batch, RAZDEL tokenizer ──────────────────────────────────────
    print(f"\n{sep}")
    print("7. BATCH (2 текста) — RAZDEL tokenizer, CoNLL-U")
    print(sep)
    batch_results_r = wrapper.parse_batch(
        batch_texts, output_format="conllu", tokenizer="razdel",
        chunk_size=chunk_size,
    )
    for idx, (bt, br) in enumerate(zip(batch_texts, batch_results_r), 1):
        print(f"\n── Текст {idx}: '{bt}'")
        _print_conllu(br)

    print(f"\n{'✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ':^72}")

if __name__ == "__main__":
    main()