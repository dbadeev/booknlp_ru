# stanza_wrapper.py
# =============================================================================
# ИЗМЕНЕНИЯ по сравнению с предыдущей версией:
#   [NEW] _split_to_chunks()         — razdel path → List[List[(text, offset)]]
#   [NEW] _split_to_sentence_chunks() — native path → List[List[str]]
#   [NEW] _merge_chunks()            — склейка результатов чанков
#   [NEW] sentence_batch_size        — параметр чанкинга в __init__
#   [CHG] parse_text() — теперь: sentenize → chunks → .map() (не .remote())
#   [CHG] parse_batch() — список текстов: sentenize × N → all_chunks → .map()
#   [REM] Всё форматирование, морфология и вывод перенесены в stanza_modal.py
#   [REM] parse_batch(batch_tokens) — удалён pretokenized API (заменён chunking)
#   [NEW] Параметр tokenizer: 'internal' | 'razdel' во всех методах
# =============================================================================

#!/usr/bin/env python3
"""
Тонкий клиент для StanzaService (Modal).

Обязанности wrapper:
  ├── _split_to_chunks()          razdel path → List[List[(text, offset)]]
  ├── _split_to_sentence_chunks() native path → List[List[str]]
  ├── _merge_chunks()             склейка результатов чанков
  │
  ├── parse_text()   sentenize → chunks → .map()
  └── parse_batch()  sentenize × N → all_chunks → .map()

Никакого форматирования, никакой морфологии, никакого вывода — только
маршрутизация и управление чанками.

Сентенизация выполняется ЗДЕСЬ, в wrapper, ДО отправки в Modal.
Оба пути (razdel / native) используют razdel.sentenize для разбивки
на предложения — Modal получает уже разбитый текст.
"""

import logging
import modal
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# [NEW] Типы — аналогично spacy_wrapper.py
TokenizerType = Literal["internal", "razdel"]
OutputFormat  = Literal["native", "conllu"]


class StanzaParser:
    """
    Тонкий клиент для StanzaService, запущенного в Modal.

    Все вычисления выполняются удалённо. Wrapper отвечает только за:
      1. Сентенизацию (razdel.sentenize) — до отправки в Modal
      2. Разбивку на чанки (sentence_batch_size) — OOM-защита на GPU
      3. Параллельную отправку чанков через .map()
      4. Склейку результатов из чанков

    Args:
        sentence_batch_size: количество предложений в одном чанке.
            Подбирается под конкретный GPU и тип текстов.
            По умолчанию: 32 (T4 16GB, средние предложения).
            Уменьшить при OOM или длинных предложениях.
    """

    def __init__(self, sentence_batch_size: int = 32):
        self.logger = logging.getLogger(__name__)
        # [NEW] sentence_batch_size — параметр чанкинга для OOM-защиты
        self.sentence_batch_size = sentence_batch_size
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-stanza", "StanzaService"
            )()
            self.logger.info("Connected to Stanza via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise

    # ─── Chunking helpers ─────────────────────────────────────────────────────

    def _split_to_chunks(
        self,
        text: str,
        base_offset: int = 0,
    ) -> List[List[Tuple[str, int]]]:
        """
        [NEW] Razdel path: сентенизация + разбивка на чанки с офсетами.

        Использует razdel.sentenize для разбивки текста на предложения.
        Каждый чанк несёт пары (sentence_text, start_char) — символьный
        офсет относительно начала исходного (полного) текста.

        Modal-метод parse_sentence_chunk() принимает именно такой формат:
        офсеты нужны для восстановления start_char/end_char токенов
        в координатах исходного текста.

        Args:
            text:        исходный текст для разбивки
            base_offset: смещение начала text в документе (для parse_batch)

        Returns:
            List[List[(sentence_text, start_char)]]
            — список чанков, каждый чанк = список пар (текст, офсет)
        """
        from razdel import sentenize

        sentences = list(sentenize(text))
        chunk_size = self.sentence_batch_size
        return [
            [
                (s.text, base_offset + s.start)
                for s in sentences[i : i + chunk_size]
            ]
            for i in range(0, len(sentences), chunk_size)
        ]

    def _split_to_sentence_chunks(
        self,
        text: str,
    ) -> List[List[str]]:
        """
        [NEW] Native (internal tokenizer) path: сентенизация + разбивка на чанки.

        Использует тот же razdel.sentenize для разбивки — только тексты,
        без офсетов. Modal-метод parse_sentence_chunk_native() принимает
        List[str] и выполняет собственную токенизацию Stanza.

        start_char/end_char токенов в результате будут относительны
        начала каждого предложения (не исходного текста) — это осознанное
        ограничение internal-пути.

        Args:
            text: исходный текст для разбивки

        Returns:
            List[List[str]] — список чанков, каждый чанк = список текстов
        """
        from razdel import sentenize

        sentences = list(sentenize(text))
        chunk_size = self.sentence_batch_size
        return [
            [s.text for s in sentences[i : i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(chunks_results: List[Any]) -> Any:
        """
        [NEW] Склейка результатов из нескольких чанков.

        Поддерживает оба формата:
          native → List[Dict]: конкатенация списков предложений
          conllu → str:        объединение строк через двойной перенос

        Args:
            chunks_results: List[результатов от parse_sentence_chunk*]

        Returns:
            Объединённый результат того же типа, что и элементы входного списка.
        """
        if not chunks_results:
            return []

        # Определяем тип по первому непустому результату
        first = next((r for r in chunks_results if r), None)
        if first is None:
            return []

        if isinstance(first, str):
            # CoNLL-U: объединяем через двойной перевод строки
            return "\n\n".join(r.strip() for r in chunks_results if r.strip()) + "\n"
        else:
            # Native: конкатенируем списки предложений
            result = []
            for chunk in chunks_results:
                if isinstance(chunk, list):
                    result.extend(chunk)
            return result

    # ─── Public API ───────────────────────────────────────────────────────────

    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        sentence_batch_size: Optional[int] = None,
    ) -> Any:
        """
        [CHG] Основной метод парсинга одного текста.

        Алгоритм:
          1. Сентенизация текста через razdel.sentenize (в wrapper)
          2. Разбивка на чанки по sentence_batch_size
          3. Отправка чанков в Modal через .map() (параллельно)
          4. Склейка результатов

        Razdel path (tokenizer='razdel'):
          chunks = List[List[(text, offset)]]
          → parse_sentence_chunk.map(chunks)

        Native path (tokenizer='internal'):
          chunks = List[List[str]]
          → parse_sentence_chunk_native.map(chunks)

        Args:
            text:               исходный текст
            output_format:      'native' | 'conllu'
            tokenizer:          'internal' | 'razdel'
            sentence_batch_size: переопределяет self.sentence_batch_size
                                 для этого вызова

        Returns:
            native → List[Dict] (список предложений)
            conllu → str
        """
        # Временное переопределение batch_size для этого вызова
        orig_batch_size = self.sentence_batch_size
        if sentence_batch_size is not None:
            self.sentence_batch_size = sentence_batch_size

        try:
            if tokenizer == "razdel":
                # ── Razdel path ──────────────────────────────────────────
                chunks = self._split_to_chunks(text, base_offset=0)
                if not chunks:
                    return [] if output_format == "native" else ""
                # Параллельная отправка чанков в Modal
                results = list(
                    self.service.parse_sentence_chunk.map(
                        chunks,
                        kwargs={"output_format": output_format},
                    )
                )
            else:
                # ── Native (internal) path ────────────────────────────────
                chunks = self._split_to_sentence_chunks(text)
                if not chunks:
                    return [] if output_format == "native" else ""
                results = list(
                    self.service.parse_sentence_chunk_native.map(
                        chunks,
                        kwargs={"output_format": output_format},
                    )
                )

            return self._merge_chunks(results)

        except Exception as e:
            self.logger.error(f"Error during Stanza parsing: {e}")
            raise
        finally:
            self.sentence_batch_size = orig_batch_size

    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        sentence_batch_size: Optional[int] = None,
    ) -> List[Any]:
        """
        [CHG] Пакетная обработка нескольких текстов.

        Для каждого текста выполняет сентенизацию и разбивку на чанки,
        затем отправляет все чанки всех текстов в Modal за один .map().

        Алгоритм:
          1. Для каждого текста: sentenize → chunks
          2. Все чанки объединяются в один список с маркерами текстов
          3. .map() по всем чанкам параллельно
          4. Склейка чанков обратно по текстам

        Args:
            texts:              список текстов для обработки
            output_format:      'native' | 'conllu'
            tokenizer:          'internal' | 'razdel'
            sentence_batch_size: переопределяет self.sentence_batch_size

        Returns:
            List[результат для каждого текста] — порядок соответствует texts
        """
        orig_batch_size = self.sentence_batch_size
        if sentence_batch_size is not None:
            self.sentence_batch_size = sentence_batch_size

        try:
            # Собираем чанки всех текстов и запоминаем границы
            all_chunks: List[Any] = []
            # text_chunk_counts[i] = количество чанков для texts[i]
            text_chunk_counts: List[int] = []

            for text in texts:
                if tokenizer == "razdel":
                    chunks = self._split_to_chunks(text, base_offset=0)
                else:
                    chunks = self._split_to_sentence_chunks(text)
                all_chunks.extend(chunks)
                text_chunk_counts.append(len(chunks))

            if not all_chunks:
                return [[] if output_format == "native" else "" for _ in texts]

            # Единый .map() по всем чанкам всех текстов
            if tokenizer == "razdel":
                all_results = list(
                    self.service.parse_sentence_chunk.map(
                        all_chunks,
                        kwargs={"output_format": output_format},
                    )
                )
            else:
                all_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        all_chunks,
                        kwargs={"output_format": output_format},
                    )
                )

            # Распределяем результаты обратно по текстам
            per_text_results: List[Any] = []
            idx = 0
            for count in text_chunk_counts:
                text_chunks = all_results[idx : idx + count]
                per_text_results.append(self._merge_chunks(text_chunks))
                idx += count

            return per_text_results

        except Exception as e:
            self.logger.error(f"Error during batch parsing: {e}")
            raise
        finally:
            self.sentence_batch_size = orig_batch_size


# ─── Тесты — аналогично spacy_wrapper.py ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = StanzaParser(sentence_batch_size=32)

    text_single = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    text_multi  = "Зло, которым пугаешь, не так зло. Москва — столица России."
    sep = "=" * 60

    # ─── Локальные хелперы вывода ─────────────────────────────────────────
    # Заголовок столбцов — только для тестовой печати, не в выводе модели.
    _CONLLU_HEADER = "\t".join(
        ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
    )

    def _print_conllu(result: str) -> None:
        """Печатает CoNLL-U с заголовком столбцов после каждой строки # text."""
        for block in result.strip().split("\n\n"):
            lines = block.strip().split("\n")
            for line in lines:
                print(line)
                if line.startswith("# text"):
                    print(_CONLLU_HEADER)
            print()

    def _print_native(results: list) -> None:
        """Выводит ВСЕ нативные поля каждого токена с подписями."""
        for sent in results:
            print(
                f"\nПредложение: '{sent['text']}' "
                f"(chars {sent['start_char']}:{sent['end_char']})"
            )
            for tok in sent["words"]:
                feats     = tok.get("feats") or {}
                feats_str = "|".join(f"{k}={v}" for k, v in feats.items()) or "_"
                sa        = tok.get("spaces_after")
                sa_s      = repr(sa) if sa not in (" ", None) else "_"
                ner       = tok.get("ner", "O")
                print(
                    f"  {tok['id']:>2}  "
                    f"{tok['form']:<16} "
                    f"upos={tok['upos']:<6} "
                    f"lemma={tok['lemma']:<16} "
                    f"xpos={tok['xpos'] or '_':<6} "
                    f"feats=[{feats_str}]  "
                    f"head={tok['head']:<3} "
                    f"deprel={tok['deprel']:<12} "
                    f"sc={tok['start_char']} ec={tok['end_char']}  "
                    f"sa={sa_s:<6} "
                    f"ner={ner}"
                )

    # ── 1. CONLL-U + RAZDEL ──────────────────────────────────────────────
    print(f"\n{sep}\n1. CONLL-U + RAZDEL (parse_text)\n{sep}")
    _print_conllu(
        parser.parse_text(text_single, output_format="conllu", tokenizer="razdel")
    )

    # ── 2. CONLL-U + INTERNAL ────────────────────────────────────────────
    print(f"\n{sep}\n2. CONLL-U + INTERNAL (parse_text)\n{sep}")
    _print_conllu(
        parser.parse_text(text_single, output_format="conllu", tokenizer="internal")
    )

    # ── 3. NATIVE + RAZDEL ───────────────────────────────────────────────
    print(f"\n{sep}\n3. NATIVE + RAZDEL (parse_text)\n{sep}")
    res_r = parser.parse_text(text_single, output_format="native", tokenizer="razdel")
    print(f"Предложений: {len(res_r)}")
    _print_native(res_r)

    # ── 4. NATIVE + INTERNAL ─────────────────────────────────────────────
    print(f"\n{sep}\n4. NATIVE + INTERNAL (parse_text)\n{sep}")
    res_i = parser.parse_text(text_single, output_format="native", tokenizer="internal")
    print(f"Предложений: {len(res_i)}")
    _print_native(res_i)

    # ── 5. NER-статистика ────────────────────────────────────────────────
    print(f"\n{sep}\n5. СТАТИСТИКА NER (native + razdel)\n{sep}")
    all_words = [tok for sent in res_r for tok in sent["words"]]
    ner_tags  = [tok.get("ner", "O") for tok in all_words]
    print(f"Всего токенов: {len(ner_tags)}")
    print(f"Персоны  (PER): {sum(1 for t in ner_tags if t and 'PER' in t)}")
    print(f"Локации  (LOC): {sum(1 for t in ner_tags if t and 'LOC' in t)}")
    print(f"Орг.     (ORG): {sum(1 for t in ner_tags if t and 'ORG' in t)}")

    # ── 6. parse_batch ───────────────────────────────────────────────────
    print(f"\n{sep}\n6. parse_batch (два текста, conllu + razdel)\n{sep}")
    for i, res in enumerate(
        parser.parse_batch([text_single, text_multi], output_format="conllu", tokenizer="razdel")
    ):
        print(f"\n# === text {i + 1} ===")
        _print_conllu(res)

    # ── 7. Сравнение токенизаторов ────────────────────────────────────────
    print(f"\n{sep}\n7. СРАВНЕНИЕ ТОКЕНИЗАТОРОВ (razdel vs internal)\n{sep}")
    sample = "Кружка-термос стоит 500р."
    print(f"Текст: '{sample}'")
    print(f"  razdel:   {[w['form'] for s in parser.parse_text(sample, tokenizer='razdel')   for w in s['words']]}")
    print(f"  internal: {[w['form'] for s in parser.parse_text(sample, tokenizer='internal') for w in s['words']]}")

    # ── 8. Ключи первого токена и предложения ────────────────────────────
    print(f"\n{sep}\n8. ВСЕ КЛЮЧИ ПЕРВОГО ТОКЕНА И ПРЕДЛОЖЕНИЯ\n{sep}")
    if res_i and res_i[0].get("words"):
        first_tok = res_i[0]["words"][0]
        print(f"Ключи токена:      {list(first_tok.keys())}")
        print(f"Ключи предложения: {list(res_i[0].keys())}")
        print("\nЗначения первого токена:")
        for k, v in first_tok.items():
            print(f"  {k}: {v}")

    print(f"\n{'✅ Тестирование завершено!':^60}")
