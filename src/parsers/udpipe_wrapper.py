# udpipe_wrapper.py
# ─────────────────────────────────────────────────────────────────────────────
# Тонкий клиент для UDPipeService (Modal).
# Единственные обязанности:
#   _split_to_sentence_chunks()  — native-путь:  List[List[str]]
#   _split_to_chunks()           — razdel-путь:  (token_chunks, offset_chunks)
#   _merge_chunks()              — склейка результатов .map()
#   parse_text()                 — sentenize → chunks → .remote() / .map()
#   parse_batch()                — sentenize × N → all_chunks → единый .map()
#
# Никакого форматирования, никакой морфологии, никакого вывода —
# только маршрутизация и управление чанками.
#
# ИЗМЕНЕНИЯ по сравнению с исходной версией помечены: # ← НОВОЕ / # ← ИЗМЕНЕНО
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import modal

from typing import Any, Dict, List, Literal, Optional, Tuple

from razdel import sentenize, tokenize as razdel_tokenize

# ← ИЗМЕНЕНО: импорт типов токенизатора из modal-модуля для согласованности
TokenizerType = Literal["native", "razdel"]
OutputFormat  = Literal["dict", "native"]

# ─── Псевдонимы типов чанков ────────────────────────────────────────────────

# ← НОВОЕ: именованные псевдонимы для читаемости сигнатур
# Один чанк native-пути: список текстов предложений (передаётся в Modal)
_NativeSentChunk  = List[str]

# Один чанк razdel-пути: список предложений, каждое — список строк-токенов
# (передаётся в Modal)
_RazdelTokenChunk = List[List[str]]

# Один чанк офсетов: список (текст_предложения, символьный_офсет_в_документе)
# (хранится в wrapper, в Modal не передаётся)
_OffsetChunk      = List[Tuple[str, int]]

# Результат одного .map()-чанка от Modal: список предложений с токенами
_ChunkResult      = List[List[Dict[str, Any]]]


MODAL_APP_NAME = "booknlp-ru-udpipe"
MODAL_CLS_NAME = "UDPipeService"


# ─── Wrapper-класс ────────────────────────────────────────────────────────────

class UDPipeParser:
    """
    Тонкий клиент для UDPipeService (Modal).

    Отвечает за:
      1. Разбиение текста на чанки (razdel.sentenize / razdel.tokenize).
      2. Управление офсетами для razdel-пути.
      3. Маршрутизацию чанков через Modal .map() / .remote().
      4. Склейку результатов из Modal в финальный список.

    Вся логика тегирования и парсинга — в UDPipeService (udpipe_modal.py).

    Параметры parse_text / parse_batch:
      tokenizer          — "native"  : UDPipe токенизирует сам
                           "razdel"  : razdel токенизирует, UDPipe — horizontal
      output_format      — "dict"    : misc как строка CoNLL-U
                           "native"  : misc как словарь Python
      sentence_batch_size — количество предложений в одном Modal-чанке;
                           подбирать под конкретный GPU и объём текста
                           (меньше → меньше памяти на контейнер, больше чанков)

    Возвращаемые типы:
      parse_text native  → List[List[Dict]]
                           плоский список предложений, каждое — список токенов
      parse_text razdel  → List[Dict{"tokens", "sent_text", "sent_start"}]
                           каждое предложение + текст + char-офсет в документе
    """

    def __init__(self, app_name: str = MODAL_APP_NAME) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Подключаемся к задеплоенному Modal-классу по имени
        _UDPipeService = modal.Cls.from_name(app_name, MODAL_CLS_NAME)
        self._service  = _UDPipeService()
        self.logger.info(
            f"UDPipeParser подключён к Modal-приложению '{app_name}'."
        )

    # ─── Chunking helpers ────────────────────────────────────────────────────

    @staticmethod
    def _split_to_sentence_chunks(
        text: str,
        chunk_size: int,
    ) -> List[_NativeSentChunk]:
        """
        ← НОВОЕ: Native-путь.

        Разбивает текст на чанки текстов предложений с помощью razdel.sentenize.
        Офсеты не сохраняются — в Modal передаются только тексты предложений.
        UDPipe выполнит токенизацию на стороне контейнера.

        Args:
            text:       входной текст любого объёма.
            chunk_size: количество предложений в одном чанке (sentence_batch_size).

        Returns:
            List[List[str]] — список чанков; каждый чанк — список текстов
            предложений для одного вызова parse_sentence_chunk.

        Raises:
            ValueError: если chunk_size <= 0.
        """
        if not text or not text.strip():
            return []
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size должен быть > 0, получено: {chunk_size!r}"
            )
        sentences = list(sentenize(text))
        return [
            [s.text for s in sentences[i : i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _split_to_chunks(
        text: str,
        chunk_size: int,
        base_offset: int = 0,
    ) -> Tuple[List[_RazdelTokenChunk], List[_OffsetChunk]]:
        """
        ← НОВОЕ: Razdel-путь.

        Разбивает текст на чанки токенов с помощью razdel.sentenize +
        razdel.tokenize. Сохраняет символьные офсеты для постобработки.

        Каждый чанк содержит chunk_size предложений. Для каждого предложения
        выполняется razdel.tokenize → список строк-токенов.

        В Modal передаётся только token_chunks (List[List[str]]).
        offset_chunks остаётся в wrapper и присоединяется к результатам
        в _merge_chunks для восстановления позиций в исходном документе.

        Args:
            text:        входной текст.
            chunk_size:  количество предложений в одном чанке.
            base_offset: смещение начала text в родительском документе
                         (используется при обработке частей большого файла).

        Returns:
            token_chunks:  List[List[List[str]]] — для Modal.
            offset_chunks: List[List[Tuple[str, int]]] — для wrapper:
                           (текст_предложения, start_char_в_документе).

        Raises:
            ValueError: если chunk_size <= 0.
        """
        if not text or not text.strip():
            return []
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size должен быть > 0, получено: {chunk_size!r}"
            )
        sentences     = list(sentenize(text))
        token_chunks:  List[_RazdelTokenChunk] = []
        offset_chunks: List[_OffsetChunk]      = []

        for i in range(0, len(sentences), chunk_size):
            batch = sentences[i : i + chunk_size]

            # Токенизируем каждое предложение чанка через razdel
            token_chunks.append([
                [tok.text for tok in razdel_tokenize(s.text)]
                for s in batch
            ])

            # Сохраняем абсолютные char-офсеты для каждого предложения чанка
            offset_chunks.append([
                (s.text, base_offset + s.start)
                for s in batch
            ])

        return token_chunks, offset_chunks

    @staticmethod
    def _merge_chunks(
        results: List[_ChunkResult],
        offset_chunks: Optional[List[_OffsetChunk]] = None,
    ) -> List[Any]:
        """
        ← НОВОЕ: Склейка результатов из Modal .map().

        Native-путь (offset_chunks=None):
            Flatten List[_ChunkResult] → List[List[Dict]]
            (плоский список предложений, каждое — список токенов)

        Razdel-путь (offset_chunks заданы):
            Каждому предложению присоединяется sent_text и sent_start
            из offset_chunks.
            Returns: List[Dict{"tokens": ..., "sent_text": ..., "sent_start": ...}]

        Контракт: len(flat_results) == len(flat_offsets) при razdel-пути.
        Нарушение контракта логируется как WARNING — постобработка продолжается
        по min(len) во избежание краша.

        Args:
            results:       список результатов чанков из Modal .map().
            offset_chunks: список офсет-чанков из _split_to_chunks (razdel-путь)
                           или None (native-путь).

        Returns:
            List[List[Dict]] (native) или
            List[Dict{"tokens", "sent_text", "sent_start"}] (razdel).
        """
        # Разворачиваем чанки в плоский список предложений
        flat_results: List[List[Dict[str, Any]]] = [
            sent
            for chunk in results
            for sent in (chunk or [])   # guard: None от Modal при ошибке
        ]

        if offset_chunks is None:
            # Native-путь: возвращаем как есть
            return flat_results

        # Razdel-путь: присоединяем офсеты
        flat_offsets: List[Tuple[str, int]] = [
            (text, start)
            for chunk in offset_chunks
            for text, start in chunk
        ]

        if len(flat_results) != len(flat_offsets):
            # Несоответствие возможно при ошибке Modal для отдельного чанка
            logging.getLogger("UDPipeParser").warning(
                f"_merge_chunks: несоответствие количества предложений "
                f"({len(flat_results)}) и офсетов ({len(flat_offsets)}). "
                f"Используется min({len(flat_results)}, {len(flat_offsets)})."
            )

        return [
            {
                "tokens":     sent,
                "sent_text":  text,
                "sent_start": start,
            }
            for sent, (text, start) in zip(flat_results, flat_offsets)
        ]

    # ─── Public API ──────────────────────────────────────────────────────────

    def parse_text(
        self,
        text: str,
        tokenizer: TokenizerType = "native",
        output_format: OutputFormat = "dict",
        sentence_batch_size: int = 32,
    ) -> List[Any]:
        """
        ← ИЗМЕНЕНО: добавлены параметры tokenizer и sentence_batch_size;
        реализован чанкинг через .map() / .remote() вместо прямого вызова.

        Парсит один текст с разбиением на чанки.

        Алгоритм:
          1. razdel.sentenize разбивает text на предложения.
          2. Предложения группируются в чанки по sentence_batch_size.
          3. Один чанк → .remote(); несколько чанков → .map() (параллельно).
          4. Результаты склеиваются через _merge_chunks.

        Args:
            text:               входной текст.
            tokenizer:          "native" | "razdel".
            output_format:      "dict" | "native".
            sentence_batch_size: количество предложений в одном Modal-чанке.

        Returns:
            native: List[List[Dict]]                                   — предложения
            razdel: List[Dict{"tokens", "sent_text", "sent_start"}]    — с офсетами
        """
        if not text or not text.strip():
            return []

        if tokenizer == "native":
            chunks = self._split_to_sentence_chunks(text, sentence_batch_size)
            if not chunks:
                return []

            # Один чанк — .remote() дешевле (нет overhead .map)
            if len(chunks) == 1:
                result = self._service.parse_sentence_chunk.remote(
                    chunks[0], output_format=output_format
                )
                return result or []

            # Несколько чанков — параллельный .map()
            results = list(
                self._service.parse_sentence_chunk.map(
                    chunks,
                    kwargs={"output_format": output_format},
                )
            )
            return self._merge_chunks(results)

        else:  # razdel
            token_chunks, offset_chunks = self._split_to_chunks(
                text, sentence_batch_size
            )
            if not token_chunks:
                return []

            if len(token_chunks) == 1:
                modal_result = (
                    self._service.parse_sentence_chunk_razdel.remote(
                        token_chunks[0], output_format=output_format
                    ) or []
                )
                return self._merge_chunks([modal_result], offset_chunks)

            results = list(
                self._service.parse_sentence_chunk_razdel.map(
                    token_chunks,
                    kwargs={"output_format": output_format},
                )
            )
            return self._merge_chunks(results, offset_chunks)

    def parse_batch(
        self,
        texts: List[str],
        tokenizer: TokenizerType = "native",
        output_format: OutputFormat = "dict",
        sentence_batch_size: int = 32,
    ) -> List[List[Any]]:
        """
        ← ИЗМЕНЕНО: добавлены tokenizer и sentence_batch_size;
        все чанки всех текстов отправляются одним .map() (было: цикл .remote()).

        Парсит пакет текстов: sentenize × N → все чанки → единый .map().

        Алгоритм:
          1. Для каждого текста строим чанки → собираем в один flat-список.
          2. Единый .map() → Modal обрабатывает все чанки параллельно.
          3. Результаты разрезаются обратно по текстам и склеиваются.

        Args:
            texts:              список входных текстов.
            tokenizer:          "native" | "razdel".
            output_format:      "dict" | "native".
            sentence_batch_size: количество предложений в одном Modal-чанке.

        Returns:
            Список результатов parse_text для каждого входного текста.
        """
        if not texts:
            return []

        if tokenizer == "native":
            all_chunks: List[_NativeSentChunk] = []
            chunk_counts: List[int] = []

            for text in texts:
                if not text or not text.strip():
                    chunk_counts.append(0)
                    continue
                chunks = self._split_to_sentence_chunks(text, sentence_batch_size)
                chunk_counts.append(len(chunks))
                all_chunks.extend(chunks)

            if not all_chunks:
                return [[] for _ in texts]

            # Единый .map() для всех чанков всех текстов
            all_results = list(
                self._service.parse_sentence_chunk.map(
                    all_chunks,
                    kwargs={"output_format": output_format},
                )
            )

            # Разрезаем результаты обратно по текстам
            output: List[List[Any]] = []
            idx = 0
            for n in chunk_counts:
                if n == 0:
                    output.append([])
                else:
                    output.append(self._merge_chunks(all_results[idx : idx + n]))
                    idx += n
            assert idx == len(all_results), (
                f"parse_batch native: idx={idx} != "
                f"len(all_results)={len(all_results)}. "
                f"Нарушен контракт chunk_counts."
            )
            return output

        else:  # razdel
            all_token_chunks:  List[_RazdelTokenChunk] = []
            all_offset_chunks: List[_OffsetChunk]      = []
            chunk_counts: List[int] = []

            for text in texts:
                if not text or not text.strip():
                    chunk_counts.append(0)
                    continue
                tc, oc = self._split_to_chunks(text, sentence_batch_size)
                chunk_counts.append(len(tc))
                all_token_chunks.extend(tc)
                all_offset_chunks.extend(oc)

            if not all_token_chunks:
                return [[] for _ in texts]

            all_results = list(
                self._service.parse_sentence_chunk_razdel.map(
                    all_token_chunks,
                    kwargs={"output_format": output_format},
                )
            )

            output: List[List[Any]] = []
            idx = 0
            for n in chunk_counts:
                if n == 0:
                    output.append([])
                else:
                    output.append(
                        self._merge_chunks(
                            all_results[idx : idx + n],
                            all_offset_chunks[idx : idx + n],
                        )
                    )
                    idx += n
            assert idx == len(all_results), (
                f"parse_batch native: idx={idx} != "
                f"len(all_results)={len(all_results)}. "
                f"Нарушен контракт chunk_counts."
            )
            return output


# ─── Вспомогательные функции вывода ──────────────────────────────────────────
# Дублируются из udpipe_modal.py для автономного запуска wrapper-тестов.

_CONLLU_HEADER = (
    f"  {'ID':>4}  "
    f"{'FORM':<14} "
    f"{'LEMMA':<14} "
    f"{'UPOS':<7} "
    f"{'XPOS':<12} "
    f"{'FEATS':<35} "
    f"{'HEAD':>4}  "
    f"{'DEPREL':<12} "
    f"{'DEPS':<6}  "
    f"MISC"
)
_CONLLU_SEP = "  " + "─" * (len(_CONLLU_HEADER) - 2)
_CONLLU_HEADER = (
    f"  {'ID':>4}  {'FORM':<14} {'LEMMA':<14} "
    f"{'UPOS':<7} {'XPOS':<12} {'HEAD':>4}  "
    f"{'DEPREL':<12} {'DEPS':<6}  MISC"
)

def _print_sentence_table(
    sent_idx: int,
    tokens: List[Dict[str, Any]],
    sent_text: str = "",
    sent_start: Optional[int] = None,
) -> None:
    """
    Выводит предложение как таблицу CoNLL-U со всеми 10 полями.

    Заголовок:
        Предложение N:  # text = <текст>  [start=<офсет>]
        ID   FORM   LEMMA   UPOS   XPOS   FEATS   HEAD   DEPREL   DEPS   MISC
        ───────────────────────────────────────────────────────────────────────
    """
    offset_str = f"  [start={sent_start}]" if sent_start is not None else ""
    if sent_text:
        print(f"\n  Предложение {sent_idx}:  # text = {sent_text}{offset_str}")
    else:
        print(f"\n  Предложение {sent_idx}:{offset_str}")

    print(_CONLLU_HEADER)
    print(_CONLLU_SEP)

    for tok in tokens:
        print(
            f"  {tok['id']:>4}  {tok['form']:<14} {tok['lemma']:<14} "
            f"{tok['upos']:<7} {tok['xpos']:<12} {tok['head']:>4}  "
            f"{tok['deprel']:<12} {tok['deps']:<6}  {tok['misc']}"
        )
        # FEATS выводится полностью отдельной строкой
        if tok["feats"] != "_":
            print(f"        ↳ feats: {tok['feats']}")


def _print_token_full(tok: Dict[str, Any]) -> None:
    """Выводит все 10 CoNLL-U полей токена в вертикальном формате."""
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 32)
    print(f"     [CoNLL-U поля — 10 стандартных полей]")
    print(f"     id:     {tok['id']}")
    print(f"     form:   {tok['form']}")
    print(f"     lemma:  {tok['lemma']}")
    print(f"     upos:   {tok['upos']}")
    print(f"     xpos:   {tok['xpos']}")
    print(f"     feats:  {tok['feats']}")
    print(f"     head:   {tok['head']}")
    print(f"     deprel: {tok['deprel']}")
    print(f"     deps:   {tok['deps']}")
    print(f"     misc:   {tok['misc']!r}")


def _print_misc_summary(results: List[Any], label: str) -> None:
    """Сводная таблица непустых MISC-значений."""
    print(f"\n  Уникальные MISC ({label}):")
    seen: Dict[str, str] = {}
    for item in results:
        # Поддерживаем оба формата: List[Dict] (native) и Dict{"tokens"} (razdel)
        tokens = item["tokens"] if isinstance(item, dict) else item
        for tok in tokens:
            m = tok["misc"]
            key = repr(m)
            if m and m != "_" and m != {} and key not in seen:
                seen[key] = tok["form"]
    if seen:
        for val, form in seen.items():
            print(f"    {form:<16} → {val}")
    else:
        print("    (нет непустых MISC)")


# ─── Юнит-тесты методов разбиения (без Modal) ────────────────────────────────

def _run_unit_tests() -> None:
    """
    ← НОВОЕ: Юнит-тесты _split_to_sentence_chunks, _split_to_chunks,
    _merge_chunks без обращения к Modal.
    """
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print("ЮНИТ-ТЕСТЫ (без Modal)")
    print(f"{'═' * 72}")

    text = (
        "Нет!\n"
        "Это невозможно,— сказал он.\n"
        "«Правда?» — спросила она.\n"
        "Он молчал.\n"
        "Она ждала."
    )
    # ── _split_to_sentence_chunks ─────────────────────────────────────────
    print(f"\n{sep}")
    print("_split_to_sentence_chunks  (chunk_size=2)")
    print(sep)
    chunks_n = UDPipeParser._split_to_sentence_chunks(text, chunk_size=2)
    print(f"Предложений всего: {sum(len(c) for c in chunks_n)}, чанков: {len(chunks_n)}")
    for i, ch in enumerate(chunks_n):
        print(f"  Чанк {i}: {ch}")

    # chunk_size=1
    chunks_n1 = UDPipeParser._split_to_sentence_chunks(text, chunk_size=1)
    print(f"chunk_size=1 → {len(chunks_n1)} чанков  "
          f"({'✅' if len(chunks_n1) == 5 else '❌'})")

    # Пустой текст
    chunks_empty = UDPipeParser._split_to_sentence_chunks("", chunk_size=32)
    print(f"Пустой текст → {chunks_empty}  "
          f"({'✅' if chunks_empty == [] else '❌'})")

    # chunk_size <= 0
    try:
        UDPipeParser._split_to_sentence_chunks(text, chunk_size=0)
        print("chunk_size=0 → ❌ (ожидался ValueError)")
    except ValueError as e:
        print(f"chunk_size=0 → ValueError: {e}  ✅")

    # ── _split_to_chunks ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("_split_to_chunks  (chunk_size=2, base_offset=100)")
    print(sep)
    tc, oc = UDPipeParser._split_to_chunks(text, chunk_size=2, base_offset=100)
    print(f"Чанков: {len(tc)}  (token_chunks: {len(tc)}, offset_chunks: {len(oc)})")
    print(f"Количества совпадают: {'✅' if len(tc) == len(oc) else '❌'}")
    for i, (t_chunk, o_chunk) in enumerate(zip(tc, oc)):
        print(f"  Чанк {i}:")
        for j, (tokens, (sent_text, start)) in enumerate(zip(t_chunk, o_chunk)):
            print(f"    [{j}] start={start:>5}  tokens={tokens[:4]}{'...' if len(tokens) > 4 else ''}")
            print(f"         text={sent_text!r}")
    print(f"base_offset применён: "
          f"{'✅' if oc[0][0][1] >= 100 else '❌'} "
          f"(первый офсет = {oc[0][0][1]})")

    # ── _merge_chunks (native-путь) ───────────────────────────────────────
    print(f"\n{sep}")
    print("_merge_chunks  (native-путь, offset_chunks=None)")
    print(sep)
    # Синтетические данные для теста без Modal
    mock_results: List[_ChunkResult] = [
        [[{"id": 1, "form": "Нет",  "lemma": "нет", "upos": "PART",
           "xpos": "_", "feats": "_", "head": 0, "deprel": "root",
           "deps": "_", "misc": "SpaceAfter=No"}]],
        [[{"id": 1, "form": "Это",  "lemma": "это", "upos": "PRON",
           "xpos": "_", "feats": "_", "head": 2, "deprel": "nsubj",
           "deps": "_", "misc": "_"}]],
    ]
    merged_n = UDPipeParser._merge_chunks(mock_results)
    print(f"Входных чанков: {len(mock_results)}, предложений после merge: {len(merged_n)}")
    print(f"Тип элемента: {type(merged_n[0]).__name__}  "
          f"({'✅' if isinstance(merged_n[0], list) else '❌ ожидался list'})")

    # ── _merge_chunks (razdel-путь) ───────────────────────────────────────
    print(f"\n{sep}")
    print("_merge_chunks  (razdel-путь, offset_chunks заданы)")
    print(sep)
    mock_offsets: List[_OffsetChunk] = [
        [("Нет!", 0)],
        [("Это невозможно.", 5)],
    ]
    merged_r = UDPipeParser._merge_chunks(mock_results, mock_offsets)
    print(f"Предложений после merge: {len(merged_r)}")
    for item in merged_r:
        print(f"  sent_start={item['sent_start']:>4}  "
              f"sent_text={item['sent_text']!r:<25}  "
              f"tokens[0]={item['tokens'][0]['form']!r}")
    ok = (
        isinstance(merged_r[0], dict) and
        "tokens" in merged_r[0] and
        "sent_text" in merged_r[0] and
        "sent_start" in merged_r[0]
    )
    print(f"Структура Dict{{tokens,sent_text,sent_start}}: {'✅' if ok else '❌'}")


# ─── Интеграционные тесты (с Modal) ─────────────────────────────────────────

def _run_integration_tests(parser: UDPipeParser) -> None:
    """
    ← НОВОЕ: Интеграционные тесты parse_text / parse_batch через Modal.
    Проверяет оба tokenizer-пути, оба формата misc, чанкинг и офсеты.
    """
    import json

    sep = "─" * 72
    SEP = "═" * 72

    text_single = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься."
    )
    text_multi = (
        "Нет!\n"
        "Это невозможно,— сказал он.\n"
        "«Правда?» — спросила она."
    )

    # ── 1. parse_text: native, dict ───────────────────────────────────────
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_text  (tokenizer='native', output_format='dict')")
    print(SEP)
    result_nd = parser.parse_text(
        text_multi, tokenizer="native", output_format="dict", sentence_batch_size=32
    )
    print(f"Предложений: {len(result_nd)}")
    for i, tokens in enumerate(result_nd, 1):
        _print_sentence_table(i, tokens)
    _print_misc_summary(result_nd, "dict")
    if result_nd:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_nd[0][0], ensure_ascii=False, indent=2))

    # ── 2. parse_text: native, native ─────────────────────────────────────
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_text  (tokenizer='native', output_format='native')")
    print(SEP)
    result_nn = parser.parse_text(
        text_multi, tokenizer="native", output_format="native"
    )
    print(f"Предложений: {len(result_nn)}")
    for i, tokens in enumerate(result_nn, 1):
        _print_sentence_table(i, tokens)
    _print_misc_summary(result_nn, "native")
    if result_nn:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_nn[0][0], ensure_ascii=False, indent=2))

    # ── 3. parse_text: razdel, dict ───────────────────────────────────────
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_text  (tokenizer='razdel', output_format='dict')")
    print(SEP)
    result_rd = parser.parse_text(
        text_multi, tokenizer="razdel", output_format="dict", sentence_batch_size=32
    )
    print(f"Предложений: {len(result_rd)}")
    for i, item in enumerate(result_rd, 1):
        _print_sentence_table(
            i, item["tokens"], item["sent_text"], item["sent_start"]
        )
    _print_misc_summary(result_rd, "dict/razdel")
    if result_rd:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_rd[0]["tokens"][0], ensure_ascii=False, indent=2))
        print(f"\n  Офсеты предложений:")
        for item in result_rd:
            print(f"    start={item['sent_start']:>5}  {item['sent_text']!r}")

    # ── 4. parse_text: razdel, native ─────────────────────────────────────
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_text  (tokenizer='razdel', output_format='native')")
    print(SEP)
    result_rn = parser.parse_text(
        text_multi, tokenizer="razdel", output_format="native"
    )
    print(f"Предложений: {len(result_rn)}")
    for i, item in enumerate(result_rn, 1):
        _print_sentence_table(
            i, item["tokens"], item["sent_text"], item["sent_start"]
        )
    _print_misc_summary(result_rn, "native/razdel")
    if result_rn:
        _print_token_full(result_rn[0]["tokens"][0])

    # ── 5. Чанкинг: chunk_size=1 (каждое предложение — отдельный Modal-вызов)
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_text  (sentence_batch_size=1, проверка чанкинга)")
    print(SEP)
    result_chunk1 = parser.parse_text(
        text_multi, tokenizer="native", output_format="dict", sentence_batch_size=1
    )
    result_chunk32 = parser.parse_text(
        text_multi, tokenizer="native", output_format="dict", sentence_batch_size=32
    )
    # Форма должна совпадать независимо от chunk_size
    forms1  = [[t["form"] for t in s] for s in result_chunk1]
    forms32 = [[t["form"] for t in s] for s in result_chunk32]
    match   = forms1 == forms32
    print(f"chunk_size=1  → {len(result_chunk1)} предл.")
    print(f"chunk_size=32 → {len(result_chunk32)} предл.")
    print(f"Формы совпадают: {'✅' if match else '❌'}")

    # ── 6. parse_batch ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("ИНТЕГРАЦИЯ: parse_batch  (оба tokenizer-пути)")
    print(SEP)
    batch = [
        "Он думал о море.",
        "Кот лежал на диване.",
        text_multi,
        "",   # пустой текст — должен вернуть []
    ]
    for tok_type in ("native", "razdel"):
        results_b = parser.parse_batch(
            batch, tokenizer=tok_type, output_format="dict", sentence_batch_size=32
        )
        print(f"\n  tokenizer='{tok_type}' → {len(results_b)} результатов:")
        for i, (txt, sents) in enumerate(zip(batch, results_b)):
            if tok_type == "native":
                n_s = len(sents)
                n_t = sum(len(s) for s in sents)
            else:
                n_s = len(sents)
                n_t = sum(len(item["tokens"]) for item in sents)
            print(f"    [{i}] {txt[:35]!r:38} → {n_s} предл., {n_t} токенов")

    # ── 7. Пустой ввод ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("ПУСТОЙ ВВОД → ожидается []")
    print(SEP)
    e1 = parser.parse_text("",    tokenizer="native")
    e2 = parser.parse_text("   ", tokenizer="razdel")
    e3 = parser.parse_batch([],   tokenizer="native")
    print(f"  parse_text('')  native: {e1}  {'✅' if e1 == [] else '❌'}")
    print(f"  parse_text(' ') razdel: {e2}  {'✅' if e2 == [] else '❌'}")
    print(f"  parse_batch([]):        {e3}  {'✅' if e3 == [] else '❌'}")

    # ── 8. Сравнение native vs parse_batch[2] ─────────────────────────────
    print(f"\n{SEP}")
    print("СРАВНЕНИЕ: parse_text vs parse_batch[2]  (native, text_multi)")
    print(SEP)
    pt = parser.parse_text(text_multi, tokenizer="native", output_format="dict")
    pb = parser.parse_batch(
        [batch[0], batch[1], text_multi], tokenizer="native", output_format="dict"
    )[2]
    forms_pt = [[t["form"] for t in s] for s in pt]
    forms_pb = [[t["form"] for t in s] for s in pb]
    match_pb = forms_pt == forms_pb
    print(f"  parse_text vs parse_batch[2]: {'✅ совпадают' if match_pb else '❌ расходятся'}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Юнит-тесты (без Modal) — всегда
    _run_unit_tests()

    # 2. Интеграционные тесты (с Modal) — требуют запущенного сервиса
    parser = UDPipeParser()
    _run_integration_tests(parser)

    print(f"\n{'=' * 72}")
    print(f"{'✅ Все тесты завершены':^72}")
    print(f"{'=' * 72}")
