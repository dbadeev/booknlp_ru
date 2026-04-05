#!/usr/bin/env python3
"""
Локальная обёртка для SpaCy парсера (Modal).

Wrapper — тонкий клиент. Три обязанности:
  1. Сентенизация текста (razdel.sentenize) и разбивка на чанки.
  2. Маршрутизация чанков в нужный Modal-метод (razdel / native ветка).
  3. Сборка результатов чанков в единый ответ.

Вся NLP-логика (токенизация, pipeline, форматирование) — в spacy_modal.py.

Три пути токенизации (сентенизация — всегда razdel.sentenize):

  razdel (внешний токенизатор):
    sentenize → List[List[(text, start_char)]] → service.parse_sentence_chunk.map(chunks)

  internal (встроенный токенизатор spaCy):
    sentenize → List[List[str]] → service.parse_sentence_chunk_native.map(chunks)

  native_ru (spaCy rule-based + SynTagRus merge patterns):  [native_ru]
    sentenize → List[List[str]] → service.parse_sentence_chunk_native.map(chunks,
                                      kwargs={..., "tokenizer": "native_ru"})

Использование:
  from spacy_wrapper import SpacyParser
  parser = SpacyParser()

  # 6 вариантов (2 формата × 3 токенизатора):
  result = parser.parse_text(text, output_format="native",  tokenizer="internal")
  result = parser.parse_text(text, output_format="native",  tokenizer="razdel")
  result = parser.parse_text(text, output_format="native",  tokenizer="native_ru")  # [native_ru]
  result = parser.parse_text(text, output_format="conllu",  tokenizer="internal")
  result = parser.parse_text(text, output_format="conllu",  tokenizer="razdel")
  result = parser.parse_text(text, output_format="conllu",  tokenizer="native_ru")  # [native_ru]

  # chunk_size подбирается под GPU и тип текста (по умолчанию 32):
  result = parser.parse_text(text, tokenizer="native_ru", chunk_size=16)
"""
import argparse
import logging
import sys

import modal
from razdel import sentenize
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union, cast, overload

import logging as _logging

_merge_logger = _logging.getLogger(__name__)

OutputFormat = Literal["native", "conllu"]
# [native_ru] Добавлено значение "native_ru" в тип TokenizerType.
TokenizerType = Literal["internal", "razdel", "native_ru"]
# noinspection DuplicatedCode
default_chunk_size: int = 32  # предложений на чанк; подбирается под GPU и тип текста


# ─── Типы для аннотаций токенов (подавляют предупреждения IDE) ──────────────
class TokenDict(TypedDict, total=False):
    id: int
    start_char: int
    end_char: int
    form: str
    norm: str
    lower: str
    shape: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    n_lefts: int
    n_rights: int
    children: List[int]
    ent_type: str
    ent_iob: str
    is_sent_start: bool
    whitespace: str
    misc: str
    is_alpha: bool
    is_digit: bool
    is_punct: bool
    is_space: bool
    is_stop: bool
    is_oov: bool
    like_num: bool
    like_url: bool
    like_email: bool
    has_vector: bool
    cluster: int
    vector_norm: float


class SentenceDict(TypedDict, total=False):
    text: str
    start_char: int
    end_char: int
    words: List[TokenDict]
    entities: List[Dict[str, Any]]


# ─── SpacyParser ──────────────────────────────────────────────────────────────
class SpacyParser:
    """
    Клиент для SpaCy парсера (Modal).

    Поддерживает:
      - 2 формата:     native (полный), conllu (стандарт UD)
      - 3 токенизатора: internal (spaCy), razdel (внешний), native_ru  [native_ru]

    Сентенизация выполняется локально (razdel.sentenize) до отправки в Modal.
    Чанки распределяются по контейнерам через .map().
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-spacy",
                "SpacyService",
            )()
            self.logger.info("✓ Connected to SpaCy via Modal.")
        except Exception as exc:
            self.logger.error(f"❌ Failed to connect to Modal: {exc}")
            raise

    # ─── Chunking ─────────────────────────────────────────────────────────
    # noinspection DuplicatedCode
    @staticmethod
    def _split_to_chunks(
        text: str,
        chunk_size: int,
        base_offset: int = 0,
    ) -> List[List[Tuple[str, int]]]:
        """
        Razdel path: разбивает текст на чанки с символьными офсетами.
        Returns: List[List[(sentence_text, start_char_in_original)]]
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
        Native / native_ru path: разбивает текст на чанки предложений (только тексты).
        Returns: List[List[str]]
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [s.text for s in sentences[i: i + chunk_size]]  # ✅ только текст
            for i in range(0, len(sentences), chunk_size)
        ]

    # noinspection DuplicatedCode
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
            # ── SpaceAfter последнего токена промежуточных предложений ──
            for i in range(len(parts) - 1):
                lines = parts[i].split("\n")
                for j in range(len(lines) - 1, -1, -1):
                    if lines[j] and not lines[j].startswith("#"):
                        cols = lines[j].split("\t")
                        if len(cols) == 10 and cols[9] == "SpaceAfter=No":
                            cols[9] = "_"
                            lines[j] = "\t".join(cols)
                        elif len(cols) != 10:
                            _merge_logger.warning(
                                "Unexpected CoNLL-U column count %d in line: %r", len(cols), lines[j]
                            )
                        break
                parts[i] = "\n".join(lines)
            return "\n\n".join(parts) + "\n" if parts else ""

        sentences = [sent for cr in chunk_results for sent in cr]
        # для native — исправляем misc последнего токена промежуточных предложений ──
        for sent in sentences[:-1]:
            if sent.get("words"):
                last_tok = sent["words"][-1]
                if last_tok.get("misc") == "SpaceAfter=No":
                    last_tok["misc"] = "_"
        return sentences

    # noinspection DuplicatedCode
    @staticmethod
    def _fix_boundary_misc(result: Any, output_format: str) -> Any:
        """
        Одиночный чанк с несколькими предложениями: фиксирует misc/SpaceAfter
        для последнего токена промежуточных предложений (не последнего).
        Вызывается только при len(chunks)==1 (multi-sentence chunk).
        """
        if output_format == "native" and isinstance(result, list) and len(result) > 1:
            for sent in result[:-1]:
                if sent.get("words"):
                    last_tok = sent["words"][-1]
                    if last_tok.get("misc") == "SpaceAfter=No":
                        last_tok["misc"] = "_"
        elif output_format == "conllu" and isinstance(result, str):
            parts = result.strip().split("\n\n")
            if len(parts) > 1:
                for i in range(len(parts) - 1):
                    lines = parts[i].split("\n")
                    for j in range(len(lines) - 1, -1, -1):
                        if lines[j] and not lines[j].startswith("#"):
                            cols = lines[j].split("\t")
                            if len(cols) == 10 and cols[9] == "SpaceAfter=No":
                                cols[9] = "_"
                                lines[j] = "\t".join(cols)
                            elif len(cols) != 10:
                                _merge_logger.warning(
                                    "Unexpected CoNLL-U column count %d in line: %r", len(cols), lines[j]
                                )
                            break
                    parts[i] = "\n".join(lines)
                result = "\n\n".join(parts) + "\n"
        return result

    # ─── Public API ───────────────────────────────────────────────────────
    @overload
    def parse_text(
            self,
            text: str,
            output_format: Literal["native"],
            tokenizer: TokenizerType = ...,
            chunk_size: int = ...,
            batch_size: int = ...,
    ) -> List[Dict[str, Any]]:
        ...

    @overload
    def parse_text(
            self,
            text: str,
            output_format: Literal["conllu"],
            tokenizer: TokenizerType = ...,
            chunk_size: int = ...,
            batch_size: int = ...,
    ) -> str:
        ...

    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        chunk_size: int = default_chunk_size,
        batch_size: int = 32,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Парсит текст через SpaCy в Modal.

        Алгоритм:
          1. Разбить текст на предложения (razdel.sentenize).
          2. Сгруппировать в чанки по chunk_size.
          3. Один чанк → .remote(); несколько → .map() (параллельно).
          4. Склеить результаты.

        Args:
            text:         Входной текст
            output_format: 'native' | 'conllu'
            tokenizer:    'internal' | 'razdel' | 'native_ru'  [native_ru]
            chunk_size:   Предложений на чанк (подбирается под GPU).
                          По умолчанию default_chunk_size = 32.
            batch_size:   Размер батча при обработке GPU (передаётся в Modal). Default: 32.
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
                    result = self.service.parse_sentence_chunk.remote(
                        chunks[0], output_format=output_format, batch_size=batch_size
                    )
                    return self._fix_boundary_misc(result, output_format)
                chunk_results = list(
                    self.service.parse_sentence_chunk.map(
                        chunks, kwargs={"output_format": output_format, "batch_size": batch_size}
                    )
                )
                return self._merge_chunks(chunk_results, output_format)
            else:
                # [native_ru] Ветка обрабатывает tokenizer="internal" и tokenizer="native_ru".
                # Оба используют _split_to_sentence_chunks (без абсолютных офсетов).
                # Значение tokenizer передаётся в Modal через kwargs, чтобы
                # parse_sentence_chunk_native направил его в _make_doc.
                chunks = self._split_to_sentence_chunks(text, chunk_size)
                if not chunks:
                    return [] if output_format == "native" else ""
                if len(chunks) == 1:
                    result = self.service.parse_sentence_chunk_native.remote(
                        chunks[0],
                        output_format=output_format,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                    )
                    return self._fix_boundary_misc(result, output_format)
                chunk_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        chunks,
                        kwargs={
                            "output_format": output_format,
                            "tokenizer": tokenizer,  # [native_ru] передаём во все чанки
                            "batch_size": batch_size,
                        },
                    )
                )
                return self._merge_chunks(chunk_results, output_format)
        except Exception as exc:
            self.logger.error(f"❌ Error during spaCy parsing: {exc}")
            raise

    # noinspection DuplicatedCode
    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        chunk_size: int = default_chunk_size,
        batch_size: int = 32,
    ) -> List[Union[List[Dict[str, Any]], str]]:
        """
        Разбивает все тексты на чанки и отправляет их единым .map() —
        Modal распределяет по доступным контейнерам.

        Args:
            texts:        Список текстов
            output_format: 'native' | 'conllu'
            tokenizer:    'internal' | 'razdel' | 'native_ru'  [native_ru]
            chunk_size:   Предложений на чанк
            batch_size:   Размер батча при обработке GPU (передаётся в Modal). Default: 32.
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
                        all_chunks, kwargs={"output_format": output_format, "batch_size": batch_size,}
                    )
                )
            else:
                # [native_ru] Ветка обрабатывает tokenizer="internal" и tokenizer="native_ru".
                # tokenizer передаётся через kwargs во все вызовы .map(),
                # чтобы parse_sentence_chunk_native применил нужный _make_doc.
                all_chunks_native: List[List[str]] = []
                for text in texts:
                    text_chunks = self._split_to_sentence_chunks(text, chunk_size)
                    chunks_per_text.append(len(text_chunks))
                    all_chunks_native.extend(text_chunks)
                if not all_chunks_native:
                    return [[] if output_format == "native" else "" for _ in texts]
                all_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        all_chunks_native,
                        kwargs={
                            "output_format": output_format,
                            "tokenizer": tokenizer,  # [native_ru] передаём "internal" или "native_ru"
                            "batch_size": batch_size,
                        },
                    )
                )

            # Reassemble: восстанавливаем результаты по текстам
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
            self.logger.error(f"❌ Error during batch parsing: {exc}")
            raise


# ─── Вспомогательная функция вывода ──────────────────────────────────────────
# noinspection DuplicatedCode
def _print_token_full(tok: TokenDict) -> None:
    """Выводит все поля токена в нативном формате spaCy."""
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"  ПОЗИЦИЯ:")
    print(f"    start_char:      {tok['start_char']}")
    print(f"    end_char:        {tok['end_char']}")
    print(f"  ФОРМА:")
    print(f"    form:            {tok['form']}")
    print(f"    norm:            {tok.get('norm', '—')}")
    print(f"    lower:           {tok.get('lower', '—')}")
    print(f"    shape:           {tok.get('shape', '—')}")
    print(f"  ЛЕММА И POS:")
    print(f"    lemma:           {tok['lemma']}")
    print(f"    upos:            {tok['upos']}")
    print(f"    xpos:            {tok['xpos']}")
    print(f"    feats:           {tok['feats']}")
    print(f"  СИНТАКСИС:")
    print(f"    head:            {tok['head']}")
    print(f"    deprel:          {tok['deprel']}")
    print(f"    n_lefts:         {tok.get('n_lefts', '—')}")
    print(f"    n_rights:        {tok.get('n_rights', '—')}")
    print(f"    children:        {tok.get('children', [])}")
    print(f"  СУЩНОСТИ:")
    print(f"    ent_type:        {tok.get('ent_type') or '—'}")
    print(f"    ent_iob:         {tok.get('ent_iob') or '—'}")
    print(f"  МЕТАДАННЫЕ:")
    print(f"    is_sent_start:   {tok.get('is_sent_start')}")
    print(f"    whitespace:      '{tok.get('whitespace', '')}'")
    print(f"    misc:            {tok.get('misc', '—')}")
    print(f"  ФЛАГИ:")
    print(f"    is_alpha:        {tok.get('is_alpha')}")
    print(f"    is_digit:        {tok.get('is_digit')}")
    print(f"    is_punct:        {tok.get('is_punct')}")
    print(f"    is_space:        {tok.get('is_space')}")
    print(f"    is_stop:         {tok.get('is_stop')}")
    print(f"    is_oov:          {tok.get('is_oov')}")
    print(f"    like_num:        {tok.get('like_num')}")
    print(f"    like_url:        {tok.get('like_url')}")
    print(f"    like_email:      {tok.get('like_email')}")
    print(f"  ВЕКТОР:")
    print(f"    has_vector:      {tok.get('has_vector')}")
    vn = tok.get("vector_norm")
    print(f"    vector_norm:     {vn if vn is not None else '—'}")

# ─── Утилита: вывод строки сравнения токенизаторов ──────────────────────────
def _print_comparison(text: str, results: Dict[str, Any]) -> None:
    """
    Выводит 3-строчный блок сравнения токенизаторов.
    results: {"internal": result, "razdel": result, "native_ru": result}
    """
    print(f"\n⚡ Сравнение всех трёх токенизаторов для: '{text}'")
    for name, res in results.items():
        forms = [w["form"] for s in res for w in s["words"]]
        print(f"   {name:<10}: {forms}")

# ─── Константа заголовка CoNLL-U ─────────────────────────────────────────────
CONLLU_HEADER = "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"


def _print_conllu(text: str, conllu: str) -> None:
    """Выводит CoNLL-U блок с текстом предложения и заголовком столбцов."""
    print(f"\n# text = {text}")
    print(CONLLU_HEADER)
    print(conllu)


# ─── __main__: тест через wrapper (с chunking) ───────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ap = argparse.ArgumentParser(description="SpaCy wrapper тест")
    ap.add_argument(
        "--tokenizer",
        # [native_ru] Добавлено значение "native_ru" в список допустимых аргументов CLI.
        choices=["internal", "razdel", "native_ru"],
        default="internal",
        help="Токенизатор (default: internal)",
    )
    ap.add_argument(
        "--output-format",
        choices=["native", "conllu"],
        default="native",
        dest="output_format",
        help="Формат вывода (default: native)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=default_chunk_size,
        dest="chunk_size",
        help=f"Предложений на чанк (default: {default_chunk_size})",
    )
    args = ap.parse_args()

    sep = "=" * 72
    print(sep)
    print("ПРОВЕРКА ДОСТУПНОСТИ MODAL-СЕРВИСА")
    print(sep)
    try:
        parser = SpacyParser()
    except Exception as e:
        print(f"⚠️ Modal-сервис недоступен: {e}")
        print("\nЗапустите сервис командой:")
        print("  modal deploy src/parsers/spacy_modal.py")
        sys.exit(1)

    text_single = "Кружка-термос стоит 500р."
    text_multi = (
        "Зло, которым пугаешь, не так зло. "
        "Москва — столица России. "
        "Кружка-термос стоит 500р."
    )
    # [native_ru] Текст с дефисными конструкциями для теста native_ru.
    text_compare = (
        "Все-таки кружка-термос стоит 500р., "
        "а какая-нибудь кресло-качалка 10 000р."
    )

    # ── Вариант 1: NATIVE + INTERNAL ──────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 1: NATIVE + INTERNAL TOKENIZER")
    print(sep)
    result_ni = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="internal",
        chunk_size=args.chunk_size,
    )
    # noinspection DuplicatedCode
    print(f"\nТекст: '{text_single}'")
    # noinspection DuplicatedCode
    for sentence in result_ni:
        sentence: SentenceDict
        print(
            f"\nПредложение: '{sentence['text']}' "
            f"(chars {sentence['start_char']}:{sentence['end_char']})"
        )
        if sentence.get("entities"):
            print(f"  Сущности: {[(e['text'], e['label']) for e in sentence['entities']]}")
        for token in sentence["words"]:
            token: TokenDict
            _print_token_full(token)

    # ── Вариант 2: NATIVE + RAZDEL ────────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 2: NATIVE + RAZDEL TOKENIZER")
    print(sep)
    result_nr = parser.parse_text(
        text_single,
        output_format="native",
        tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    print(f"\n⚡ Сравнение токенизаторов: '{text_single}'")
    # noinspection DuplicatedCode
    native_sents = cast(List[Dict[str, Any]], result_ni)
    razdel_sents = cast(List[Dict[str, Any]], result_nr)
    print(f"  internal: {[w['form'] for s in native_sents for w in s['words']]}")
    print(f"  razdel:   {[w['form'] for s in razdel_sents for w in s['words']]}")
    for sentence in result_nr:
        sentence: SentenceDict
        print(
            f"\nПредложение: '{sentence['text']}' "
            f"(chars {sentence['start_char']}:{sentence['end_char']})"
        )
        for token in sentence["words"]:
            _print_token_full(token)

    # ── 2b. — 3-way сравнение через parse.remote ─────────────────────────────────────────────
    result_cmp_int = parser.parse_text(text_compare, "native", "internal")
    result_cmp_rz = parser.parse_text(text_compare, "native", "razdel")
    result_cmp_nru = parser.parse_text(text_compare, "native", "native_ru")
    _print_comparison(text_compare, {
        "internal": result_cmp_int,
        "razdel": result_cmp_rz,
        "native_ru": result_cmp_nru,
    })

    # ── Вариант 3: CONLL-U + INTERNAL ─────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 3: CONLL-U + INTERNAL TOKENIZER")
    print(sep)
    _print_conllu(
        text_multi,
        parser.parse_text(
            text_multi,
            output_format="conllu",
            tokenizer="internal",
            chunk_size=args.chunk_size,
        ),
    )

    # ── Вариант 4: CONLL-U + RAZDEL ───────────────────────────────────────
    print(f"\n{sep}")
    print("ВАРИАНТ 4: CONLL-U + RAZDEL TOKENIZER")
    print(sep)
    _print_conllu(
        text_multi,
        parser.parse_text(
            text_multi,
            output_format="conllu",
            tokenizer="razdel",
            chunk_size=args.chunk_size,
        ),
    )

    # ── Вариант 5: NATIVE + NATIVE_RU  [native_ru] ──────────────────────────
    # ⚡ Полное сравнение всех трёх токенизаторов на text_compare
    print(f"\n{sep}")
    print("ВАРИАНТ 5: NATIVE + NATIVE_RU TOKENIZER  [native_ru]")
    print(sep)

    # Результаты для text_compare уже получены в Варианте 2b — переиспользуем.
    # cast() только для аннотации типа, не меняет данные во время выполнения.
    nru_sents  = cast(List[Dict[str, Any]], result_cmp_nru)
    ni_c_sents = cast(List[Dict[str, Any]], result_cmp_int)
    rz_c_sents = cast(List[Dict[str, Any]], result_cmp_rz)

    # 3-строчный блок сравнения: форм-списки для всех трёх токенизаторов
    _print_comparison(text_compare, {
        "internal":  ni_c_sents,
        "razdel":    rz_c_sents,
        "native_ru": nru_sents,
    })

    # Ожидаемые различия — ориентир при ручной проверке результатов
    print(f"\n  Ожидаемый результат:")
    print(f"    Все-таки → native_ru/razdel: 1 токен | internal: 3 токена")
    print(f"    кружка-термос  → razdel: 1 токен | internal/native_ru: 3 токена")
    print(f"    какая-нибудь   → native_ru/razdel: 1 токен | internal: 3 токена")
    print(f"    кресло-качалка → razdel: 1 токен | internal/native_ru: 3 токена")
    print(f"    10 000р.       → разбивка числа и единицы зависит от токенизатора")

    # Детальный вывод всех атрибутов токенов native_ru для text_compare
    print(f"\n--- Детальный вывод (native_ru, '{text_compare}') ---")
    for sentence in nru_sents:
        sentence: SentenceDict
        print(
            f"\nПредложение: '{sentence['text']}' "
            f"(chars {sentence['start_char']}:{sentence['end_char']})"
        )
        if sentence.get("entities"):
            print(f"  Сущности: {[(e['text'], e['label']) for e in sentence['entities']]}")
        for token in sentence["words"]:
            _print_token_full(token)

    # ── Вариант 6: CONLL-U + NATIVE_RU ────────────────────────────────────
    # [native_ru] Тест CoNLL-U вывода с токенизатором native_ru через wrapper.
    print(f"\n{sep}")
    print("ВАРИАНТ 6: CONLL-U + NATIVE_RU TOKENIZER  [native_ru]")
    print(sep)
    _print_conllu(
        text_compare,
        parser.parse_text(
            text_compare,
            output_format="conllu",
            tokenizer="native_ru",
            chunk_size=args.chunk_size,
        ),
    )

    # ── Вариант 7: MULTI-SENTENCE + INTERNAL (char_offset regression test) ───
    print(f"\n{sep}")
    print("ВАРИАНТ 7: MULTI-SENTENCE char_offset REGRESSION (internal)")
    print(sep)
    result_multi_int = parser.parse_text(
        text_multi,
        output_format="native",
        tokenizer="internal",
        chunk_size=args.chunk_size,
    )
    print(f"\nТекст: '{text_multi}'")
    for s in result_multi_int:
        print(f"  Предложение: '{s['text']}' (chars {s['start_char']}:{s['end_char']})")
        sent_forms = [w["form"] for w in s["words"]]
        print(f"    Токены: {sent_forms}")
        last_word = s["words"][-1]
        print(f"    Последний токен misc: {last_word.get('misc')} (ожидается '_' у промежуточных)")

    print(
        "\n  ⚠️  ОГРАНИЧЕНИЕ internal/native_ru: start_char=0 для всех предложений.\n"
        "      В razdel-пути start_char корректен (передаётся из sentenize).\n"
        "      В internal/native_ru-пути каждое предложение парсится без\n"
        "      исходного смещения — offset не передаётся в Modal."
    )

    # Regression assertion: start_char "Москва" == 35 (после "Зло, которым пугаешь, не так зло. ")
    # Для internal-пути start_char всегда 0 (предложение изолировано) — это осознанное ограничение.
    # Тест проверяет корректность misc у промежуточных предложений.
    if len(result_multi_int) > 1:
        mid_sent = result_multi_int[0]
        last_misc = mid_sent["words"][-1].get("misc")
        assert last_misc == "_", (
            f"❌ REGRESSION: misc последнего токена промежуточного предложения "
            f"должен быть '_', получено '{last_misc}'"
        )
        print("\n✅ misc regression PASSED")

    # То же для native_ru:
    result_multi_nru = parser.parse_text(
        text_multi,
        output_format="native",
        tokenizer="native_ru",
        chunk_size=args.chunk_size,
    )
    if len(result_multi_nru) > 1:
        last_misc_nru = result_multi_nru[0]["words"][-1].get("misc")
        assert last_misc_nru == "_", (
            f"❌ REGRESSION native_ru: misc='{last_misc_nru}', ожидается '_'"
        )
        print("✅ misc regression PASSED (native_ru)")

    print(f"\n{sep}")
    print("BATCH: CONLL-U + RAZDEL (2 текста)")
    print(sep)
    batch_texts = [text_single, "Зло, которым пугаешь, не так зло."]
    batch_results = parser.parse_batch(
        batch_texts,
        output_format="conllu",
        tokenizer="razdel",
        chunk_size=args.chunk_size,
    )
    for idx, (batch_text, batch_res) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"\n── Текст {idx}: '{batch_text}'")
        _print_conllu(batch_text, batch_res)

    # ── parse_batch + native_ru ────────────────────────────────────────────
    # [native_ru] Тест parse_batch с токенизатором native_ru.
    # Проверяет корректную передачу tokenizer="native_ru" через .map() kwargs
    # и правильную сборку результатов через _merge_chunks.
    print(f"\n{sep}")
    print("BATCH: NATIVE + NATIVE_RU (2 текста)  [native_ru]")
    print(sep)
    batch_texts_nru = [text_single, text_compare]
    batch_results_nru = parser.parse_batch(
        batch_texts_nru,
        output_format="native",
        tokenizer="native_ru",
        chunk_size=args.chunk_size,
    )
    for idx, (bt, br) in enumerate(zip(batch_texts_nru, batch_results_nru), 1):
        print(f"\n── Текст {idx}: '{bt}'")
        for sentence in br:
            tokens_forms = [w["form"] for w in sentence["words"]]
            print(f"   Токены: {tokens_forms}")

    # ── Regression: parse_batch с 1 текстом (razdel) ────────────────────
    batch_one = parser.parse_batch([text_single], output_format="native", tokenizer="razdel")
    assert isinstance(batch_one, list) and isinstance(batch_one[0], list), (
        f"❌ parse_batch должен возвращать List[List], получено {type(batch_one[0])}"
    )
    print("✅ parse_batch single-text type regression PASSED")

    print(f"\n{'✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ':^72}")
