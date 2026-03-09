# stanza_modal.py
# =============================================================================
# ИЗМЕНЕНИЯ по сравнению с предыдущей версией:
#   [NEW] Параметр tokenizer: TokenizerType = "internal" | "razdel" во всех
#         production-методах — аналогично spacy_modal.py.
#   [NEW] Единственный пайплайн self.nlp (вместо двух: nlp_raw + nlp_pretokenized).
#         Выбор пути токенизации производится динамически в _make_doc().
#   [NEW] RazdelTokenizer — внешний токенизатор на базе razdel (аналог spaCy).
#   [NEW] parse_sentence_chunk() — production-метод, razdel path:
#         принимает List[(sentence_text, start_char)], возвращает CoNLL-U | native.
#   [NEW] parse_sentence_chunk_native() — production-метод, internal path:
#         принимает List[str], возвращает CoNLL-U | native.
#   [NEW] batch_size во всех production-методах для управления GPU-памятью.
#   [NEW] Поля text/start_char/end_char на уровне предложения в _format_native().
#   [NEW] Параметр tokenizer добавлен в parse() и parse_batch() (backward compat).
#   [CHG] parse_batch() переименован семантически: теперь принимает List[str] текстов,
#         а не List[List[str]] токенов (pretokenized path → parse_sentence_chunk).
#   [REM] nlp_pretokenized удалён — заменён единым _make_doc() + tokenize_pretokenized.
#   [REM] _parse_misc_to_dict() перенесена из глобальной функции в @staticmethod.
# =============================================================================

import modal
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# ─── Типы ─────────────────────────────────────────────────────────────────────
# [NEW] TokenizerType — аналогично spacy_modal.py
TokenizerType = Literal["internal", "razdel"]
OutputFormat  = Literal["native", "conllu"]

# ─── Modal image ──────────────────────────────────────────────────────────────
# [CHG] Добавлен razdel в pip_install для внешней токенизации
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("stanza", "torch", "razdel>=0.5.0")
    .run_commands("python -c 'import stanza; stanza.download(\"ru\")'")
)

app = modal.App("booknlp-ru-stanza")


# ─── Service ───────────────────────────────────────────────────────────────────
@app.cls(image=image, gpu="T4", timeout=600, scaledown_window=300)
class StanzaService:
    """
    Modal-сервис для морфо-синтаксического анализа русского текста через Stanza.

    Поддерживает два режима токенизации:
      internal — встроенный нейросетевой токенизатор Stanza (tokenize processor)
      razdel   — внешний ML-токенизатор razdel; подаётся как pretokenized ввод

    Два формата вывода:
      native — полный набор атрибутов Stanza (List[Dict])
      conllu — строка в стандарте Universal Dependencies (str)

    ВАЖНО: Сентенизация выполняется в wrapper ДО отправки в Modal.
    Production-методы принимают уже разбитые предложения (чанки).
    Повторная сентенизация внутри Modal исключена (no_ssplit=True для internal,
    pretokenized=True для razdel).

    Основные методы (production path из wrapper):
      parse_sentence_chunk        — razdel path:   List[(text, start_char)]
      parse_sentence_chunk_native — internal path: List[str]

    Вспомогательные методы (local_entrypoint / прямые вызовы):
      parse       — одиночный текст целиком
      parse_batch — список текстов целиком
    """

    @modal.enter()
    def setup(self):
        import stanza
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StanzaService")

        # ── Единственный пайплайн ─────────────────────────────────────────
        # [CHG] Вместо двух пайплайнов (nlp_raw + nlp_pretokenized) — один.
        # Режим токенизации выбирается динамически в _make_doc().
        #
        # Процессоры:
        #   tokenize  — токенизация и сентенизация (отключается в production)
        #   pos       — части речи (UPOS + XPOS)
        #   lemma     — лемматизация
        #   depparse  — синтаксический разбор
        #   ner       — именованные сущности (доступен для ru)
        #
        # Недоступные для русского процессоры (только en/zh/de):
        #   sentiment    — анализ тональности
        #   constituency — дерево составляющих
        self.nlp = stanza.Pipeline(
            "ru",
            processors="tokenize,pos,lemma,depparse,ner",
            verbose=False,
            use_gpu=True,
        )
        self.logger.info("Stanza loaded (Single Pipeline with NER)!")

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    # [CHG] Перенесена из глобальной функции в @staticmethod класса
    def _parse_misc_to_dict(misc_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Конвертирует CoNLL-U MISC / FEATS строку в dict для native-режима."""
        if not misc_str or misc_str == "_":
            return None
        result: Dict[str, Any] = {}
        for item in misc_str.split("|"):
            if "=" in item:
                key, val = item.split("=", 1)
                result[key] = val
            else:
                result[item] = True  # булевый флаг без значения
        return result

    def _make_doc_internal(self, sentences: List[str]):
        """
        [NEW] Internal path: передаёт список текстов предложений в Stanza
        как Document с уже выполненной сентенизацией (no_ssplit=True эмулируется
        через предварительно созданный Document с заданными предложениями).

        Stanza не пересентенизирует: каждый элемент списка = одно предложение.
        """
        import stanza

        # Создаём Document с явно заданными предложениями — Stanza не будет
        # запускать собственный сентенизатор поверх уже разбитых предложений.
        doc_input = [stanza.Document([], text=s) for s in sentences]
        # Обрабатываем каждый документ через пайплайн
        # (для батчевой обработки используется _process_batch)
        return doc_input

    def _make_doc_razdel(self, sentences_with_offsets: List[Tuple[str, int]]):
        """
        [NEW] Razdel path: каждое предложение уже токенизировано razdel в wrapper.
        Передаём в Stanza как pretokenized ввод — внутренний токенизатор
        Stanza не задействуется.

        sentences_with_offsets: List[(sentence_text, start_char_in_original)]
        Возвращает: (List[stanza.Document], List[int]) — docs и char_offsets.
        """
        import stanza
        from razdel import tokenize as razdel_tokenize

        docs = []
        char_offsets = []
        for sent_text, char_offset in sentences_with_offsets:
            # Токенизируем razdel → получаем список токенов
            tokens = list(razdel_tokenize(sent_text))
            if not tokens:
                continue
            token_texts = [[t.text for t in tokens]]  # List[List[str]] для Stanza
            # tokenize_pretokenized=True: Stanza принимает токены как готовые,
            # не запускает собственную токенизацию и сентенизацию
            doc = stanza.Document(token_texts, text=sent_text)
            docs.append(doc)
            char_offsets.append(char_offset)
        return docs, char_offsets

    def _run_pipeline_batch(self, docs: list, batch_size: int = 16) -> list:
        """
        [NEW] Пакетно прогоняет список Document через пайплайн Stanza.

        Stanza Pipeline поддерживает батчевую обработку через
        nlp(list_of_docs). batch_size управляет GPU-памятью.

        Args:
            docs:       список stanza.Document
            batch_size: размер батча (подбирается под GPU и тип текстов)
        """
        results = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            # Stanza Pipeline принимает список Document и обрабатывает батчем
            processed = self.nlp(batch) if len(batch) > 1 else [self.nlp(batch[0])]
            results.extend(processed if isinstance(processed, list) else [processed])
        return results

    def _run_pipeline_razdel_batch(
        self, docs: list, batch_size: int = 16
    ) -> list:
        """
        [NEW] Батчевая обработка pretokenized документов (razdel path).
        Использует отдельный Pipeline с tokenize_pretokenized=True.
        Создаётся lazily при первом вызове.
        """
        import stanza

        # Lazy init: создаём претокенизированный пайплайн один раз
        if not hasattr(self, "_nlp_pretokenized"):
            self._nlp_pretokenized = stanza.Pipeline(
                "ru",
                processors="tokenize,pos,lemma,depparse,ner",
                verbose=False,
                use_gpu=True,
                tokenize_pretokenized=True,
            )
            self.logger.info("Stanza pretokenized pipeline initialized (lazy).")

        results = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            processed = (
                self._nlp_pretokenized(batch)
                if len(batch) > 1
                else [self._nlp_pretokenized(batch[0])]
            )
            results.extend(processed if isinstance(processed, list) else [processed])
        return results

    # ─── Форматирование вывода ────────────────────────────────────────────────

    def _format_conllu(self, doc) -> str:
        """
        CoNLL-U формат вывода.
        Поля: id, form, lemma, upos, xpos, feats, head, deprel, deps, misc.
        start_char/end_char/spaces_after отсутствуют (не входят в стандарт CoNLL-U).
        """
        lines = []
        for sent in doc.sentences:
            for word in sent.words:
                # Восстанавливаем SpaceAfter из token.spaces_after
                misc = "_"
                # Ищем соответствующий Token для этого Word
                for token in sent.tokens:
                    if any(int(w.id) == int(word.id) for w in token.words):
                        sa = getattr(token, "spaces_after", None)
                        if sa == "":
                            misc = "SpaceAfter=No"
                        break
                lines.append(
                    f"{int(word.id)}\t"
                    f"{word.text}\t"
                    f"{word.lemma}\t"
                    f"{word.upos}\t"
                    f"{word.xpos or '_'}\t"
                    f"{word.feats or '_'}\t"
                    f"{int(word.head)}\t"
                    f"{word.deprel}\t"
                    f"_\t"  # DEPS (enhanced deps — Stanza не поддерживает для ru)
                    f"{misc}"
                )
            lines.append("")  # пустая строка между предложениями
        return "\n".join(lines)

    def _format_native(self, doc, char_offset: int = 0) -> List[Dict[str, Any]]:
        """
        [CHG] Нативный формат вывода — максимально полное представление Doc.

        ИЗМЕНЕНИЯ:
          [NEW] text, start_char, end_char на уровне предложения (аналог spaCy).
          [NEW] char_offset — смещение символов от начала исходного текста
                (передаётся из parse_sentence_chunk для razdel-пути).
          [CHG] _parse_misc_to_dict перенесена в @staticmethod.

        Поля каждого токена (word_dict):
          id, form, lemma, upos, xpos — стандарт CoNLL-U
          feats       — dict {"Case": "Nom", "Number": "Sing", ...}
          head, deprel — синтаксис
          start_char  — word.start_char + char_offset (None при pretokenized)
          end_char    — word.end_char   + char_offset (None при pretokenized)
          spaces_after — token.spaces_after (Stanza v1.4+):
                         '' = нет пробела (≈ SpaceAfter=No в CoNLL-U)
                         ' ' = обычный пробел (норма)
                         None = при pretokenized (исходная строка недоступна)
          misc (опц.) — dict прочих MISC-полей из token.misc (Translit и др.)
          ner  (опц.) — token.ner: тег NER (B-PER, I-LOC, O и т.д.)
        """
        result: List[Dict[str, Any]] = []

        for sent in doc.sentences:
            # ── Маппинг NER, MISC, SpaceAfter из Token → Word ──────────────
            word_to_ner: Dict[int, Optional[str]] = {}
            word_to_misc: Dict[int, Optional[Dict]] = {}
            word_to_spaces_after: Dict[int, Optional[str]] = {}

            for token in sent.tokens:
                ner_tag = token.ner if hasattr(token, "ner") else None
                misc_dict = self._parse_misc_to_dict(token.misc)
                sa = token.spaces_after if hasattr(token, "spaces_after") else None
                # Для MWT: spaces_after и misc присваиваются только последнему слову
                last_word_id = (
                    max(int(w.id) for w in token.words)
                    if len(token.words) > 1
                    else None
                )
                for word in token.words:
                    wid = int(word.id)
                    word_to_ner[wid] = ner_tag
                    if last_word_id is None or wid == last_word_id:
                        word_to_misc[wid] = misc_dict
                        word_to_spaces_after[wid] = sa

            # ── Формируем список токенов предложения ───────────────────────
            sent_parsed: List[Dict[str, Any]] = []
            for word in sent.words:
                wid = int(word.id)

                # start_char / end_char: корректируем на char_offset
                # None при tokenize_pretokenized=True (символьные позиции недоступны)
                sc = word.start_char
                ec = word.end_char
                word_dict: Dict[str, Any] = {
                    "id":         wid,
                    "form":       word.text,
                    "lemma":      word.lemma,
                    "upos":       word.upos,
                    "xpos":       word.xpos,
                    "feats":      self._parse_misc_to_dict(word.feats),
                    "head":       int(word.head),
                    "deprel":     word.deprel,
                    # [NEW] char_offset учитывается для razdel-пути
                    "start_char": (sc + char_offset) if sc is not None else None,
                    "end_char":   (ec + char_offset) if ec is not None else None,
                }

                # spaces_after: '' = нет пробела, ' ' = норма, None = pretokenized
                if wid in word_to_spaces_after:
                    word_dict["spaces_after"] = word_to_spaces_after[wid]

                # misc: прочие MISC-поля (Translit и др.), без SpaceAfter
                if wid in word_to_misc and word_to_misc[wid] is not None:
                    word_dict["misc"] = word_to_misc[wid]

                # ner: тег NER из token.ner
                if wid in word_to_ner and word_to_ner[wid] is not None:
                    word_dict["ner"] = word_to_ner[wid]

                sent_parsed.append(word_dict)

            # [NEW] Поля text/start_char/end_char на уровне предложения
            first_token = next(
                (t for t in sent.tokens if t.words), None
            )
            last_token = next(
                (t for t in reversed(sent.tokens) if t.words), None
            )
            sc_sent = (
                (first_token.words[0].start_char + char_offset)
                if first_token and first_token.words[0].start_char is not None
                else None
            )
            ec_sent = (
                (last_token.words[-1].end_char + char_offset)
                if last_token and last_token.words[-1].end_char is not None
                else None
            )

            sentence_data: Dict[str, Any] = {
                "text":       sent.text,
                "start_char": sc_sent,
                "end_char":   ec_sent,
                "words":      sent_parsed,
            }

            # sentiment / constituency — недоступны для русского
            if hasattr(sent, "sentiment") and sent.sentiment is not None:
                sentence_data["sentiment"] = sent.sentiment
            if hasattr(sent, "constituency") and sent.constituency is not None:
                sentence_data["constituency"] = str(sent.constituency)

            result.append(sentence_data)

        return result

    # ─── Production methods: принимают pre-split чанки из wrapper ─────────────

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: OutputFormat = "native",
        batch_size: int = 16,
    ) -> Any:
        """
        [NEW] Razdel path — production-метод.

        Принимает чанк пар (sentence_text, start_char_in_original).
        Каждое предложение уже сентенизировано и токенизировано razdel в wrapper.
        Stanza получает готовые токены (tokenize_pretokenized=True) и не запускает
        собственную токенизацию/сентенизацию.

        start_char используется для вычисления символьных позиций токенов
        относительно исходного (полного) текста.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format:          'native' | 'conllu'
            batch_size:             размер батча (подбирается под GPU, OOM-защита)

        Returns:
            native → List[Dict] (список предложений с полными атрибутами)
            conllu → str
        """
        docs, char_offsets = self._make_doc_razdel(sentences_with_offsets)
        docs = self._run_pipeline_razdel_batch(docs, batch_size=batch_size)

        if output_format == "conllu":
            return "\n\n".join(
                self._format_conllu(doc).strip() for doc in docs
            ) + "\n"

        result = []
        for doc, char_offset in zip(docs, char_offsets):
            result.extend(self._format_native(doc, char_offset=char_offset))
        return result

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: OutputFormat = "native",
        batch_size: int = 16,
    ) -> Any:
        """
        [NEW] Internal path — production-метод.

        Принимает чанк текстов предложений (без символьных офсетов).
        Stanza выполняет собственную токенизацию каждого предложения.
        Повторная сентенизация исключена: каждый элемент = одно предложение
        (передаётся как отдельный Document).

        start_char/end_char токенов — относительны начала каждого предложения
        (char_offset=0), поскольку исходные смещения недоступны.

        Args:
            sentences:     List[str] — тексты предложений чанка
            output_format: 'native' | 'conllu'
            batch_size:    размер батча (подбирается под GPU, OOM-защита)

        Returns:
            native → List[Dict]
            conllu → str
        """
        import stanza

        # Создаём отдельный Document для каждого предложения — Stanza
        # не будет объединять их и пересентенизировать.
        docs = [stanza.Document([], text=s) for s in sentences]
        docs = self._run_pipeline_batch(docs, batch_size=batch_size)

        if output_format == "conllu":
            return "\n\n".join(
                self._format_conllu(doc).strip() for doc in docs
            ) + "\n"

        result = []
        for doc in docs:
            result.extend(self._format_native(doc, char_offset=0))
        return result

    # ─── Backward compat / local_entrypoint ───────────────────────────────────

    @modal.method()
    def parse(
        self,
        text: str,
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
    ) -> Any:
        """
        [CHG] Добавлен параметр tokenizer: 'internal' | 'razdel'.
        Парсит текст целиком. Для local_entrypoint и прямых вызовов.
        Сентенизация выполняется внутри Stanza (не рекомендуется для
        production с длинными текстами — используйте parse_sentence_chunk).
        """
        import stanza

        if tokenizer == "razdel":
            # Razdel: токенизируем всё предложение как один блок
            from razdel import tokenize as razdel_tokenize
            tokens = list(razdel_tokenize(text))
            token_texts = [[t.text for t in tokens]]
            doc_input = stanza.Document(token_texts, text=text)
            doc = self._run_pipeline_razdel_batch([doc_input], batch_size=1)[0]
        else:
            doc = self.nlp(text)

        if output_format == "conllu":
            return self._format_conllu(doc)
        return self._format_native(doc)

    @modal.method()
    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        batch_size: int = 16,
    ) -> List[Any]:
        """
        [CHG] Переработан: принимает List[str] текстов (не List[List[str]] токенов).
              Добавлены параметры tokenizer и batch_size.
        Пакетная обработка текстов целиком. Backward compat.
        """
        import stanza

        if tokenizer == "razdel":
            from razdel import tokenize as razdel_tokenize
            docs_input = []
            for text in texts:
                tokens = list(razdel_tokenize(text))
                token_texts = [[t.text for t in tokens]]
                docs_input.append(stanza.Document(token_texts, text=text))
            docs = self._run_pipeline_razdel_batch(docs_input, batch_size=batch_size)
        else:
            docs_input = [stanza.Document([], text=t) for t in texts]
            docs = self._run_pipeline_batch(docs_input, batch_size=batch_size)

        if output_format == "conllu":
            return [self._format_conllu(doc) for doc in docs]
        return [self._format_native(doc) for doc in docs]


# ─── local_entrypoint: тест Modal-сервиса напрямую ────────────────────────────
@app.local_entrypoint()
def main():
    """
    Тестирует StanzaService напрямую — без wrapper, без chunking.
    Проверяет:
      - модель загружена, оба токенизатора работают
      - оба формата вывода корректны
      - production-методы (parse_sentence_chunk, parse_sentence_chunk_native)
      - backward compat (parse, parse_batch)
    """
    from razdel import sentenize

    service = StanzaService()
    text_single = "Коля сказал: «Привет!» И ушёл."
    text_multi  = "Зло, которым пугаешь, не так зло. Москва — столица России."
    sep = "=" * 72

    # ── 1. NATIVE + INTERNAL (parse.remote) ──────────────────────────────
    print(f"\n{sep}")
    print("1. NATIVE + INTERNAL (parse.remote)")
    print(sep)
    result = service.parse.remote(text_single, output_format="native", tokenizer="internal")
    for sent in result:
        print(f"\nПредложение: '{sent['text']}' "
              f"(chars {sent['start_char']}:{sent['end_char']})")
        for tok in sent["words"]:
            ner  = f" [NER: {tok['ner']}]" if "ner" in tok else ""
            sa   = tok.get("spaces_after")
            sa_s = f" [sa: {repr(sa)}]" if sa != " " and sa is not None else ""
            print(f"  {tok['id']:>2}  {tok['form']:<15} {tok['upos']:<6}"
                  f"  lemma={tok['lemma']}{ner}{sa_s}")

    # ── 2. NATIVE + RAZDEL (parse.remote) ────────────────────────────────
    print(f"\n{sep}")
    print("2. NATIVE + RAZDEL (parse.remote)")
    print(sep)
    result_r = service.parse.remote(text_single, output_format="native", tokenizer="razdel")
    print(f"\n⚡ Сравнение токенизаторов для: '{text_single}'")
    print(f"  internal: {[w['form'] for s in result  for w in s['words']]}")
    print(f"  razdel:   {[w['form'] for s in result_r for w in s['words']]}")

    # ── 3. CONLL-U + INTERNAL ────────────────────────────────────────────
    print(f"\n{sep}")
    print("3. CONLL-U + INTERNAL (parse.remote)")
    print(sep)
    result_ci = service.parse.remote(text_multi, output_format="conllu", tokenizer="internal")
    print(f"# text = {text_multi}")
    print(result_ci)

    # ── 4. CONLL-U + RAZDEL ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("4. CONLL-U + RAZDEL (parse.remote)")
    print(sep)
    result_cr = service.parse.remote(text_multi, output_format="conllu", tokenizer="razdel")
    print(f"# text = {text_multi}")
    print(result_cr)

    # ── 5. parse_sentence_chunk — razdel path (production method) ────────
    print(f"\n{sep}")
    print("5. parse_sentence_chunk (razdel path, pre-split chunk)")
    print(sep)
    sentences = list(sentenize(text_multi))
    chunk = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk)} предложений): {[c[0] for c in chunk]}")
    result_chunk = service.parse_sentence_chunk.remote(chunk, output_format="conllu")
    print(result_chunk)

    # ── 6. parse_sentence_chunk_native — internal path (production method)
    print(f"\n{sep}")
    print("6. parse_sentence_chunk_native (internal path, pre-split chunk)")
    print(sep)
    chunk_texts = [s.text for s in sentences]
    print(f"Чанк ({len(chunk_texts)} предложений): {chunk_texts}")
    result_cn = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    for sent in result_cn:
        print(f"\nПредложение: '{sent['text']}' "
              f"(chars {sent['start_char']}:{sent['end_char']})")
        if sent.get("entities"):
            print(f"  Сущности: {[(e['text'], e['label']) for e in sent['entities']]}")
        for tok in sent["words"]:
            ner = f" [NER: {tok['ner']}]" if "ner" in tok else ""
            print(f"  {tok['id']:>2}  {tok['form']:<15} {tok['upos']:<6}"
                  f"  lemma={tok['lemma']}{ner}")

    print(f"\n{'✅ Тестирование завершено!':^72}")

