import modal
import logging
from typing import Any, Dict, List, Literal, Tuple

# ─── Modal image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "spacy>=3.7.0",
        "pymorphy3>=1.2.0",
        "pymorphy3-dicts-ru>=2.4.0",
        "spacy-conll>=4.0.0",
        "razdel>=0.5.0",
        # [native_ru] Установка через zip-архив GitHub — не требует git в образе.
        "spacy_russian_tokenizer @ https://github.com/aatimofeev/spacy_russian_tokenizer/archive/refs/heads/master.zip",
    )
    .run_commands("python -m spacy download ru_core_news_lg")
)

app = modal.App("booknlp-ru-spacy")

# [native_ru] Добавлено значение "native_ru" в тип TokenizerType.
# native_ru = spaCy rule-based tokenizer + мёрж дефисных конструкций (SynTagRus).
TokenizerType = Literal["internal", "razdel", "native_ru"]
OutputFormat = Literal["native", "conllu"]


# ─── Service ─────────────────────────────────────────────────────────────────
@app.cls(image=image, timeout=600, scaledown_window=300)
class SpacyService:
    """
    Modal-сервис для морфо-синтаксического анализа с использованием
    официальной модели ru_core_news_lg.

    Поддерживает три токенизатора:
      internal  — встроенный rule-based токенизатор spaCy
      razdel    — внешний ML-токенизатор razdel (рекомендован для русского)
      native_ru — spaCy rule-based + мёрж дефисных конструкций  [native_ru]
                  (spacy_russian_tokenizer: MERGE_PATTERNS + SYNTAGRUS_RARE_CASES)

    Два формата вывода:
      native  — полный набор атрибутов spaCy (List[Dict])
      conllu  — стандарт Universal Dependencies (str)

    Сентенизация выполняется в wrapper ДО отправки в Modal.
    Основные методы принимают уже разбитые предложения (чанки).

    Основные методы (production path из wrapper):
      parse_sentence_chunk        — razdel path: List[(text, start_char)]
      parse_sentence_chunk_native — internal/native_ru path: List[str]  [native_ru]

    Вспомогательные методы (local_entrypoint / прямые вызовы):
      parse       — одиночный текст целиком
      parse_batch — список текстов целиком
    """

    # noinspection DuplicatedCode
    @modal.enter()
    def setup(self):
        import spacy
        from razdel import tokenize as razdel_tokenize

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SpacyService")

        self.nlp = spacy.load("ru_core_news_lg")
        self.original_tokenizer = self.nlp.tokenizer

        # Внешний токенизатор: razdel → Doc (без запуска pipeline)
        class RazdelTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab

            # noinspection DuplicatedCode
            def __call__(self, text: str):
                from spacy.tokens import Doc

                tokens = list(razdel_tokenize(text))
                if not tokens:
                    return Doc(self.vocab, words=[], spaces=[])
                words = [t.text for t in tokens]
                spaces = [
                    tokens[i].stop < tokens[i + 1].start
                    for i in range(len(tokens) - 1)
                ] + [False]
                return Doc(self.vocab, words=words, spaces=spaces)

        # noinspection DuplicatedCode
        self.razdel_tokenizer = RazdelTokenizer(self.nlp.vocab)

        # [native_ru] Инициализация компонента spacy_russian_tokenizer.
        # Используется собственный _RuMatcher (совместим с spaCy v3 API):
        # RussianTokenizer из библиотеки использует устаревший v2 API.

        from spacy.matcher import Matcher as SpacyMatcher
        from spacy.util import filter_spans
        from spacy_russian_tokenizer import MERGE_PATTERNS, SYNTAGRUS_RARE_CASES

        class _RuMatcher:
            def __init__(self, vocab, patterns):
                self.matcher = SpacyMatcher(vocab)
                self.matcher.add("MERGE_HYPHEN", patterns)  # v3: list, не *args

            def __call__(self, doc):
                matches = self.matcher(doc)
                if not matches:
                    return doc
                spans = filter_spans([doc[start:end] for _, start, end in matches])
                with doc.retokenize() as retokenizer:
                    for span in spans:
                        retokenizer.merge(span)  # v3: retokenize() вместо span.merge()
                return doc

        self.ru_tokenizer_component = _RuMatcher(
            self.nlp.vocab, MERGE_PATTERNS + SYNTAGRUS_RARE_CASES
        )


        if "conll_formatter" not in self.nlp.pipe_names:
            config = {
                "ext_names": {
                    "conll_str": "conll_str",
                    "conll": "conll",
                    "conll_pd": "conll_pd",
                },
                "conversion_maps": {
                    "UPOS": {},
                    "XPOS": {},
                    "FEATS": {},
                    "DEPREL": {},
                },
                # spacy-conll >= 4.0 объявляет field_names как Dict[str, str] (не Optional).
                # Валидатор confection отклоняет дефолтный None.
                # Пустой словарь {} валиден — formatter подставит стандартные
                # CoNLL-U имена (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC).
                "field_names": {},
            }
            self.nlp.add_pipe("conll_formatter", config=config, last=True)

        # Отключаем senter/sentencizer: каждый doc уже содержит ровно одно
        # предложение (сентенизация выполнена снаружи — razdel или wrapper).
        # Без этого spaCy может пересентенизировать doc, что сломает:
        # - нумерацию токенов в _format_native
        # - разбивку на .sents в spacy-conll (_format_conllu)
        for pipe_name in ("senter", "sentencizer"):
            if pipe_name in self.nlp.pipe_names:
                self.nlp.remove_pipe(pipe_name)
                self.logger.info(f"Removed pipe: {pipe_name}")

        self.logger.info("SpaCy loaded (ru_core_news_lg)!")
        self.logger.info(f"Pipeline components: {self.nlp.pipe_names}")
        # [native_ru] Обновлено лог-сообщение: добавлен native_ru.
        self.logger.info(
            "Tokenizers: internal (spaCy rule-based), "
            "razdel (ML), "
            "native_ru (spaCy rule-based + SynTagRus merge patterns)"
        )

    # ─── Internal helpers ────────────────────────────────────────────────────
    def _make_doc(self, text: str, tokenizer_type: TokenizerType):
        """Токенизирует текст выбранным токенизатором без запуска pipeline."""
        if tokenizer_type == "razdel":
            return self.razdel_tokenizer(text)
        # [native_ru] Новая ветка токенизатора native_ru.
        # Шаг 1: стандартная spaCy-токенизация (prefix/suffix/infix правила).
        # Шаг 2: ru_tokenizer_component(doc) мёржит токены по MERGE_PATTERNS +
        #         SYNTAGRUS_RARE_CASES — результат ближе к аннотации SynTagRus,
        #         на которой обучена ru_core_news_lg (улучшает parser и NER).
        if tokenizer_type == "native_ru":
            doc = self.original_tokenizer(text)
            doc = self.ru_tokenizer_component(doc)
            return doc
        return self.original_tokenizer(text)

    def _run_pipeline(self, doc):
        """Прогоняет Doc через все компоненты pipeline поодиночке."""
        for _, pipe in self.nlp.pipeline:
            doc = pipe(doc)
        return doc

    # noinspection DuplicatedCode
    def _run_pipeline_batch(self, docs: list, batch_size: int) -> list:
        """
        Пакетно прогоняет список Doc через pipeline.
        Компоненты с .pipe() (tok2vec) обрабатываются GPU-батчами,
        остальные — поодиночке.
        """
        current_docs: list = list(docs)
        for _, pipe in self.nlp.pipeline:
            if hasattr(pipe, "pipe"):
                try:
                    current_docs = list(pipe.pipe(current_docs, batch_size=batch_size))
                except TypeError:
                    current_docs = list(pipe.pipe(current_docs))
            else:
                current_docs = [pipe(doc) for doc in current_docs]
        return current_docs

    # @staticmethod
    # def _format_native(doc, char_offset: int = 0) -> List[Dict[str, Any]]:
    #     """
    #     Полный нативный формат spaCy — все атрибуты токена.
    #     char_offset: смещение начала текста в исходном документе.
    #     Передаётся из parse_sentence_chunk для корректных start_char/end_char
    #     относительно всего исходного текста (не только текущего предложения).
    #     """
    #     result = []
    #     for sent in doc.sents:
    #         sent_data = {
    #             "text": sent.text,
    #             "start_char": sent.start_char + char_offset,
    #             "end_char": sent.end_char + char_offset,
    #             "words": [],
    #         }
    #         sent_token_offset = sent.start  # индекс первого токена предложения в Doc
    #         for token in sent:
    #             word_dict = {
    #                 # ── Позиция ─────────────────────────────────────────────
    #                 "id": token.i - sent_token_offset + 1,
    #                 "start_char": token.idx + char_offset,
    #                 "end_char": token.idx + len(token.text) + char_offset,
    #                 # ── Форма ───────────────────────────────────────────────
    #                 "form": token.text,
    #                 "norm": token.norm_,
    #                 "lower": token.lower_,
    #                 "shape": token.shape_,
    #                 # ── Лемма и POS ─────────────────────────────────────────
    #                 "lemma": token.lemma_,
    #                 "upos": token.pos_,
    #                 "xpos": token.tag_,
    #                 "feats": str(token.morph) if token.morph.to_dict() else "_",
    #                 # ── Синтаксис ───────────────────────────────────────────
    #                 "head": (
    #                     token.head.i - sent_token_offset + 1
    #                     if token.head.i != token.i
    #                     else 0
    #                 ),
    #                 "deprel": token.dep_,
    #                 "n_lefts": token.n_lefts,
    #                 "n_rights": token.n_rights,
    #                 "children": [c.i - sent_token_offset + 1 for c in token.children],
    #                 # ── Именованные сущности ────────────────────────────────
    #                 "ent_type": token.ent_type_ or None,
    #                 "ent_iob": token.ent_iob_ if token.ent_iob_ != "O" else None,
    #                 # ── Метаданные ──────────────────────────────────────────
    #                 "is_sent_start": token.is_sent_start,
    #                 "whitespace": token.whitespace_,
    #                 "misc": "SpaceAfter=No" if not token.whitespace_ else "_",
    #                 # ── Лексические флаги ───────────────────────────────────
    #                 "is_alpha": token.is_alpha,
    #                 "is_digit": token.is_digit,
    #                 "is_punct": token.is_punct,
    #                 "is_space": token.is_space,
    #                 "is_stop": token.is_stop,
    #                 "is_oov": token.is_oov,
    #                 "like_num": token.like_num,
    #                 "like_url": token.like_url,
    #                 "like_email": token.like_email,
    #                 # ── Вектор ──────────────────────────────────────────────
    #                 "has_vector": token.has_vector,
    #                 "cluster": token.cluster,
    #                 "vector_norm": (
    #                     round(float(token.vector_norm), 6) if token.has_vector else None
    #                 ),
    #             }
    #             sent_data["words"].append(word_dict)
    #         ents = [
    #             {
    #                 "text": ent.text,
    #                 "start": ent.start - sent.start,
    #                 "end": ent.end - sent.start,
    #                 "label": ent.label_,
    #                 "start_char": ent.start_char + char_offset,
    #                 "end_char": ent.end_char + char_offset,
    #             }
    #             for ent in sent.ents
    #         ]
    #         if ents:
    #             sent_data["entities"] = ents
    #         result.append(sent_data)
    #     return result
    #
    # @staticmethod
    # def format_native(doc, char_offset: int = 0) -> List[Dict[str, Any]]:
    #     """
    #     Форматирует spaCy Doc в List[SentenceDict].
    #     Doc считается одним предложением — сентенизация уже выполнена снаружи
    #     (razdel в wrapper или split_to_sentence_chunks).
    #     Итерация по doc.sents намеренно не используется во избежание
    #     рассинхронизации с внешней сентенизацией.
    #
    #     Args:
    #         doc: spaCy Doc (одно предложение)
    #         char_offset: смещение символов относительно исходного текста
    #                      (передаётся из parse_sentence_chunk для razdel-пути)
    #     Returns:
    #         List с одним SentenceDict, или [] если doc пустой
    #     """
    #     tokens = list(doc)
    #     if not tokens:
    #         return []
    #
    #     words: List[Dict[str, Any]] = []
    #     for token in tokens:
    #         morph_str = str(token.morph) if token.morph.to_dict() else ""
    #         word_dict: Dict[str, Any] = {
    #             # Позиция в предложении (1-based, CoNLL-совместимо)
    #             "id": token.i + 1,
    #             "start_char": token.idx + char_offset,
    #             "end_char": token.idx + len(token.text) + char_offset,
    #             # Формы
    #             "form": token.text,
    #             "norm": token.norm_,
    #             "lower": token.lower_,
    #             "shape": token.shape_,
    #             # Морфология
    #             "lemma": token.lemma_,
    #             "upos": token.pos_,
    #             "xpos": token.tag_,
    #             "feats": morph_str if morph_str else "_",   # "_" как в стандарте CoNLL-
    #             # Синтаксис
    #             # head=0 означает root (токен указывает на себя)
    #             "head": token.head.i + 1 if token.head.i != token.i else 0,
    #             "deprel": token.dep_,
    #             "n_lefts": token.n_lefts,
    #             "n_rights": token.n_rights,
    #             "children": [c.i + 1 for c in token.children],
    #             # NER
    #             "ent_type": token.ent_type_ or None,
    #             "ent_iob": token.ent_iob_ if token.ent_iob_ != "O" else None,
    #             # Границы предложений
    #             # Для первого токена всегда True (doc = одно предложение)
    #             "is_sent_start": token.i == 0,
    #             # Пробелы / CoNLL misc
    #             "whitespace": token.whitespace_,
    #             "misc": "SpaceAfter=No" if not token.whitespace_ else "_",
    #             # Булевы флаги
    #             "is_alpha": token.is_alpha,
    #             "is_digit": token.is_digit,
    #             "is_punct": token.is_punct,
    #             "is_space": token.is_space,
    #             "is_stop": token.is_stop,
    #             "is_oov": token.is_oov,
    #             "like_num": token.like_num,
    #             "like_url": token.like_url,
    #             "like_email": token.like_email,
    #             # Векторы
    #             "has_vector": token.has_vector,
    #             "cluster": token.cluster,
    #             "vector_norm": (
    #                 round(float(token.vector_norm), 6) if token.has_vector else None
    #             ),
    #         }
    #         words.append(word_dict)
    #
    #     # NER на уровне предложения — из doc.ents
    #     entities: List[Dict[str, Any]] = [
    #         {
    #             "text": ent.text,
    #             "start": ent.start,  # индекс первого токена сущности в doc
    #             "end": ent.end,      # индекс после последнего токена
    #             "label": ent.label_,
    #             "start_char": ent.start_char + char_offset,
    #             "end_char": ent.end_char + char_offset,
    #         }
    #         for ent in doc.ents
    #     ]
    #
    #     sent_data: Dict[str, Any] = {
    #         "text": doc.text,
    #         "start_char": tokens[0].idx + char_offset,
    #         "end_char": tokens[-1].idx + len(tokens[-1].text) + char_offset,
    #         "words": words,
    #         "entities": entities,
    #     }
    #     return [sent_data]

    # noinspection DuplicatedCode
    @staticmethod
    def _format_native_doc(
            doc,
            char_offset: int = 0,
            single_sentence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        single_sentence=True  → doc = одно предложение (production path)
        single_sentence=False → doc = несколько предложений (parse / parse_batch)
        """
        if single_sentence:
            # текущая логика format_native

            # Форматирует spaCy Doc в List[SentenceDict].
            # Doc считается одним предложением — сентенизация уже выполнена снаружи
            # (razdel в wrapper или split_to_sentence_chunks).
            # Итерация по doc.sents намеренно не используется во избежание
            # рассинхронизации с внешней сентенизацией.
            #
            # Args:
            #     doc: spaCy Doc (одно предложение)
            #     char_offset: смещение символов относительно исходного текста
            #                  (передаётся из parse_sentence_chunk для razdel-пути)
            # Returns:
            #     List с одним SentenceDict, или [] если doc пустой

            tokens = list(doc)
            if not tokens:
                return []

            words: List[Dict[str, Any]] = []
            for token in tokens:
                morph_str = str(token.morph) if token.morph.to_dict() else ""
                word_dict: Dict[str, Any] = {
                    # Позиция в предложении (1-based, CoNLL-совместимо)
                    "id": token.i - tokens[0].i + 1,
                    "start_char": token.idx + char_offset,
                    "end_char": token.idx + len(token.text) + char_offset,
                    # Формы
                    "form": token.text,
                    "norm": token.norm_,
                    "lower": token.lower_,
                    "shape": token.shape_,
                    # Морфология
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "feats": morph_str if morph_str else "_",  # "_" как в стандарте CoNLL-
                    # Синтаксис
                    # head=0 означает root (токен указывает на себя)
                    "head": token.head.i - tokens[0].i + 1 if token.head.i != token.i else 0,
                    "deprel": token.dep_,
                    "n_lefts": token.n_lefts,
                    "n_rights": token.n_rights,
                    "children": [c.i - tokens[0].i + 1 for c in token.children],
                    # NER
                    "ent_type": token.ent_type_ or None,
                    "ent_iob": token.ent_iob_ if token.ent_iob_ != "O" else None,
                    # Границы предложений
                    # Для первого токена всегда True (doc = одно предложение)
                    "is_sent_start": token.i == 0,
                    # Пробелы / CoNLL misc
                    "whitespace": token.whitespace_,
                    "misc": "SpaceAfter=No" if not token.whitespace_ else "_",
                    # Булевы флаги
                    "is_alpha": token.is_alpha,
                    "is_digit": token.is_digit,
                    "is_punct": token.is_punct,
                    "is_space": token.is_space,
                    "is_stop": token.is_stop,
                    "is_oov": token.is_oov,
                    "like_num": token.like_num,
                    "like_url": token.like_url,
                    "like_email": token.like_email,
                    # Векторы
                    "has_vector": token.has_vector,
                    "cluster": token.cluster,
                    "vector_norm": (
                        round(float(token.vector_norm), 6) if token.has_vector else None
                    ),
                }
                words.append(word_dict)

            # NER на уровне предложения — из doc.ents
            entities: List[Dict[str, Any]] = [
                {
                    "text": ent.text,
                    "start": ent.start,  # индекс первого токена сущности в doc
                    "end": ent.end,  # индекс после последнего токена
                    "label": ent.label_,
                    "start_char": ent.start_char + char_offset,
                    "end_char": ent.end_char + char_offset,
                }
                for ent in doc.ents
            ]

            sent_data: Dict[str, Any] = {
                "text": doc.text,
                "start_char": tokens[0].idx + char_offset,
                "end_char": tokens[-1].idx + len(tokens[-1].text) + char_offset,
                "words": words,
            }
            if entities:
                sent_data["entities"] = entities
            return [sent_data]
        else:
            # текущая логика _format_native
            """
            Полный нативный формат spaCy — все атрибуты токена.
            char_offset: смещение начала текста в исходном документе.
            Передаётся из parse_sentence_chunk для корректных start_char/end_char
            относительно всего исходного текста (не только текущего предложения).
            """
            result = []
            for sent in doc.sents:
                sent_data = {
                    "text": sent.text,
                    "start_char": sent.start_char + char_offset,
                    "end_char": sent.end_char + char_offset,
                    "words": [],
                }
                sent_token_offset = sent.start  # индекс первого токена предложения в Doc
                for token in sent:
                    word_dict = {
                        # ── Позиция ─────────────────────────────────────────────
                        "id": token.i - sent_token_offset + 1,
                        "start_char": token.idx + char_offset,
                        "end_char": token.idx + len(token.text) + char_offset,
                        # ── Форма ───────────────────────────────────────────────
                        "form": token.text,
                        "norm": token.norm_,
                        "lower": token.lower_,
                        "shape": token.shape_,
                        # ── Лемма и POS ─────────────────────────────────────────
                        "lemma": token.lemma_,
                        "upos": token.pos_,
                        "xpos": token.tag_,
                        "feats": str(token.morph) if token.morph.to_dict() else "_",
                        # ── Синтаксис ───────────────────────────────────────────
                        "head": (
                            token.head.i - sent_token_offset + 1
                            if token.head.i != token.i
                            else 0
                        ),
                        "deprel": token.dep_,
                        "n_lefts": token.n_lefts,
                        "n_rights": token.n_rights,
                        "children": [c.i - sent_token_offset + 1 for c in token.children],
                        # ── Именованные сущности ────────────────────────────────
                        "ent_type": token.ent_type_ or None,
                        "ent_iob": token.ent_iob_ if token.ent_iob_ != "O" else None,
                        # ── Метаданные ──────────────────────────────────────────
                        "is_sent_start": token.is_sent_start if token.is_sent_start is not None
                        else (token.i == sent.start),
                        "whitespace": token.whitespace_,
                        "misc": "SpaceAfter=No" if not token.whitespace_ else "_",
                        # ── Лексические флаги ───────────────────────────────────
                        "is_alpha": token.is_alpha,
                        "is_digit": token.is_digit,
                        "is_punct": token.is_punct,
                        "is_space": token.is_space,
                        "is_stop": token.is_stop,
                        "is_oov": token.is_oov,
                        "like_num": token.like_num,
                        "like_url": token.like_url,
                        "like_email": token.like_email,
                        # ── Вектор ──────────────────────────────────────────────
                        "has_vector": token.has_vector,
                        "cluster": token.cluster,
                        "vector_norm": (
                            round(float(token.vector_norm), 6) if token.has_vector else None
                        ),
                    }
                    sent_data["words"].append(word_dict)
                ents = [
                    {
                        "text": ent.text,
                        "start": ent.start - sent.start,
                        "end": ent.end - sent.start,
                        "label": ent.label_,
                        "start_char": ent.start_char + char_offset,
                        "end_char": ent.end_char + char_offset,
                    }
                    for ent in sent.ents
                ]
                if ents:
                    sent_data["entities"] = ents
                result.append(sent_data)
            return result

    @staticmethod
    def _format_conllu(doc) -> str:
        """CoNLL-U через spacy-conll (doc._.conll_str заполняется в pipeline)."""
        # noinspection PyProtectedMember
        # return doc._.conll_str  # type: ignore[attr-defined]
        return doc._.conll_str.strip() + "\n"

    # ─── Production methods: принимают pre-split чанки из wrapper ───────────
    # noinspection DuplicatedCode
    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: str = "native",
        batch_size: int = 32,
    ) -> Any:
        """
        Razdel path.
        Принимает чанк пар (sentence_text, start_char_in_original).
        start_char используется для вычисления символьных позиций токенов
        относительно исходного текста.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format: 'native' | 'conllu'
            batch_size: int
        Returns:
            native → List[Dict]
            conllu → str
        """
        docs = []
        char_offsets = []
        for sent_text, char_offset in sentences_with_offsets:
            docs.append(self._make_doc(sent_text, "razdel"))
            char_offsets.append(char_offset)
        docs = self._run_pipeline_batch(docs, batch_size=batch_size)
        if output_format == "conllu":
            # noinspection PyProtectedMember
            return "\n\n".join(
                doc._.conll_str.strip() for doc in docs  # type: ignore[attr-defined]
            ) + "\n"
        result = []
        for doc, char_offset in zip(docs, char_offsets):
            result.extend(self._format_native_doc(doc, char_offset=char_offset))

        # Исправляем SpaceAfter=No у последнего токена промежуточных предложений.
        # Аналогично parse_sentence_chunk_native и _merge_chunks в wrapper.
        for sent in result[:-1]:
            if sent.get("words"):
                last_tok = sent["words"][-1]
                if last_tok.get("misc") == "SpaceAfter=No":
                    last_tok["misc"] = "_"
        return result

    # noinspection DuplicatedCode
    @modal.method()
    def parse_sentence_chunk_native(
            self,
            sentences: List[Tuple[str, int]],  # (текст предложения, start_char в исходном тексте)
            output_format: str = "native",
            batch_size: int = 32,
            tokenizer: TokenizerType = "internal",
    ) -> Any:
        """
        Internal / native_ru path.
        Принимает чанк предложений вместе с абсолютными символьными офсетами.

        Args:
            sentences:     List[Tuple[str, int]] — (текст предложения, start_char в исходном тексте)
            output_format: 'native' | 'conllu'
            batch_size:    int
            tokenizer:     'internal' | 'native_ru'  [native_ru]
        Returns:
            native → List[Dict] с корректными абсолютными start_char/end_char
            conllu → str (char-офсеты в CoNLL-U не используются, без изменений)

        ⚠️  misc последнего токена промежуточных предложений чанка возвращается
        as-is (SpaceAfter=No / _). Нормализацию выполняет wrapper:
        SpacyParser._merge_chunks / _fix_boundary_misc.
        При прямых вызовах сервиса вне wrapper корректируйте misc самостоятельно.
        """
        # [native_ru] tokenizer пробрасывается в _make_doc:
        #   "internal"  → self.original_tokenizer(text) (spaCy rule-based)
        #   "native_ru" → original_tokenizer + ru_tokenizer_component (мёрж паттернов)
        if tokenizer not in ("internal", "native_ru"):
            raise ValueError(
                f"parse_sentence_chunk_native: tokenizer must be "
                f"'internal' or 'native_ru', got {tokenizer!r}"
            )
        sent_texts = [text for text, _ in sentences]
        base_offsets = [offset for _, offset in sentences]

        docs = [self._make_doc(s, tokenizer) for s in sent_texts]
        docs = self._run_pipeline_batch(docs, batch_size=batch_size)
        if output_format == "conllu":
            # char-офсеты в CoNLL-U не используются — возвращаем as-is.
            # noinspection PyProtectedMember
            return "\n\n".join(
                doc._.conll_str.strip() for doc in docs  # type: ignore[attr-defined]
            ) + "\n"
        result = []
        for doc, base_offset in zip(docs, base_offsets):
            result.extend(self._format_native_doc(doc, char_offset=base_offset))

        # Исправляем SpaceAfter=No у последнего токена промежуточных предложений.
        # В wrapper _fix_boundary_misc / _merge_chunks делают то же самое, поэтому
        # двойное применение безопасно (повторный вызов идемпотентен: _ → _ без изменений).
        for sent in result[:-1]:
            if sent.get("words"):
                last_tok = sent["words"][-1]
                if last_tok.get("misc") == "SpaceAfter=No":
                    last_tok["misc"] = "_"
        return result

    # ─── Backward compat / local_entrypoint ─────────────────────────────────
    @modal.method()
    def parse(
        self,
        text: str,
        output_format: str = "native",
        tokenizer: TokenizerType = "internal",
    ) -> Any:
        """Парсит текст целиком. Для local_entrypoint и прямых вызовов."""
        doc = self._make_doc(text, tokenizer)
        doc = self._run_pipeline(doc)
        if output_format == "conllu":
            return self._format_conllu(doc)
        return self._format_native_doc(doc, single_sentence=False)

    @modal.method()
    def parse_batch(
        self,
        texts: List[str],
        output_format: str = "native",
        tokenizer: TokenizerType = "internal",
        batch_size: int = 32,
    ) -> List[Any]:
        """Пакетная обработка текстов целиком. Backward compat."""
        docs = [self._make_doc(text, tokenizer) for text in texts]
        docs = self._run_pipeline_batch(docs, batch_size)
        if output_format == "conllu":
            return [self._format_conllu(doc) for doc in docs]
        return [self._format_native_doc(doc, single_sentence=False) for doc in docs]


# ─── Вспомогательная функция вывода ──────────────────────────────────────────
# noinspection DuplicatedCode
def _print_token_full(tok: Dict[str, Any]) -> None:
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


# ─── Константа заголовка CoNLL-U ─────────────────────────────────────────────
CONLLU_HEADER = "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"

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


def _print_conllu(text: str, conllu: str) -> None:
    """Выводит CoNLL-U блок с текстом предложения и заголовком столбцов."""
    print(f"\n# text = {text}")
    print(CONLLU_HEADER)
    print(conllu)


# ─── local_entrypoint: тест Modal-сервиса напрямую ───────────────────────────
# noinspection DuplicatedCode
@app.local_entrypoint()
def main():
    """
    Тестирует SpaCyService напрямую — без wrapper, без chunking.
    Проверяет: модель загружена, все три токенизатора работают,
    оба формата вывода корректны, все production-методы работают.

    Тест-кейсы:
      1–2:  native  + internal / razdel    (parse.remote)
      3–4:  conllu  + internal / razdel    (parse.remote)
      5:    parse_sentence_chunk           (razdel path, production)
      6:    parse_sentence_chunk_native    (internal path, production)
      7–8:  native/conllu + native_ru      (parse.remote)           [native_ru]
      9:    parse_sentence_chunk_native    (native_ru path, production) [native_ru]
    """
    from razdel import sentenize

    service = SpacyService()
    text_single = "Кружка-термос стоит 500р."
    text_multi = "Зло, которым пугаешь, не так зло. Москва — столица России."
    text_compare = (
        "Все-таки кружка-термос стоит 500р., "
        "а какая-нибудь кресло-качалка 10 000р."
    )
    sep = "=" * 72

    # ── 1. NATIVE + INTERNAL ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("1. NATIVE + INTERNAL (parse.remote)")
    print(sep)
    result = service.parse.remote(text_single, output_format="native", tokenizer="internal")
    for sent in result:
        print(f"\nПредложение: '{sent['text']}'")
        for tok in sent["words"]:
            _print_token_full(tok)

    # ── 2. NATIVE + RAZDEL ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("2. NATIVE + RAZDEL (parse.remote)")
    print(sep)
    result_r = service.parse.remote(text_single, output_format="native", tokenizer="razdel")
    print(f"\n⚡ Сравнение токенизаторов для: '{text_single}'")
    print(f"   internal:  {[w['form'] for s in result for w in s['words']]}")
    print(f"   razdel:    {[w['form'] for s in result_r for w in s['words']]}")
    for sent in result_r:
        print(f"\nПредложение: '{sent['text']}'")
        for tok in sent["words"]:
            _print_token_full(tok)

    # ── 2b. — 3-way сравнение через parse.remote ─────────────────────────────────────────────
    result_cmp_int = service.parse.remote(text_compare, "native", "internal")
    result_cmp_rz = service.parse.remote(text_compare, "native", "razdel")
    result_cmp_nru = service.parse.remote(text_compare, "native", "native_ru")
    print(f"\n{sep}")
    print("ВАРИАНТ 2b: СРАВНЕНИЕ ВСЕХ ТРЁХ ТОКЕНИЗАТОРОВ")
    print(sep)
    _print_comparison(text_compare, {
        "internal": result_cmp_int,
        "razdel": result_cmp_rz,
        "native_ru": result_cmp_nru,
    })

    # ── 3. CONLL-U + INTERNAL ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("3. CONLL-U + INTERNAL (parse.remote)")
    print(sep)
    result_conllu_i = service.parse.remote(
        text_multi, output_format="conllu", tokenizer="internal"
    )
    _print_conllu(text_multi, result_conllu_i)

    # ── 4. CONLL-U + RAZDEL ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("4. CONLL-U + RAZDEL (parse.remote)")
    print(sep)
    result_conllu_r = service.parse.remote(
        text_multi, output_format="conllu", tokenizer="razdel"
    )
    _print_conllu(text_multi, result_conllu_r)

    # ── 5. parse_sentence_chunk — razdel path (production method) ─────────
    print(f"\n{sep}")
    print("5. parse_sentence_chunk (razdel path, pre-split chunk)")
    print(sep)
    sentences = list(sentenize(text_multi))
    chunk = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk)} предложений): {[c[0] for c in chunk]}")
    result_chunk = service.parse_sentence_chunk.remote(chunk, output_format="conllu")
    print(f"\n{CONLLU_HEADER}")
    print(result_chunk)

    # ── 6. parse_sentence_chunk_native — internal path (production method) ─
    print(f"\n{sep}")
    print("6. parse_sentence_chunk_native (internal path, pre-split chunk)")
    print(sep)
    chunk_native = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk_native)} предложений): {[c[0] for c in chunk_native]}")
    result_chunk_native = service.parse_sentence_chunk_native.remote(
        chunk_native, output_format="native", tokenizer="internal"
    )
    for sent in result_chunk_native:
        print(
            f"\nПредложение: '{sent['text']}' "
            f"(chars {sent['start_char']}:{sent['end_char']})"
        )
        if sent.get("entities"):
            print(f"  Сущности: {[(e['text'], e['label']) for e in sent['entities']]}")
        for tok in sent["words"]:
            _print_token_full(tok)

    # ── 7. NATIVE + NATIVE_RU (parse.remote) ─────────────────────────────
    # [native_ru] Тест нового токенизатора native_ru через parse.remote.
    # Проверка: 'Все-таки' и 'какая-нибудь' → 1 токен (только native_ru/razdel).
    # 'Кружка-термос' → 3 токена (как и internal): MERGE_PATTERNS не содержит
    # паттерна Noun-Noun (только частицы и местоимения из SynTagRus).
    # native_ru НЕ мержит Noun-Noun конструкции ('кружка-термос', 'кресло-качалка').
    # Мержатся только паттерны из SYNTAGRUS: частицы ('все-таки'),
    # местоимения ('какая-нибудь') и аналогичные.
    print(f"\n{sep}")
    print("7. NATIVE + NATIVE_RU (parse.remote)  [native_ru]")
    print(sep)
    result_nru = service.parse.remote(
        text_single, output_format="native", tokenizer="native_ru"
    )
    print(f"\n⚡ Сравнение всех трёх токенизаторов для: '{text_single}'")
    print(f"   internal:  {[w['form'] for s in result   for w in s['words']]}")
    print(f"   razdel:    {[w['form'] for s in result_r for w in s['words']]}")
    print(f"   native_ru: {[w['form'] for s in result_nru for w in s['words']]}")
    print("   native_ru: 'Кружка-термос' → 3 токена (ожидаемо: Noun-Noun не входит в MERGE_PATTERNS)")
    for sent in result_nru:
        print(f"\nПредложение: '{sent['text']}'")
        for tok in sent["words"]:
            _print_token_full(tok)

    # ── 8. CONLL-U + NATIVE_RU (parse.remote) ─────────────────────────────
    # [native_ru] Тест CoNLL-U вывода с токенизатором native_ru.
    # Смерженные токены не генерируют multi-word token строки (1-2 CoNLL-U),
    # т.к. мёрж происходит на уровне spaCy Doc до передачи в pipeline.
    print(f"\n{sep}")
    print("8. CONLL-U + NATIVE_RU (parse.remote)  [native_ru]")
    print(sep)
    result_conllu_nru = service.parse.remote(
        text_multi, output_format="conllu", tokenizer="native_ru"
    )
    _print_conllu(text_multi, result_conllu_nru)

    # ── 9. parse_sentence_chunk_native — native_ru path [native_ru] ──────────
    print(f"\n{sep}")
    print("9. parse_sentence_chunk_native (native_ru path) [native_ru]")
    print(sep)

    # Двухпредложный чанк: проверяет и дефисные конструкции, и сборку нескольких предложений.
    # ВАЖНО: sentenize razdel не ставит границу предложения после однобуквенных
    # аббревиатур с точкой (р., г., т.д., т.п.).
    # Текст вида "...500р. Москва..." воспринимается как одно предложение.
    # Для теста используется текст без таких аббревиатур на границе предложений.
    text_chunk9 = (
        "Все-таки кружка-термос стоит 500р., а какая-нибудь кресло-качалка сильно дороже. "
        "Москва — столица России."
    )
    sentences_9 = list(sentenize(text_chunk9))
    chunk_9 = [(s.text, s.start) for s in sentences_9]     # единый формат для native и razdel
                                # razdel path — тот же список (офсеты уже есть)

    print(f"Чанк ({len(chunk_9)} предложений): {[c[0] for c in chunk_9]}")

    result_nru_9 = service.parse_sentence_chunk_native.remote(
        chunk_9, output_format="native", tokenizer="native_ru"
    )
    result_int_9 = service.parse_sentence_chunk_native.remote(
        chunk_9, output_format="native", tokenizer="internal"
    )
    result_rz_9 = service.parse_sentence_chunk.remote(
        chunk_9, output_format="native"
    )

    print(f"\n⚡ Сравнение токенизаторов (pre-split chunk, {len([c[0] for c in chunk_9])} предложения):")
    for i, (s_int, s_rz, s_nru) in enumerate(
            zip(result_int_9, result_rz_9, result_nru_9), 1
    ):
        print(f"\n  Предложение {i}: '{s_int['text']}'")
        print(f"    internal  : {[w['form'] for w in s_int['words']]}")
        print(f"    razdel    : {[w['form'] for w in s_rz['words']]}")
        print(f"    native_ru : {[w['form'] for w in s_nru['words']]}")
        # Проверка misc последнего токена промежуточных предложений
        if i < len([c[0] for c in chunk_9]):
            for name, res in [("internal", s_int), ("razdel", s_rz), ("native_ru", s_nru)]:
                last_misc = res["words"][-1].get("misc")
                status = "✅" if last_misc == "_" else "❌"
                print(f"    {status} {name}: последний токен misc='{last_misc}' (ожидается '_')")

    # Ожидаемые результаты токенизации для предложения 1:
    print(f"\n  Ожидаемые результаты (предложение 1):")
    print(f"    Все-таки      → 1 токен (native_ru, razdel) / 3 токена (internal)")
    print(f"    кружка-термос → 3 токена (internal, native_ru) / 1 токен (razdel)")
    print(f"    какая-нибудь  → 1 токен (native_ru, razdel) / 3 токена (internal)")


    print(f"\n{'✅ Тестирование завершено!':^72}")
