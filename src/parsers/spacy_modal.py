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
    )
    .run_commands("python -m spacy download ru_core_news_lg")
)

app = modal.App("booknlp-ru-spacy")

TokenizerType = Literal["internal", "razdel"]
OutputFormat  = Literal["native", "conllu"]


# ─── Service ─────────────────────────────────────────────────────────────────

@app.cls(image=image, timeout=600, scaledown_window=300)
class SpacyService:
    """
    Modal-сервис для морфо-синтаксического анализа с использованием
    официальной модели ru_core_news_lg.

    Поддерживает два токенизатора:
      internal — встроенный rule-based токенизатор spaCy
      razdel   — внешний ML-токенизатор razdel (рекомендован для русского)

    Два формата вывода:
      native — полный набор атрибутов spaCy (List[Dict])
      conllu — стандарт Universal Dependencies (str)

    Сентенизация выполняется в wrapper ДО отправки в Modal.
    Основные методы принимают уже разбитые предложения (чанки).

    Основные методы (production path из wrapper):
      parse_sentence_chunk        — razdel path: List[(text, start_char)]
      parse_sentence_chunk_native — internal path: List[str]

    Вспомогательные методы (local_entrypoint / прямые вызовы):
      parse       — одиночный текст целиком
      parse_batch — список текстов целиком
    """

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

            def __call__(self, text: str):
                from spacy.tokens import Doc
                tokens = list(razdel_tokenize(text))
                if not tokens:
                    return Doc(self.vocab, words=[], spaces=[])
                words  = [t.text for t in tokens]
                spaces = [
                    tokens[i].stop < tokens[i + 1].start
                    for i in range(len(tokens) - 1)
                ] + [False]
                return Doc(self.vocab, words=words, spaces=spaces)

        self.razdel_tokenizer = RazdelTokenizer(self.nlp.vocab)

        if "conll_formatter" not in self.nlp.pipe_names:
            config = {
                "ext_names": {
                    "conll_str": "conll_str",
                    "conll":     "conll",
                    "conll_pd":  "conll_pd",
                },
                "conversion_maps": {
                    "UPOS": {}, "XPOS": {}, "FEATS": {}, "DEPREL": {},
                },
            }
            self.nlp.add_pipe("conll_formatter", config=config, last=True)

        self.logger.info("SpaCy loaded (ru_core_news_lg)!")
        self.logger.info(f"Pipeline components: {self.nlp.pipe_names}")
        self.logger.info("Tokenizers: internal (spaCy rule-based), razdel (ML)")

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _make_doc(self, text: str, tokenizer_type: TokenizerType):
        """Токенизирует текст выбранным токенизатором без запуска pipeline."""
        if tokenizer_type == "razdel":
            return self.razdel_tokenizer(text)
        return self.original_tokenizer(text)

    def _run_pipeline(self, doc):
        """Прогоняет Doc через все компоненты pipeline поодиночке."""
        for _, pipe in self.nlp.pipeline:
            doc = pipe(doc)
        return doc

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

    @staticmethod
    def _format_native(doc, char_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Полный нативный формат spaCy — все атрибуты токена.

        char_offset: смещение начала текста в исходном документе.
        Передаётся из parse_sentence_chunk для корректных start_char/end_char
        относительно всего исходного текста (не только текущего предложения).
        """
        result = []
        for sent in doc.sents:
            sent_data = {
                "text":       sent.text,
                "start_char": sent.start_char + char_offset,
                "end_char":   sent.end_char   + char_offset,
                "words":      [],
            }
            sent_token_offset = sent.start  # индекс первого токена предложения в Doc
            for token in sent:
                word_dict = {
                    # ── Позиция ──────────────────────────────────────────
                    "id":         token.i - sent_token_offset + 1,
                    "start_char": token.idx + char_offset,
                    "end_char":   token.idx + len(token.text) + char_offset,
                    # ── Форма ────────────────────────────────────────────
                    "form":  token.text,
                    "norm":  token.norm_,
                    "lower": token.lower_,
                    "shape": token.shape_,
                    # ── Лемма и POS ──────────────────────────────────────
                    # lemma_ вычислен через pymorphy3 (режим lemmatizer=pymorphy3
                    # задокументирован для русского языка в spaCy)
                    "lemma": token.lemma_,
                    "upos":  token.pos_,
                    "xpos":  token.tag_,
                    "feats": str(token.morph) if token.morph.to_dict() else "_",
                    # ── Синтаксис ─────────────────────────────────────────
                    "head":     (token.head.i - sent_token_offset + 1
                                 if token.head.i != token.i else 0),
                    "deprel":   token.dep_,
                    "n_lefts":  token.n_lefts,
                    "n_rights": token.n_rights,
                    "children": [c.i - sent_token_offset + 1
                                 for c in token.children],
                    # ── Именованные сущности ──────────────────────────────
                    "ent_type": token.ent_type_ or None,
                    "ent_iob":  token.ent_iob_ if token.ent_iob_ != "O" else None,
                    # ── Метаданные ────────────────────────────────────────
                    "is_sent_start": token.is_sent_start,
                    "whitespace":    token.whitespace_,
                    "misc":          "SpaceAfter=No" if not token.whitespace_ else "_",
                    # ── Лексические флаги ────────────────────────────────
                    "is_alpha":   token.is_alpha,
                    "is_digit":   token.is_digit,
                    "is_punct":   token.is_punct,
                    "is_space":   token.is_space,
                    "is_stop":    token.is_stop,
                    "is_oov":     token.is_oov,
                    "like_num":   token.like_num,
                    "like_url":   token.like_url,
                    "like_email": token.like_email,
                    # ── Вектор ────────────────────────────────────────────
                    "has_vector":  token.has_vector,
                    "cluster":     token.cluster,
                    "vector_norm": round(float(token.vector_norm), 6)
                                   if token.has_vector else None,
                }
                sent_data["words"].append(word_dict)

            ents = [
                {
                    "text":       ent.text,
                    "start":      ent.start      - sent.start,
                    "end":        ent.end         - sent.start,
                    "label":      ent.label_,
                    "start_char": ent.start_char + char_offset,
                    "end_char":   ent.end_char   + char_offset,
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
        return doc._.conll_str  # type: ignore[attr-defined]

    # ─── Production methods: принимают pre-split чанки из wrapper ────────────

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
        docs         = []
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
            result.extend(self._format_native(doc, char_offset=char_offset))
        return result

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: str = "native",
        batch_size: int = 32,
    ) -> Any:
        """
        Internal/native path.

        Принимает чанк текстов предложений (без символьных офсетов).
        start_char/end_char токенов — относительны каждого предложения.

        Args:
            sentences:     List[str] — тексты предложений чанка
            output_format: 'native' | 'conllu'
            batch_size: int
        Returns:
            native → List[Dict]
            conllu → str
        """
        docs = [self._make_doc(s, "internal") for s in sentences]
        docs = self._run_pipeline_batch(docs, batch_size=batch_size)

        if output_format == "conllu":
            # noinspection PyProtectedMember
            return "\n\n".join(
                doc._.conll_str.strip() for doc in docs  # type: ignore[attr-defined]
            ) + "\n"

        result = []
        for doc in docs:
            result.extend(self._format_native(doc))
        return result

    # ─── Backward compat / local_entrypoint ──────────────────────────────────

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
        return self._format_native(doc)

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
        return [self._format_native(doc) for doc in docs]


# ─── Вспомогательная функция вывода ─────────────────────────────────────────

def _print_token_full(tok: Dict[str, Any]) -> None:
    """Выводит все поля токена в нативном формате spaCy."""
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"  ПОЗИЦИЯ:")
    print(f"    start_char:    {tok['start_char']}")
    print(f"    end_char:      {tok['end_char']}")
    print(f"  ФОРМА:")
    print(f"    form:          {tok['form']}")
    print(f"    norm:          {tok.get('norm', '—')}")
    print(f"    lower:         {tok.get('lower', '—')}")
    print(f"    shape:         {tok.get('shape', '—')}")
    print(f"  ЛЕММА И POS:")
    print(f"    lemma:         {tok['lemma']}")
    print(f"    upos:          {tok['upos']}")
    print(f"    xpos:          {tok['xpos']}")
    print(f"    feats:         {tok['feats']}")
    print(f"  СИНТАКСИС:")
    print(f"    head:          {tok['head']}")
    print(f"    deprel:        {tok['deprel']}")
    print(f"    n_lefts:       {tok.get('n_lefts', '—')}")
    print(f"    n_rights:      {tok.get('n_rights', '—')}")
    print(f"    children:      {tok.get('children', [])}")
    print(f"  СУЩНОСТИ:")
    print(f"    ent_type:      {tok.get('ent_type') or '—'}")
    print(f"    ent_iob:       {tok.get('ent_iob') or '—'}")
    print(f"  МЕТАДАННЫЕ:")
    print(f"    is_sent_start: {tok.get('is_sent_start')}")
    print(f"    whitespace:    '{tok.get('whitespace', '')}'")
    print(f"    misc:          {tok.get('misc', '—')}")
    print(f"  ФЛАГИ:")
    print(f"    is_alpha:      {tok.get('is_alpha')}")
    print(f"    is_digit:      {tok.get('is_digit')}")
    print(f"    is_punct:      {tok.get('is_punct')}")
    print(f"    is_space:      {tok.get('is_space')}")
    print(f"    is_stop:       {tok.get('is_stop')}")
    print(f"    is_oov:        {tok.get('is_oov')}")
    print(f"    like_num:      {tok.get('like_num')}")
    print(f"    like_url:      {tok.get('like_url')}")
    print(f"    like_email:    {tok.get('like_email')}")
    print(f"  ВЕКТОР:")
    print(f"    has_vector:    {tok.get('has_vector')}")
    vn = tok.get("vector_norm")
    print(f"    vector_norm:   {vn if vn is not None else '—'}")

    # ─── Константа заголовка CoNLL-U ──────────────────────────────────────────
CONLLU_HEADER = "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"

def _print_conllu(text: str, conllu: str) -> None:
    """Выводит CoNLL-U блок с текстом предложения и заголовком столбцов."""
    print(f"\n# text = {text}")
    print(CONLLU_HEADER)
    print(conllu)

# ─── local_entrypoint: тест Modal-сервиса напрямую ───────────────────────────

@app.local_entrypoint()
def main():
    """
    Тестирует SpaCyService напрямую — без wrapper, без chunking.
    Проверяет: модель загружена, оба токенизатора работают,
    оба формата вывода корректны, оба production-метода работают.
    """
    from razdel import sentenize

    service = SpacyService()

    text_single = "Кружка-термос стоит 500р."
    text_multi  = "Зло, которым пугаешь, не так зло. Москва — столица России."

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

    # ── 2. NATIVE + RAZDEL ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("2. NATIVE + RAZDEL (parse.remote)")
    print(sep)
    result_r = service.parse.remote(text_single, output_format="native", tokenizer="razdel")
    print(f"\n⚡ Сравнение токенизаторов для: '{text_single}'")
    print(f"  internal: {[w['form'] for s in result   for w in s['words']]}")
    print(f"  razdel:   {[w['form'] for s in result_r for w in s['words']]}")
    for sent in result_r:
        print(f"\nПредложение: '{sent['text']}'")
        for tok in sent["words"]:
            _print_token_full(tok)

    # ── 3. CONLL-U + INTERNAL ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("3. CONLL-U + INTERNAL (parse.remote)")
    print(sep)
    result_conllu_i = service.parse.remote(
        text_multi, output_format="conllu", tokenizer="internal"
    )
    _print_conllu(text_multi, result_conllu_i)

    # ── 4. CONLL-U + RAZDEL ──────────────────────────────────────────────
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
    chunk_texts = [s.text for s in sentences]
    print(f"Чанк ({len(chunk_texts)} предложений): {chunk_texts}")
    result_chunk_native = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    for sent in result_chunk_native:
        print(f"\nПредложение: '{sent['text']}' "
              f"(chars {sent['start_char']}:{sent['end_char']})")
        if sent.get("entities"):
            print(f"  Сущности: {[(e['text'], e['label']) for e in sent['entities']]}")
        for tok in sent["words"]:
            _print_token_full(tok)

    print(f"\n{'✅ Тестирование завершено!':^72}")
