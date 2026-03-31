import modal
from typing import Any, Dict, List, Literal, Tuple

# ─── Modal image ──────────────────────────────────────────────────────────────
def _download_models():
    """
    rupostagger использует ruword2tags.db, вшитый через add_local_file —
    сетевой вызов не нужен, load() в @modal.enter() читает из образа.
    Здесь загружаем только rulemma, которая скачивает свои модели через gdown.
    """
    import rulemma
    print("Загружаем rulemma...")
    rulemma.Lemmatizer().load()
    print("✓ rulemma загружена.")

# noinspection DuplicatedCode
koziev_image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential")
    .pip_install(
        "python-crfsuite",
        "gdown",
        "git+https://github.com/Koziev/rusyllab",
        "git+https://github.com/Koziev/ruword2tags",
        "git+https://github.com/Koziev/rutokenizer",
        "git+https://github.com/Koziev/rupostagger",
        "git+https://github.com/Koziev/rulemma",
    )
    .add_local_file(
        "./src/parsers/ruword2tags.db",
        "/usr/local/lib/python3.11/site-packages/ruword2tags/ruword2tags.db",
        copy=True,
    )
    .run_function(_download_models)
)

app = modal.App("booknlp-ru-koziev-service")

OutputFormat = Literal["native", "conllu"]

CONLLU_HEADER = "ID  \tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"


# ─── Service ──────────────────────────────────────────────────────────────────
@app.cls(image=koziev_image, timeout=600, scaledown_window=300)
class KozievService:
    """
    Modal-сервис морфологического анализа (Козиев).

    Сентенизация — всегда в wrapper (razdel.sentenize), ДО отправки в Modal.
    Токенизация слов — на выбор:

    tokenizer="native" (rutokenizer, внутренний):
        wrapper отправляет текст предложений → Modal токенизирует слова сам.
        Методы:
          parse_sentence_chunk         List[(sent_text, start_char)] → с офсетами
          parse_sentence_chunk_native  List[str]                     → без офсетов

    tokenizer="razdel" (razdel.tokenize, внешний):
        wrapper токенизирует слова ЛОКАЛЬНО, отправляет готовые токены в Modal.
        Методы:
          parse_pretokenized_chunk         List[(sent_text, tokens, start_char)] → с офсетами
          parse_pretokenized_chunk_native  List[(sent_text, tokens)]             → без офсетов

    Вспомогательный:
        parse  — одиночное предложение (local_entrypoint / прямые вызовы)
    """

    # noinspection DuplicatedCode
    @modal.enter()
    def load_models(self):
        import rutokenizer, rupostagger, rulemma
        import logging

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("KozievService")

        self.tokenizer = rutokenizer.Tokenizer()
        self.tokenizer.load()
        self.tagger = rupostagger.RuPosTagger()
        self.tagger.load()
        self.lemmatizer = rulemma.Lemmatizer()
        self.lemmatizer.load()

        self.logger.info(
            "✓ Koziev models loaded. "
            "native path: rutokenizer→rupostagger→rulemma; "
            "razdel path: (razdel.tokenize в wrapper)→rupostagger→rulemma."
        )

    # ─── Internal helpers ──────────────────────────────────────────────────────
    def _analyze_with_native(self, sent_text: str):
        """
        Полный pipeline с ВНУТРЕННИМ токенизатором rutokenizer.
        Используется в методах parse_sentence_chunk*.
        """
        tokens = self.tokenizer.tokenize(sent_text)
        return self._tag_and_lemmatize(tokens)

    def _tag_and_lemmatize(self, tokens: List[str]):
        """
        Тегирование + лемматизация готового списка токенов.
        Вызывается как из _analyze_with_native (токены от rutokenizer),
        так и из parse_pretokenized_chunk* (токены от razdel.tokenize).
        """
        tags = self.tagger.tag(tokens)
        return self.lemmatizer.lemmatize(tags)

    # noinspection DuplicatedCode
    @staticmethod
    def _parse_pos_tags(pos_tags: str):
        """
        Разбирает строку rupostagger → (UPOS, FEATS).
        Формат: "UPOS|Feat1=Val1|Feat2=Val2|..."
        """
        if not pos_tags:
            return "_", "_"
        parts = pos_tags.split("|")
        upos = parts[0] if parts[0] else "_"
        feats = "|".join(parts[1:]) if len(parts) > 1 else "_"
        return upos, feats

    # noinspection DuplicatedCode
    @staticmethod
    def _lemmas_to_native(
        sent_text: str,
        lemmas,
        char_offset: int = 0,
    ) -> Dict[str, Any]:
        words: List[Dict[str, Any]] = []
        for i, (word, pos_tags, lemma, *_) in enumerate(lemmas, start=1):
            upos, feats = KozievService._parse_pos_tags(pos_tags)
            words.append({
                "id": i,
                "form": word,
                "lemma": lemma,
                "upos": upos,
                "feats": feats,
            })
        return {"text": sent_text, "start_char": char_offset, "words": words}

    # noinspection DuplicatedCode
    @staticmethod
    def _lemmas_to_conllu(sent_text: str, lemmas, sent_id: int = 1) -> str:
        """
        CoNLL-U блок одного предложения.
        Структура:
          # sent_id = N
          # text = ...
          # ID  FORM  ...  (CONLLU_HEADER)
          1  токен  ...

        HEAD=0 / DEPREL=root — заглушки (синтаксис не реализован).
        """
        lines = [
            f"# sent_id = {sent_id}",
            f"# text = {sent_text}",
            CONLLU_HEADER,
        ]
        for i, (word, pos_tags, lemma, *_) in enumerate(lemmas, start=1):
            upos, feats = KozievService._parse_pos_tags(pos_tags)
            lines.append(
                f"{i}\t{word}\t{lemma}\t{upos}\t_\t{feats}\t0\troot\t_\t_"
            )
        return "\n".join(lines)

    # ─── NATIVE tokenizer methods (rutokenizer inside Modal) ──────────────────

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: str = "conllu",
    ) -> Any:
        """
        NATIVE токенизатор, с символьными офсетами.
        rutokenizer работает внутри Modal.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format: Формат вывода — 'native' | 'conllu'
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, (sent_text, char_offset) in enumerate(
            sentences_with_offsets, start=1
        ):
            lemmas = self._analyze_with_native(sent_text)
            if output_format == "native":
                native_results.append(
                    self._lemmas_to_native(sent_text, lemmas, char_offset)
                )
            else:
                conllu_blocks.append(
                    self._lemmas_to_conllu(sent_text, lemmas, sent_id)
                )

        if output_format == "conllu":
            return "\n\n".join(conllu_blocks) + "\n"
        return native_results

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: str = "conllu",
    ) -> Any:
        """
        NATIVE токенизатор, без символьных офсетов.
        rutokenizer работает внутри Modal.

        Args:
            sentences: List[str]
            output_format: Формат вывода — 'native' | 'conllu'
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, sent_text in enumerate(sentences, start=1):
            lemmas = self._analyze_with_native(sent_text)
            if output_format == "native":
                native_results.append(
                    self._lemmas_to_native(sent_text, lemmas, char_offset=0)
                )
            else:
                conllu_blocks.append(
                    self._lemmas_to_conllu(sent_text, lemmas, sent_id)
                )

        if output_format == "conllu":
            return "\n\n".join(conllu_blocks) + "\n"
        return native_results

    # ─── RAZDEL tokenizer methods (razdel.tokenize tokens from wrapper) ───────

    @modal.method()
    def parse_pretokenized_chunk(
        self,
        sentences: List[Tuple[str, List[str], int]],
        output_format: str = "conllu",
    ) -> Any:
        """
        RAZDEL токенизатор (внешний), с символьными офсетами.
        Токены уже получены в wrapper через razdel.tokenize — rutokenizer пропускается.

        Args:
            sentences: List[(sent_text, tokens, start_char)]
                sent_text  — исходный текст предложения (для # text = ...)
                tokens     — токены от razdel.tokenize(sent_text)
                start_char — символьный офсет начала предложения в документе
            output_format: Формат вывода — 'native' | 'conllu'
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, (sent_text, tokens, char_offset) in enumerate(
            sentences, start=1
        ):
            lemmas = self._tag_and_lemmatize(tokens)
            if output_format == "native":
                native_results.append(
                    self._lemmas_to_native(sent_text, lemmas, char_offset)
                )
            else:
                conllu_blocks.append(
                    self._lemmas_to_conllu(sent_text, lemmas, sent_id)
                )

        if output_format == "conllu":
            return "\n\n".join(conllu_blocks) + "\n"
        return native_results

    @modal.method()
    def parse_pretokenized_chunk_native(
        self,
        sentences: List[Tuple[str, List[str]]],
        output_format: str = "conllu",
    ) -> Any:
        """
        RAZDEL токенизатор (внешний), без символьных офсетов.
        Токены уже получены в wrapper через razdel.tokenize — rutokenizer пропускается.

        Args:
            sentences: List[(sent_text, tokens)]
            output_format: Формат вывода — 'native' | 'conllu'
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, (sent_text, tokens) in enumerate(sentences, start=1):
            lemmas = self._tag_and_lemmatize(tokens)
            if output_format == "native":
                native_results.append(
                    self._lemmas_to_native(sent_text, lemmas, char_offset=0)
                )
            else:
                conllu_blocks.append(
                    self._lemmas_to_conllu(sent_text, lemmas, sent_id)
                )

        if output_format == "conllu":
            return "\n\n".join(conllu_blocks) + "\n"
        return native_results

    # ─── Direct call ──────────────────────────────────────────────────────────

    @modal.method()
    def parse(self, text: str, output_format: str = "conllu") -> Any:
        """
        Одиночное предложение, native tokenizer.
        Для local_entrypoint и прямых вызовов.
        """
        lemmas = self._analyze_with_native(text)
        if output_format == "conllu":
            return self._lemmas_to_conllu(text, lemmas, sent_id=1)
        return self._lemmas_to_native(text, lemmas, char_offset=0)


# ─── Вспомогательные функции вывода ───────────────────────────────────────────
# noinspection DuplicatedCode
def _print_token(tok: Dict[str, Any]) -> None:
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"     form:  {tok['form']}")
    print(f"     lemma: {tok['lemma']}")
    print(f"     upos:  {tok['upos']}")
    print(f"     feats: {tok['feats']}")


def _print_conllu(conllu: str) -> None:
    """Выводит CoNLL-U. CONLLU_HEADER вшит внутри каждого блока предложения."""
    print(conllu)


# ─── local_entrypoint: тесты Modal-сервиса напрямую ──────────────────────────
@app.local_entrypoint()
def main():
    """
    Тестирует KozievService напрямую — без wrapper, без chunking.
    Проверяет все 4 production-метода и вспомогательный parse.

    Тест 1–2: native tokenizer (rutokenizer внутри Modal)
    Тест 3–4: razdel tokenizer (razdel.tokenize локально, токены → Modal)
    Тест 5:   parse() — одиночное предложение напрямую
    """
    from razdel import sentenize, tokenize as razdel_tokenize

    service = KozievService()

    text_multi = "Зло, которым пугаешь, не так зло. Москва — столица России."
    text_single = "Кружка-термос стоит 500р."
    sep = "=" * 72

    # ── 1. parse_sentence_chunk — NATIVE, с офсетами, NATIVE format ───────────
    print(f"\n{sep}")
    print("1. parse_sentence_chunk — NATIVE tokenizer, с офсетами, NATIVE format")
    # noinspection DuplicatedCode
    print(sep)
    sents = list(sentenize(text_multi))
    chunk = [(s.text, s.start) for s in sents]
    print(f"Чанк ({len(chunk)} предл.): {[c[0] for c in chunk]}")
    result = service.parse_sentence_chunk.remote(chunk, output_format="native")
    for sent in result:
        print(f"\nПредложение: '{sent['text']}' (start_char={sent['start_char']})")
        for tok in sent["words"]:
            _print_token(tok)

    # ── 2. parse_sentence_chunk — NATIVE, с офсетами, CONLL-U format ──────────
    print(f"\n{sep}")
    print("2. parse_sentence_chunk — NATIVE tokenizer, с офсетами, CoNLL-U")
    print(sep)
    result_conllu = service.parse_sentence_chunk.remote(chunk, output_format="conllu")
    _print_conllu(result_conllu)

    # ── 3. parse_sentence_chunk_native — NATIVE, без офсетов, NATIVE format ───
    print(f"\n{sep}")
    print("3. parse_sentence_chunk_native — NATIVE tokenizer, без офсетов, NATIVE")
    # noinspection DuplicatedCode
    print(sep)
    chunk_texts = [s.text for s in sents]
    print(f"Чанк ({len(chunk_texts)} предл.): {chunk_texts}")
    result_n = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    for sent in result_n:
        print(f"\nПредложение: '{sent['text']}' (start_char={sent['start_char']})")
        for tok in sent["words"]:
            _print_token(tok)

    # ── 4. parse_sentence_chunk_native — NATIVE, без офсетов, CONLL-U ─────────
    print(f"\n{sep}")
    print("4. parse_sentence_chunk_native — NATIVE tokenizer, без офсетов, CoNLL-U")
    print(sep)
    _print_conllu(
        service.parse_sentence_chunk_native.remote(
            chunk_texts, output_format="conllu"
        )
    )

    # ── 5. parse_pretokenized_chunk — RAZDEL, с офсетами, NATIVE format ───────
    print(f"\n{sep}")
    print("5. parse_pretokenized_chunk — RAZDEL tokenizer, с офсетами, NATIVE")
    print(sep)
    # Токенизируем слова ЛОКАЛЬНО через razdel.tokenize
    razdel_chunk = [
        (s.text, [t.text for t in razdel_tokenize(s.text)], s.start)
        for s in sents
    ]
    print(f"Чанк ({len(razdel_chunk)} предл.):")
    for txt, toks, off in razdel_chunk:
        print(f"  '{txt}' offset={off} → tokens={toks}")
    result_r = service.parse_pretokenized_chunk.remote(
        razdel_chunk, output_format="native"
    )
    for sent in result_r:
        print(f"\nПредложение: '{sent['text']}' (start_char={sent['start_char']})")
        for tok in sent["words"]:
            _print_token(tok)

    # ── 6. parse_pretokenized_chunk — RAZDEL, с офсетами, CONLL-U ─────────────
    print(f"\n{sep}")
    print("6. parse_pretokenized_chunk — RAZDEL tokenizer, с офсетами, CoNLL-U")
    print(sep)
    _print_conllu(
        service.parse_pretokenized_chunk.remote(razdel_chunk, output_format="conllu")
    )

    # ── 7. parse_pretokenized_chunk_native — RAZDEL, без офсетов, NATIVE ──────
    print(f"\n{sep}")
    print("7. parse_pretokenized_chunk_native — RAZDEL tokenizer, без офсетов, NATIVE")
    print(sep)
    razdel_chunk_native = [
        (s.text, [t.text for t in razdel_tokenize(s.text)])
        for s in sents
    ]
    result_rn = service.parse_pretokenized_chunk_native.remote(
        razdel_chunk_native, output_format="native"
    )
    print("⚡ Сравнение результатов (native vs razdel tokenizer):")
    for sn, sr in zip(result_n, result_rn):
        n_forms = [w["form"] for w in sn["words"]]
        r_forms = [w["form"] for w in sr["words"]]
        print(f"\n  Предложение: '{sn['text']}'")
        print(f"    native:  {n_forms}")
        print(f"    razdel:  {r_forms}")
        if n_forms != r_forms:
            print(f"    ⚠️  Токенизация различается!")

    # ── 8. parse_pretokenized_chunk_native — RAZDEL, без офсетов, CONLL-U ─────
    print(f"\n{sep}")
    print("8. parse_pretokenized_chunk_native — RAZDEL tokenizer, без офсетов, CoNLL-U")
    print(sep)
    _print_conllu(
        service.parse_pretokenized_chunk_native.remote(
            razdel_chunk_native, output_format="conllu"
        )
    )

    # ── 9. parse — одиночное предложение (direct call) ────────────────────────
    print(f"\n{sep}")
    print("9. parse — одиночное предложение (direct call, native tokenizer)")
    print(sep)
    print(f"Текст: '{text_single}'")
    result_single = service.parse.remote(text_single, output_format="native")
    print(f"\nПредложение: '{result_single['text']}'")
    for tok in result_single["words"]:
        _print_token(tok)
    print()
    _print_conllu(service.parse.remote(text_single, output_format="conllu"))

    print(f"\n{'✅ Тестирование завершено!':^72}")