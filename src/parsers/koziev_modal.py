import modal
import logging
from typing import Any, Dict, List, Literal, Tuple

# ─── Modal image ──────────────────────────────────────────────────────────────
koziev_image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential")   # build-essential на случай сборки из исходников
    .pip_install(
        "python-crfsuite",
        "git+https://github.com/Koziev/rutokenizer",
        "git+https://github.com/Koziev/rupostagger",
        "git+https://github.com/Koziev/rulemma",
    )
)

app = modal.App("booknlp-ru-koziev-service")

OutputFormat = Literal["native", "conllu"]


# ─── Service ──────────────────────────────────────────────────────────────────
@app.cls(image=koziev_image, timeout=600, scaledown_window=300)
class KozievService:
    """
    Modal-сервис для морфологического анализа с использованием
    инструментов Козиева (rutokenizer + rupostagger + rulemma).

    Сентенизация выполняется в wrapper (razdel.sentenize) ДО отправки в Modal.
    Два production-метода (вызываются из wrapper):

      parse_sentence_chunk        — razdel path:  List[(text, start_char)]
      parse_sentence_chunk_native — native path:  List[str]

    Вспомогательный метод (local_entrypoint / прямые вызовы):
      parse — одиночное предложение целиком
    """

    @modal.enter()
    def load_models(self):
        """Загрузка моделей при старте контейнера."""
        import rutokenizer
        import rupostagger
        import rulemma

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("KozievService")

        self.tokenizer = rutokenizer.Tokenizer()
        self.tokenizer.load()
        self.tagger = rupostagger.RuPosTagger()
        self.tagger.load()
        self.lemmatizer = rulemma.Lemmatizer()
        self.lemmatizer.load()

        self.logger.info(
            "Koziev models loaded (rutokenizer + rupostagger + rulemma)!"
        )
        self.logger.info(
            "Ready: razdel path → parse_sentence_chunk, "
            "native path → parse_sentence_chunk_native"
        )

    # ─── Internal helpers ──────────────────────────────────────────────────────
    def _analyze(self, sent_text: str):
        """Прогоняет одно предложение через полный pipeline Козиева."""
        tokens = self.tokenizer.tokenize(sent_text)
        tags = self.tagger.tag(tokens)
        return self.lemmatizer.lemmatize(tags)

    @staticmethod
    def _parse_pos_tags(pos_tags: str):
        """
        Разбирает строку тегов rupostagger на UPOS и FEATS.

        Формат: "UPOS|Feat1=Val1|Feat2=Val2|..."
        Возвращает (upos: str, feats: str) совместимые с CoNLL-U.

        ИСПРАВЛЕНИЕ критического бага оригинала:
            было:   upos = pos_tags.split('|')   → возвращал list, CoNLL-U ломался
            стало:  upos = pos_tags.split('|')[0] → строка, как требует стандарт
        """
        if not pos_tags:
            return "_", "_"
        parts = pos_tags.split("|")
        upos = parts[0] if parts[0] else "_"
        feats = "|".join(parts[1:]) if len(parts) > 1 else "_"
        return upos, feats

    @staticmethod
    def _lemmas_to_native(
        sent_text: str,
        lemmas,
        char_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Преобразует результат lemmatize() в sentence-dict нативного формата.

        char_offset: символьное смещение начала предложения в исходном тексте
                     (razdel path: из parse_sentence_chunk;
                      native path: всегда 0 — осознанное ограничение).
        """
        words: List[Dict[str, Any]] = []
        for i, (word, pos_tags, lemma, *_) in enumerate(lemmas, start=1):
            upos, feats = KozievService._parse_pos_tags(pos_tags)
            words.append(
                {
                    "id": i,
                    "form": word,
                    "lemma": lemma,
                    "upos": upos,
                    "feats": feats,
                }
            )
        return {
            "text": sent_text,
            "start_char": char_offset,
            "words": words,
        }

    @staticmethod
    def _lemmas_to_conllu(sent_text: str, lemmas, sent_id: int = 1) -> str:
        """
        Преобразует результат lemmatize() в CoNLL-U блок.

        Включает стандартные комментарии # sent_id и # text.
        HEAD=0 / DEPREL=root для всех токенов — заглушка:
        синтаксический разбор в pipeline Козиева отсутствует.
        """
        lines = [f"# sent_id = {sent_id}", f"# text = {sent_text}"]
        for i, (word, pos_tags, lemma, *_) in enumerate(lemmas, start=1):
            upos, feats = KozievService._parse_pos_tags(pos_tags)
            # XPOS="_" — rupostagger не разделяет UPOS/XPOS
            lines.append(
                f"{i}\t{word}\t{lemma}\t{upos}\t_\t{feats}\t0\troot\t_\t_"
            )
        return "\n".join(lines)

    # ─── Production methods ────────────────────────────────────────────────────
    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: str = "conllu",
    ) -> Any:
        """
        Razdel path.
        Принимает чанк пар (sentence_text, start_char_in_original).
        start_char используется для поля start_char в native-формате.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format: 'native' | 'conllu'

        Returns:
            native → List[Dict]  (sentence-dict: text / start_char / words)
            conllu → str
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, (sent_text, char_offset) in enumerate(
            sentences_with_offsets, start=1
        ):
            lemmas = self._analyze(sent_text)
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
        Native path.
        Принимает чанк текстов предложений (без символьных офсетов).
        start_char в native-формате всегда 0 — осознанное ограничение.

        Args:
            sentences: List[str]
            output_format: 'native' | 'conllu'

        Returns:
            native → List[Dict]
            conllu → str
        """
        native_results: List[Dict[str, Any]] = []
        conllu_blocks: List[str] = []

        for sent_id, sent_text in enumerate(sentences, start=1):
            lemmas = self._analyze(sent_text)
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

    # ─── Backward compat / local_entrypoint ───────────────────────────────────
    @modal.method()
    def parse(
        self,
        text: str,
        output_format: str = "conllu",
    ) -> Any:
        """Парсит одно предложение целиком. Для local_entrypoint и прямых вызовов."""
        lemmas = self._analyze(text)
        if output_format == "conllu":
            return self._lemmas_to_conllu(text, lemmas, sent_id=1)
        return self._lemmas_to_native(text, lemmas, char_offset=0)


# ─── Вспомогательные функции вывода ───────────────────────────────────────────
def _print_token(tok: Dict[str, Any]) -> None:
    """Выводит поля токена в нативном формате Козиева."""
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 30)
    print(f"     form:  {tok['form']}")
    print(f"     lemma: {tok['lemma']}")
    print(f"     upos:  {tok['upos']}")
    print(f"     feats: {tok['feats']}")


CONLLU_HEADER = "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"


def _print_conllu(text: str, conllu: str) -> None:
    """Выводит CoNLL-U блок с текстом предложения и заголовком столбцов."""
    print(f"\n# text = {text}")
    print(CONLLU_HEADER)
    print(conllu)


# ─── local_entrypoint: тест Modal-сервиса напрямую ────────────────────────────
@app.local_entrypoint()
def main():
    """
    Тестирует KozievService напрямую — без wrapper, без chunking.
    Проверяет оба production-метода, оба формата вывода и вспомогательный parse.
    """
    from razdel import sentenize

    service = KozievService()
    text_single = "Кружка-термос стоит 500р."
    text_multi = "Зло, которым пугаешь, не так зло. Москва — столица России."

    sep = "=" * 72

    # ── 1. parse_sentence_chunk — razdel path, NATIVE ─────────────────────────
    print(f"\n{sep}")
    print("1. parse_sentence_chunk — RAZDEL PATH, NATIVE (с символьными офсетами)")
    print(sep)
    sentences = list(sentenize(text_multi))
    chunk = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk)} предложений): {[c[0] for c in chunk]}")
    result = service.parse_sentence_chunk.remote(chunk, output_format="native")
    for sent in result:
        print(f"\nПредложение: '{sent['text']}' (start_char={sent['start_char']})")
        for tok in sent["words"]:
            _print_token(tok)

    # ── 2. parse_sentence_chunk — razdel path, CONLL-U ────────────────────────
    print(f"\n{sep}")
    print("2. parse_sentence_chunk — RAZDEL PATH, CONLL-U")
    print(sep)
    result_conllu = service.parse_sentence_chunk.remote(chunk, output_format="conllu")
    _print_conllu(text_multi, result_conllu)

    # ── 3. parse_sentence_chunk_native — native path, NATIVE ──────────────────
    print(f"\n{sep}")
    print("3. parse_sentence_chunk_native — NATIVE PATH, NATIVE (без офсетов)")
    print(sep)
    chunk_texts = [s.text for s in sentences]
    print(f"Чанк ({len(chunk_texts)} предложений): {chunk_texts}")
    result_native = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    for sent in result_native:
        print(f"\nПредложение: '{sent['text']}' (start_char={sent['start_char']})")
        for tok in sent["words"]:
            _print_token(tok)

    # ── 4. parse_sentence_chunk_native — native path, CONLL-U ─────────────────
    print(f"\n{sep}")
    print("4. parse_sentence_chunk_native — NATIVE PATH, CONLL-U")
    print(sep)
    result_native_conllu = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="conllu"
    )
    _print_conllu(text_multi, result_native_conllu)

    # ── 5. parse — одиночное предложение напрямую ─────────────────────────────
    print(f"\n{sep}")
    print("5. parse — одиночное предложение (direct call)")
    print(sep)
    result_single_native = service.parse.remote(text_single, output_format="native")
    print(f"\nПредложение: '{result_single_native['text']}'")
    for tok in result_single_native["words"]:
        _print_token(tok)

    result_single_conllu = service.parse.remote(text_single, output_format="conllu")
    _print_conllu(text_single, result_single_conllu)

    print(f"\n{'✅ Тестирование завершено!':^72}")