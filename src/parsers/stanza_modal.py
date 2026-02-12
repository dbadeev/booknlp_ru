import modal
import logging

image = (
    modal.Image.debian_slim()
    .pip_install("stanza", "torch")
    .run_commands("python -c 'import stanza; stanza.download(\"ru\")'")
)

app = modal.App("booknlp-ru-stanza")


@app.cls(image=image, gpu="T4", timeout=600, container_idle_timeout=300)
class StanzaService:
    @modal.enter()
    def setup(self):
        import stanza
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StanzaService")

        # Инициализируем ДВА пайплайна (они делят одну модель в памяти GPU):

        # 1. Для сырого текста (обычный режим)
        # tokenize_pretokenized=False (по умолчанию)
        self.nlp_raw = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=True)

        # 2. Для готовых токенов (режим аудита)
        # tokenize_pretokenized=True заставляет Stanza верить нашим токенам
        self.nlp_pretokenized = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse', verbose=False,
                                                use_gpu=True, tokenize_pretokenized=True)

        self.logger.info("Stanza loaded (Dual Mode)!")

    @modal.method()
    def parse(self, text: str) -> list[list[dict]]:
        """Вход: сырой текст. Использует nlp_raw."""
        doc = self.nlp_raw(text)
        return self._format_output(doc)

    @modal.method()
    def parse_batch(self, batch_tokens: list[list[str]]) -> list[list[dict]]:
        """Вход: готовые токены. Использует nlp_pretokenized."""
        # Stanza с tokenize_pretokenized=True ожидает список списков строк
        doc = self.nlp_pretokenized(batch_tokens)
        return self._format_output(doc)

    def _format_output(self, doc) -> list[list[dict]]:
        """
        Вспомогательный метод (бывший _doc_to_list).
        Конвертирует внутренний объект Stanza Doc в наш формат списка словарей.
        Используется обоими методами выше, чтобы избежать дублирования кода.
        """
        result = []
        for sent in doc.sentences:
            sent_parsed = []
            for word in sent.words:
                sent_parsed.append({
                    "id": int(word.id),
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": int(word.head),
                    "deprel": word.deprel,
                    "start_char": word.start_char,
                    "end_char": word.end_char
                })
            result.append(sent_parsed)
        return result