import modal
import logging

# Образ для Stanza (PyTorch based)
image = (
    modal.Image.debian_slim()
    .pip_install("stanza", "torch")
    # Предзагрузка модели для русского языка, чтобы не качать при каждом старте
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
        self.logger.info("Initializing Stanza pipeline...")

        # Инициализируем пайплайн. Отключаем NER и Sentiment для скорости, оставляем синтаксис.
        # pretokenized=True важно, если мы хотим сами управлять токенизацией (Razdel),
        # но Stanza лучше работает со своим токенизатором.
        # Для чистоты эксперимента ("End-to-End") позволим Stanza самой токенизировать текст,
        # а потом выровняем через FuzzyAligner.
        self.nlp = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=True)
        self.logger.info("Stanza loaded!")

    @modal.method()
    def parse(self, text: str) -> list[list[dict]]:
        """Возвращает список предложений в нашем унифицированном формате."""
        doc = self.nlp(text)

        result = []
        for sent in doc.sentences:
            sent_parsed = []
            for word in sent.words:
                token_data = {
                    "id": int(word.id),
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": int(word.head),
                    "deprel": word.deprel,
                    "start_char": word.start_char,  # Stanza возвращает оффсеты!
                    "end_char": word.end_char
                }
                sent_parsed.append(token_data)
            result.append(sent_parsed)

        return result