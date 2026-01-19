import modal
import logging

# UDPipe требует компиляции (C++), поэтому ставим build-essential и swig
image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "build-essential", "swig", "g++")
    .pip_install("ufal.udpipe")
    # Скачиваем модель Russian-SynTagRus 2.5 (стабильный сервер LINDAT)
    .run_commands(
        "curl -L -o /root/russian-syntagrus.udpipe https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-syntagrus-ud-2.5-191206.udpipe"
    )
)

app = modal.App("booknlp-ru-udpipe")


@app.cls(image=image, gpu="T4", timeout=600)
class UDPipeService:
    @modal.enter()
    def setup(self):
        from ufal.udpipe import Model, Pipeline
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UDPipeService")

        self.logger.info("Loading UDPipe model...")
        self.model = Model.load("/root/russian-syntagrus.udpipe")
        if not self.model:
            raise RuntimeError("Cannot load UDPipe model file!")

        # Pipeline: Tokenizer=default, Tagger=default, Parser=default
        # Output format: conllu
        self.pipeline = Pipeline(self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        self.logger.info("UDPipe loaded!")

    @modal.method()
    def parse(self, text: str) -> list[list[dict]]:
        # UDPipe принимает сырой текст и возвращает строку в формате CoNLL-U
        processed = self.pipeline.process(text)

        result = []
        current_sent = []

        for line in processed.split('\n'):
            line = line.strip()
            # Пропускаем комментарии
            if not line or line.startswith('#'):
                if current_sent:
                    result.append(current_sent)
                    current_sent = []
                continue

            parts = line.split('\t')
            if len(parts) < 10: continue

            # Парсим строку CoNLL
            token = {
                "id": int(parts[0]),
                "form": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
                "head": int(parts[6]),
                "deprel": parts[7],
                # UDPipe 1.2 не всегда возвращает char offsets в Misc,
                # но наш Aligner на клиенте это исправит.
                "start_char": 0,
                "end_char": 0
            }
            current_sent.append(token)

        if current_sent:
            result.append(current_sent)

        return result
    