import modal
import logging

# Образ: Python + ufal.udpipe
image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "build-essential", "swig", "g++")
    .pip_install("ufal.udpipe")
    # Скачивание модели Russian-SynTagRus 2.5 с LINDAT
    .run_commands(
        "curl -L -o /root/russian-syntagrus.udpipe "
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/"
        "russian-syntagrus-ud-2.5-191206.udpipe"
    )
)

app = modal.App("booknlp-ru-udpipe")


@app.cls(image=image, timeout=600)  # UDPipe работает на CPU
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

        # Пайплайн: tokenize + tagger + parser, вывод в CoNLL-U
        self.pipeline = Pipeline(
            self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
        )
        self.logger.info("UDPipe loaded!")

    def parse_text(self, text: str):
        """
        Парсит текст и возвращает список предложений (каждое - список токенов).
        """
        if not text or not text.strip():
            return []

        try:
            # UDPipe возвращает CoNLL-U строку
            processed = self.pipeline.process(text)

            result = []
            current_sent = []

            for line in processed.split('\n'):
                line = line.strip()

                # Пропускаем комментарии и пустые строки
                if not line or line.startswith('#'):
                    if current_sent:
                        result.append(current_sent)
                        current_sent = []
                    continue

                # ===== ИСПРАВЛЕНО: ИЗВЛЕЧЕНИЕ ВСЕХ 10 ПОЛЕЙ CoNLL-U =====
                parts = line.split('\t')
                if len(parts) >= 10:  # Полный CoNLL-U формат
                    # CoNLL-U: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                    token = {
                        'id': int(parts[0]) if parts[0].isdigit() else 0,
                        'form': parts[1],
                        'lemma': parts[2],
                        'upos': parts[3],
                        'xpos': parts[4],  # ← НОВОЕ: добавлено XPOS
                        'feats': parts[5],  # ← НОВОЕ: добавлено FEATS
                        'head': int(parts[6]) if parts[6].isdigit() else 0,
                        'deprel': parts[7],
                        'deps': parts[8],  # ← НОВОЕ: Enhanced UD
                        'misc': parts[9],  # ← НОВОЕ: MISC поля
                        'startchar': 0,  # TODO: извлечь из MISC если есть TokenRange
                        'endchar': 0
                    }
                    current_sent.append(token)
                # ===== КОНЕЦ ИСПРАВЛЕНИЙ =====

            if current_sent:
                result.append(current_sent)

            return result

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    @modal.method()
    def parse(self, text: str):
        """Публичный метод для вызова через Modal."""
        return self.parse_text(text)

    @modal.method()
    def parse_batch(self, texts: list):
        """Батч-обработка списка текстов."""
        return [self.parse_text(text) for text in texts]


@app.local_entrypoint()
def main():
    test_text = "Мама мыла раму."
    print("🚀 Testing UDPipe service...")

    service = UDPipeService()
    result = service.parse.remote(test_text)

    print(f"\n📄 Result: {len(result)} sentences")
    for s_id, sent in enumerate(result, 1):
        print(f"\nSentence {s_id}: {len(sent)} tokens")
        for tok in sent:
            print(f"  {tok['id']}\t{tok['form']}\t{tok['lemma']}\t{tok['upos']}\t"
                  f"{tok['xpos']}\t{tok['feats']}\t{tok['head']}\t{tok['deprel']}")  # ← НОВОЕ: выводим все поля

    print("\n✅ Test completed!")
