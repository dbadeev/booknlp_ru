import modal
import logging

# Образ с установкой зависимостей через DeepPavlov
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "deeppavlov",
        "tensorflow==2.15.0",
        "pandas",
        "transformers"
    )
    .run_commands(
        "python -m deeppavlov install syntax_ru_syntagrus_bert"
    )
)

app = modal.App("booknlp-ru-deeppavlov")

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
)
class DeepPavlovService:
    @modal.enter()
    def setup(self):
        """
        Инициализация при старте контейнера.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DeepPavlovService")
        self.logger.info("Initializing DeepPavlov container...")

        try:
            from deeppavlov import build_model
            self.logger.info("Import successful. Building model syntax_ru_syntagrus_bert...")

            # download=True гарантирует наличие весов
            # self.model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)
            self.model = build_model("syntax_ru_syntagrus_bert", download=True)
            self.logger.info("Model loaded successfully!")

        except Exception as e:
            self.logger.error(f"CRITICAL INIT ERROR: {e}", exc_info=True)
            raise  # Прокидываем исключение, чтобы контейнер не запустился

    @modal.method()
    def parse_batch(self, batch_tokens: list[list[str]]) -> list[list[dict]]:
        """
        Парсинг батча предложений.
        """
        if not batch_tokens:
            return []

        try:
            # 2. Инференс
            outputs = self.model(batch_tokens)

            parsed_batch = []

            # 3. Разбор ответа
            for i, conll_output in enumerate(outputs):
                sentence_parsed = []
                lines = conll_output.strip().split('\n')

                for line in lines:
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) < 10:
                        continue

                    token_data = {
                        "id": int(parts[0]),
                        "form": parts[1],
                        "lemma": parts[2],
                        "upos": parts[3],
                        "xpos": parts[4],
                        "feats": parts[5],
                        "head": int(parts[6]),
                        "deprel": parts[7],
                        "deps": parts[8],
                        "misc": parts[9]
                    }
                    sentence_parsed.append(token_data)

                parsed_batch.append(sentence_parsed)

            return parsed_batch

        except Exception as e:
            self.logger.error(f"Error during inference: {e}", exc_info=True)
            raise


@app.local_entrypoint()
def main():
    print("Deploying and testing...")
    tokens = ["Мама", "мыла", "раму", "."]
    service = DeepPavlovService()
    try:
        result = service.parse_batch.remote([tokens])
        print("Success! Result sample:")
        print(result[0][0])
    except Exception as e:
        print(f"Remote call failed: {e}")
