import modal
import logging
import os

# Локальный путь для модели
LOCAL_MODEL_PATH = "/root/local_models/xlm-roberta-large"
LANG = "russian"
TITLE = "xlm-roberta-large"

# Образ с предзагруженной моделью
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "wget", "unzip")
    .pip_install(
        "six",
        "torch==2.0.0",
        "numpy<2",
        "trankit==1.1.1",
        "transformers==4.39.0",
        "huggingface-hub")
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}")
    .run_commands(
        f"huggingface-cli download {TITLE} "
        f"--local-dir {LOCAL_MODEL_PATH} "
        f"--local-dir-use-symlinks False"
    )
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}/{LANG}")
    .run_commands(
        f"wget https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/{TITLE}/{LANG}.zip "
        f"-O /tmp/russian.zip"
    )
    .run_commands(f"unzip -j /tmp/russian.zip -d {LOCAL_MODEL_PATH}/{LANG}")
    .run_commands("rm /tmp/russian.zip")
    .run_commands(f"touch {LOCAL_MODEL_PATH}/{LANG}/.downloaded")
)

app = modal.App("booknlp-ru-trankit")


@app.cls(image=image, gpu="T4", timeout=600)
class TrankitService:
    """Trankit dependency parsing на Modal с поддержкой CUDA."""

    @modal.enter()
    def setup(self):
        import trankit
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TrankitService")

        self.logger.info(f"Setting up Trankit with local model at {LOCAL_MODEL_PATH} ({TITLE})")

        # Белый список для локальных моделей
        if LOCAL_MODEL_PATH not in trankit.supported_embeddings:
            trankit.supported_embeddings.append(LOCAL_MODEL_PATH)
            self.logger.info(f"Added {LOCAL_MODEL_PATH} to whitelist.")

        # Отключаем онлайн-загрузку
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        try:
            self.nlp = trankit.Pipeline(
                LANG,
                embedding=LOCAL_MODEL_PATH,
                gpu=True,
                cache_dir=LOCAL_MODEL_PATH
            )
            self.logger.info("Trankit loaded successfully from local files!")
        except Exception as e:
            self.logger.error(f"Failed to initialize Trankit: {e}")
            try:
                import glob
                files = glob.glob(f"{LOCAL_MODEL_PATH}/**")
                self.logger.info(f"Files in {LOCAL_MODEL_PATH}: {files}")
                lang_files = glob.glob(f"{LOCAL_MODEL_PATH}/{LANG}/**")
                self.logger.info(f"Files in {LOCAL_MODEL_PATH}/{LANG}: {lang_files}")
            except:
                pass
            self.nlp = None

    def parse_text(self, text: str):
        """
        Парсит текст и возвращает список предложений (каждое - список токенов).
        """
        if self.nlp is None:
            return []

        if not text.strip():
            return []

        try:
            doc = self.nlp(text)
            result = []

            if 'sentences' not in doc:
                return []

            for sent in doc['sentences']:
                sent_res = []

                for t in sent['tokens']:
                    # ===== ИСПРАВЛЕНО: ПРАВИЛЬНОЕ ИЗВЛЕЧЕНИЕ ПОЛЕЙ =====

                    # ID токена (может быть списком для multi-word tokens)
                    tid = t.get('id', 0)
                    if isinstance(tid, list):
                        tid = tid[0] if tid else 0

                    # Позиции в тексте
                    start_char, end_char = 0, 0
                    if 'dspan' in t:
                        dspan = t['dspan']
                        if isinstance(dspan, (list, tuple)) and len(dspan) == 2:
                            start_char, end_char = dspan

                    # ИСПРАВЛЕНО: Trankit возвращает 'upos', а не 'pos'
                    # Но для совместимости проверяем оба варианта
                    upos = t.get('upos') or t.get('pos', '_')

                    # НОВОЕ: Извлечение FEATS
                    # Trankit возвращает feats как строку или None
                    feats = t.get('feats', '_')
                    if feats is None or feats == '':
                        feats = '_'

                    # НОВОЕ: Извлечение XPOS (может отсутствовать для русского)
                    xpos = t.get('xpos', '_')
                    if xpos is None or xpos == '':
                        xpos = '_'

                    sent_res.append({
                        'id': int(tid) if isinstance(tid, int) or str(tid).isdigit() else 0,
                        'form': t.get('text', ''),
                        'lemma': t.get('lemma', t.get('text', '')),
                        'upos': upos,  # ← ИСПРАВЛЕНО: было 'pos'
                        'xpos': xpos,  # ← НОВОЕ
                        'feats': feats,  # ← НОВОЕ
                        'head': int(t.get('head', 0)),
                        'deprel': t.get('deprel', 'root'),
                        'start_char': start_char,  # ← НОВОЕ: сохраняем позиции
                        'end_char': end_char
                    })
                    # ===== КОНЕЦ ИСПРАВЛЕНИЙ =====

                if sent_res:
                    result.append(sent_res)

            return result

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    @modal.method()
    def parse(self, text: str):
        return self.parse_text(text)

    @modal.method()
    def parse_batch(self, texts: list):
        return [self.parse_text(text) for text in texts]


@app.local_entrypoint()
def main():
    test_texts = [
        "Trankit работает.",
        "Проверка на GPU."
    ]

    print("Testing Trankit service...")
    service = TrankitService()
    results = service.parse_batch.remote(test_texts)

    for i, doc in enumerate(results):
        print(f"\nDocument {i + 1}:")
        for sent_idx, sent in enumerate(doc):
            print(f"  Sentence {sent_idx + 1}: {len(sent)} tokens")
            for tok in sent[:3]:
                # НОВОЕ: выводим все поля для проверки
                print(f"    {tok['form']} -> {tok['lemma']} ({tok['upos']}) "
                      f"[xpos={tok['xpos']}, feats={tok['feats']}]")

    print("Test completed!")
