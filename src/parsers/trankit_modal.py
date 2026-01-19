import modal
import logging
import os

# Используем максимально простой путь без скрытых папок
# В этой папке будут лежать config.json, pytorch_model.bin и т.д.
LOCAL_MODEL_PATH = "/root/local_models/xlm-roberta-large"
LANG = "russian"

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "wget", "unzip")
    .pip_install(
        "six",
        "torch>=2.0.0",
        "trankit",
        "transformers==4.39.0",
        "huggingface_hub"
    )
    # 1. Создаем папку для модели
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}")

    # 2. Скачиваем БАЗОВУЮ модель (файлы ложатся прямо в папку)
    # --local-dir-use-symlinks False гарантирует реальные файлы
    .run_commands(
        f"huggingface-cli download xlm-roberta-large --local-dir {LOCAL_MODEL_PATH} --local-dir-use-symlinks False"
    )

    # 3. Скачиваем РУССКИЙ адаптер.
    # Trankit ищет адаптер строго в подпапке <path>/<lang>/
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}/{LANG}")
    .run_commands(
        f"wget https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/xlm-roberta-large/{LANG}.zip -O /tmp/russian.zip"
    )
    .run_commands(
        f"unzip -j /tmp/russian.zip -d {LOCAL_MODEL_PATH}/{LANG}/"
    )
    .run_commands("rm /tmp/russian.zip")

    # 4. Создаем флаг, что скачивание завершено (для успокоения Trankit)
    .run_commands(f"touch {LOCAL_MODEL_PATH}/{LANG}.downloaded")
)

app = modal.App("booknlp-ru-trankit")


@app.cls(image=image, gpu="T4", timeout=600)
class TrankitService:
    @modal.enter()
    def setup(self):
        import trankit
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TrankitService")
        self.logger.info(f"Setting up Trankit with local model at: {LOCAL_MODEL_PATH}")

        # === ГЛАВНЫЙ ХАК ===
        # Trankit имеет жесткий список разрешенных имен моделей.
        # Мы добавляем наш локальный путь в этот список, чтобы пройти проверку.
        if LOCAL_MODEL_PATH not in trankit.supported_embeddings:
            trankit.supported_embeddings.append(LOCAL_MODEL_PATH)
            self.logger.info(f"Added {LOCAL_MODEL_PATH} to whitelist.")

        # Переводим Transformers в оффлайн-режим для надежности
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        try:
            # Теперь мы можем передать ПУТЬ в embedding.
            # Transformers поймет, что это путь, и загрузит файлы напрямую.
            self.nlp = trankit.Pipeline(
                LANG,
                embedding=LOCAL_MODEL_PATH,
                gpu=True,
                cache_dir=LOCAL_MODEL_PATH  # Формальность
            )
            self.logger.info("Trankit loaded successfully from local files!")
        except Exception as e:
            self.logger.error(f"Failed to initialize Trankit: {e}")
            # Отладочный вывод содержимого папок
            try:
                import glob
                files = glob.glob(f"{LOCAL_MODEL_PATH}/*")
                self.logger.info(f"Files in {LOCAL_MODEL_PATH}: {files}")
            except:
                pass
            self.nlp = None

    @modal.method()
    def parse(self, text: str):
        if self.nlp is None:
            return []

        try:
            doc = self.nlp(text)
            result = []
            if 'sentences' in doc:
                for sent in doc['sentences']:
                    sent_res = []
                    for t in sent['tokens']:
                        tid = t['id']
                        if isinstance(tid, list): tid = tid[0]

                        start_char, end_char = 0, 0
                        if 'dspan' in t:
                            start_char, end_char = t['dspan']

                        sent_res.append({
                            "id": int(tid),
                            "form": t['text'],
                            "head": int(t.get('head', 0)),
                            "deprel": t.get('deprel', 'root'),
                            "start_char": start_char,
                            "end_char": end_char
                        })
                    result.append(sent_res)
            return result
        except Exception as e:
            print(f"Error parsing sentence: {e}")
            return []