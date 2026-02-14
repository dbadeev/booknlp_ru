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

    def parse_text(self, text: str, output_format: str = "simplified"):
        """
        Парсит текст и возвращает список предложений (каждое - список токенов).

        Аргументы:
            text (str): Входной текст для разбора.
            output_format (str): Формат выхода - "simplified" (текущий формат) или "native" (нативный формат Trankit).

        Возвращает: List[List[Dict]] - список предложений с токенами.

        При output_format="simplified":
            Каждый токен: {id, form, lemma, upos, xpos, feats, head, deprel, start_char, end_char}

        При output_format="native":
            Возвращает полную нативную структуру Trankit с сохранением всех полей:
            {id, text, lemma, upos, xpos, feats, head, deprel, span, dspan, ner, expanded}
        """
        if self.nlp is None:
            return []

        if not text.strip():
            return []

        try:
            doc = self.nlp(text)

            # ============================================================
            # БЛОК: Выбор формата выхода в зависимости от параметра
            # ============================================================
            if output_format == "native":
                # Нативный формат: возвращаем полную структуру Trankit
                return self._process_native(doc)
            else:
                # Упрощенный формат (текущая логика): возвращаем базовые поля
                return self._process_simplified(doc)

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    # ============================================================
    # БЛОК: Подготовка нативного выхода модели Trankit
    # ============================================================
    def _process_native(self, doc: dict):
        """
        Подготавливает нативный выход модели Trankit.

        Возвращает полную структуру вложенного словаря Trankit со всеми полями:
        - id: идентификатор токена (может быть списком для multi-word tokens)
        - text: исходный текст токена
        - lemma: нормальная форма (лемма)
        - upos: универсальная часть речи
        - xpos: язык-специфичная часть речи (опционально)
        - feats: морфологические признаки
        - head: индекс главного слова
        - deprel: тип синтаксической связи
        - span: локальное смещение в предложении (tuple)
        - dspan: глобальное смещение в документе (tuple)
        - ner: теги именованных сущностей (B-PER, O и т.д.)
        - expanded: расшифровка мультитокенов (MWT) - список словарей
        - lang: язык предложения (для многоязычных пайплайнов)  # ← ДОБАВЛЕНО!

        Аргументы:
            doc (dict): Нативный вывод от trankit.Pipeline()

        Возвращает:
            List[List[Dict]]: Список предложений с полными нативными полями
        """
        result = []

        if 'sentences' not in doc:
            return []

        for sent in doc['sentences']:
            sent_res = []
            # ============================================================
            # Извлекаем язык предложения (для многоязычных пайплайнов)
            # ============================================================
            sent_lang = sent.get('lang', None)  # ← ДОБАВЛЕНО!

            for t in sent['tokens']:
                # ============================================================
                # Сохраняем ВСЕ нативные поля токена без преобразований
                # ============================================================
                native_token = {
                    'id': t.get('id'),  # Может быть int или list для MWT
                    'text': t.get('text', ''),
                    'lemma': t.get('lemma', ''),
                    'upos': t.get('upos', '_'),
                    'xpos': t.get('xpos', '_'),
                    'feats': t.get('feats', '_'),
                    'head': t.get('head', 0),
                    'deprel': t.get('deprel', '_'),
                    'span': t.get('span'),  # Локальное смещение (tuple)
                    'dspan': t.get('dspan'),  # Глобальное смещение (tuple)
                    'ner': t.get('ner', 'O'),  # Теги NER (критическое поле!)
                    'expanded': t.get('expanded', []),  # Расшифровка MWT (критическое поле!)
                    'lang': sent_lang  # ← ДОБАВЛЕНО! Сохраняем язык предложения
                }

                sent_res.append(native_token)

            if sent_res:
                result.append(sent_res)

        return result

    # ============================================================
    # БЛОК: Упрощенный формат (текущая логика без изменений)
    # ============================================================
    def _process_simplified(self, doc: dict):
        """
        Подготавливает упрощенный выход (текущий формат).

        Возвращает токены с базовыми полями для совместимости:
        {id, form, lemma, upos, xpos, feats, head, deprel, start_char, end_char}

        Аргументы:
            doc (dict): Нативный вывод от trankit.Pipeline()

        Возвращает:
            List[List[Dict]]: Список предложений с упрощенными полями
        """
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

    @modal.method()
    def parse(self, text: str, output_format: str = "simplified"):
        """
        Удаленный метод для парсинга одного текста.

        Аргументы:
            text (str): Текст для разбора.
            output_format (str): Формат выхода - "simplified" или "native".
        """
        return self.parse_text(text, output_format)

    @modal.method()
    def parse_batch(self, texts: list, output_format: str = "simplified"):
        """
        Удаленный метод для парсинга пакета текстов.

        Аргументы:
            texts (list): Список текстов для разбора.
            output_format (str): Формат выхода - "simplified" или "native".
        """
        return [self.parse_text(text, output_format) for text in texts]


@app.local_entrypoint()
def main():
    test_texts = [
        "Trankit работает.",
        "Проверка на GPU."
    ]

    print("Testing Trankit service...")
    service = TrankitService()

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("\n" + "=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)
    results = service.parse_batch.remote(test_texts, output_format="simplified")

    for i, doc in enumerate(results):
        print(f"\nDocument {i + 1}:")
        for sent_idx, sent in enumerate(doc):
            print(f"  Sentence {sent_idx + 1}: {len(sent)} tokens")
            for tok in sent[:3]:
                print(f"    {tok['form']} -> {tok['lemma']} ({tok['upos']}) "
                      f"[xpos={tok['xpos']}, feats={tok['feats']}]")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 60)
    results_native = service.parse_batch.remote(test_texts[:1], output_format="native")

    for i, doc in enumerate(results_native):
        print(f"\nDocument {i + 1}:")
        for sent_idx, sent in enumerate(doc):
            print(f"  Sentence {sent_idx + 1}: {len(sent)} tokens")
            for tok in sent[:3]:
                print(f"    Token: {tok['text']}")
                print(f"      lemma: {tok['lemma']}, upos: {tok['upos']}")
                print(f"      span: {tok['span']}, dspan: {tok['dspan']}")
                print(f"      ner: {tok['ner']}, expanded: {tok['expanded']}")

    print("\nTest completed!")
