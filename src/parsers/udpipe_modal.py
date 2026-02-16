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

    # ============================================================================
    # БЛОК ПОДГОТОВКИ НАТИВНОГО ВЫХОДА МОДЕЛИ (CoNLL-U формат)
    # ============================================================================
    def _format_native_output(self, sentences: list) -> str:
        """
        Преобразует список предложений (список словарей) в нативный CoNLL-U формат.

        Формат CoNLL-U (10 колонок):
        1. ID - порядковый номер токена
        2. FORM - словоформа
        3. LEMMA - лемма
        4. UPOS - универсальный POS-тег
        5. XPOS - языково-специфичный тег
        6. FEATS - морфологические признаки
        7. HEAD - индекс главного слова
        8. DEPREL - тип синтаксической связи
        9. DEPS - вторичные зависимости (Enhanced UD)
        10. MISC - дополнительная информация

        :param sentences: список предложений (каждое - список токенов-словарей)
        :return: строка в формате CoNLL-U (предложения разделены пустой строкой)
        """
        conllu_blocks = []

        for sent in sentences:
            lines = []
            for token in sent:
                # Формируем строку CoNLL-U (10 колонок через табуляцию)
                line = "\t".join([
                    str(token.get('id', 0)),           # 1. ID
                    token.get('form', '_'),            # 2. FORM
                    token.get('lemma', '_'),           # 3. LEMMA
                    token.get('upos', '_'),            # 4. UPOS
                    token.get('xpos', '_'),            # 5. XPOS
                    token.get('feats', '_'),           # 6. FEATS
                    str(token.get('head', 0)),         # 7. HEAD
                    token.get('deprel', '_'),          # 8. DEPREL
                    token.get('deps', '_'),            # 9. DEPS
                    token.get('misc', '_')             # 10. MISC
                ])
                lines.append(line)

            # Добавляем предложение (с пустой строкой после него)
            conllu_blocks.append('\n'.join(lines))

        # Объединяем все предложения через двойной перенос строки (стандарт CoNLL-U)
        return '\n\n'.join(conllu_blocks)
    # ============================================================================

    def parse_text(self, text: str, output_format: str = 'dict'):
        """
        Парсит текст и возвращает результат в указанном формате.

        :param text: входной текст
        :param output_format: формат выхода - 'dict' (по умолчанию) или 'native'
            - 'dict': список предложений (каждое - список словарей с токенами)
            - 'native': строка в нативном формате CoNLL-U
        :return: разобранный текст в указанном формате
        """
        if not text or not text.strip():
            return [] if output_format == 'dict' else ''

        try:
            # UDPipe возвращает CoNLL-U строку
            processed = self.pipeline.process(text)

            # ========================================================================
            # ПАРСИНГ CoNLL-U В ПРОМЕЖУТОЧНЫЙ ФОРМАТ (список словарей)
            # ========================================================================
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
            # ========================================================================

            # ========================================================================
            # ВЫБОР ФОРМАТА ВЫХОДА: нативный (CoNLL-U) или текущий (dict)
            # ========================================================================
            if output_format == 'native':
                # Генерируем нативный CoNLL-U формат
                return self._format_native_output(result)
            else:
                # Возвращаем текущий формат (список словарей)
                return result
            # ========================================================================

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [] if output_format == 'dict' else ''

    @modal.method()
    def parse(self, text: str, output_format: str = 'dict'):
        """
        Публичный метод для вызова через Modal.

        :param text: текст для разбора
        :param output_format: 'dict' или 'native'
        :return: результат в указанном формате
        """
        return self.parse_text(text, output_format=output_format)

    @modal.method()
    def parse_batch(self, texts: list, output_format: str = 'dict'):
        """
        Батч-обработка списка текстов.

        :param texts: список текстов
        :param output_format: 'dict' или 'native'
        :return: список результатов в указанном формате
        """
        return [self.parse_text(text, output_format=output_format) for text in texts]


@app.local_entrypoint()
def main():
    test_text = "Мама мыла раму."
    print("🚀 Testing UDPipe service...")
    service = UDPipeService()

    # Тест 1: Текущий формат (dict)
    print("\n" + "="*80)
    print("ТЕСТ 1: Текущий формат (output_format='dict')")
    print("="*80)
    result_dict = service.parse.remote(test_text, output_format='dict')
    print(f"\n📄 Result: {len(result_dict)} sentences")
    for s_id, sent in enumerate(result_dict, 1):
        print(f"\nSentence {s_id}: {len(sent)} tokens")
        for tok in sent:
            print(f"  {tok['id']}\t{tok['form']}\t{tok['lemma']}\t{tok['upos']}\t"
                  f"{tok['xpos']}\t{tok['feats']}\t{tok['head']}\t{tok['deprel']}")

    # Тест 2: Нативный формат (CoNLL-U)
    print("\n" + "="*80)
    print("ТЕСТ 2: Нативный формат (output_format='native')")
    print("="*80)
    result_native = service.parse.remote(test_text, output_format='native')
    print(f"\n📄 CoNLL-U format:\n")
    print(result_native)

    print("\n✅ Test completed!")
