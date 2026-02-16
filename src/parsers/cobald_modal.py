import modal
import logging
import sys
import os

LOCALCOBALDDIR = "src/cobald_parser"
REMOTEROOT = "/root/booknlp_ru"
REMOTESRC = f"{REMOTEROOT}/src"

# Образ для CoBaLD
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub",
        "numpy",
        "razdel",
        "torch==2.10.0",
        "transformers==4.35.2",
    )
    .env({
        "PYTHONPATH": f"{REMOTEROOT}:{REMOTESRC}:$PYTHONPATH",
        "ACCELERATE_DISABLE_MAPPING": "1",
        "ACCELERATE_USE_CPU": "0",
    })
    .add_local_dir(LOCALCOBALDDIR, remote_path=f"{REMOTESRC}/cobald_parser", copy=True)
)

app = modal.App("booknlp-ru-cobald")

@app.cls(image=image, gpu="T4", timeout=600)
class CobaldService:
    @modal.enter()
    def setup(self):
        import torch
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CobaldService")

        if REMOTEROOT not in sys.path:
            sys.path.append(REMOTEROOT)
        if REMOTESRC not in sys.path:
            sys.path.append(REMOTESRC)

        from src.cobald_parser.modeling_parser import CobaldParser
        from src.cobald_parser.configuration import CobaldParserConfig
        from src.cobald_parser.pipeline import ConlluTokenClassificationPipeline
        from razdel import tokenize as razdel_tokenize, sentenize

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "CoBaLD/xlm-roberta-base-cobald-parser-ru"

        config = CobaldParserConfig.from_pretrained(model_name)
        model = CobaldParser.from_pretrained(model_name, config=config)
        model.to(self.device)
        model.eval()

        # Используем Pipeline для декодирования
        self.pipeline = ConlluTokenClassificationPipeline(
            model=model,
            tokenizer=lambda text: [tok.text for tok in razdel_tokenize(text)],
            sentenizer=lambda text: [sent.text for sent in sentenize(text)]
        )

        self.vocab = config.vocabulary
        self.logger.info(f"CoBaLD pipeline loaded on {self.device}!")

    # ============================================================================
    # БЛОК ПОДГОТОВКИ НАТИВНОГО ВЫХОДА МОДЕЛИ (CoNLL-Plus формат)
    # ============================================================================
    def _format_native_output(self, sentence_data: dict) -> str:
        """
        Преобразует выход pipeline в нативный CoNLL-Plus формат (12 колонок).

        Формат CoNLL-Plus:
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
        11. SC (Semantic Class) - семантический класс
        12. DS (Deep Slot) - глубинный слот

        :param sentence_data: словарь с данными предложения от pipeline
        :return: строка в формате CoNLL-Plus (таблица с табуляцией)
        """
        lines = []

        # ===== СОЗДАЁМ МАППИНГ СТАРЫХ ID -> НОВЫХ ID (аналогично dict-формату) =====
        # Старые ID: 1 ([CLS]), 2 (Мама), 3 (мыла), ...
        # Новые ID: 1 (Мама), 2 (мыла), 3 (раму), ...
        id_mapping = {}  # старый_id -> новый_id
        new_id = 0

        for i, word_id in enumerate(sentence_data['ids']):
            word = sentence_data['words'][i]
            if word == '[CLS]':
                # [CLS] маппится на 0 (root)
                id_mapping[str(word_id)] = 0
            else:
                new_id += 1
                id_mapping[str(word_id)] = new_id
        # =============================================================================

        # Обрабатываем каждый токен
        for i, word_id in enumerate(sentence_data['ids']):
            word = sentence_data['words'][i]

            # Пропускаем служебный токен [CLS]
            if word == '[CLS]':
                continue

            # Колонка 1: ID (новый ID из маппинга)
            token_id = id_mapping[str(word_id)]

            # Колонка 2: FORM
            form = word

            # Колонка 3: LEMMA
            lemma = sentence_data.get('lemmas', ['_'] * len(sentence_data['words']))[i] or '_'

            # Колонка 4: UPOS
            upos = sentence_data.get('upos', ['_'] * len(sentence_data['words']))[i]

            # Колонка 5: XPOS
            xpos = sentence_data.get('xpos', ['_'] * len(sentence_data['words']))[i]

            # Колонка 6: FEATS
            feats = sentence_data.get('feats', ['_'] * len(sentence_data['words']))[i]

            # Колонки 7-8: HEAD и DEPREL (базовый UD)
            head = 0
            deprel = '_'
            if 'deps_ud' in sentence_data:
                for arc_from, arc_to, rel in sentence_data['deps_ud']:
                    if arc_to == word_id:
                        # Используем маппинг для корректного HEAD
                        head = id_mapping.get(str(arc_from), 0)
                        deprel = rel
                        break

            # Колонка 9: DEPS (Enhanced UD)
            deps = '_'
            if 'deps_eud' in sentence_data:
                eud_list = []
                for arc_from, arc_to, rel in sentence_data['deps_eud']:
                    if arc_to == word_id:
                        # Используем маппинг для корректного HEAD
                        eud_head = id_mapping.get(str(arc_from), 0)
                        eud_list.append(f"{eud_head}:{rel}")
                if eud_list:
                    deps = '|'.join(eud_list)

            # Колонка 10: MISC
            misc = sentence_data.get('miscs', ['_'] * len(sentence_data['words']))[i] if 'miscs' in sentence_data else '_'

            # Колонка 11: SC (Semantic Class) - нативное поле CoBaLD
            sc = sentence_data.get('semclasses', ['_'] * len(sentence_data['words']))[i] if 'semclasses' in sentence_data else '_'

            # Колонка 12: DS (Deep Slot) - нативное поле CoBaLD
            ds = sentence_data.get('deepslots', ['_'] * len(sentence_data['words']))[i] if 'deepslots' in sentence_data else '_'

            # Формируем строку (12 колонок через табуляцию)
            line = f"{token_id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}\t{sc}\t{ds}"
            lines.append(line)

        # Возвращаем таблицу как единую строку (строки разделены \n)
        return '\n'.join(lines)
    # ============================================================================

    @modal.method()
    def parse_batch(self, batch_tokens: list[list[str]], output_format: str = 'dict'):
        """
        batch_tokens: список предложений (каждое — список токенов).
        output_format: формат выхода - 'dict' (текущий) или 'native' (CoNLL-Plus).

        Возвращает:
        - Если output_format='dict': список предложений с максимально полным разбором (текущий формат).
        - Если output_format='native': список строк в нативном формате CoNLL-Plus.
        """
        if not batch_tokens:
            return []

        all_results = []

        for tokens in batch_tokens:
            if not tokens:
                all_results.append([] if output_format == 'dict' else '')
                continue

            try:
                # Склеиваем токены обратно в текст для pipeline
                text = " ".join(tokens)

                # Pipeline возвращает List[Dict] - список предложений
                decoded_sentences = self.pipeline(text, output_format='list')

                # Берём первое предложение
                if not decoded_sentences:
                    all_results.append([] if output_format == 'dict' else '')
                    continue

                sentence_data = decoded_sentences[0]

                # ========================================================================
                # ВЫБОР ФОРМАТА ВЫХОДА: нативный (CoNLL-Plus) или текущий (dict)
                # ========================================================================
                if output_format == 'native':
                    # Генерируем нативный CoNLL-Plus формат
                    native_output = self._format_native_output(sentence_data)
                    all_results.append(native_output)
                else:
                    # Текущая логика формирования dict (без изменений)
                    # ===== НОВОЕ: СОЗДАЁМ МАППИНГ СТАРЫХ ID -> НОВЫХ ID =====
                    # Старые ID: "1" ([CLS]), "2" (Мама), "3" (мыла), ...
                    # Новые ID: "1" (Мама), "2" (мыла), "3" (раму), ...
                    id_mapping = {}  # старый_id -> новый_id
                    new_id = 0

                    for i, word_id in enumerate(sentence_data['ids']):
                        word = sentence_data['words'][i]
                        if word == '[CLS]':
                            # [CLS] (id=1) маппится на 0 (root)
                            id_mapping['1'] = '0'
                        else:
                            new_id += 1
                            id_mapping[str(word_id)] = str(new_id)
                    # =========================================================

                    # Преобразуем в формат токенов
                    sent_tokens = []

                    for i, word_id in enumerate(sentence_data['ids']):
                        word = sentence_data['words'][i]

                        # Фильтруем служебный [CLS] токен
                        if word == '[CLS]':
                            continue

                        # ===== НОВОЕ: ИСПОЛЬЗУЕМ НОВЫЙ ID =====
                        new_token_id = id_mapping[str(word_id)]
                        # =====================================

                        token = {
                            'id': new_token_id,  # ИСПРАВЛЕНО
                            'form': word,
                            'lemma': sentence_data.get('lemmas', [''] * len(sentence_data['words']))[i] or '_',
                            'upos': sentence_data.get('upos', ['_'] * len(sentence_data['words']))[i],
                            'xpos': sentence_data.get('xpos', ['_'] * len(sentence_data['words']))[i],
                            'feats': sentence_data.get('feats', ['_'] * len(sentence_data['words']))[i],
                            'head': 0,
                            'deprel': '_',
                            'deps': '_',
                            'misc': sentence_data.get('miscs', ['_'] * len(sentence_data['words']))[
                                i] if 'miscs' in sentence_data else '_',
                        }

                        # Добавляем синтаксис из deps_ud
                        if 'deps_ud' in sentence_data:
                            for arc_from, arc_to, deprel in sentence_data['deps_ud']:
                                if arc_to == word_id:
                                    # ===== НОВОЕ: ИСПОЛЬЗУЕМ МАППИНГ =====
                                    old_head = str(arc_from)
                                    new_head = id_mapping.get(old_head, '0')
                                    token['head'] = int(new_head)
                                    # ======================================
                                    token['deprel'] = deprel
                                    break

                        # Enhanced deps
                        if 'deps_eud' in sentence_data:
                            eud_list = []
                            for arc_from, arc_to, deprel in sentence_data['deps_eud']:
                                if arc_to == word_id:
                                    # ===== НОВОЕ: ИСПОЛЬЗУЕМ МАППИНГ =====
                                    old_head = str(arc_from)
                                    new_head = id_mapping.get(old_head, '0')
                                    eud_list.append(f"{new_head}:{deprel}")
                                    # ======================================
                            if eud_list:
                                token['deps'] = '|'.join(eud_list)

                        # Семантика
                        if 'deepslots' in sentence_data and i < len(sentence_data['deepslots']):
                            token['deepslot'] = sentence_data['deepslots'][i]
                        if 'semclasses' in sentence_data and i < len(sentence_data['semclasses']):
                            token['semclass'] = sentence_data['semclasses'][i]

                        sent_tokens.append(token)

                    all_results.append(sent_tokens)
                # ========================================================================

            except Exception as e:
                self.logger.error(f"CoBaLD error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                all_results.append([] if output_format == 'dict' else '')

        return all_results

    @modal.method()
    def parse(self, tokens: list[str], output_format: str = 'dict'):
        """
        Парсинг одного предложения.

        :param tokens: список токенов
        :param output_format: 'dict' или 'native'
        :return: результат разбора в указанном формате
        """
        batch_result = self.parse_batch.remote([tokens], output_format=output_format)
        return batch_result[0] if batch_result else ([] if output_format == 'dict' else '')

@app.local_entrypoint()
def main():
    test_tokens = [
        ["Мама", "мыла", "раму", "."],
        ["CoBaLD", "работает", "на", "GPU", "."],
    ]

    print("🚀 Testing CoBaLD service...")
    service = CobaldService()

    # Тест 1: Текущий формат (dict)
    print("\n" + "="*80)
    print("ТЕСТ 1: Текущий формат (output_format='dict')")
    print("="*80)
    results_dict = service.parse_batch.remote(test_tokens, output_format='dict')
    for i, sent in enumerate(results_dict):
        print(f"\n📄 Sentence {i + 1}: {' '.join(test_tokens[i])}")
        if not sent:
            print("  [Empty result]")
            continue
        print(f"  Tokens: {len(sent)}")
        for tok in sent:
            print(
                f"  {tok['id']}\t{tok['form']}\t{tok['lemma']}\t{tok['upos']}\t"
                f"{tok.get('xpos', '_')}\t{tok.get('feats', '_')}\t"
                f"{tok['head']}\t{tok['deprel']}"
            )

    # Тест 2: Нативный формат (CoNLL-Plus)
    print("\n" + "="*80)
    print("ТЕСТ 2: Нативный формат (output_format='native')")
    print("="*80)
    results_native = service.parse_batch.remote(test_tokens, output_format='native')
    for i, sent_native in enumerate(results_native):
        print(f"\n📄 Sentence {i + 1}: {' '.join(test_tokens[i])}")
        if not sent_native:
            print("  [Empty result]")
            continue
        print("  CoNLL-Plus format (12 columns):")
        print(sent_native)

    print("\n✅ Test completed!")
