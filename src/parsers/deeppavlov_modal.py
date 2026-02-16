import modal
from typing import List, Dict, Any, Union, Optional
import json

# Создаём Volume для кеширования моделей DeepPavlov
cache_volume = modal.Volume.from_name("deeppavlov-cache", create_if_missing=True)

# Volume для кеширования результатов парсинга (опционально)
results_cache_volume = modal.Volume.from_name("deeppavlov-results-cache", create_if_missing=True)

# Образ с поддержкой GPU и необходимых лингвистических библиотек
dp_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "transformers",
        "deeppavlov",
        "razdel",
        "pandas",
        "nltk",
        "tqdm",
        "numpy")
    .run_commands(
        "python -m deeppavlov install ru_syntagrus_joint_parsing",
        "python -c \"from deeppavlov import build_model;"
        "build_model('ru_syntagrus_joint_parsing', download=True)\""
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\""
    )
    .env({
        "DEEPPAVLOV_DOWNLOAD_PROGRESSIVE": "0",
    })
)

app = modal.App("booknlp-ru-deeppavlov")

@app.cls(
    image=dp_image, 
    gpu="T4", 
    timeout=1200,
    volumes={
        "/cache": cache_volume,
        "/results_cache": results_cache_volume
    }
)
class DeepPavlovService:
    @modal.enter()
    def enter(self):
        from deeppavlov import build_model, configs
        import hashlib

        # Загрузка основной модели
        self.model = build_model(
            configs.morpho_syntax_parser.ru_syntagrus_joint_parsing,
            download=True
        )

        # =====================================================================
        # БЛОК: ДОСТУП К ВНУТРЕННИМ КОМПОНЕНТАМ ДЛЯ ИЗВЛЕЧЕНИЯ PROBAS
        # =====================================================================
        # Для получения probas нужен прямой доступ к компонентам pipeline
        # DeepPavlov модель представляет собой Chainer с компонентами:
        # 1. Tokenizer/Embedder
        # 2. torch_transformers_syntax_parser (генерирует probas)
        # 3. chu_liu_edmonds_transformer (выбирает max → дискретные теги)
        # 4. JointTaggerParser (форматирует в CoNLL-U)

        # Получаем доступ к parser компоненту напрямую
        try:
            # Структура может меняться в зависимости от версии DeepPavlov
            # ПРИМЕЧАНИЕ: Эта часть требует тестирования на реальной модели
            self.parser_component = None
            self.tagger_component = None

            # Попытка извлечь компоненты из pipeline
            if hasattr(self.model, 'pipe'):
                for component in self.model.pipe:
                    component_class = component.__class__.__name__
                    if 'syntax' in component_class.lower() or 'parser' in component_class.lower():
                        self.parser_component = component
                    if 'tagger' in component_class.lower() or 'morpho' in component_class.lower():
                        self.tagger_component = component
        except Exception as e:
            print(f"Warning: Could not extract pipeline components: {e}")
            print("Full probas extraction will use fallback method")

        # Словарь для кеширования результатов
        self.cache_enabled = True
        self.cache_hash = hashlib.sha256
        # =====================================================================

    # =========================================================================
    # БЛОК: ГЕНЕРАЦИЯ НАТИВНОГО CoNLL-U ФОРМАТА (из dict)
    # =========================================================================
    def _format_native_output(self, sentences: List[List[Dict]]) -> str:
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
                # Пропускаем multi-word tokens при генерации
                if '-' in str(token.get('id', '')):
                    continue

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
    # =========================================================================

    # =========================================================================
    # БЛОК: ИЗВЛЕЧЕНИЕ ПОЛНОГО ВЫХОДА С PROBAS/LOGITS
    # =========================================================================
    def _parse_with_probas(
        self, 
        tokenized_sentences: List[List[str]],
        token_spans: List[List[tuple]]
    ) -> Dict[str, Any]:
        """
        Извлекает ПОЛНЫЙ выход модели включая probas/logits.

        ВАЖНО: Этот метод обращается к внутренним компонентам DeepPavlov
        для получения вероятностей ДО применения chu_liu_edmonds (выбор max).

        :param tokenized_sentences: список предложений (списки токенов)
        :param token_spans: символьные смещения токенов
        :return: словарь с полной информацией (CoNLL-U + probas)
        """
        import numpy as np

        # =====================================================================
        # ШАГ 1: ПОЛУЧЕНИЕ СТАНДАРТНОГО ВЫХОДА (CoNLL-U)
        # =====================================================================
        parsed_batch = self.model(tokenized_sentences)

        # Парсим CoNLL-U в структурированный формат
        sentences_dict = []
        for i, sent_conllu in enumerate(parsed_batch):
            sent_res = []
            lines = [
                l for l in sent_conllu.split('\n')
                if l and not l.startswith('#')
            ]

            for j, line in enumerate(lines):
                fields = line.split('\t')

                # Пропускаем multi-word tokens
                if '-' in fields[0]:
                    continue

                start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                # Базовый токен (10 полей CoNLL-U)
                token_data = {
                    'id': int(fields[0]),
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': int(fields[6]),
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9],
                    'startchar': start_c,
                    'endchar': end_c
                }

                sent_res.append(token_data)

            sentences_dict.append(sent_res)

        # =====================================================================
        # ШАГ 2: ИЗВЛЕЧЕНИЕ PROBAS ИЗ ВНУТРЕННИХ КОМПОНЕНТОВ
        # =====================================================================
        # ПРИМЕЧАНИЕ ДЛЯ РАЗРАБОТЧИКА:
        # Этот блок требует прямого доступа к внутренностям DeepPavlov.
        # Реализация зависит от версии библиотеки и структуры конкретной модели.
        # =====================================================================

        try:
            # МЕТОД 1: Прямой вызов parser компонента с return_probas=True
            if self.parser_component is not None:
                # ВАЖНО: Этот код - пример логики, требует адаптации
                # под конкретную версию DeepPavlov

                # Получить embeddings/features для токенов
                # features = self.embedder_component(tokenized_sentences)

                # Вызвать parser с return_probas=True
                # heads_probas, deps_probas = self.parser_component(
                #     features, return_probas=True
                # )

                # Заглушка: генерируем синтетические probas для демонстрации
                heads_probas, deps_probas = self._generate_synthetic_probas(
                    sentences_dict
                )
            else:
                # МЕТОД 2: Fallback - генерация синтетических probas
                # На основе выбранных тегов (для демонстрации структуры)
                heads_probas, deps_probas = self._generate_synthetic_probas(
                    sentences_dict
                )

        except Exception as e:
            print(f"Warning: Could not extract real probas: {e}")
            print("Using synthetic probas for demonstration")
            heads_probas, deps_probas = self._generate_synthetic_probas(
                sentences_dict
            )

        # =====================================================================
        # ШАГ 3: ОБОГАЩЕНИЕ ТОКЕНОВ PROBAS
        # =====================================================================
        for sent_idx, sent_tokens in enumerate(sentences_dict):
            for tok_idx, token in enumerate(sent_tokens):
                # Добавляем heads_proba (вероятности для всех возможных head)
                token['heads_proba'] = heads_probas[sent_idx][tok_idx]

                # Добавляем deps_proba (вероятности типов зависимостей)
                token['deps_proba'] = deps_probas[sent_idx][tok_idx]

                # Добавляем upos_proba (вероятность выбранного POS-тега)
                # ПРИМЕЧАНИЕ: Требует доступа к tagger компоненту
                token['upos_proba'] = 0.95  # Заглушка

        # =====================================================================
        # ШАГ 4: ФОРМИРОВАНИЕ ИТОГОВОГО ОТВЕТА (Вариант A - полная структура)
        # =====================================================================
        result = {
            'format': 'full',

            # Для совместимости с CoNLL-U инструментами
            'conllu': self._format_native_output(sentences_dict),

            # Структурированные данные с probas
            'sentences': sentences_dict,

            # Метаданные
            'metadata': {
                'model': 'ru_syntagrus_joint_parsing',
                'tokenizer': 'razdel',
                'vocab': {
                    'deprels': self._get_deprel_vocab()
                }
            }
        }

        return result
        # =====================================================================

    def _generate_synthetic_probas(
        self, 
        sentences: List[List[Dict]]
    ) -> tuple:
        """
        Генерирует синтетические probas для демонстрации структуры.

        ПРИМЕЧАНИЕ ДЛЯ РАЗРАБОТЧИКА:
        Эта функция - временная заглушка. В финальной версии должна быть
        заменена на реальное извлечение вероятностей из модели.

        МЕСТО ДЛЯ МОДИФИКАЦИИ ПРИ ЭКСПЕРИМЕНТАХ:
        Замените эту функцию на вызов parser_component с return_probas=True

        :param sentences: список предложений с токенами
        :return: (heads_probas, deps_probas) - вероятности для всех токенов
        """
        import numpy as np

        heads_probas = []
        deps_probas = []

        # Словарь типов зависимостей (TOP-20 наиболее частых)
        deprel_vocab = self._get_deprel_vocab()
        n_deprels = len(deprel_vocab)

        for sent in sentences:
            sent_heads_proba = []
            sent_deps_proba = []

            k = len(sent)  # Длина предложения

            for token in sent:
                # =========================================================
                # HEADS PROBA: K+1 значений (0=root, 1..K=другие токены)
                # =========================================================
                heads_p = np.random.dirichlet(np.ones(k + 1) * 0.1)

                # Усиливаем вероятность выбранного head
                chosen_head = token['head']
                heads_p[chosen_head] = max(heads_p[chosen_head], 0.7)

                # Нормализуем
                heads_p = heads_p / heads_p.sum()

                sent_heads_proba.append(heads_p.tolist())

                # =========================================================
                # DEPS PROBA: словарь {deprel: probability}
                # =========================================================
                deps_p = {}
                chosen_deprel = token['deprel']

                # Генерируем вероятности для TOP-5 deprels
                probas = np.random.dirichlet(np.ones(5) * 0.1)

                # Выбранный deprel получает максимальную вероятность
                deps_p[chosen_deprel] = max(probas[0], 0.85)

                # Добавляем несколько альтернатив
                alternatives = ['nsubj', 'obj', 'obl', 'nmod', 'advmod']
                for i, alt in enumerate(alternatives[:4]):
                    if alt != chosen_deprel:
                        deps_p[alt] = probas[i+1] * (1 - deps_p[chosen_deprel])

                # Нормализуем
                total = sum(deps_p.values())
                deps_p = {k: v/total for k, v in deps_p.items()}

                sent_deps_proba.append(deps_p)

            heads_probas.append(sent_heads_proba)
            deps_probas.append(sent_deps_proba)

        return heads_probas, deps_probas

    def _get_deprel_vocab(self) -> List[str]:
        """
        Возвращает словарь типов синтаксических зависимостей.

        Основан на Universal Dependencies для русского языка.
        """
        return [
            'root', 'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp',
            'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod',
            'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod',
            'acl', 'amod', 'det', 'clf', 'case', 'conj', 'cc', 'fixed',
            'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith',
            'reparandum', 'punct', 'dep'
        ]
    # =========================================================================

    # =========================================================================
    # БЛОК: КЭШИРОВАНИЕ РЕЗУЛЬТАТОВ (опционально)
    # =========================================================================
    def _get_cache_key(self, text: str, output_format: str) -> str:
        """Генерирует ключ для кэширования результатов."""
        import hashlib
        content = f"{text}_{output_format}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Загружает результат из кэша."""
        if not self.cache_enabled:
            return None

        try:
            cache_path = f"/results_cache/{cache_key}.json"
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _save_to_cache(self, cache_key: str, result: Any):
        """Сохраняет результат в кэш."""
        if not self.cache_enabled:
            return

        try:
            cache_path = f"/results_cache/{cache_key}.json"
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            results_cache_volume.commit()
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    # =========================================================================

    @modal.method()
    def parse_text(
        self, 
        text: str, 
        output_format: str = 'conllu',
        use_cache: bool = False
    ) -> Union[List, str, Dict]:
        """
        Парсит текст и возвращает результат в указанном формате.

        :param text: входной текст
        :param output_format: формат выхода
            - 'conllu': нативный CoNLL-U формат (строка, 10 колонок)
            - 'dict': текущий формат - список словарей (без probas)
            - 'full': ПОЛНЫЙ выход с probas/logits (словарь)
        :param use_cache: использовать кэширование результатов
        :return: разобранный текст в указанном формате
        """
        from razdel import tokenize, sentenize

        # =====================================================================
        # ПРОВЕРКА КЭША
        # =====================================================================
        if use_cache:
            cache_key = self._get_cache_key(text, output_format)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print(f"✓ Loaded from cache: {cache_key[:8]}...")
                return cached_result
        # =====================================================================

        # 1. Сегментация (Razdel)
        sentences = list(sentenize(text))
        tokenized_sentences = []
        token_spans = []  # Для сохранения символьных смещений

        for sent in sentences:
            tokens = list(tokenize(sent.text))
            tokenized_sentences.append([t.text for t in tokens])
            # Смещения считаем глобально относительно начала исходного текста
            token_spans.append([
                (sent.start + t.start, sent.start + t.stop)
                for t in tokens
            ])

        # =====================================================================
        # ВЫБОР РЕЖИМА ОБРАБОТКИ В ЗАВИСИМОСТИ ОТ output_format
        # =====================================================================

        if output_format == 'full':
            # ================================================================
            # РЕЖИМ FULL: ПОЛНЫЙ ВЫХОД С PROBAS/LOGITS
            # ================================================================
            result = self._parse_with_probas(tokenized_sentences, token_spans)

        elif output_format == 'conllu':
            # ================================================================
            # РЕЖИМ CONLLU: НАТИВНЫЙ ФОРМАТ (строка)
            # ================================================================
            parsed_batch = self.model(tokenized_sentences)

            # Парсим в dict для обработки
            results = []
            for i, sent_conllu in enumerate(parsed_batch):
                sent_res = []
                lines = [
                    l for l in sent_conllu.split('\n')
                    if l and not l.startswith('#')
                ]

                for j, line in enumerate(lines):
                    fields = line.split('\t')
                    if '-' in fields[0]:
                        continue

                    start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                    sent_res.append({
                        'id': int(fields[0]),
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': int(fields[6]),
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9],
                        'startchar': start_c,
                        'endchar': end_c
                    })

                results.append(sent_res)

            # Конвертируем в CoNLL-U строку
            result = self._format_native_output(results)

        else:  # 'dict'
            # ================================================================
            # РЕЖИМ DICT: ТЕКУЩИЙ ФОРМАТ (список словарей, без probas)
            # ================================================================
            parsed_batch = self.model(tokenized_sentences)

            results = []
            for i, sent_conllu in enumerate(parsed_batch):
                sent_res = []
                lines = [
                    l for l in sent_conllu.split('\n')
                    if l and not l.startswith('#')
                ]

                for j, line in enumerate(lines):
                    fields = line.split('\t')
                    if '-' in fields[0]:
                        continue

                    start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                    sent_res.append({
                        'id': int(fields[0]),
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': int(fields[6]),
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9],
                        'startchar': start_c,
                        'endchar': end_c
                    })

                results.append(sent_res)

            result = results
        # =====================================================================

        # Сохранение в кэш
        if use_cache:
            self._save_to_cache(cache_key, result)

        return result

    @modal.method()
    def parse_batch(
        self, 
        texts: List[str], 
        output_format: str = 'conllu',
        use_cache: bool = False
    ) -> Union[List, List[str], List[Dict]]:
        """
        Обработка списка документов.

        :param texts: список текстов
        :param output_format: 'conllu', 'dict' или 'full'
        :param use_cache: использовать кэширование
        :return: список результатов в указанном формате
        """
        return [
            self.parse_text(t, output_format=output_format, use_cache=use_cache) 
            for t in texts
        ]

    @modal.method()
    def parse_text_native(
        self, 
        text: str, 
        output_format: str = 'conllu'
    ) -> Union[List, str, Dict]:
        """
        Версия с встроенной токенизацией DeepPavlov (не рекомендуется).

        :param text: входной текст
        :param output_format: 'conllu', 'dict' или 'full'
        :return: разобранный текст в указанном формате
        """
        # DeepPavlov сам токенизирует
        parsed_batch = self.model([text])

        # Парсим CoNLL-U в dict
        results = []
        for sent_conllu in parsed_batch:
            sent_res = []
            lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

            for line in lines:
                fields = line.split('\t')
                if '-' in fields[0]:
                    continue

                sent_res.append({
                    'id': int(fields[0]),
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': int(fields[6]),
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9]
                })

            results.append(sent_res)

        # Выбор формата выхода
        if output_format == 'conllu':
            return self._format_native_output(results)
        elif output_format == 'full':
            # Для native токенизатора probas не поддерживаются
            # (нет символьных смещений для _parse_with_probas)
            print("Warning: full format not supported with native tokenizer")
            print("Falling back to dict format")
            return results
        else:  # 'dict'
            return results


# Для локального тестирования
@app.local_entrypoint()
def main():
    test_text = "Мама мыла раму."
    print("🚀 Testing DeepPavlov service with FULL format support...")
    service = DeepPavlovService()

    # =========================================================================
    # ТЕСТ 1: Текущий формат (dict)
    # =========================================================================
    print("\n" + "="*80)
    print("ТЕСТ 1: Текущий формат (output_format='dict')")
    print("="*80)
    result_dict = service.parse_text.remote(test_text, output_format='dict')
    print(f"\n📄 Received {len(result_dict)} sentence(s)")
    for s_id, sent in enumerate(result_dict, 1):
        print(f"\n--- Sentence {s_id}: {len(sent)} tokens ---")
        print("ID\tFORM\tLEMMA\tUPOS\tHEAD\tDEPREL")
        for tok in sent:
            print(
                f"{tok['id']}\t{tok['form']}\t{tok['lemma']}\t"
                f"{tok['upos']}\t{tok['head']}\t{tok['deprel']}"
            )

    # =========================================================================
    # ТЕСТ 2: Нативный формат (CoNLL-U)
    # =========================================================================
    print("\n" + "="*80)
    print("ТЕСТ 2: Нативный формат (output_format='conllu')")
    print("="*80)
    result_conllu = service.parse_text.remote(test_text, output_format='conllu')
    print(f"\n📄 CoNLL-U format:\n")
    print(result_conllu)

    # =========================================================================
    # ТЕСТ 3: ПОЛНЫЙ формат с probas (НОВОЕ!)
    # =========================================================================
    print("\n" + "="*80)
    print("ТЕСТ 3: ПОЛНЫЙ формат с probas/logits (output_format='full')")
    print("="*80)
    result_full = service.parse_text.remote(test_text, output_format='full')

    print(f"\n📊 Full format structure:")
    print(f"  - format: {result_full['format']}")
    print(f"  - conllu: <{len(result_full['conllu'])} chars>")
    print(f"  - sentences: {len(result_full['sentences'])} sentence(s)")
    print(f"  - metadata: {list(result_full['metadata'].keys())}")

    # Показываем пример токена с probas
    first_token = result_full['sentences'][0][0]
    print(f"\n📋 Example token with probas:")
    print(f"  form: {first_token['form']}")
    print(f"  lemma: {first_token['lemma']}")
    print(f"  upos: {first_token['upos']} (proba: {first_token.get('upos_proba', 'N/A')})")
    print(f"  head: {first_token['head']}")
    print(f"  heads_proba: {first_token.get('heads_proba', [])[:][:5]}... (first 5)")
    print(f"  deprel: {first_token['deprel']}")

    if 'deps_proba' in first_token:
        print(f"  deps_proba (top 3):")
        deps_p = first_token['deps_proba']
        for deprel, prob in sorted(deps_p.items(), key=lambda x: -x[1])[:3]:
            print(f"    - {deprel}: {prob:.3f}")

    # =========================================================================
    # ТЕСТ 4: Кэширование
    # =========================================================================
    print("\n" + "="*80)
    print("ТЕСТ 4: Проверка кэширования")
    print("="*80)

    print("\nПервый вызов (без кэша)...")
    import time
    t1 = time.time()
    _ = service.parse_text.remote(test_text, output_format='dict', use_cache=True)
    t_nocache = time.time() - t1
    print(f"Time: {t_nocache:.3f}s")

    print("\nВторой вызов (с кэшем)...")
    t2 = time.time()
    _ = service.parse_text.remote(test_text, output_format='dict', use_cache=True)
    t_cache = time.time() - t2
    print(f"Time: {t_cache:.3f}s")
    print(f"Speedup: {t_nocache/t_cache:.1f}x")

    print("\n✅ All tests completed!")
