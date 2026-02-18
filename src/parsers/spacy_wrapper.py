#!/usr/bin/env python3
"""
Локальная обертка для spaCy парсера.

Использование:
    from spacy_wrapper import SpacyParser, SpacyPymorphy3Parser

    # Базовый парсер (только spaCy)
    parser = SpacyParser()
    result = parser.parse_text(text, output_format="simplified")
    result = parser.parse_text(text, output_format="native")
    result = parser.parse_text(text, output_format="conllu")

    # Обогащенный парсер (spaCy + pymorphy3)
    parser_enriched = SpacyPymorphy3Parser()
    result = parser_enriched.parse_text_enriched(
        text,
        output_format="simplified",
        include_lexeme=True
    )

    # Пакетная обработка
    results = parser.parse_batch(texts, output_format="simplified")
"""
import logging
import modal
from typing import List, Dict, Any, Union, Literal, Optional

logger = logging.getLogger(__name__)

OutputFormat = Literal["simplified", "native", "conllu"]


class SpacyParser:
    """
    Клиент для spaCy парсера, запущенного в Modal.

    Использует официальную модель ru_core_news_lg (CNN/Tok2Vec архитектура)
    для морфо-синтаксического анализа русского языка.
    """

    def __init__(self):
        """
        Инициализация парсера.

        Raises:
            Exception: Если не удалось подключиться к Modal
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-spacy",
                "SpacyService"
            )()
            self.logger.info("✓ Connected to SpaCy via Modal.")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Modal: {e}")
            raise e

    def parse_text(
            self,
            text: str,
            output_format: OutputFormat = "simplified"
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]], str]:
        """
        Парсит текст с использованием spaCy.

        Args:
            text: Входной текст для анализа
            output_format: Формат вывода
                - 'simplified': упрощенный формат (список предложений → список токенов)
                - 'native': полный нативный формат spaCy со всеми атрибутами
                - 'conllu': строка в стандартном формате CoNLL-U

        Returns:
            Результат парсинга в выбранном формате

        Форматы вывода:

        simplified (List[List[Dict]]):
            [
                [  # Предложение 1
                    {
                        "id": 1,
                        "form": "Коля",
                        "lemma": "коля",
                        "upos": "PROPN",
                        "xpos": "PROPN",
                        "feats": "Case=Nom|Gender=Masc|Number=Sing",
                        "head": 2,
                        "deprel": "nsubj",
                        "start_char": 0,
                        "end_char": 4
                    },
                    {...}
                ],
                [...]  # Предложение 2
            ]

        native (List[Dict]):
            [
                {
                    "text": "Коля сказал...",
                    "start_char": 0,
                    "end_char": 20,
                    "words": [
                        {
                            "id": 1,
                            "form": "Коля",
                            "lemma": "коля",
                            "upos": "PROPN",
                            "xpos": "PROPN",
                            "feats": "Case=Nom|Gender=Masc|Number=Sing",
                            "head": 2,
                            "deprel": "nsubj",
                            "start_char": 0,
                            "end_char": 4,
                            "ent_type": "PER",
                            "ent_iob": "B",
                            "is_sent_start": True,
                            "whitespace": " ",
                            "shape": "Xxxx",
                            "is_alpha": True,
                            "is_punct": False,
                            "like_num": False
                        },
                        {...}
                    ],
                    "entities": [
                        {
                            "text": "Коля",
                            "start": 0,
                            "end": 1,
                            "label": "PER"
                        }
                    ]
                },
                {...}
            ]

        conllu (str):
            # sent_id = 1
            # text = Коля сказал...
            1	Коля	коля	PROPN	PROPN	Case=Nom|Gender=Masc|Number=Sing	2	nsubj	_	_
            2	сказал	сказать	VERB	VERB	...	0	root	_	SpaceAfter=No
            ...
        """
        try:
            return self.service.parse.remote(text, output_format=output_format)
        except Exception as e:
            self.logger.error(f"❌ Error during spaCy parsing: {e}")
            raise e

    def parse_batch(
            self,
            texts: List[str],
            output_format: OutputFormat = "simplified",
            batch_size: int = 32
    ) -> List[Union[List[List[Dict[str, Any]]], List[Dict[str, Any]], str]]:
        """
        Пакетная обработка текстов для повышения производительности.

        Args:
            texts: Список текстов для анализа
            output_format: Формат вывода
            batch_size: Размер батча (по умолчанию 32)

        Returns:
            Список результатов для каждого текста в том же формате,
            что и parse_text()
        """
        try:
            return self.service.parse_batch.remote(
                texts,
                output_format=output_format,
                batch_size=batch_size
            )
        except Exception as e:
            self.logger.error(f"❌ Error during batch parsing: {e}")
            raise e


# ========== ИНТЕГРАЦИЯ С PYMORPHY3 ==========
class SpacyPymorphy3Parser:
    """
    Расширенная версия парсера с полной информацией из pymorphy3.

    Использует spaCy для качественного синтаксиса и добавляет
    ПОЛНУЮ детальную морфологию из pymorphy3.

    Преимущества перед standalone pymorphy3_wrapper:
    - ✅ Правильное синтаксическое дерево (от spaCy)
    - ✅ NER (от spaCy)
    - ✅ Полная парадигма слова (от pymorphy3)
    - ✅ Вероятности и методы разбора (от pymorphy3)
    - ✅ Сравнение контекстной и бесконтекстной морфологии
    """

    def __init__(self):
        """Инициализация парсера с интеграцией pymorphy3."""
        import pymorphy3

        self.spacy_parser = SpacyParser()
        self.morph = pymorphy3.MorphAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.logger.info("✓ SpaCy+Pymorphy3 parser initialized.")

    def parse_text_enriched(
            self,
            text: str,
            output_format: OutputFormat = "simplified",
            include_lexeme: bool = False,
            include_all_parses: bool = False
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Парсит текст с полным обогащением данными из pymorphy3.

        Добавляет к каждому токену ВСЕ доступные поля pymorphy3:

        Базовые поля (всегда):
        - pymorphy3_word: форма слова из pymorphy3 (lowercase!)
        - pymorphy3_lemma: нормальная форма из pymorphy3
        - pymorphy3_tag: полный тег OpenCorpora
        - pymorphy3_score: вероятность данного разбора (0.0-1.0)
        - pymorphy3_is_known: True если слово есть в словаре

        Расширенные поля (всегда):
        - pymorphy3_methods_stack: список методов, которые pymorphy3
          использовал для разбора (например, [('DictAnalyzer', 1.0)])
        - pymorphy3_normalized: информация о нормализованной форме
          {word, tag, score}

        Опциональные поля:
        - pymorphy3_lexeme: полная парадигма слова - все словоформы
          (если include_lexeme=True, может быть 10-30+ форм)
        - pymorphy3_all_parses: все возможные разборы слова
          (если include_all_parses=True, полезно для омонимов)

        Args:
            text: Входной текст
            output_format: Формат вывода ('simplified' или 'native')
                           'conllu' не поддерживается
            include_lexeme: Включать ли полную парадигму слова
            include_all_parses: Включать ли все варианты разбора

        Returns:
            Обогащенный результат парсинга

        Примечания:
            Отличия значений полей spaCy vs pymorphy3:
            - lemma (spaCy) — контекстно-зависимая лемматизация
            - pymorphy3_lemma — первый (наиболее вероятный) разбор без контекста
            - form (spaCy) — оригинальный регистр
            - pymorphy3_word — всегда lowercase
            - feats (spaCy) — формат Universal Dependencies
            - pymorphy3_tag — формат OpenCorpora
        """
        # Получаем базовый результат от spaCy
        spacy_result = self.spacy_parser.parse_text(
            text,
            output_format=output_format
        )

        # Для CoNLL-U не добавляем дополнительные поля
        if output_format == "conllu":
            self.logger.warning(
                "CoNLL-U format doesn't support enrichment. "
                "Returning original spaCy result."
            )
            return spacy_result

        # Обогащаем данными из pymorphy3
        enriched_result = []
        for sent in spacy_result:
            enriched_sent = []

            # Получаем список токенов (зависит от формата)
            words_list = sent if output_format == "simplified" else sent.get("words", [])

            for token_dict in words_list:
                form = token_dict.get("form", "")

                # Получаем ВСЕ разборы
                all_parses = self.morph.parse(form)
                p = all_parses[0]  # Наиболее вероятный

                # Копируем токен и добавляем поля pymorphy3
                token_dict_enriched = token_dict.copy()

                # ========== БАЗОВЫЕ ПОЛЯ ==========
                token_dict_enriched.update({
                    # ВАЖНО: p.word всегда lowercase!
                    "pymorphy3_word": p.word,
                    "pymorphy3_lemma": p.normal_form,
                    "pymorphy3_tag": str(p.tag),
                    "pymorphy3_score": p.score,
                    "pymorphy3_is_known": p.is_known,
                })

                # ========== РАСШИРЕННЫЕ ПОЛЯ ==========
                # methods_stack показывает как именно слово было разобрано
                # Примеры:
                # - [('DictAnalyzer', 1.0)] - найдено в словаре
                # - [('FakeDictionary', 0.1)] - неизвестное слово
                # - [('KnownPrefixAnalyzer', 0.5)] - по известным префиксам
                token_dict_enriched["pymorphy3_methods_stack"] = p.methods_stack

                # Информация о нормализованной форме
                token_dict_enriched["pymorphy3_normalized"] = {
                    "word": p.normalized.word,
                    "tag": str(p.normalized.tag),
                    "score": p.normalized.score
                }

                # ========== ОПЦИОНАЛЬНЫЕ ПОЛЯ ==========

                # Полная парадигма (опционально, так как может быть большой)
                if include_lexeme:
                    # lexeme содержит ВСЕ формы слова
                    # Например для "идти": иду, идешь, идет, шел, шла, пошёл ...
                    token_dict_enriched["pymorphy3_lexeme"] = [
                        {
                            "word": form.word,
                            "tag": str(form.tag)
                        }
                        for form in p.lexeme
                    ]
                else:
                    # Возвращаем только количество форм в парадигме
                    token_dict_enriched["pymorphy3_lexeme_count"] = len(p.lexeme)

                # Все возможные разборы (для омонимов)
                if include_all_parses and len(all_parses) > 1:
                    token_dict_enriched["pymorphy3_all_parses"] = [
                        {
                            "normal_form": parse.normal_form,
                            "tag": str(parse.tag),
                            "score": parse.score,
                            "is_known": parse.is_known
                        }
                        for parse in all_parses
                    ]
                    token_dict_enriched["pymorphy3_parses_count"] = len(all_parses)

                enriched_sent.append(token_dict_enriched)

            # Формируем результат в зависимости от формата
            if output_format == "native":
                sent_copy = sent.copy()
                sent_copy["words"] = enriched_sent
                enriched_result.append(sent_copy)
            else:
                enriched_result.append(enriched_sent)

        return enriched_result


# ========== ТЕСТОВЫЕ ПРИМЕРЫ ==========
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Тестовые тексты
    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    test_ambiguous = "мой брат"  # омоним: мой (местоимение) vs мыть
    test_name = "Москва - столица России."

    print("=" * 80)
    print("ТЕСТ 1: БАЗОВЫЙ SPACY PARSER")
    print("=" * 80)

    parser = SpacyParser()

    # 1.1 Simplified format
    print("\n1.1 SIMPLIFIED FORMAT:")
    print("-" * 80)
    result_simple = parser.parse_text(test_text, output_format="simplified")

    print(f"Текст: '{test_text}'")
    print(f"Предложений: {len(result_simple)}\n")

    for sent in result_simple:
        print(f"Токенов в предложении: {len(sent)}\n")
        print("ID\tFORM\t\tLEMMA\t\tUPOS\tDEPREL")
        print("-" * 80)
        for tok in sent[:7]:
            print(f"{tok['id']}\t{tok['form']:<12}\t{tok['lemma']:<12}\t"
                  f"{tok['upos']}\t{tok['deprel']}")
        if len(sent) > 7:
            print(f"... (всего {len(sent)} токенов)")

    # 1.2 Native format
    print("\n1.2 NATIVE FORMAT с NER:")
    print("-" * 80)
    result_native = parser.parse_text(test_name, output_format="native")

    for sent_data in result_native:
        print(f"\nПредложение: '{sent_data['text']}'")
        print(f"Границы: [{sent_data['start_char']}, {sent_data['end_char']}]")

        if "entities" in sent_data and sent_data["entities"]:
            print("\nИменованные сущности:")
            for ent in sent_data["entities"]:
                print(f"  - '{ent['text']}' [{ent['label']}]")
        else:
            print("\nИменованные сущности: не найдены")

        print("\nПервые 3 токена (детальная информация):")
        for tok in sent_data["words"]:
        # for tok in sent_data["words"][:3]:
            print(f"\n  Токен: '{tok['form']}'")
            print(f"    Lemma: {tok['lemma']}")
            print(f"    POS: {tok['upos']} / {tok['xpos']}")
            print(f"    Feats: {tok['feats']}")
            print(f"    Head: {tok['head']}, Deprel: {tok['deprel']}")
            print(f"    Shape: {tok['shape']}, Alpha: {tok['is_alpha']}, Punct: {tok['is_punct']}")
            if tok.get('ent_type'):
                print(f"    Entity: {tok['ent_type']} ({tok['ent_iob']})")
            if tok.get('misc'):
                print(f"    Misc: {tok['misc']}")

    # 1.3 CoNLL-U format
    print("\n1.3 CONLL-U FORMAT:")
    print("-" * 80)
    result_conllu = parser.parse_text(test_text, output_format="conllu")
    print("\nВывод в формате CoNLL-U (первые 800 символов):")
    print(result_conllu[:800])
    if len(result_conllu) > 800:
        print("... (обрезано)")

    # ========================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 2: SPACY + PYMORPHY3 (БАЗОВОЕ ОБОГАЩЕНИЕ)")
    print("=" * 80)

    parser_enriched = SpacyPymorphy3Parser()
    result_enriched = parser_enriched.parse_text_enriched(
        test_text,
        output_format="simplified",
        include_lexeme=False
    )

    print(f"\nТекст: '{test_text}'\n")
    print("Сравнение spaCy и pymorphy3 для первых 3 токенов:")
    print("-" * 80)

    for sent in result_enriched:
        for tok in sent:
        # for tok in sent[:3]:
            print(f"\n{'=' * 70}")
            print(f"Токен: '{tok['form']}'")
            print(f"{'=' * 70}")

            print(f"\n📊 SPACY (контекстный анализ):")
            print(f"  Form: {tok['form']}")
            print(f"  Lemma: {tok['lemma']}")
            print(f"  POS: {tok['upos']} (Universal)")
            print(f"  Feats: {tok['feats']}")
            print(f"  Head: {tok['head']}, Deprel: {tok['deprel']}")

            print(f"\n🔍 PYMORPHY3 (бесконтекстный анализ):")
            print(f"  Word: {tok['pymorphy3_word']} ← (всегда lowercase!)")
            print(f"  Lemma: {tok['pymorphy3_lemma']}")
            print(f"  Tag OpenCorpora: {tok['pymorphy3_tag']}")
            print(f"  Score: {tok['pymorphy3_score']:.4f}")
            print(f"  Is known: {tok['pymorphy3_is_known']}")

            print(f"\n⚙️  МЕТОДЫ РАЗБОРА:")
            print(f"  Methods stack: {tok['pymorphy3_methods_stack']}")
            if tok['pymorphy3_methods_stack']:
                method_name = tok['pymorphy3_methods_stack'][0][0]
                if method_name == 'DictAnalyzer':
                    print(f"  ↳ Найдено в словаре pymorphy3")
                elif method_name == 'FakeDictionary':
                    print(f"  ↳ Неизвестное слово, использована эвристика")
                elif method_name == 'KnownPrefixAnalyzer':
                    print(f"  ↳ Разбор по известным префиксам")

            print(f"\n📝 НОРМАЛИЗОВАННАЯ ФОРМА:")
            normalized = tok['pymorphy3_normalized']
            print(f"  Word: {normalized['word']}")
            print(f"  Tag: {normalized['tag']}")
            print(f"  Score: {normalized['score']:.4f}")

            print(f"\n📚 ПАРАДИГМА:")
            print(f"  Всего форм: {tok['pymorphy3_lexeme_count']}")

            # Сравнение лемм
            if tok['lemma'] != tok['pymorphy3_lemma']:
                print(f"\n⚠️  РАСХОЖДЕНИЕ В ЛЕММАХ:")
                print(f"  spaCy:      '{tok['lemma']}' (с учётом контекста)")
                print(f"  pymorphy3:  '{tok['pymorphy3_lemma']}' (первый разбор)")

    # ========================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 3: ПОЛНАЯ ПАРАДИГМА (lexeme)")
    print("=" * 80)

    test_word = "книга"
    result_lexeme = parser_enriched.parse_text_enriched(
        test_word,
        output_format="simplified",
        include_lexeme=True
    )

    print(f"\nТекст: '{test_word}'\n")

    for sent in result_lexeme:
        for tok in sent:
            print(f"Токен: '{tok['form']}'")
            print(f"Lemma (spaCy): {tok['lemma']}")
            print(f"Lemma (pymorphy3): {tok['pymorphy3_lemma']}")
            print(f"\nПолная парадигма ({len(tok['pymorphy3_lexeme'])} форм):")
            print("-" * 60)

            # Выводим все формы
            for i, form in enumerate(tok['pymorphy3_lexeme'], 1):
                print(f"  {i:2d}. {form['word']:<15} [{form['tag']}]")

    # ========================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 4: ОМОНИМЫ (все разборы)")
    print("=" * 80)

    result_ambiguous = parser_enriched.parse_text_enriched(
        test_ambiguous,
        output_format="simplified",
        include_all_parses=True
    )

    print(f"\nТекст: '{test_ambiguous}'\n")
    print("Анализ омонимов:")
    print("-" * 80)

    for sent in result_ambiguous:
        for tok in sent:
            print(f"\nТокен: '{tok['form']}'")

            # Результат spaCy (контекстный)
            print(f"\n  ✅ spaCy выбрал (с учётом контекста):")
            print(f"     Lemma: {tok['lemma']}, POS: {tok['upos']}")

            # Результат pymorphy3 (первый разбор)
            print(f"\n  📊 pymorphy3 предложил (первый по вероятности):")
            print(f"     Lemma: {tok['pymorphy3_lemma']}")
            print(f"     Tag: {tok['pymorphy3_tag']}")
            print(f"     Score: {tok['pymorphy3_score']:.4f}")

            # Все возможные разборы
            if 'pymorphy3_all_parses' in tok:
                print(f"\n  🔍 Все возможные разборы ({tok['pymorphy3_parses_count']}):")
                for i, parse in enumerate(tok['pymorphy3_all_parses'], 1):
                    print(f"     {i}. {parse['normal_form']:<12} "
                          f"[{parse['tag']:<30}] score={parse['score']:.4f}")

            # Сравнение
            if tok['lemma'] != tok['pymorphy3_lemma']:
                print(f"\n  ⚠️  КОНТЕКСТ ПОМОГ: spaCy выбрал '{tok['lemma']}' вместо '{tok['pymorphy3_lemma']}'")

    # ========================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 5: РЕГИСТР (form vs pymorphy3_word)")
    print("=" * 80)

    result_case = parser_enriched.parse_text_enriched(
        test_name,
        output_format="simplified"
    )

    print(f"\nТекст: '{test_name}'\n")
    print("Сравнение регистра:")
    print("-" * 80)

    for sent in result_case:
        for tok in sent:
        # for tok in sent[:3]:
            print(f"\nТокен:")
            print(f"  form (spaCy):          '{tok['form']}'  ← оригинальный регистр")
            print(f"  pymorphy3_word:        '{tok['pymorphy3_word']}'  ← всегда lowercase")

            if tok['form'] != tok['pymorphy3_word']:
                print(f"  ⚠️  РАЗЛИЧИЕ В РЕГИСТРЕ!")

    # ========================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 6: BATCH PROCESSING")
    print("=" * 80)

    test_texts_batch = [
        # "Москва - столица России.",
        # "Петр купил книгу в магазине.",
        # "Она читает интересную газету.",
        "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    ]

    print("\n6.1 Batch Processing (spaCy):")
    print("-" * 80)
    results_batch = parser.parse_batch(
        test_texts_batch,
        output_format="simplified",
        batch_size=32
    )

    print(f"\nОбработано текстов: {len(results_batch)}\n")
    for i, text_result in enumerate(results_batch):
        print(f"Текст {i + 1}: '{test_texts_batch[i]}'")
        for sent in text_result:
            tokens_str = " → ".join([
                f"{tok['form']}({tok['upos']})"
                for tok in sent
                # for tok in sent[:3]
            ])
            print(f"  Токенов: {len(sent)}, Первые 3: {tokens_str}")

    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 80)

    print("\n📊 СВОДКА:")
    print("-" * 80)
    print("✓ SpaCy parser: 3 формата (simplified, native, conllu)")
    print("✓ SpaCy+Pymorphy3: полное обогащение")
    print("  - Базовые поля: lemma, tag, score, is_known")
    print("  - Расширенные: methods_stack, normalized")
    print("  - Опциональные: lexeme, all_parses")
    print("✓ Batch processing: оптимизация производительности")
    print("✓ Выявление расхождений: контекст, регистр, омонимы")
    print("-" * 80)

    print("\n💡 КЛЮЧЕВЫЕ ОТЛИЧИЯ spaCy vs pymorphy3:")
    print("-" * 80)
    print("1. ЛЕММА:")
    print("   - spaCy: контекстно-зависимая (точнее для омонимов)")
    print("   - pymorphy3: первый по вероятности (без контекста)")
    print("\n2. РЕГИСТР:")
    print("   - spaCy (form): оригинальный регистр из текста")
    print("   - pymorphy3 (word): всегда lowercase")
    print("\n3. МОРФОЛОГИЯ:")
    print("   - spaCy (feats): формат Universal Dependencies")
    print("   - pymorphy3 (tag): формат OpenCorpora")
    print("\n4. УНИКАЛЬНЫЕ ВОЗМОЖНОСТИ pymorphy3:")
    print("   - Полная парадигма слова (все формы)")
    print("   - Все варианты разбора (для омонимов)")
    print("   - Методы разбора (как слово было проанализировано)")
    print("-" * 80)
