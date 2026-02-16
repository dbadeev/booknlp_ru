#!/usr/bin/env python3
"""
Slovnet Wrapper для booknlp_ru проекта.

Использует:
1. Razdel для токенизации (стандарт для русского NLP)
2. Modal для удаленного выполнения Slovnet+Natasha парсинга
3. Возвращает полный разбор: морфология, синтаксис, NER с normal и fact

Аналог cobald_wrapper.py
"""

import logging
import modal
from typing import List, Dict, Any
from razdel import tokenize as razdel_tokenize

logger = logging.getLogger(__name__)


class SlovnetParser:
    """
    Wrapper для Slovnet парсера с поддержкой Modal.

    Архитектура:
    - Локально: токенизация через Razdel
    - Remote (Modal): полный парсинг через Slovnet + Natasha
    """

    def __init__(self):
        """Инициализация соединения с Modal."""
        self.logger = logging.getLogger(__name__)

        try:
            # Подключаемся к Modal сервису
            self.service = modal.Cls.from_name("booknlp-ru-slovnet", "SlovnetService")()
            self.logger.info("✓ Connected to Slovnet via Modal.")
        except Exception as e:
            self.logger.error(f"✗ Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str, include_ner: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Токенизирует текст (Razdel) и отправляет в Slovnet для парсинга.

        Parameters:
        -----------
        text : str
            Исходный текст для парсинга
        include_ner : bool, optional
            Включать ли NER spans (с normal и fact через Natasha)
            По умолчанию True

        Returns:
        --------
        List[List[Dict[str, Any]]]
            Список предложений, каждое - список токенов с разметкой:
            [
                [  # Предложение 1
                    {
                        "id": 1,
                        "form": "Александр",
                        "lemma": "_",  # Slovnet не предоставляет
                        "upos": "PROPN",
                        "xpos": "_",
                        "feats": "{'Animacy': 'Anim', 'Case': 'Nom', ...}",
                        "head": 4,
                        "deprel": "nsubj",
                        "deps": "_",
                        "misc": "_"
                    },
                    ...
                ],
                ...
            ]

        Notes:
        ------
        - Токенизация выполняется локально через Razdel
        - Парсинг выполняется удаленно через Modal
        - Для получения NER spans используйте parse_text_with_ner()
        """
        try:
            # 1. Токенизация через Razdel (стандарт для русского NLP)
            tokens_gen = razdel_tokenize(text)
            tokens = [_.text for _ in tokens_gen]

            if not tokens:
                self.logger.warning("Empty tokens after tokenization")
                return []

            # 2. Отправка в Modal для парсинга
            # Возвращает список словарей в CoNLL-U подобном формате
            parsed_sent = self.service.parse.remote(tokens, text if include_ner else None)

            if not parsed_sent:
                self.logger.warning("Empty result from Modal service")
                return []

            # 3. Возвращаем как список предложений (одно предложение)
            return [parsed_sent]

        except Exception as e:
            self.logger.error(f"Error during Slovnet parsing: {e}")
            return []

    def parse_text_with_ner(self, text: str) -> Dict[str, Any]:
        """
        Полный парсинг с NER spans (включая normal и fact).

        Parameters:
        -----------
        text : str
            Исходный текст для парсинга

        Returns:
        --------
        Dict[str, Any]
            {
                "tokens": [список токенов с разметкой],
                "spans": [
                    {
                        "start": 0,
                        "stop": 26,
                        "type": "PER",
                        "text": "Александр Сергеевич Пушкин",
                        "normal": "Александр Сергеевич Пушкин",
                        "fact": {
                            "first": "Александр",
                            "middle": "Сергеевич",
                            "last": "Пушкин"
                        }
                    },
                    ...
                ]
            }

        Notes:
        ------
        Использует Natasha для извлечения normal и fact
        """
        try:
            # 1. Токенизация
            tokens_gen = razdel_tokenize(text)
            tokens = [_.text for _ in tokens_gen]

            if not tokens:
                return {"tokens": [], "spans": []}

            # 2. Полный парсинг с NER через Modal
            result = self.service.parse_with_ner.remote(tokens, text)

            return result

        except Exception as e:
            self.logger.error(f"Error during Slovnet NER parsing: {e}")
            return {"tokens": [], "spans": []}

    def parse_batch(self, texts: List[str], include_ner: bool = False) -> List[List[List[Dict[str, Any]]]]:
        """
        Батч-парсинг нескольких текстов.

        Parameters:
        -----------
        texts : List[str]
            Список текстов для парсинга
        include_ner : bool, optional
            Включать ли NER spans

        Returns:
        --------
        List[List[List[Dict[str, Any]]]]
            Список результатов для каждого текста
        """
        results = []

        for text in texts:
            try:
                result = self.parse_text(text, include_ner=include_ner)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error parsing text in batch: {e}")
                results.append([])

        return results


# ============================================================
# ТЕСТ
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("ТЕСТ SLOVNET WRAPPER")
    print("="*70)

    # Инициализация
    parser = SlovnetParser()

    # Тестовый текст
    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    print(f"\nТекст: {test_text}")

    # 1. Базовый парсинг (без NER)
    print("\n" + "-"*70)
    print("1. БАЗОВЫЙ ПАРСИНГ (CoNLL-U формат)")
    print("-"*70)

    result = parser.parse_text(test_text, include_ner=False)

    if result and result[0]:
        for token in result[0]:
            print(f"[{token['id']}] {token['form']:15} "
                  f"pos={token['upos']:8} "
                  f"head={token['head']} "
                  f"deprel={token['deprel']}")

    # 2. Парсинг с NER
    print("\n" + "-"*70)
    print("2. ПОЛНЫЙ ПАРСИНГ С NER")
    print("-"*70)

    result_ner = parser.parse_text_with_ner(test_text)

    print(f"\nТокенов: {len(result_ner.get('tokens', []))}")
    print(f"Spans: {len(result_ner.get('spans', []))}")

    if result_ner.get('spans'):
        print("\nИменованные сущности:")
        for i, span in enumerate(result_ner['spans'], 1):
            print(f"\n  Сущность #{i}:")
            print(f"    type:   {span.get('type')}")
            print(f"    text:   \"{span.get('text')}\"")
            if span.get('normal'):
                print(f"    normal: \"{span.get('normal')}\"")
            if span.get('fact'):
                print(f"    fact:   {span.get('fact')}")

    print("\n" + "="*70)
    print("ТЕСТ ЗАВЕРШЕН ✓")
    print("="*70)
