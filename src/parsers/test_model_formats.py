#!/usr/bin/env python3
"""
Тестовый скрипт для определения фактической структуры вывода всех моделей.
"""

import json
import logging
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MODEL_FORMAT_TEST")

TEST_TEXT = "Мама мыла раму."


def test_pymorphy3():
    """Тест Pymorphy3."""
    from pymorphy3_wrapper import Pymorphy3Parser

    logger.info("=== Тестирование Pymorphy3 ===")
    parser = Pymorphy3Parser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- Pymorphy3 результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_stanza():
    """Тест Stanza."""
    from stanza_wrapper import StanzaParser

    logger.info("=== Тестирование Stanza ===")
    parser = StanzaParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- Stanza результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_cobald():
    """Тест CoBaLD."""
    from cobald_wrapper import CobaldParser

    logger.info("=== Тестирование CoBaLD ===")
    parser = CobaldParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- CoBaLD результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_udpipe():
    """Тест UDPipe."""
    from udpipe_wrapper import UDPipeParser

    logger.info("=== Тестирование UDPipe ===")
    parser = UDPipeParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- UDPipe результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_deeppavlov():
    """Тест DeepPavlov."""
    from deeppavlov_wrapper import DeepPavlovParser

    logger.info("=== Тестирование DeepPavlov ===")
    parser = DeepPavlovParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- DeepPavlov результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_trankit():
    """Тест Trankit."""
    from trankit_wrapper import TrankitParser

    logger.info("=== Тестирование Trankit ===")
    parser = TrankitParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- Trankit результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_mystem():
    """Тест Mystem."""
    from mystem_wrapper import MystemParser

    logger.info("=== Тестирование Mystem ===")
    parser = MystemParser()
    result = parser.parse_text(TEST_TEXT)

    print("\n--- Mystem результат ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result and result[0]:
        print("\n--- Поля первого токена ---")
        print(f"Доступные ключи: {list(result[0][0].keys())}")

    return result


def test_rubic2():
    """Тест Rubic2 (лемматизатор, не парсер!)."""
    from rubic2_wrapper import Rubic2Lemmatizer

    logger.info("=== Тестирование Rubic2 (только лемматизация) ===")
    lemmatizer = Rubic2Lemmatizer()

    # Rubic2 требует слова + UPOS теги
    words = ["Мама", "мыла", "раму", "."]
    upos = ["NOUN", "VERB", "NOUN", "PUNCT"]

    lemmas = lemmatizer.get_lemmas(words, upos)

    result = [[{
        "id": i + 1,
        "form": w,
        "lemma": l,
        "upos": p,
        "head": 0,
        "deprel": "_"
    } for i, (w, l, p) in enumerate(zip(words, lemmas, upos))]]

    print("\n--- Rubic2 результат (эмулированный) ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def main():
    """Запускает все тесты последовательно."""

    models = {
        "pymorphy3": test_pymorphy3,
        "stanza": test_stanza,
        "cobald": test_cobald,
        "udpipe": test_udpipe,
        "deeppavlov": test_deeppavlov,
        "trankit": test_trankit,
        "mystem": test_mystem,
        "rubic2": test_rubic2,  # Только лемматизатор!
    }

    results = {}

    for model_name, test_func in models.items():
        try:
            logger.info(f"Запуск теста для {model_name}...")
            results[model_name] = test_func()
            logger.info(f"✓ {model_name} успешно протестирован")
        except Exception as e:
            logger.error(f"✗ {model_name} завершился с ошибкой: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None

    # Сохраняем результаты
    with open("model_formats_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("\n=== Итоги тестирования ===")
    for model_name, result in results.items():
        status = "✓ OK" if result is not None else "✗ FAIL"
        logger.info(f"{model_name}: {status}")

    logger.info(f"\nПолные результаты сохранены в model_formats_test_results.json")


if __name__ == "__main__":
    main()
