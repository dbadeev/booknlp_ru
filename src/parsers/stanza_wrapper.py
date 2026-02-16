#!/usr/bin/env python3
"""
Обёртка для Stanza (через Modal) с поддержкой нативного формата.
Включает корректную обработку полей misc (SpaceAfter) и ner.
"""
import logging
import modal
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class StanzaParser:
    """
    Клиент для Stanza, запущенного в Modal.
    Stanza выполняет полный морфо-синтаксический анализ + NER для русского языка.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-stanza", "StanzaService")()
            self.logger.info("Connected to Stanza via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str, native_format: bool = False) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Метод принимает сырой текст (str), так как Stanza сама делает сегментацию.

        Параметры:
        ----------
        text : str
            Входной текст для парсинга
        native_format : bool, optional (default=False)
            Если False - возвращает текущий упрощенный формат
            Если True - возвращает нативный формат Stanza со всеми полями

        Возвращает:
        -----------
        Если native_format=False: List[List[Dict[str, Any]]]
            Текущий формат - список предложений, каждое - список токенов

        Если native_format=True: List[Dict[str, Any]]
            Нативный формат с полями:
            - "words": токены с {id, form, lemma, upos, xpos, feats, head, deprel,
                               start_char, end_char, misc, ner}
            - misc: SpaceAfter=No (когда токен не имеет пробела после себя)
            - ner: теги именованных сущностей (B-PER, I-LOC, O и т.д.)

        ВАЖНО:
        ------
        Поле misc заполняется корректно только в нативном формате.
        Информация берется из token.spaces_after (Stanza v1.4+).
        """
        try:
            return self.service.parse.remote(text, native_format=native_format)
        except Exception as e:
            self.logger.error(f"Error during Stanza parsing: {e}")
            raise e



# ============================================================
# БЛОК: Тестовые примеры использования wrapper
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = StanzaParser()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    # test_text = 'Коля сказал:"Привет!"И ушёл.'

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)

    result = parser.parse_text(test_text, native_format=False)

    print("Stanza Test:")
    for sent in result:
        for tok in sent:
            print(f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t"
                  f"{tok.get('upos')}\t{tok.get('head')}\t{tok.get('deprel')}")

    # ============================================================
    # Демонстрация работы в нативном формате с NER
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native) с NER:")
    print("=" * 60)

    result_native = parser.parse_text(test_text, native_format=True)

    print("Stanza Test (Native):")
    for sent_data in result_native:
        words = sent_data.get("words", [])

        # Показываем метаданные предложения
        print(f"\nПредложение содержит {len(words)} токенов")

        # ========== ПОЯСНЕНИЕ ПРО SENTIMENT И CONSTITUENCY ==========
        # Для русского языка эти процессоры не доступны
        if "sentiment" in sent_data:
            print(f"Sentiment: {sent_data['sentiment']}")
        else:
            print(f"Sentiment: не доступен для русского языка (только en/zh/de)")

        if "constituency" in sent_data:
            print(f"Constituency: {sent_data['constituency'][:50]}...")
        else:
            print(f"Constituency: не доступен для русского языка")
        # ============================================================

        # Показываем первые 5 токенов с NER
        print("\nПервые 5 токенов:")
        for tok in words:
        # for tok in words[:5]:
            print(f"\nText: {tok.get('form')}")
            print(f"  id: {tok.get('id')}")
            print(f"  lemma: {tok.get('lemma')}, upos: {tok.get('upos')}, xpos: {tok.get('xpos')}")
            print(f"  feats: {tok.get('feats')}")
            print(f"  head: {tok.get('head')}, deprel: {tok.get('deprel')}")
            print(f"  start_char: {tok.get('start_char')}, end_char: {tok.get('end_char')}")

            # Дополнительные нативные поля
            if 'misc' in tok:
                print(f"  misc: {tok.get('misc')}")
            else:
                print(f"  misc: None")

            # ========== NER ТЕГА ==========
            # Если NER процессор был включен, здесь будут теги
            # B-PER, I-PER (персона), B-LOC, I-LOC (локация), B-ORG, I-ORG (организация), O (вне сущности)
            if 'ner' in tok:
                print(f"  ner: {tok.get('ner')}")
            else:
                print(f"  ner: O (вне именованной сущности)")
            # ==============================

    # ============================================================
    # Дополнительная проверка: подсчет именованных сущностей
    # ============================================================
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ИМЕНОВАННЫХ СУЩНОСТЕЙ:")
    print("=" * 60)

    if result_native and result_native[0] and result_native[0].get("words"):
        ner_tags = [tok.get('ner', 'O') for tok in result_native[0]["words"]]
        person_count = sum(1 for tag in ner_tags if tag and tag.endswith('PER'))
        location_count = sum(1 for tag in ner_tags if tag and tag.endswith('LOC'))
        org_count = sum(1 for tag in ner_tags if tag and tag.endswith('ORG'))

        print(f"Всего токенов: {len(ner_tags)}")
        print(f"Персоны (PER): {person_count}")
        print(f"Локации (LOC): {location_count}")
        print(f"Организации (ORG): {org_count}")

        print("\nТокены с NER тегами (кроме O):")
        for tok in result_native[0]["words"]:
            ner = tok.get('ner', 'O')
            if ner != 'O':
                print(f"  {tok.get('form')}: {ner}")

    # ============================================================
    # Все ключи первого токена и предложения
    # ============================================================
    print("\n" + "=" * 60)
    print("ВСЕ КЛЮЧИ ПЕРВОГО ТОКЕНА:")
    print("=" * 60)

    if result_native and result_native[0] and result_native[0].get("words"):
        first_token = result_native[0]["words"][0]
        print(f"Ключи: {list(first_token.keys())}")
        print(f"Значения:")
        for key, value in first_token.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("ВСЕ КЛЮЧИ ПРЕДЛОЖЕНИЯ:")
    print("=" * 60)

    if result_native and result_native[0]:
        first_sentence = result_native[0]
        print(f"Ключи предложения: {list(first_sentence.keys())}")
