#!/usr/bin/env python3
"""
Обёртка для Trankit (через Modal).
"""

import logging
import modal
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TrankitParser:
    """
    Клиент для Trankit, запущенного в Modal.
    Trankit выполняет полный морфо-синтаксический анализ с NER.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-trankit", "TrankitService")()
            self.logger.info("Connected to Trankit via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str, output_format: str = "simplified") -> List[List[Dict[str, Any]]]:
        """
        Парсит текст через удалённый Trankit.

        Аргументы:
            text (str): Входной текст для разбора.
            output_format (str): Формат выхода - "simplified" (текущий формат) или "native" (нативный формат Trankit).

        Возвращает: List[List[Dict]] - список предложений с токенами.

        При output_format="simplified":
            Каждый токен: {id, form, lemma, upos, xpos, feats, head, deprel, start_char, end_char}

        При output_format="native":
            Каждый токен: {id, text, lemma, upos, xpos, feats, head, deprel, span, dspan, ner, expanded}
            где:
            - span: локальное смещение в предложении (tuple)
            - dspan: глобальное смещение в документе (tuple)
            - ner: теги именованных сущностей (B-PER, I-PER, O и т.д.)
            - expanded: список словарей для multi-word tokens (MWT)
        """
        try:
            # ============================================================
            # БЛОК: Передача параметра output_format в удаленный сервис
            # ============================================================
            return self.service.parse.remote(text, output_format=output_format)

        except Exception as e:
            self.logger.error(f"Error during Trankit parsing: {e}")
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = TrankitParser()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)
    result = parser.parse_text(test_text, output_format="simplified")

    print("Trankit Test:")
    for sent in result:
        for tok in sent:
            print(f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t{tok.get('upos')}\t{tok.get('head')}\t{tok.get('deprel')}")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 60)
    result_native = parser.parse_text(test_text, output_format="native")

    print("Trankit Test (Native):")
    for sent in result_native:
        for tok in sent[:25]:  # Показываем первые 5 токенов
            print(f"\nText: {tok.get('text')}")
            print(f"  id: {tok.get('id')}")
            print(f"  lemma: {tok.get('lemma')}, upos: {tok.get('upos')}, xpos: {tok.get('xpos')}")
            print(f"  feats: {tok.get('feats')}")
            print(f"  head: {tok.get('head')}, deprel: {tok.get('deprel')}")  # ← ДОБАВЛЕНО!
            print(f"  span: {tok.get('span')}, dspan: {tok.get('dspan')}")
            print(f"  ner: {tok.get('ner')}")
            print(f"  lang: {tok.get('lang')}")
            if tok.get('expanded'):
                print(f"  expanded (MWT): {tok.get('expanded')}")
            else:
                print(f"  expanded: []")  # ← ПОКАЗЫВАЕМ ЯВНО

        # ============================================================
        # Дополнительная проверка: выведем все ключи первого токена
        # ============================================================
    print("\n" + "=" * 60)
    print("ВСЕ КЛЮЧИ ПЕРВОГО ТОКЕНА:")
    print("=" * 60)
    if result_native and result_native[0]:
        first_token = result_native[0][0]
        print(f"Ключи: {list(first_token.keys())}")
        print(f"Значения:")
        for key, value in first_token.items():
            print(f"  {key}: {value}")
