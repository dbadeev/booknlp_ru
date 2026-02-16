#!/usr/bin/env python3
"""
Обёртка для Mystem (через Modal).
"""

import logging
import modal
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MystemParser:
    """
    Клиент для Mystem, запущенного в Modal.
    Mystem возвращает только морфологию (id, form, lemma, upos).
    НЕТ синтаксического разбора (head/deprel).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-mystem", "MystemService")()
            self.logger.info("Connected to Mystem via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Mystem Modal app: {e}")
            raise e

    def parse_text(self, text: str, output_format: str = "simplified") -> List[List[Dict[str, Any]]]:
        """
        Парсит текст через удалённый Mystem.

        Аргументы:
            text (str): Входной текст для разбора.
            output_format (str): Формат выхода - "simplified" (текущий формат) или "native" (нативный формат Mystem).

        Возвращает: List[List[Dict]] - список предложений с токенами.

        При output_format="simplified":
            Каждый токен: {id, form, lemma, upos}

        При output_format="native":
            Каждый токен: {id, text, analysis}
            где analysis - список всех гипотез разбора (омонимов) с полями:
            - lex: лемма
            - gr: грамматическая строка
            - wt: вес (вероятность) гипотезы
            - qual: маркер качества
        """
        try:
            # ============================================================
            # БЛОК: Передача параметра output_format в удаленный сервис
            # ============================================================
            # parse_batch принимает список текстов, возвращает список документов
            # Каждый документ = список предложений
            results = self.service.parse_batch.remote([text], output_format=output_format)

            if not results or not results[0]:
                return []

            # results[0] = первый документ (наш единственный текст)
            # Это уже List[List[Dict]]
            return results[0]

        except Exception as e:
            self.logger.error(f"Error during Mystem parsing: {e}")
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = MystemParser()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)
    result = parser.parse_text(test_text, output_format="simplified")

    print("Mystem Test:")
    for sent in result:
        for tok in sent:
            print(f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t{tok.get('upos')}")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 60)
    result_native = parser.parse_text(test_text, output_format="native")

    print("Mystem Test (Native):")
    for sent in result_native:
        for tok in sent:
        # for tok in sent[:3]: # Показываем первые три токена
            print(f"  Text: {tok['text']}")
            print(f"    Analysis variants: {len(tok['analysis'])}")
            for j, variant in enumerate(tok['analysis'][:2]):
                # Безопасное обращение к опциональному полю qual
                qual_marker = variant.get('qual', 'dictionary')  # <-- ДОБАВИТЬ
                print(
                    f"      [{j + 1}] lex={variant.get('lex')}, gr={variant.get('gr')}, wt={variant.get('wt')}, qual={qual_marker}")