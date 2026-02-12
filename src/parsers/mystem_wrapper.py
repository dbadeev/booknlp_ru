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

    def parse_text(self, text: str) -> List[List[Dict[str, Any]]]:
        """
        Парсит текст через удалённый Mystem.

        Возвращает: List[List[Dict]] - список предложений с токенами.
        Каждый токен: {id, form, lemma, upos}
        """
        try:
            # parse_batch принимает список текстов, возвращает список документов
            # Каждый документ = список предложений
            results = self.service.parse_batch.remote([text])

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
    test_text = "Мама мыла раму."
    result = parser.parse_text(test_text)
    print("Mystem Test:")
    for sent in result:
        for tok in sent[:5]:
            print(f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t{tok.get('upos')}")
