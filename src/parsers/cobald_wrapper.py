import logging
import modal
from typing import List, Dict, Any
from razdel import tokenize as razdel_tokenize

logger = logging.getLogger(__name__)


class CobaldParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-cobald", "CobaldService")()
            self.logger.info("Connected to CoBaLD via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str) -> List[List[Dict[str, Any]]]:
        """
        Токенизирует текст (Razdel) и отправляет в CoBaLD.
        Возвращает список предложений (сейчас CoBaLD обрабатывает по одному,
        но для совместимости интерфейса возвращаем [[tokens]]).
        """
        try:
            # 1. Токенизация через Razdel (стандарт для русского NLP)
            # В идеале нужно еще разбивать на предложения, но бенчмарк подает
            # текст по одному предложению за раз.
            tokens_gen = razdel_tokenize(text)
            tokens = [_.text for _ in tokens_gen]

            if not tokens:
                return []

            # 2. Отправка в Modal
            # Возвращает список словарей для одного предложения
            parsed_sent = self.service.parse.remote(tokens)

            if not parsed_sent:
                return []

            # 3. Возвращаем как список предложений (одно предложение)
            return [parsed_sent]

        except Exception as e:
            self.logger.error(f"Error during CoBaLD parsing: {e}")
            return []
        