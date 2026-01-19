import logging
import modal
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StanzaParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-stanza", "StanzaService")()
            self.logger.info("Connected to Stanza via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str) -> List[List[Dict[str, Any]]]:
        """
        Метод принимает сырой текст (str), так как Stanza сама делает сегментацию.
        """
        try:
            return self.service.parse.remote(text)
        except Exception as e:
            self.logger.error(f"Error during Stanza parsing: {e}")
            raise e
        