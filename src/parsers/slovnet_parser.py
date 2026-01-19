import logging
from pathlib import Path
from navec import Navec
from slovnet import Syntax
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SlovnetParser:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.navec_path = self.models_dir / "navec_news_v1_1B_250K_300d_100q.tar"
        self.slovnet_path = self.models_dir / "slovnet_syntax_news_v1.tar"

        self._load_models()

    def _load_models(self):
        if not self.navec_path.exists() or not self.slovnet_path.exists():
            raise FileNotFoundError(
                f"Models not found in {self.models_dir}. "
                "Run 'python scripts/download_models.py' first."
            )

        logger.info("Loading Navec embeddings...")
        self.navec = Navec.load(self.navec_path)

        logger.info("Loading Slovnet syntax model...")
        self.syntax = Syntax.load(self.slovnet_path)
        self.syntax.navec(self.navec)
        logger.info("Slovnet loaded successfully.")

    def parse(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Парсинг одного предложения.
        """
        if not tokens:
            return []

        # Инференс: передаем токены напрямую (библиотека сама обработает список строк)
        markup = self.syntax(tokens)

        parsed_tokens = []

        for token in markup.tokens:
            # Безопасное преобразование head_id
            head_id = int(token.head_id) if token.head_id and token.head_id.isdigit() else 0

            # Безопасное извлечение атрибутов (pos и feats отсутствуют в синтаксической модели)
            # Используем getattr с дефолтным значением "_"
            pos = getattr(token, "pos", "_")
            feats = getattr(token, "feats", "_")
            rel = getattr(token, "rel", "_")

            token_data = {
                "id": int(token.id),
                "form": token.text,
                "lemma": "_",
                "upos": pos if pos else "_",
                "xpos": "_",
                "feats": feats if feats else "_",
                "head": head_id,
                "deprel": rel if rel else "_",
                "deps": "_",
                "misc": "_"
            }
            parsed_tokens.append(token_data)

        return parsed_tokens

    def parse_batch(self, batch_tokens: List[List[str]]) -> List[List[Dict[str, Any]]]:
        """
        Парсинг батча предложений.
        """
        if not batch_tokens:
            return []

        batch_results = []

        # Используем .map() для обработки списка списков (батча)
        for sent_markup in self.syntax.map(batch_tokens):
            sent_parsed = []
            for token in sent_markup.tokens:
                head_id = int(token.head_id) if token.head_id and token.head_id.isdigit() else 0

                pos = getattr(token, "pos", "_")
                feats = getattr(token, "feats", "_")
                rel = getattr(token, "rel", "_")

                token_data = {
                    "id": int(token.id),
                    "form": token.text,
                    "lemma": "_",
                    "upos": pos if pos else "_",
                    "xpos": "_",
                    "feats": feats if feats else "_",
                    "head": head_id,
                    "deprel": rel if rel else "_",
                    "deps": "_",
                    "misc": "_"
                }
                sent_parsed.append(token_data)
            batch_results.append(sent_parsed)

        return batch_results
