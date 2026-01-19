import logging
from typing import List, Dict, Any, Literal
from dataclasses import asdict

# Импорты наших модулей из предыдущих задач
from src.segmentation import RazdelSegmenter, Sentence
from src.parsers.slovnet_parser import SlovnetParser

# Опциональный импорт DeepPavlov (чтобы не падать, если Modal не настроен)
try:
    from src.parsers.deeppavlov_wrapper import DeepPavlovParser
except ImportError:
    DeepPavlovParser = None

logger = logging.getLogger(__name__)


class BookNLP:
    """
    Главный класс-оркестратор.
    Объединяет сегментацию (Razdel) и парсинг (Slovnet/DeepPavlov).
    """

    def __init__(self, model_type: Literal["fast", "accurate"] = "fast"):
        self.model_type = model_type
        self.segmenter = RazdelSegmenter()

        logger.info(f"Initializing BookNLP pipeline with mode='{model_type}'...")

        if model_type == "fast":
            self.parser = SlovnetParser()
        elif model_type == "accurate":
            if DeepPavlovParser is None:
                raise ImportError("DeepPavlov wrapper not available. Check dependencies.")
            self.parser = DeepPavlovParser()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def process(self, text: str) -> List[Dict[str, Any]]:
        """
        Полный цикл обработки текста.
        Возвращает список предложений с токенами, морфологией, синтаксисом и оффсетами.
        """
        # 1. Сегментация (получаем предложения с глобальными оффсетами)
        sentences: List[Sentence] = self.segmenter.split_sentences(text)

        if not sentences:
            return []

        # 2. Подготовка батча для парсера (список списков токенов-строк)
        batch_tokens_str = []
        for sent in sentences:
            # Извлекаем текст каждого токена
            batch_tokens_str.append([t.text for t in sent.tokens])

        # 3. Парсинг (Slovnet или DeepPavlov)
        # Возвращает список списков словарей (conll-like)
        logger.info(f"Parsing {len(sentences)} sentences...")
        parsed_batch = self.parser.parse_batch(batch_tokens_str)

        # 4. Слияние (Merge) результатов парсинга с оффсетами сегментатора
        final_output = []

        for i, sentence_obj in enumerate(sentences):
            parsed_info = parsed_batch[i]  # Результат парсера для этого предложения
            original_tokens = sentence_obj.tokens  # Токены с оффсетами

            # Проверка на рассинхрон (санити-чек)
            if len(parsed_info) != len(original_tokens):
                logger.warning(
                    f"Token mismatch in sent {i}: "
                    f"Segmenter={len(original_tokens)}, Parser={len(parsed_info)}. "
                    "Aligning by min length."
                )

            processed_tokens = []
            for j, orig_token in enumerate(original_tokens):
                if j >= len(parsed_info):
                    break

                p_token = parsed_info[j]

                # Собираем богатый объект токена
                rich_token = {
                    "id": p_token["id"],
                    "text": orig_token.text,
                    "lemma": p_token.get("lemma", "_"),
                    "pos": p_token.get("upos", "_"),
                    "feats": p_token.get("feats", "_"),
                    "head_id": p_token["head"],
                    "rel": p_token.get("deprel", "_"),
                    # Самое важное для задачи 2.1 - Оффсеты
                    "start_char": orig_token.start_char,
                    "end_char": orig_token.end_char
                }
                processed_tokens.append(rich_token)

            final_output.append({
                "sent_id": i + 1,
                "text": sentence_obj.text,
                "start_char": sentence_obj.start_char,
                "end_char": sentence_obj.end_char,
                "tokens": processed_tokens
            })

        return final_output
    