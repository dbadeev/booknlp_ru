import argparse
import logging
import csv
import difflib
import json
from pathlib import Path
from collections import Counter
from conllu import parse_incr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelAdapters:
    """Адаптеры для унификации вызова разных токенизаторов."""

    @staticmethod
    def load_razdel():
        try:
            from razdel import tokenize
            return lambda text: [t.text for t in tokenize(text)]
        except ImportError:
            logger.error("Razdel not installed.")
            return None

    @staticmethod
    def load_mystem():
        try:
            from pymystem3 import Mystem
            m = Mystem()
            # Фильтруем пробелы, которые возвращает Mystem
            return lambda text: [x['text'] for x in m.analyze(text) if x.get('text', '').strip()]
        except ImportError:
            logger.error("Pymystem3 not installed.")
            return None

    @staticmethod
    def load_slovnet():
        # Slovnet использует Razdel (navec). Для чистоты эксперимента используем razdel.
        return ModelAdapters.load_razdel()

    @staticmethod
    def load_spacy():
        try:
            import spacy
            # Отключаем лишнее для скорости
            if not spacy.util.is_package("ru_core_news_sm"):
                logger.warning("Spacy model 'ru_core_news_sm' not found.")
                return None
            nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner", "lemmatizer"])
            return lambda text: [t.text for t in nlp(text)]
        except ImportError:
            logger.error("Spacy not installed.")
            return None

    @staticmethod
    def load_stanza():
        try:
            import stanza
            # Stanza Tokenizer (MWT expand=True по умолчанию)
            nlp = stanza.Pipeline(lang='ru', processors='tokenize', verbose=False, use_gpu=True)
            return lambda text: [token.text for sentence in nlp(text).sentences for token in sentence.tokens]
        except Exception as e:
            logger.error(f"Stanza load error: {e}")
            return None

    @staticmethod
    def load_trankit():
        try:
            from trankit import Pipeline
            # Trankit требует кэш
            nlp = Pipeline(lang='russian', gpu=True, cache_dir='./cache/trankit')
            return lambda text: [token['text'] for token in nlp.tokenize(text, is_sent=True)['tokens']]
        except Exception as e:
            logger.error(f"Trankit load error: {e}")
            return None

    @staticmethod
    def load_deeppavlov():
        """
        Используем RuBERT Tokenizer (база для DeepPavlov NER/Syntax).
        Логика: WordPiece -> склейка ##subwords.
        """
        try:
            from transformers import AutoTokenizer
            # Стандартная модель для DeepPavlov
            tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

            def _tokenize(text):
                tokens = tokenizer.tokenize(text)
                words = []
                current_word = ""
                for t in tokens:
                    if t.startswith("##"):
                        current_word += t[2:]
                    else:
                        if current_word:
                            words.append(current_word)
                        current_word = t
                if current_word:
                    words.append(current_word)
                return words

            return _tokenize
        except Exception as e:
            logger.error(f"Transformers (DeepPavlov proxy) load error: {e}")
            return None

    @staticmethod
    def load_udpipe():
        try:
            from ufal.udpipe import Model, Pipeline, ProcessingError
            # Ищем модель в папке models/
            model_paths = list(Path("models").glob("russian-syntagrus-*.udpipe"))
            if not model_paths:
                logger.error("UDPipe model file not found in 'models/' directory.")
                return None
            model = Model.load(str(model_paths[0]))
            pipeline = Pipeline(model, "tokenize", Model.DEFAULT, Model.DEFAULT, "conllu")

            def _tokenize(text):
                processed = pipeline.process(text)
                from conllu import parse
                sentences = parse(processed)
                tokens = []
                for s in sentences:
                    tokens.extend([t['form'] for t in s])
                return tokens

            return _tokenize
        except ImportError:
            logger.error("ufal.udpipe library not installed.")
            return None


class TokenizationAuditor:
    def __init__(self, models_to_test):
        self.models = {}
        for name in models_to_test:
            logger.info(f"Loading {name}...")
            loader = getattr(ModelAdapters, f"load_{name}", None)
            if loader:
                func = loader()
                if func: self.models[name] = func
            else:
                logger.warning(f"No adapter for {name}")

    def _classify_diff(self, gold_seg, model_seg):
        if not model_seg: return "MISSING"
        if not gold_seg: return "HALLUCINATION"
        if len(model_seg) == 1 and len(gold_seg) > 1: return "MERGE_UNDER_TOK"
        if len(model_seg) > 1 and len(gold_seg) == 1: return "SPLIT_OVER_TOK"
        if len(model_seg) == len(gold_seg): return "TEXT_MISMATCH"
        return "COMPLEX_MISMATCH"

    def audit_file(self, input_path, output_dir, limit=None):
        input_path = Path(input_path)
        dataset_name = input_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Auditing dataset: {dataset_name}")

        with open(input_path, "r", encoding="utf-8") as f:
            sentences = list(parse_incr(f))
            if limit: sentences = sentences[:limit]

        # Для каждой модели создаем отдельный CSV
        for model_name, tokenizer in self.models.items():
            csv_path = output_dir / f"errors_{dataset_name}_{model_name}.csv"
            errors = []

            for sent in tqdm(sentences, desc=f"{model_name}"):
                text = sent.metadata.get("text")
                sent_id = sent.metadata.get("sent_id_new", sent.metadata.get("sent_id", "UNK"))

                if not text: continue

                gold_tokens = [t["form"] for t in sent if isinstance(t["id"], int)]

                try:
                    pred_tokens = tokenizer(text)
                except Exception as e:
                    logger.error(f"Error {model_name} on {sent_id}: {e}")
                    continue

                matcher = difflib.SequenceMatcher(None, gold_tokens, pred_tokens)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal': continue

                    gold_seg = gold_tokens[i1:i2]
                    pred_seg = pred_tokens[j1:j2]
                    error_type = self._classify_diff(gold_seg, pred_seg)

                    errors.append({
                        "dataset": dataset_name,
                        "sent_id": sent_id,
                        "model": model_name,
                        "error_type": error_type,
                        "gold": str(gold_seg),
                        "pred": str(pred_seg),
                        "context": text
                    })

            if errors:
                with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=errors[0].keys())
                    writer.writeheader()
                    writer.writerows(errors)
                logger.info(f"Saved {len(errors)} errors to {csv_path.name}")
            else:
                logger.info(f"No errors found for {model_name}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+",
                        default=["razdel", "mystem", "spacy", "slovnet", "deeppavlov", "stanza", "trankit", "udpipe"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    auditor = TokenizationAuditor(args.models)
    auditor.audit_file(args.input, args.output_dir, args.limit)