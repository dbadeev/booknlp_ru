import yaml
import logging
from src.ingress import IngressManager
from src.normalizer import Normalizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path='config/sources.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    logger.info("Starting Sprint 1: Card 1.1 Ingress Pipeline")
    config = load_config()

    # --- Step 1: Ingress (Download) ---
    logger.info(">>> Phase 1: Ingress (Downloading Raw Data)")
    ingress = IngressManager(config)
    # Загружаем файлы
    raw_files = ingress.run(config['sources'])

    # --- Step 2: Normalization ---
    logger.info(">>> Phase 2: Normalization (ETL)")
    norm = Normalizer(config)

    # 2.1 Processing SynTagRus (Split)
    sr_train_files = [
        raw_files['syntagrus']['train_a'],
        raw_files['syntagrus']['train_b'],
        raw_files['syntagrus']['train_c']
    ]
    norm.normalize_split_corpus(sr_train_files, "syntagrus_train_full.conllu", "SR")

    norm.normalize_standard([raw_files['syntagrus']['dev']], "syntagrus_dev.conllu", "SR")
    norm.normalize_standard([raw_files['syntagrus']['test']], "syntagrus_test.conllu", "SR")

    # 2.2 Processing Taiga (Теперь тоже Split!)
    tg_train_files = [
        raw_files['taiga']['train_a'],
        raw_files['taiga']['train_b'],
        raw_files['taiga']['train_c'],
        raw_files['taiga']['train_d'],
        raw_files['taiga']['train_e']
    ]
    norm.normalize_split_corpus(tg_train_files, "taiga_train_full.conllu", "TG")

    # Taiga Test обычно одним файлом
    norm.normalize_standard([raw_files['taiga']['test']], "taiga_test.conllu", "TG")

    # 2.3 Processing CoBaLD
    norm.normalize_cobald(raw_files['cobald']['train'], "cobald_train_adapted.conllu")
    norm.normalize_cobald(raw_files['cobald']['dev'], "cobald_dev_adapted.conllu")

    logger.info("Sprint 1 Task 1.1 Completed Successfully. Data ready in data/interim/")


if __name__ == "__main__":
    main()
