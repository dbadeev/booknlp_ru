import json
import logging
from pathlib import Path
from conllu import parse_incr
from tqdm import tqdm
from src.profiler import SentenceProfiler

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_profiling():
    # Конфигурация путей
    input_dir = Path("data/interim")
    output_file = Path("corpus_profile.json")

    # Файлы для анализа (берем результаты спринта 1.1)
    target_files = [
        "syntagrus_train_full.conllu",
        "taiga_train_full.conllu",
        # Можно добавить dev/test файлы, если нужно профилировать и их
        "syntagrus_test.conllu",
        "taiga_test.conllu"
    ]

    profiler = SentenceProfiler()
    full_profile = {}  # Dict: sent_id -> features

    logger.info("Starting Linguistic Profiling (Task 1.2)...")

    for filename in target_files:
        file_path = input_dir / filename
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Skipping.")
            continue

        logger.info(f"Profiling {filename}...")

        # Подсчет строк для tqdm (грубо)
        # Для точного прогресс-бара можно предварительно посчитать, но это IO

        with open(file_path, "r", encoding="utf-8") as f:
            # parse_incr эффективен по памяти
            for sentence in tqdm(parse_incr(f), desc=filename):
                try:
                    profile = profiler.profile_sentence(sentence)
                    sent_id = profile["id"]

                    # Добавляем метку источника для аналитики
                    profile["source_file"] = filename

                    full_profile[sent_id] = profile
                except Exception as e:
                    logger.error(f"Error profiling sentence in {filename}: {e}")

    # [cite_start]Сохранение результатов [cite: 226]
    logger.info(f"Saving profiles for {len(full_profile)} sentences to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_profile, f, ensure_ascii=False, indent=2)

    logger.info("Profiling completed successfully.")


if __name__ == "__main__":
    run_profiling()
    