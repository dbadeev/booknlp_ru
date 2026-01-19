import logging
import argparse
from pathlib import Path
from src.sampler import StratifiedSampler
from src.eda import DatasetAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Настройки
    data_dir = "data/interim"
    output_dir = "data/processed"  # Финальные датасеты кладем сюда или в корень
    profile_path = "corpus_profile.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Сэмплирование
    sampler = StratifiedSampler(profile_path, data_dir)

    # 1.1 Fiction_Dialogue (из SynTagRus)
    dialogues = sampler.sample_dialogues("syntagrus_train_full.conllu", limit=1000)
    path_dialogue = f"{output_dir}/val_dialogue.conllu"
    sampler.save_dataset(dialogues, path_dialogue)

    # 1.2 Fiction_Complex (из Taiga - используем полный train)
    # Taiga содержит поэзию и соцсети, ищем там сложные конструкции
    complex_sents = sampler.sample_complex("taiga_train_full.conllu", limit=500)
    path_complex = f"{output_dir}/val_complex.conllu"
    sampler.save_dataset(complex_sents, path_complex)

    # 1.3 Baseline (из SynTagRus для чистоты эксперимента сравнения с новостями)
    baseline = sampler.sample_baseline("syntagrus_train_full.conllu", limit=1000)
    path_baseline = f"{output_dir}/val_baseline.conllu"
    sampler.save_dataset(baseline, path_baseline)

    # 2. Анализ и отчетность (EDA)
    logger.info("Generating EDA report...")
    analyzer = DatasetAnalyzer(output_dir)

    analyzer.analyze_file("val_dialogue", path_dialogue)
    analyzer.analyze_file("val_complex", path_complex)
    analyzer.analyze_file("val_baseline", path_baseline)

    analyzer.generate_report("dataset_stats.md")
    logger.info("Sprint 1 Card 1.3 Completed. Check data/processed/ for results.")


if __name__ == "__main__":
    main()
