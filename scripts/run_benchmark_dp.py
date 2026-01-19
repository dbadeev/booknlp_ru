import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

# Добавляем корневую директорию в путь, чтобы видеть модули src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Импорты
try:
    from src.parsers.deeppavlov_wrapper import DeepPavlovParser
    from src.evaluation.alignment import FuzzyAligner
    from src.evaluation.metrics import MetricsCalculator
except ImportError as e:
    print(f"Ошибка импорта: {e}. Убедитесь, что вы находитесь в корне проекта или пути настроены верно.")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkDP")


class BenchmarkRunnerDP:
    def __init__(self, data_path: str, output_file: str = "benchmark_dp_results.csv"):
        self.data_path = Path(data_path)
        self.output_file = output_file
        self.aligner = FuzzyAligner()
        self.calculator = MetricsCalculator()

        # Инициализация вашего враппера
        logger.info("Инициализация DeepPavlovParser...")
        self.parser = DeepPavlovParser()

    def load_gold_data(self) -> list:
        """Загрузка валидационного датасета."""
        sentences = []
        if not self.data_path.exists():
            logger.error(f"Файл данных не найден: {self.data_path}")
            # Для теста вернем фиктивные данные, если файла нет
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                text = sent.metadata.get("text", "")
                if not text: continue

                tokens = []
                for t in sent:
                    tokens.append({
                        "id": t["id"],
                        "form": t["form"],
                        "head_id": t["head"],
                        "deprel": t["deprel"],
                        # Если в conllu нет оффсетов, их нужно восстановить,
                        # но FuzzyAligner.reconstruct_gold_offsets делает это сам
                    })

                # Восстанавливаем оффсеты для Gold, если их нет
                rich_tokens = self.aligner.reconstruct_gold_offsets(tokens, text)

                sentences.append({
                    "id": sent.metadata.get("sent_id"),
                    "text": text,
                    "tokens": rich_tokens
                })
        return sentences

    def run(self):
        gold_sentences = self.load_gold_data()
        if not gold_sentences:
            logger.warning("Нет данных для тестирования.")
            return

        logger.info(f"Начинаем бенчмарк на {len(gold_sentences)} предложениях.")

        results = []
        total_uas = 0.0
        total_las = 0.0
        processed_count = 0
        start_time = time.time()

        for gold_sent in tqdm(gold_sentences, desc="DeepPavlov Parsing"):
            text = gold_sent["text"]

            try:
                # 1. Токенизация (простая эвристика для бенчмарка, если нет Razdel в DP wrapper)
                # В идеале DeepPavlovParser.parse должен принимать список токенов.
                # Сделаем простую токенизацию по пробелам или используем Razdel, если есть.
                # Но ваш wrapper принимает list[str].
                from razdel import tokenize
                tokens_str = [t.text for t in tokenize(text)]

                # 2. Инференс (вызов вашего враппера)
                # Важно: ваш wrapper возвращает список словарей {'id', 'form', 'head', 'deprel', ...}
                sys_output = self.parser.parse(tokens_str)

                if not sys_output:
                    continue

                # Приводим формат к ожидаемому метриками
                sys_tokens_clean = []
                for t in sys_output:
                    sys_tokens_clean.append({
                        "id": t['id'],
                        "form": t['form'],
                        "head_id": t['head'],
                        "rel": t['deprel'],  # MetricsCalculator ожидает 'rel' или 'deprel'
                        "start_char": 0,  # Оффсеты восстановим ниже
                        "end_char": 0
                    })

                # Восстанавливаем оффсеты для System вывода (для Fuzzy Alignment)
                sys_tokens_enriched = self.aligner.reconstruct_gold_offsets(sys_tokens_clean, text)

                # 3. Выравнивание и Метрики
                alignments = self.aligner.align(sys_tokens_enriched, gold_sent["tokens"])
                metrics = self.calculator.calc_soft_metrics(alignments, sys_tokens_enriched, gold_sent["tokens"])

                total_uas += metrics['soft_uas']
                total_las += metrics['soft_las']
                processed_count += 1

            except Exception as e:
                logger.error(f"Ошибка на предложении {gold_sent.get('id')}: {e}")

        total_duration = time.time() - start_time

        if processed_count > 0:
            avg_uas = total_uas / processed_count
            avg_las = total_las / processed_count
            speed = processed_count / total_duration

            print("\n=== Результаты DeepPavlov ===")
            print(f"Обработано предложений: {processed_count}")
            print(f"Скорость: {speed:.2f} предл./сек")
            print(f"Soft UAS: {avg_uas:.4f}")
            print(f"Soft LAS: {avg_las:.4f}")

            # Сохранение
            pd.DataFrame([{
                "Model": "DeepPavlov",
                "UAS": avg_uas,
                "LAS": avg_las,
                "Speed": speed
            }]).to_csv(self.output_file, index=False)
            logger.info(f"Результаты сохранены в {self.output_file}")
        else:
            logger.error("Бенчмарк не завершен.")


if __name__ == "__main__":
    # Укажите путь к вашему файлу данных
    gold_path = "data/processed/val_complex.conllu"

    # Если файла нет, создадим заглушку для проверки работоспособности
    if not Path(gold_path).exists():
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        # TODO: Заменить на реальную загрузку или скачивание
        logger.warning(f"Файл {gold_path} не найден. Убедитесь, что данные сгенерированы.")

    runner = BenchmarkRunnerDP(gold_path)
    runner.run()
