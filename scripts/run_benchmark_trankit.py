import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

# Добавляем корневую директорию проекта в sys.path
# Это позволяет импортировать модули из src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Импорты из локальных модулей
# Убедитесь, что файл src/parsers/trankit_wrapper.py существует (см. ниже, если нет)
try:
    from src.parsers.trankit_wrapper import TrankitParser
    from src.evaluation.alignment import FuzzyAligner
    from src.evaluation.metrics import MetricsCalculator
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете скрипт из корня проекта или папки scripts/.")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BenchmarkTrankit")


def run():
    # Путь к валидационному датасету (SynTagRus)
    gold_path = PROJECT_ROOT / "data" / "processed" / "val_complex.conllu"

    if not gold_path.exists():
        logger.error(f"Файл с данными не найден: {gold_path}")
        logger.info("Пожалуйста, убедитесь, что данные распакованы в data/processed/")
        return

    logger.info("Инициализация TrankitParser (через Modal)...")
    try:
        parser = TrankitParser()
        # Быстрый тест соединения
        test_res = parser.parse_text("Тест.")
        if not test_res:
            logger.warning("Trankit вернул пустой результат на тестовом запросе.")
    except Exception as e:
        logger.critical(f"Не удалось подключиться к Modal сервису: {e}")
        return

    logger.info("Инициализация инструментов оценки...")
    aligner = FuzzyAligner()
    calculator = MetricsCalculator()

    # 1. Загрузка золотого стандарта
    logger.info(f"Загрузка данных из {gold_path}...")
    gold_data = []
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                text = sent.metadata.get("text", "")
                if not text:
                    continue

                # Извлекаем золотые токены
                tokens = [{
                    "id": t["id"],
                    "form": t["form"],
                    "head_id": t["head"],
                    "deprel": t["deprel"]
                } for t in sent]

                # Восстанавливаем оффсеты для золотого стандарта (для корректного align)
                tokens = aligner.reconstruct_gold_offsets(tokens, text)

                gold_data.append({
                    "text": text,
                    "tokens": tokens,
                    "id": sent.metadata.get("sent_id", "unknown")
                })
    except Exception as e:
        logger.error(f"Ошибка при чтении CoNLLU файла: {e}")
        return

    logger.info(f"Загружено предложений: {len(gold_data)}")

    # 2. Основной цикл бенчмарка
    total_uas = 0
    total_las = 0
    processed_count = 0

    start_time = time.time()

    # tqdm для прогресс-бара
    for sent in tqdm(gold_data, desc="Бенчмарк Trankit"):
        try:
            # Отправка текста в Modal
            # Trankit может вернуть список предложений (list of lists)
            sys_output_batched = parser.parse_text(sent['text'])

            if not sys_output_batched:
                # Если парсер вернул пустоту (например, ошибка внутри контейнера)
                continue

            # "Выпрямляем" список (Flatten), так как Trankit делает сплиттинг
            sys_tokens_flat = []
            for s in sys_output_batched:
                for t in s:
                    sys_tokens_flat.append({
                        "id": t['id'],
                        "form": t['form'],
                        "head_id": t['head'],
                        "rel": t['deprel'],
                        # Trankit может возвращать 'dspan', но мы пересчитаем aligner'ом для надежности
                        "start_char": 0,
                        "end_char": 0
                    })

            if not sys_tokens_flat:
                continue

            # Восстанавливаем оффсеты для предсказания системы
            sys_tokens_enriched = aligner.reconstruct_gold_offsets(sys_tokens_flat, sent['text'])

            # Выравниваем токены (System vs Gold)
            alignments = aligner.align(sys_tokens_enriched, sent['tokens'])

            # Считаем метрики
            metrics = calculator.calc_soft_metrics(alignments, sys_tokens_enriched, sent['tokens'])

            total_uas += metrics['soft_uas']
            total_las += metrics['soft_las']
            processed_count += 1

        except Exception as e:
            # Логируем ошибку, но не прерываем весь бенчмарк
            logger.error(f"Ошибка на предложении {sent['id']}: {e}")
            # Можно добавить небольшую паузу, если ошибка сетевая
            # time.sleep(1)

    duration = time.time() - start_time

    # 3. Вывод результатов
    print("\n" + "=" * 30)
    print("=== Trankit (Modal) Results ===")
    print("=" * 30)

    if processed_count > 0:
        speed = processed_count / duration
        avg_uas = total_uas / processed_count
        avg_las = total_las / processed_count

        print(f"Processed: {processed_count}/{len(gold_data)}")
        print(f"Time:      {duration:.2f} sec")
        print(f"Speed:     {speed:.2f} sent/sec")
        print(f"Soft UAS:  {avg_uas:.4f}")
        print(f"Soft LAS:  {avg_las:.4f}")

        # Сохранение результатов
        output_file = PROJECT_ROOT / "benchmark_trankit.csv"
        pd.DataFrame([{
            "Model": "Trankit (XLM-R Large)",
            "UAS": avg_uas,
            "LAS": avg_las,
            "Speed": speed,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]).to_csv(output_file, index=False)
        logger.info(f"Результаты сохранены в {output_file}")
    else:
        logger.error("Не обработано ни одного предложения! Проверьте логи Modal (modal logs -f booknlp-ru-trankit).")


if __name__ == "__main__":
    run()