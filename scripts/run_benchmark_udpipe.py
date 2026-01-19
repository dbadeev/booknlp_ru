import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

# 1. Настройка путей
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Импорты (оборачиваем в try-except для надежности)
try:
    from src.parsers.udpipe_wrapper import UDPipeParser
    from src.evaluation.alignment import FuzzyAligner
    from src.evaluation.metrics import MetricsCalculator
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BenchmarkUDPipe")


def run():
    # Путь к данным
    gold_path = PROJECT_ROOT / "data" / "processed" / "val_complex.conllu"

    if not gold_path.exists():
        logger.error(f"Файл с данными не найден: {gold_path}")
        return

    logger.info("Инициализация UDPipeParser...")
    try:
        parser = UDPipeParser()
    except Exception as e:
        logger.critical(f"Ошибка инициализации парсера: {e}")
        return

    aligner = FuzzyAligner()
    calculator = MetricsCalculator()

    # Загрузка данных
    logger.info(f"Загрузка данных из {gold_path}...")
    gold_data = []
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                text = sent.metadata.get("text", "")
                if not text: continue

                tokens = [{"id": t["id"], "form": t["form"], "head_id": t["head"], "deprel": t["deprel"]} for t in sent]
                tokens = aligner.reconstruct_gold_offsets(tokens, text)
                gold_data.append({
                    "text": text,
                    "tokens": tokens,
                    "id": sent.metadata.get("sent_id", "unknown")
                })
    except Exception as e:
        logger.error(f"Ошибка чтения файла данных: {e}")
        return

    logger.info(f"Загружено предложений: {len(gold_data)}")

    # Основной цикл
    total_uas = 0
    total_las = 0
    start_time = time.time()
    processed_count = 0

    for sent in tqdm(gold_data, desc="Бенчмарк UDPipe"):
        try:
            sys_output = parser.parse_text(sent['text'])

            if not sys_output:
                continue

            # Flatten sentences (UDPipe может разбить предложение на несколько)
            sys_tokens_flat = []
            for s in sys_output:
                for t in s:
                    sys_tokens_flat.append({
                        "id": t['id'],
                        "form": t['form'],
                        "head_id": t['head'],
                        "rel": t['deprel'],
                        "start_char": 0,  # Оффсеты восстановим aligner'ом
                        "end_char": 0
                    })

            if not sys_tokens_flat:
                continue

            # Восстановление оффсетов и выравнивание
            sys_tokens_enriched = aligner.reconstruct_gold_offsets(sys_tokens_flat, sent['text'])
            alignments = aligner.align(sys_tokens_enriched, sent['tokens'])
            metrics = calculator.calc_soft_metrics(alignments, sys_tokens_enriched, sent['tokens'])

            total_uas += metrics['soft_uas']
            total_las += metrics['soft_las']
            processed_count += 1

        except Exception as e:
            logger.error(f"Ошибка на предложении {sent['id']}: {e}")

    duration = time.time() - start_time

    # Вывод результатов
    print("\n" + "=" * 30)
    print("=== UDPipe Results ===")
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

        # Сохранение в CSV
        output_file = PROJECT_ROOT / "benchmark_udpipe.csv"
        pd.DataFrame([{
            "Model": "UDPipe 2.0",
            "UAS": avg_uas,
            "LAS": avg_las,
            "Speed": speed,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]).to_csv(output_file, index=False)

        logger.info(f"Результаты сохранены в {output_file}")
    else:
        logger.error("Не обработано ни одного предложения.")


if __name__ == "__main__":
    run()
