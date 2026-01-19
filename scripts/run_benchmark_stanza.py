import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

# 1. Настройка путей (как в Trankit скрипте)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Импорты из локальных модулей
try:
    from src.parsers.stanza_wrapper import StanzaParser
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
logger = logging.getLogger("BenchmarkStanza")


def run():
    # Путь к данным
    gold_path = PROJECT_ROOT / "data" / "processed" / "val_complex.conllu"

    if not gold_path.exists():
        logger.error(f"Файл с данными не найден: {gold_path}")
        return

    logger.info("Инициализация StanzaParser...")
    try:
        parser = StanzaParser()
    except Exception as e:
        logger.critical(f"Не удалось инициализировать Stanza: {e}")
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

                # Восстанавливаем оффсеты
                tokens = aligner.reconstruct_gold_offsets(tokens, text)
                gold_data.append({"text": text, "tokens": tokens, "id": sent.metadata.get("sent_id", "unknown")})
    except Exception as e:
        logger.error(f"Ошибка чтения CoNLLU: {e}")
        return

    logger.info(f"Загружено предложений: {len(gold_data)}")

    # 2. Основной цикл
    total_uas = 0
    total_las = 0
    processed_count = 0
    start_time = time.time()

    for sent in tqdm(gold_data, desc="Бенчмарк Stanza"):
        try:
            # Stanza выполняет токенизацию внутри
            sys_output = parser.parse_text(sent['text'])

            if not sys_output:
                continue

            # "Выпрямляем" список (Flatten), если Stanza разбила на несколько предложений
            sys_tokens_flat = []
            for s in sys_output:
                for t in s:
                    sys_tokens_flat.append({
                        "id": t['id'],
                        "form": t['form'],
                        "head_id": t['head'],
                        "rel": t['deprel'],
                        "start_char": t.get('start_char', 0),
                        "end_char": t.get('end_char', 0)
                    })

            if not sys_tokens_flat:
                continue

            # Если Stanza не вернула оффсеты (зависит от версии), восстанавливаем их
            if sys_tokens_flat[0]['start_char'] == 0 and sys_tokens_flat[0]['end_char'] == 0:
                sys_tokens_flat = aligner.reconstruct_gold_offsets(sys_tokens_flat, sent['text'])

            # Выравнивание и метрики
            alignments = aligner.align(sys_tokens_flat, sent['tokens'])
            metrics = calculator.calc_soft_metrics(alignments, sys_tokens_flat, sent['tokens'])

            total_uas += metrics['soft_uas']
            total_las += metrics['soft_las']
            processed_count += 1

        except Exception as e:
            logger.error(f"Ошибка на {sent['id']}: {e}")

    duration = time.time() - start_time

    # 3. Вывод и сохранение результатов
    print("\n" + "=" * 30)
    print("=== Stanza Results ===")
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
        output_file = PROJECT_ROOT / "benchmark_stanza.csv"
        pd.DataFrame([{
            "Model": "Stanza",
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