import argparse
import json
import logging
import pandas as pd
from pathlib import Path
from conllu import parse_incr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_metrics(file_path):
    data = []
    file_path = Path(file_path)
    # Пытаемся определить имя датасета из имени файла (например, TAIGA из taiga_full...)
    filename_stem = file_path.stem.upper()
    if "TAIGA" in filename_stem:
        dataset_name = "TAIGA"
    elif "SYNTAGRUS" in filename_stem:
        dataset_name = "SYNTAGRUS"
    elif "COBALD" in filename_stem:
        dataset_name = "COBALD"
    else:
        dataset_name = filename_stem

    logger.info(f"Анализ файла: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for sent in tqdm(parse_incr(f), desc=f"Reading {dataset_name}"):
            meta = sent.metadata
            metrics_str = meta.get("metrics")

            if not metrics_str: continue

            try:
                metrics = json.loads(metrics_str)
            except:
                continue

            # Извлекаем жанр (если есть)
            genre = meta.get("genre") or meta.get("source") or "unknown"

            row = {
                "dataset": dataset_name,
                "sent_id": meta.get("sent_id_new"),
                "genre": genre,
                "length": metrics.get("length", 0),
                "is_dialogue": metrics.get("is_dialogue", False),
                "score": metrics.get("score", 0.0),
                "punct_clusters": metrics.get("punct_clusters", 0),
                "abbr_count": metrics.get("abbr_count", 0),
                "propn_ratio": metrics.get("propn_ratio", 0.0)
            }
            data.append(row)

    return data


def analyze_datasets(input_files, output_file):
    all_data = []
    for f in input_files:
        all_data.extend(parse_metrics(f))

    df = pd.DataFrame(all_data)

    # Настройки отображения Pandas для полного вывода в файл
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Подготовка записи в файл
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:

        # Хелпер для дублирования вывода (Tee-like behavior)
        def report(obj="", title=None):
            content = str(obj)
            if title:
                header = f"\n{'=' * 50}\n{title}\n{'=' * 50}\n"
                print(header, end="")
                f_out.write(header)

            print(content)
            f_out.write(content + "\n")

        if df.empty:
            report("Данные не найдены или файлы пусты.", title="ERROR")
            return

        report(f"Анализ файлов: {input_files}", title="METRICS EXPLORATION REPORT")

        # 1. Распределение по Жанрам
        report(
            df.groupby(['dataset', 'genre']).size().reset_index(name='count'),
            title="[1] Распределение по жанрам/доменам"
        )

        # 2. Процентили по Длине
        report(
            df.groupby('dataset')['length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
            title="[2] Статистика длины предложений (Percentiles)"
        )

        # 3. Диалоги
        dialogue_stats = df.groupby('dataset')['is_dialogue'].mean() * 100
        report(
            dialogue_stats.to_frame(name="% Dialogues"),
            title="[3] Доля диалогов (%)"
        )

        # 4. Сложность (Score) - 90-й перцентиль
        score_quant = df.groupby('dataset')['score'].quantile(0.9)
        report(
            score_quant.to_frame(name="Score (90th percentile)"),
            title="[4] Интегральная сложность (Score 90%)"
        )

        # 5. Рекомендации
        report("", title="[5] АВТОМАТИЧЕСКИЕ РЕКОМЕНДАЦИИ (Для corpus_sampler)")

        # Ищем Taiga Fiction/Prose
        taiga_lit = df[
            (df['dataset'] == 'TAIGA') &
            (df['genre'].astype(str).str.contains('fiction|lit|prose|social', case=False, regex=True))
            ]

        if not taiga_lit.empty:
            len_90 = int(taiga_lit['length'].quantile(0.90))
            score_90 = int(taiga_lit['score'].quantile(0.90))

            rec_msg = (
                f"На основе анализа Taiga (Fiction/Social):\n"
                f"Для создания 'Hard Literary' датасета рекомендуются пороги:\n"
                f"  --query \"genre == 'fiction' and length > {len_90} and score > {score_90} and is_dialogue == True\"\n\n"
                f"Для 'Medium Literary':\n"
                f"  --query \"genre == 'fiction' and 10 < length < {len_90}\""
            )
            report(rec_msg)
        else:
            report("[WARNING] Не удалось автоматически выделить Fiction в Taiga. Проверьте таблицу жанров выше.")

    logger.info(f"\n✅ Отчет сохранен в: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis of interim .conllu files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Список .conllu файлов")
    parser.add_argument("--output", required=True, help="Путь к файлу отчета (напр. reports/metrics_stats.txt)")
    args = parser.parse_args()

    analyze_datasets(args.inputs, args.output)
