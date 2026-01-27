import argparse
import logging
import json
import pandas as pd
from pathlib import Path
from conllu import parse_incr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metadata_df(input_path):
    """
    Проходит по файлу и собирает метаданные и метрики в DataFrame.
    Не загружает токены в память, чтобы экономить ресурсы.
    """
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, sent in enumerate(tqdm(parse_incr(f), desc="Indexing corpus")):
            meta = sent.metadata

            # Базовые поля
            row = {
                "idx": i,  # Сохраняем индекс для быстрого доступа
                "sent_id": meta.get("sent_id_new") or meta.get("sent_id"),
                "genre": meta.get("genre", "unknown"),
                "text": meta.get("text", "")
            }

            # Распаковка метрик из JSON
            metrics_str = meta.get("metrics")
            if metrics_str:
                try:
                    metrics = json.loads(metrics_str)
                    row.update(metrics)  # length, is_dialogue, score и т.д.
                except:
                    pass

            data.append(row)

    return pd.DataFrame(data)


def sample_corpus(input_path, output_path, query, limit=None):
    logger.info(f"Reading metadata from: {input_path}")

    # 1. Создаем DataFrame для фильтрации
    df = load_metadata_df(input_path)

    if df.empty:
        logger.error("DataFrame is empty. Check input file format.")
        return

    logger.info(f"Total sentences: {len(df)}")

    # 2. Применяем фильтр (Pandas Query)
    try:
        # engine='python' нужен для сложных строковых операций
        filtered_df = df.query(query, engine='python')
    except Exception as e:
        logger.error(f"Query Syntax Error: {e}")
        return

    logger.info(f"Found {len(filtered_df)} matches for query: {query}")

    # 3. Лимитирование (Sampling)
    if limit and len(filtered_df) > limit:
        # Берем случайные N предложений, чтобы выборка была разнообразной
        filtered_df = filtered_df.sample(n=limit, random_state=42)
        logger.info(f"Sampled random {limit} sentences.")

    # Собираем индексы, которые нужно сохранить (для скорости используем set)
    target_indices = set(filtered_df['idx'].values)

    # 4. Второй проход: запись выбранных предложений
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:

        for i, sent in enumerate(tqdm(parse_incr(f_in), desc="Writing output", total=len(df))):
            if i in target_indices:
                f_out.write(sent.serialize())
                saved_count += 1

    logger.info(f"Saved {saved_count} sentences to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter CoNLL-U corpus by metrics")
    parser.add_argument("--input", required=True, help="Input .conllu file (enriched)")
    parser.add_argument("--output", required=True, help="Output .conllu file")
    parser.add_argument("--query", required=True,
                        help="Pandas query string (e.g. 'length > 10 and is_dialogue == True')")
    parser.add_argument("--limit", type=int, default=None, help="Max sentences to save")

    args = parser.parse_args()

    sample_corpus(args.input, args.output, args.query, args.limit)
