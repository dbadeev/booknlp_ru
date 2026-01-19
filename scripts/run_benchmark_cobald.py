import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Импорт обертки
try:
    from src.parsers.cobald_wrapper import CobaldParser
    from src.evaluation.metrics import MetricsCalculator
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkCoBaLD")


def print_comparison(sent_id, forms, sys_tokens, gold_tokens):
    """Выводит таблицу сравнения для отладки"""
    print(f"\n🔍 DEBUG: Sentence {sent_id}")
    print(
        f"{'ID':<4} {'WORD':<15} | {'GOLD HEAD':<10} {'SYS HEAD':<10} | {'GOLD REL':<10} {'SYS REL':<10} | {'MATCH?'}")
    print("-" * 90)

    for i, form in enumerate(forms):
        # Gold
        g_head = gold_tokens[i]['head_id']
        g_rel = gold_tokens[i]['deprel']

        # System
        s_head = sys_tokens[i]['head']
        s_rel = sys_tokens[i]['deprel']

        match = "✅" if (g_head == s_head and g_rel == s_rel) else "❌"
        if g_head == s_head and g_rel != s_rel: match = "⚠️ (Rel)"
        if g_head != s_head: match = "❌ (Head)"

        print(f"{i + 1:<4} {form:<15} | {g_head:<10} {s_head:<10} | {g_rel:<10} {s_rel:<10} | {match}")
    print("-" * 90 + "\n")


def run():
    gold_path = PROJECT_ROOT / "data" / "processed" / "val_complex.conllu"

    if not gold_path.exists():
        logger.error(f"Файл с данными не найден: {gold_path}")
        return

    logger.info("Инициализация CoBaLD Parser...")
    try:
        wrapper = CobaldParser()
        service = wrapper.service
        service.parse.remote(["Тест"])  # Warmup
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        return

    logger.info(f"Загрузка данных из {gold_path}...")
    gold_data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            if not sent: continue

            tokens = []
            forms = []
            for t in sent:
                if isinstance(t['id'], int):
                    tokens.append({
                        "id": t["id"],
                        "head_id": t["head"],
                        "deprel": t["deprel"]
                    })
                    forms.append(t["form"])

            if forms:
                gold_data.append({
                    "forms": forms,
                    "gold_tokens": tokens,
                    "id": sent.metadata.get("sent_id", "unknown")
                })

    logger.info(f"Загружено предложений: {len(gold_data)}")

    total_uas = 0
    total_las = 0
    processed_count = 0

    # Счетчик для отладочного вывода (выведем первые 3)
    debug_limit = 10

    start_time = time.time()

    for item in tqdm(gold_data, desc="Бенчмарк CoBaLD"):
        try:
            sys_output = service.parse.remote(item['forms'])

            if not sys_output: continue
            if len(sys_output) != len(item['gold_tokens']): continue

            # --- БЛОК ОТЛАДКИ ---
            if processed_count < debug_limit:
                print_comparison(item['id'], item['forms'], sys_output, item['gold_tokens'])
            # --------------------

            correct_heads = 0
            correct_labels = 0
            length = len(sys_output)

            for sys_tok, gold_tok in zip(sys_output, item['gold_tokens']):
                if sys_tok['head'] == gold_tok['head_id']:
                    correct_heads += 1
                    if sys_tok['deprel'] == gold_tok['deprel']:
                        correct_labels += 1

            total_uas += correct_heads / length
            total_las += correct_labels / length
            processed_count += 1

        except Exception as e:
            logger.error(f"Ошибка на {item['id']}: {e}")

    duration = time.time() - start_time

    print("\n" + "=" * 30)
    print("=== CoBaLD Results (Oracle) ===")
    print("=" * 30)

    if processed_count > 0:
        speed = processed_count / duration
        avg_uas = total_uas / processed_count
        avg_las = total_las / processed_count
        print(f"Soft UAS:  {avg_uas:.4f}")
        print(f"Soft LAS:  {avg_las:.4f}")

        pd.DataFrame([{
            "Model": "CoBaLD (Oracle)",
            "UAS": avg_uas,
            "LAS": avg_las,
            "Speed": speed,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]).to_csv(PROJECT_ROOT / "benchmark_cobald.csv", index=False)


if __name__ == "__main__":
    run()