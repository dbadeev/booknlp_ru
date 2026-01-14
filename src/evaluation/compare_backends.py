import sys
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table

# Импорты движков
from src.engines.natasha_engine import NatashaEngine
from src.engines.deeppavlov_engine import DeepPavlovEngine
from src.engines.cobald_engine import CobaldEngine
from src.data.conllu_reader import load_gold_standard

# Импорты структур
from src.core.data_structures import Token

console = Console()


class Benchmarker:
    def __init__(self, engines: Dict[str, Any]):
        self.engines = engines

    def align_tokens(self, gold_sent: List[Token], sys_sent: List[Token]) -> List[tuple]:
        """
        Маппинг токенов Gold -> System на основе пересечения координат.
        Возвращает список пар (GoldToken, SystemToken | None).
        """
        alignment = []
        sys_cursor = 0

        for g_tok in gold_sent:
            match = None

            # Ищем системный токен, который пересекается с золотым
            # Эвристика: Intersection > 50% длины золотого токена
            g_start, g_end = g_tok.char_start, g_tok.char_end
            g_len = g_end - g_start

            best_iou = 0.0
            best_cand_idx = -1

            # Проверяем окно кандидатов (чтобы не перебирать всё)
            for i in range(max(0, sys_cursor - 5), min(len(sys_sent), sys_cursor + 5)):
                s_tok = sys_sent[i]

                # Пересечение интервалов
                inter_start = max(g_start, s_tok.char_start)
                inter_end = min(g_end, s_tok.char_end)

                if inter_end > inter_start:
                    intersection = inter_end - inter_start
                    # Union (для IoU) или просто покрытие
                    coverage = intersection / g_len

                    if coverage > 0.5:  # Порог из Jira [cite: 132]
                        if coverage > best_iou:
                            best_iou = coverage
                            match = s_tok
                            best_cand_idx = i

            if match:
                sys_cursor = best_cand_idx  # Двигаем курсор

            alignment.append((g_tok, match))

        return alignment

    def compute_metrics(self, alignment: List[tuple]) -> Dict[str, float]:
        total = len(alignment)
        if total == 0: return {}

        matched_cnt = 0
        pos_correct = 0
        uas_correct = 0
        las_correct = 0
        sem_coverage = 0  # Для CoBaLD

        for g_tok, s_tok in alignment:
            if s_tok is None:
                continue

            matched_cnt += 1

            # 1. POS (UPOS)
            if g_tok.pos == s_tok.pos:
                pos_correct += 1

            # 2. Syntax (Head matching)
            # Сложность: head_id в Gold и System ссылаются на свои индексы.
            # Нам нужно проверить, ссылаются ли они на ОДНО И ТО ЖЕ СЛОВО.
            # Для строгого подсчета нужно выравнивание HEAD-ов.
            # УПРОЩЕНИЕ для MVP: Сравниваем просто ID, надеясь, что токенизация близка.
            # (В продакшене нужно мапить g_tok.head_id -> g_head_tok -> aligned_s_head_tok -> id)

            head_match = (g_tok.head_id == s_tok.head_id)
            if head_match:
                uas_correct += 1
                # 3. LAS (Label matching)
                if g_tok.rel == s_tok.rel:
                    las_correct += 1

            # 4. Semantics (Check if misc has fields)
            if s_tok.misc.get("sem_class") or s_tok.misc.get("deep_slot"):
                sem_coverage += 1

        return {
            "Tokenization F1": matched_cnt / total,  # Упрощенно
            "POS Accuracy": pos_correct / matched_cnt if matched_cnt else 0,
            "UAS": uas_correct / matched_cnt if matched_cnt else 0,
            "LAS": las_correct / matched_cnt if matched_cnt else 0,
            "Sem Coverage": sem_coverage / total
        }

    def run(self, gold_data: List[List[Token]]) -> pd.DataFrame:
        results = defaultdict(list)

        # Восстанавливаем текст для подачи в движки
        # (Предполагаем, что предложения обрабатываются по одному)

        for name, engine in self.engines.items():
            console.print(f"🚀 Running {name}...")

            agg_metrics = defaultdict(float)
            n_sents = 0

            for gold_sent in gold_data:
                # 1. Получаем текст предложения
                # Используем реальный текст токенов с пробелами
                # Важно: В реальности движки могут сегментировать иначе.
                # Тут мы подаем предложение за предложением.
                text = " ".join([t.text for t in gold_sent])

                # 2. Процессинг
                try:
                    # process возвращает List[List[Token]], берем [0] так как подаем 1 предложение
                    sys_output = engine.process(text)
                    if not sys_output: continue
                    sys_sent = sys_output[0]  # Берем первое (и единственное) предложение

                    # 3. Выравнивание
                    alignment = self.align_tokens(gold_sent, sys_sent)

                    # 4. Метрики
                    m = self.compute_metrics(alignment)

                    for k, v in m.items():
                        agg_metrics[k] += v
                    n_sents += 1

                except Exception as e:
                    console.print(f"⚠️ Error in {name}: {e}")

            # Усреднение
            if n_sents > 0:
                for k, v in agg_metrics.items():
                    results[k].append(v / n_sents)
            else:
                for k in ["Tokenization F1", "POS Accuracy", "UAS", "LAS", "Sem Coverage"]:
                    results[k].append(0.0)

            results["Engine"].append(name)

        return pd.DataFrame(results).set_index("Engine")


def main():
    # 1. Загрузка Gold Standard (например, тестовый фрагмент SynTagRus)
    gold_path = "data/test_samples.conllu"  # Создайте этот файл или укажите путь
    try:
        gold_data = load_gold_standard(gold_path)
        console.print(f"✅ Loaded {len(gold_data)} sentences from Gold Standard")
    except FileNotFoundError:
        console.print(f"❌ File {gold_path} not found. Create dummy data.")
        return

    # 2. Инициализация движков
    # Можно закомментировать те, что не установлены
    engines = {}

    try:
        engines["Natasha (CPU)"] = NatashaEngine()
    except Exception as e:
        print(f"Skip Natasha: {e}")

    # DeepPavlov требует установки (см. предыдущий шаг)
    try:
        engines["DeepPavlov (RuBERT)"] = DeepPavlovEngine(install=False)
    except Exception as e:
        print(f"Skip DeepPavlov: {e}")

    # CoBaLD (требует веса с HuggingFace)
    try:
        engines["CoBaLD (Semantics)"] = CobaldEngine()
    except Exception as e:
        print(f"Skip CoBaLD: {e}")

    if not engines:
        console.print("❌ No engines available!")
        return

    # 3. Запуск бенчмарка
    bencher = Benchmarker(engines)
    df = bencher.run(gold_data)

    # 4. Вывод таблицы
    console.print("\n🏆 Results Board:")

    table = Table(title="Engine Comparison")
    table.add_column("Engine", style="cyan")
    for col in df.columns:
        table.add_column(col, justify="right")

    for index, row in df.iterrows():
        vals = [f"{row[c]:.2%}" for c in df.columns]
        table.add_row(index, *vals)

    console.print(table)

    # 5. Сохранение
    df.to_csv("results/benchmark_report.csv")
    print("💾 Report saved to results/benchmark_report.csv")


if __name__ == "__main__":
    main()
