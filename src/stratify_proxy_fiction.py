import argparse
import logging
import json
import networkx as nx
import pandas as pd
from pathlib import Path
from conllu import parse_incr
from conllu.models import TokenList
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Попытка импорта визуализации
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Визуализация недоступна: {e}. Графики не будут построены.")
    VISUALIZATION_AVAILABLE = False
except Exception as e:
    logger.warning(f"Ошибка совместимости визуализации: {e}. Графики не будут построены.")
    VISUALIZATION_AVAILABLE = False


class AdvancedSentenceProfiler:
    """
    Профилировщик лингвистической сложности.
    Реализует метрики Карточки 1.2: Dialogue Score, Syntax Complexity, Non-projectivity.
    """

    def __init__(self, is_cobald=False):
        # Глаголы речи для детекции диалогов
        self.speech_verbs = {
            "сказать", "говорить", "ответить", "отвечать", "спросить", "спрашивать",
            "промолвить", "заявить", "подумать", "воскликнуть", "шептать", "пояснить",
            "заметить", "продолжить", "добавить", "прошептать", "крикнуть"
        }
        self.is_cobald = is_cobald

    def _is_dialogue(self, sentence: TokenList) -> bool:
        """
        Определяет, является ли предложение репликой диалога.
        Эвристика: начинается с тире И содержит глагол речи.
        """
        text = sentence.metadata.get("text", "").strip()
        # 1. Проверка на маркер диалога (тире, дефис, длинное тире)
        if not text.startswith(("-", "—", "–", "−")):
            return False

        # 2. Проверка наличия глагола речи (в леммах)
        tokens = [t for t in sentence if isinstance(t['id'], int)]
        has_speech_verb = False
        for t in tokens:
            lemma = t.get("lemma", "").lower() if t.get("lemma") else str(t.get("form", "")).lower()
            if lemma in self.speech_verbs:
                has_speech_verb = True
                break

        return has_speech_verb

    def _is_non_projective(self, sentence: TokenList) -> bool:
        """Проверка на пересечение дуг зависимостей (непроективность)."""
        arcs = []
        for t in sentence:
            if isinstance(t['id'], int) and t['head'] is not None and t['head'] > 0:
                arcs.append(tuple(sorted((t['id'], t['head']))))

        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                s1, e1 = arcs[i]
                s2, e2 = arcs[j]
                if (s1 < s2 < e1 < e2) or (s2 < s1 < e2 < e1):
                    return True
        return False

    def _calculate_tree_depth(self, sentence: TokenList) -> int:
        """Расчет максимальной глубины синтаксического дерева."""
        edges = []
        for t in sentence:
            if isinstance(t['id'], int) and t['head'] is not None:
                edges.append((t['head'], t['id']))

        if not edges: return 0

        G = nx.DiGraph(edges)
        try:
            return nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            return 0

    def profile_sentence(self, sentence: TokenList) -> dict:
        tokens = [t for t in sentence if isinstance(t['id'], int)]

        # 1. Вычисление базовых метрик
        depth = self._calculate_tree_depth(sentence)
        is_non_proj = self._is_non_projective(sentence)
        is_dialogue_bool = self._is_dialogue(sentence)

        pronoun_count = sum(1 for t in tokens if t['upos'] == 'PRON')
        propn_count = sum(1 for t in tokens if t['upos'] == 'PROPN')
        verb_count = sum(1 for t in tokens if t['upos'] == 'VERB')
        length = len(tokens)

        # 2. Формирование специфических показателей Карточки 1.2
        # Dialogue Score: 1.0 если диалог, иначе 0.0 (можно использовать веса)
        dialogue_score = 1.0 if is_dialogue_bool else 0.0

        # Syntax Complexity: Глубина + штраф за непроективность
        #
        syntax_complexity = depth + (10.0 if is_non_proj else 0.0)

        # 3. Интегральный Total Score для сортировки
        # Веса: Синтаксис + Диалог + Лексика + Бонус CoBaLD
        total_score = (
                syntax_complexity * 1.5 +  # Упор на сложный синтаксис
                dialogue_score * 15.0 +  # Сильный буст диалогам
                pronoun_count * 2.0 +  # Сложность кореференции
                verb_count * 1.0 +
                (20.0 if self.is_cobald else 0.0)
        )

        return {
            "score": total_score,
            "dialogue_score": dialogue_score,  # <--- Добавлено в output
            "syntax_complexity": syntax_complexity,  # <--- Добавлено в output
            "depth": depth,
            "non_proj": is_non_proj,
            "pronouns": pronoun_count,
            "propn": propn_count,
            "verbs": verb_count,
            "length": length
        }


class ProxyFictionStratifier:
    def __init__(self, input_path, output_dir, test_size=0.1):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size

        # Авто-определение CoBaLD
        is_cobald = "cobald" in self.input_path.name.lower()
        self.profiler = AdvancedSentenceProfiler(is_cobald=is_cobald)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        records = []
        logger.info(f"Профилирование: {self.input_path.name} (is_cobald={self.profiler.is_cobald})")

        with open(self.input_path, "r", encoding="utf-8") as f:
            for sentence in tqdm(parse_incr(f), desc="Profiling"):
                metrics = self.profiler.profile_sentence(sentence)

                # Сохраняем метрики в метаданные предложения
                sentence.metadata["complexity_score"] = f"{metrics['score']:.2f}"
                sentence.metadata["metrics"] = json.dumps(metrics)

                records.append({**metrics, "conllu": sentence.serialize()})

        df = pd.DataFrame(records).sort_values(by="score", ascending=False)

        # Разбиение
        split_idx = int(len(df) * self.test_size)
        proxy_df = df.iloc[:split_idx].copy()
        train_df = df.iloc[split_idx:].copy()

        proxy_df["set"] = "Proxy-Fiction (Dev)"
        train_df["set"] = "Train"

        logger.info(f"Разбиение завершено. Proxy-Fiction: {len(proxy_df)}, Train: {len(train_df)}")

        # Сохранение результатов
        self._save_conllu(proxy_df, "proxy_fiction_dev")
        self._save_conllu(train_df, "train")
        self._generate_reports(train_df, proxy_df)

    def _save_conllu(self, df, suffix):
        clean_name = self.input_path.stem.replace("_full", "").replace("_adapted", "").replace("_clean", "")
        out_path = self.output_dir / f"{clean_name}_{suffix}.conllu"
        with open(out_path, "w", encoding="utf-8") as f:
            for text in df["conllu"]:
                f.write(text)

    def _generate_reports(self, train_df, proxy_df):
        # CSV отчет (теперь содержит dialogue_score и syntax_complexity)
        full_df = pd.concat([proxy_df, train_df]).drop(columns=["conllu"])
        csv_path = self.output_dir / f"{self.input_path.stem}_complexity_report.csv"
        full_df.to_csv(csv_path, index=False)

        # Текстовое саммари
        summary = []
        for name, df in [("Train", train_df), ("Proxy-Fiction", proxy_df)]:
            summary.append({
                "Dataset": name,
                "Size": len(df),
                "Mean Score": df["score"].mean(),
                "Mean Syntax Complexity": df["syntax_complexity"].mean(),  # <--- Новое поле
                "Dialogue Ratio": df["dialogue_score"].mean(),  # <--- Новое поле (доля диалогов)
                "Mean Pronouns": df["pronouns"].mean()
            })

        summary_df = pd.DataFrame(summary).set_index("Dataset")
        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ ПО СТРАТИФИКАЦИИ (КАРТОЧКА 1.2)")
        print("=" * 60)
        print(summary_df.to_string())
        print("=" * 60)

        with open(self.output_dir / f"{self.input_path.stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_df.to_string())

        if VISUALIZATION_AVAILABLE:
            try:
                self._plot_distributions(full_df)
            except Exception as e:
                logger.error(f"Ошибка графиков: {e}")

    def _plot_distributions(self, df):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Syntax Complexity Distribution
        sns.histplot(data=df, x="syntax_complexity", hue="set", kde=True, ax=axes[0])
        axes[0].set_title("Syntax Complexity Distribution")

        # 2. Dialogue Score (Bar plot, т.к. это 0 или 1)
        sns.barplot(data=df, x="set", y="dialogue_score", ax=axes[1])
        axes[1].set_title("Dialogue Ratio (Mean Score)")

        # 3. Total Complexity ECDF
        sns.ecdfplot(data=df, x="score", hue="set", ax=axes[2])
        axes[2].set_title("Total Complexity Score ECDF")

        plt.tight_layout()
        plot_path = self.output_dir / f"{self.input_path.stem}_distributions.png"
        plt.savefig(plot_path)
        logger.info(f"Графики сохранены: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Стратификация корпуса (Карточка 1.2).")
    parser.add_argument("--input", required=True, help="Путь к conllu файлу")
    parser.add_argument("--output", default="data/processed", help="Папка вывода")
    parser.add_argument("--test_size", type=float, default=0.15, help="Доля Proxy-Fiction")

    args = parser.parse_args()

    ProxyFictionStratifier(args.input, args.output, args.test_size).run()