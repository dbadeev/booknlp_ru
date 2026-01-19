import logging
import matplotlib.pyplot as plt
from pathlib import Path
from conllu import parse_incr

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = []

    def analyze_file(self, name: str, filepath: str):
        """Сбор статистики по длине предложений"""
        lengths = []
        count = 0
        if not Path(filepath).exists():
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            for sent in parse_incr(f):
                # Считаем реальные токены (исключая пунктуацию, если нужно, но для длины берем все)
                tokens = [t for t in sent if isinstance(t['id'], int)]
                lengths.append(len(tokens))
                count += 1

        if not lengths:
            return

        avg_len = sum(lengths) / len(lengths)
        self.stats.append({
            "name": name,
            "count": count,
            "avg_len": round(avg_len, 2),
            "max_len": max(lengths),
            "lengths": lengths
        })

        self._plot_histogram(name, lengths)

    def _plot_histogram(self, name: str, lengths: list):
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"Sentence Length Distribution: {name}")
        plt.xlabel("Tokens per Sentence")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)

        img_path = self.output_dir / f"{name}_hist.png"
        plt.savefig(img_path)
        plt.close()

    def generate_report(self, filename: str = "dataset_stats.md"):
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Statistics Report\n\n")
            f.write("| Dataset | Sentences | Avg Length | Max Length |\n")
            f.write("|---------|-----------|------------|------------|\n")

            for s in self.stats:
                f.write(f"| {s['name']} | {s['count']} | {s['avg_len']} | {s['max_len']} |\n")

            f.write("\n## Distributions\n")
            for s in self.stats:
                f.write(f"### {s['name']}\n")
                f.write(f"![Histogram]({s['name']}_hist.png)\n\n")

        logger.info(f"Report generated at {report_path}")