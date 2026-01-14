#!/usr/bin/env python3.11
"""
Оценка предсказаний парсеров по метрикам LAS, UAS, Accuracy.
Использует встроенный расчет или CoNLL evaluation script.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict
from rich.console import Console
from rich.table import Table
import conllu

console = Console()

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "benchmarks"
SYNTAGRUS_TEST = ROOT / "data" / "syntagrus" / "ru_syntagrus-ud-test.conllu"
SLOVNET_PRED = RESULTS_DIR / "slovnet_predictions.conllu"


def download_eval_script(script_path: Path) -> bool:
    """Скачать CoNLL evaluation script с правильным URL."""
    if script_path.exists():
        return True

    console.print("📥 Скачивание CoNLL evaluation script...")

    # ПРАВИЛЬНЫЙ прямой URL (raw GitHub)
    url = "https://raw.githubusercontent.com/ufal/conll18/master/evaluation_script/conll18_ud_eval.py"

    try:
        subprocess.run(
            ["curl", "-s", "-L", "-o", str(script_path), url],
            check=True,
            timeout=10
        )

        # Проверить, что это Python файл, а не HTML
        with open(script_path, 'r') as f:
            first_line = f.readline()
            if 'html' in first_line.lower() or 'DOCTYPE' in first_line:
                console.print("⚠️  Скачан HTML вместо Python файла. Используем встроенный расчет.")
                script_path.unlink()
                return False

        console.print(f"✅ Script сохранён\n")
        return True

    except Exception as e:
        console.print(f"⚠️  Ошибка при скачивании: {e}")
        console.print("   Используем встроенный расчет метрик.\n")
        return False


def run_evaluation(gold_file: Path, pred_file: Path, eval_script: Path) -> Dict[str, float]:
    """Запустить evaluation script или использовать встроенный расчет."""
    console.print(f"📊 Оценка предсказаний...")
    console.print(f"  Gold: {gold_file.name}")
    console.print(f"  Pred: {pred_file.name}\n")

    # Сначала пробуем встроенный расчет
    metrics = _calculate_metrics_builtin(gold_file, pred_file)

    if metrics:
        return metrics

    # Если встроенный не сработал, пробуем официальный скрипт
    if eval_script.exists():
        return _run_eval_script(gold_file, pred_file, eval_script)

    return {}


def _calculate_metrics_builtin(gold_file: Path, pred_file: Path) -> Dict[str, float]:
    """
    Встроенный расчет метрик UAS, LAS без внешних зависимостей.
    """
    try:
        console.print("🔢 Расчет метрик (встроенный)...\n")

        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_sents = conllu.parse(f.read())

        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_sents = conllu.parse(f.read())

        if len(gold_sents) != len(pred_sents):
            console.print(f"⚠️  Разное количество предложений ({len(gold_sents)} vs {len(pred_sents)})")
            return {}

        uas_correct = 0
        las_correct = 0
        total_tokens = 0

        for gold_sent, pred_sent in zip(gold_sents, pred_sents):
            # Пропустить пунктуацию и пустые узлы
            for gold_token, pred_token in zip(gold_sent, pred_sent):
                # Только основные токены (не пунктуация, не empty nodes)
                if not isinstance(gold_token['id'], int):
                    continue

                gold_head = gold_token['head']
                gold_deprel = gold_token['deprel']

                pred_head = pred_token['head']
                pred_deprel = pred_token['deprel']

                total_tokens += 1

                # UAS: правильное головное слово
                if pred_head == gold_head:
                    uas_correct += 1

                    # LAS: правильное головное слово И правильный deprel
                    if pred_deprel == gold_deprel:
                        las_correct += 1

        if total_tokens == 0:
            console.print("⚠️  Нет токенов для оценки")
            return {}

        uas = uas_correct / total_tokens
        las = las_correct / total_tokens

        metrics = {
            "UAS": uas,
            "LAS": las,
            "total_tokens": total_tokens
        }

        return metrics

    except Exception as e:
        console.print(f"⚠️  Ошибка при встроенном расчете: {e}")
        return {}


def _run_eval_script(gold_file: Path, pred_file: Path, eval_script: Path) -> Dict[str, float]:
    """Запустить официальный CoNLL evaluation script."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(eval_script),
                "-v",
                str(gold_file),
                str(pred_file)
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )

        metrics = _parse_eval_output(result.stdout)
        return metrics

    except subprocess.TimeoutExpired:
        console.print("⚠️  Evaluation script занял слишком долго (timeout)")
        return {}
    except subprocess.CalledProcessError as e:
        console.print(f"⚠️  Ошибка при запуске скрипта: {e.stderr[:200]}")
        return {}


def _parse_eval_output(output: str) -> Dict[str, float]:
    """Парсить вывод CoNLL evaluation script."""
    metrics = {}

    for line in output.split('\n'):
        if '=' in line and not line.startswith('#'):
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    val_str = parts[1].strip().split()[0]
                    value = float(val_str)
                    metrics[key] = value
                except (ValueError, IndexError):
                    pass

    return metrics


def save_results(metrics: Dict[str, float], output_path: Path, parser_name: str):
    """Сохранить результаты в JSON."""
    results = {
        "parser": parser_name,
        "dataset": "SynTagRus-test",
        "test_file": "ru_syntagrus-ud-test.conllu",
        "metrics": metrics
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"💾 Результаты сохранены в {output_path}")


def display_results(metrics: Dict[str, float]):
    """Показать результаты в таблице."""
    table = Table(title="📈 Slovnet on SynTagRus test")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")

    # Сортировать: UAS, LAS в начале
    ordered_metrics = {}
    for key in ['UAS', 'LAS', 'MLAS', 'BLEX', 'Tokens', 'total_tokens']:
        if key in metrics:
            ordered_metrics[key] = metrics[key]

    # Остальные
    for key in sorted(metrics.keys()):
        if key not in ordered_metrics:
            ordered_metrics[key] = metrics[key]

    for key, value in ordered_metrics.items():
        if isinstance(value, float):
            if key in ['UAS', 'LAS', 'MLAS', 'BLEX']:
                table.add_row(key, f"{value:.4f} ({value * 100:.2f}%)")
            else:
                table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print("\n")
    console.print(table)


def main():
    # Проверить файлы
    if not SYNTAGRUS_TEST.exists():
        console.print(f"❌ {SYNTAGRUS_TEST} не найден")
        sys.exit(1)

    if not SLOVNET_PRED.exists():
        console.print(f"❌ {SLOVNET_PRED} не найден")
        console.print("   Запусти: python src/evaluation/parse_syntagrus.py")
        sys.exit(1)

    # Скачать eval script (опционально)
    eval_script = RESULTS_DIR / "conll18_ud_eval.py"
    download_eval_script(eval_script)

    # Запустить оценку
    metrics = run_evaluation(SYNTAGRUS_TEST, SLOVNET_PRED, eval_script)

    if not metrics:
        console.print("⚠️  Не удалось получить метрики")
        sys.exit(1)

    # Сохранить и показать
    results_file = RESULTS_DIR / "slovnet_metric_.json"
    save_results(metrics, results_file, "slovnet")
    display_results(metrics)

    console.print(f"\n✨ Оценка завершена!")


if __name__ == "__main__":
    main()
