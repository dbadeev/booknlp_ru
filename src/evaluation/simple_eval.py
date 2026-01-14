#!/usr/bin/env python3.11
"""
Простой расчет UAS/LAS метрик без CoNLL-U парсинга.
Напрямую сравнивает gold и pred файлы.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
import json

console = Console()

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "benchmarks"
SYNTAGRUS_TEST = ROOT / "data" / "syntagrus" / "ru_syntagrus-ud-test.conllu"
SLOVNET_PRED = RESULTS_DIR / "slovnet_predictions.conllu"


def parse_conllu_simple(filepath: Path) -> list:
    """
    Парсить CoNLL-U файл простым способом (без conllu библиотеки).
    Возвращает список предложений, каждое — список токенов.
    """
    sentences = []
    current_sent = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Пропустить пустые строки и комментарии между предложениями
            if not line or line.startswith('#'):
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                continue

            # Парсить строку токена
            parts = line.split('\t')
            if len(parts) < 7:
                continue

            token_id = parts[0]
            form = parts[1]
            lemma = parts[2]
            upos = parts[3]
            head = parts[6]
            deprel = parts[7]

            # Пропустить пустые узлы (id типа '1_1')
            if '-' in token_id or '_' in token_id:
                continue

            try:
                token_id = int(token_id)
                head = int(head)
            except ValueError:
                continue

            current_sent.append({
                'id': token_id,
                'form': form,
                'head': head,
                'deprel': deprel
            })

    # Добавить последнее предложение
    if current_sent:
        sentences.append(current_sent)

    return sentences


def calculate_metrics(gold_sentences: list, pred_sentences: list) -> dict:
    """Расчет UAS и LAS метрик."""
    if len(gold_sentences) != len(pred_sentences):
        console.print(f"⚠️  Разное количество предложений:")
        console.print(f"   Gold: {len(gold_sentences)}")
        console.print(f"   Pred: {len(pred_sentences)}")
        return {}

    uas_correct = 0
    las_correct = 0
    total_tokens = 0
    errors = 0

    for sent_idx, (gold_sent, pred_sent) in enumerate(zip(gold_sentences, pred_sentences)):
        # Может быть разное количество токенов (если парсер пропустил что-то)
        if len(gold_sent) != len(pred_sent):
            errors += 1
            continue

        for gold_token, pred_token in zip(gold_sent, pred_sent):
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
        "uas_correct": uas_correct,
        "las_correct": las_correct,
        "total_tokens": total_tokens,
        "sentences_with_errors": errors
    }

    return metrics


def display_results(metrics: dict):
    """Показать результаты в таблице."""
    if not metrics:
        return

    table = Table(title="📈 Slovnet on SynTagRus-test")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")

    table.add_row("UAS", f"{metrics['UAS']:.4f} ({metrics['UAS'] * 100:.2f}%)")
    table.add_row("LAS", f"{metrics['LAS']:.4f} ({metrics['LAS'] * 100:.2f}%)")
    table.add_row("Правильных UAS", str(metrics['uas_correct']))
    table.add_row("Правильных LAS", str(metrics['las_correct']))
    table.add_row("Всего токенов", str(metrics['total_tokens']))

    if metrics['sentences_with_errors'] > 0:
        table.add_row("Ошибки разбора", str(metrics['sentences_with_errors']))

    # ИСПРАВЛЕНО: просто выводим table, без конкатенации
    console.print("\n")
    console.print(table)


def save_results(metrics: dict, output_path: Path):
    """Сохранить результаты в JSON."""
    results = {
        "parser": "slovnet",
        "dataset": "SynTagRus-test",
        "test_file": "ru_syntagrus-ud-test.conllu",
        "metrics": {
            "UAS": metrics['UAS'],
            "LAS": metrics['LAS']
        },
        "details": {
            "uas_correct": metrics['uas_correct'],
            "las_correct": metrics['las_correct'],
            "total_tokens": metrics['total_tokens']
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"💾 Результаты сохранены в {output_path}")


def main():
    console.print("📊 Оценка Slovnet на SynTagRus-test\n")

    # Проверить файлы
    if not SYNTAGRUS_TEST.exists():
        console.print(f"❌ {SYNTAGRUS_TEST} не найден")
        sys.exit(1)

    if not SLOVNET_PRED.exists():
        console.print(f"❌ {SLOVNET_PRED} не найден")
        sys.exit(1)

    # Парсить файлы
    console.print("📂 Загружаю золотой стандарт (SynTagRus)...")
    gold_sentences = parse_conllu_simple(SYNTAGRUS_TEST)
    console.print(f"✅ Загружено {len(gold_sentences)} предложений\n")

    console.print("📂 Загружаю предсказания (Slovnet)...")
    pred_sentences = parse_conllu_simple(SLOVNET_PRED)
    console.print(f"✅ Загружено {len(pred_sentences)} предложений\n")

    # Расчет метрик
    console.print("🔢 Расчет метрик...\n")
    metrics = calculate_metrics(gold_sentences, pred_sentences)

    if not metrics:
        console.print("❌ Не удалось расчитать метрики")
        sys.exit(1)

    # Показать результаты
    display_results(metrics)

    # Сохранить результаты
    results_file = RESULTS_DIR / "slovnet_metrics_simple.json"
    save_results(metrics, results_file)

    console.print(f"\n✨ Оценка завершена!")


if __name__ == "__main__":
    main()
