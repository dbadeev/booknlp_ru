#!/usr/bin/env python3.11
"""
CoBaLD Parser Benchmark на SynTagRus test set.
"""

import sys
from pathlib import Path

# Определить ROOT в самом начале
ROOT = Path(__file__).resolve().parents[2]

# Добавить src в PYTHONPATH
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from typing import List
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import json

console = Console()

DATA_DIR = ROOT / "data"
SYNTAGRUS_DIR = DATA_DIR / "syntagrus"
RESULTS_DIR = ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Пути к файлам
SYNTAGRUS_TEST = SYNTAGRUS_DIR / "ru_syntagrus-ud-test.conllu"
COBALD_OUTPUT = RESULTS_DIR / "cobald_predictions.conllu"



def load_cobald_pipeline():
    """Загрузить CoBaLD Parser через кастомный Pipeline."""
    console.print("🔍 Загрузка CoBaLD Parser...\n")

    try:
        # Добавить src в PYTHONPATH
        src_path = ROOT / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        console.print(f"📦 PYTHONPATH: {src_path}\n")

        # Импортировать кастомные классы
        from cobald_parser import (
            CobaldParser,
            CobaldParserConfig,
            ConlluTokenClassificationPipeline
        )
        console.print("✅ Импортированы кастомные классы CoBaLD\n")

        model_name = "CoBaLD/xlm-roberta-base-cobald-parser-ru"

        console.print(f"📥 Загружаю модель из {model_name}...")
        console.print("   (это может занять несколько минут)\n")

        # Загрузить модель
        model = CobaldParser.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model.eval()

        console.print("✅ Модель загружена\n")

        # Настроить tokenizer/sentenizer
        console.print("📦 Настройка токенизации...")

        try:
            from razdel import sentenize, tokenize

            def sentenizer(text):
                return [s.text for s in sentenize(text)]

            def tokenizer(text):
                return [t.text for t in tokenize(text)]

            console.print("✅ Используем razdel для сегментации\n")

        except ImportError:
            console.print("⚠️  razdel не установлен")
            console.print("   pip install razdel\n")
            console.print("   Используем простую токенизацию\n")

            def sentenizer(text):
                import re
                # Простая сегментация
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]

            def tokenizer(text):
                # Простая токенизация
                return text.split()

        # Создать pipeline
        pipeline = ConlluTokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            sentenizer=sentenizer
        )

        console.print("✅ CoBaLD Pipeline готов\n")
        return pipeline

    except ImportError as e:
        console.print(f"❌ Ошибка импорта: {e}\n")
        console.print("Отладка:")
        console.print(f"  ROOT: {ROOT}")
        console.print(f"  src path: {ROOT / 'src'}")
        console.print(f"  cobald_parser exists: {(ROOT / 'src' / 'cobald_parser').exists()}")
        console.print(f"  __init__.py exists: {(ROOT / 'src' / 'cobald_parser' / '__init__.py').exists()}")
        console.print(f"\n  Файлы в cobald_parser:")
        cobald_dir = ROOT / 'src' / 'cobald_parser'
        if cobald_dir.exists():
            for f in cobald_dir.glob('*.py'):
                console.print(f"    - {f.name}")
        console.print("\nПроверь:")
        console.print("  1. Все файлы скопированы в src/cobald_parser/")
        console.print("  2. __init__.py создан правильно")
        console.print("  3. Установлены зависимости:")
        console.print("     pip install transformers torch razdel\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        console.print(f"❌ Ошибка загрузки модели: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_conllu_for_text(filepath: Path, limit: int = None) -> List[str]:
    """Извлечь тексты предложений из CoNLL-U."""
    texts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text ='):
                text = line.split('=', 1)[1].strip()
                texts.append(text)

                if limit and len(texts) >= limit:
                    break

    return texts


def run_cobald_parsing(pipeline, texts: List[str]) -> str:
    """Запустить CoBaLD Parser на текстах."""
    console.print(f"🔍 Парсинг {len(texts)} предложений...\n")

    all_results = []
    errors = 0

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Обработка предложений...",
            total=len(texts)
        )

        for i, text in enumerate(texts, 1):
            try:
                # Запустить pipeline
                result = pipeline(text, output_format='str')
                all_results.append(result)

            except Exception as e:
                errors += 1
                console.print(f"\n⚠️  Ошибка на предложении {i}: {text[:50]}...")
                console.print(f"   {str(e)[:100]}")

                # Добавить placeholder
                all_results.append(f"# ERROR on sentence {i}: {e}")

            progress.update(task, advance=1)

    if errors > 0:
        console.print(f"\n⚠️  Ошибок при парсинге: {errors}/{len(texts)}\n")

    return "\n\n".join(all_results)


def parse_conllu_simple(filepath: Path) -> list:
    """Простой парсер CoNLL-U (для оценки)."""
    sentences = []
    current_sent = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line or line.startswith('#'):
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                continue

            parts = line.split('\t')
            if len(parts) < 8:
                continue

            token_id = parts[0]

            # Пропустить range и null tokens
            if '-' in token_id or '.' in token_id:
                continue

            try:
                token_id = int(token_id)
                head = int(parts[6])
                deprel = parts[7]

                current_sent.append({
                    'id': token_id,
                    'form': parts[1],
                    'head': head,
                    'deprel': deprel
                })
            except (ValueError, IndexError):
                continue

    if current_sent:
        sentences.append(current_sent)

    return sentences


def calculate_metrics(gold_sentences: list, pred_sentences: list) -> dict:
    """Расчет UAS и LAS."""
    if len(gold_sentences) != len(pred_sentences):
        console.print(f"⚠️  Разное количество предложений:")
        console.print(f"   Gold: {len(gold_sentences)}")
        console.print(f"   Pred: {len(pred_sentences)}")
        # Используем минимум
        min_len = min(len(gold_sentences), len(pred_sentences))
        gold_sentences = gold_sentences[:min_len]
        pred_sentences = pred_sentences[:min_len]

    uas_correct = 0
    las_correct = 0
    total_tokens = 0

    for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
        # Align tokens by id
        gold_dict = {t['id']: t for t in gold_sent}
        pred_dict = {t['id']: t for t in pred_sent}

        for token_id in gold_dict:
            if token_id not in pred_dict:
                continue

            total_tokens += 1

            if pred_dict[token_id]['head'] == gold_dict[token_id]['head']:
                uas_correct += 1

                if pred_dict[token_id]['deprel'] == gold_dict[token_id]['deprel']:
                    las_correct += 1

    if total_tokens == 0:
        return {}

    return {
        "UAS": uas_correct / total_tokens,
        "LAS": las_correct / total_tokens,
        "uas_correct": uas_correct,
        "las_correct": las_correct,
        "total_tokens": total_tokens
    }


def evaluate_cobald_predictions(gold_file: Path, pred_file: Path):
    """Оценить предсказания CoBaLD."""
    console.print("\n🔢 Расчет метрик...\n")

    try:
        gold_sentences = parse_conllu_simple(gold_file)
        pred_sentences = parse_conllu_simple(pred_file)

        console.print(f"✅ Gold: {len(gold_sentences)} предложений")
        console.print(f"✅ Pred: {len(pred_sentences)} предложений\n")

        metrics = calculate_metrics(gold_sentences, pred_sentences)

        if not metrics:
            console.print("❌ Не удалось рассчитать метрики")
            return None

        return metrics

    except Exception as e:
        console.print(f"❌ Ошибка оценки: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_results(metrics: dict):
    """Показать результаты."""
    table = Table(title="📈 CoBaLD Parser on SynTagRus-test")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")

    table.add_row("UAS", f"{metrics['UAS']:.4f} ({metrics['UAS'] * 100:.2f}%)")
    table.add_row("LAS", f"{metrics['LAS']:.4f} ({metrics['LAS'] * 100:.2f}%)")
    table.add_row("Правильных UAS", str(metrics['uas_correct']))
    table.add_row("Правильных LAS", str(metrics['las_correct']))
    table.add_row("Всего токенов", str(metrics['total_tokens']))

    console.print("\n")
    console.print(table)


def main():
    console.print("=" * 80)
    console.print("CoBaLD Parser Benchmark".center(80))
    console.print("=" * 80 + "\n")

    # Проверить файлы
    cobald_dir = ROOT / "src" / "cobald_parser"
    if not cobald_dir.exists():
        console.print(f"❌ {cobald_dir} не найдена")
        sys.exit(1)

    console.print(f"✅ Папка CoBaLD: {cobald_dir}")
    console.print(f"   Файлов: {len(list(cobald_dir.glob('*.py')))}\n")

    if not SYNTAGRUS_TEST.exists():
        console.print(f"❌ {SYNTAGRUS_TEST} не найден")
        sys.exit(1)

    # Режим работы
    console.print("Выбери режим:\n")
    console.print("  1. Быстрый тест (10 предложений)")
    console.print("  2. Полный бенчмарк (8800 предложений)\n")

    mode = input("Режим [1/2]: ").strip()

    if mode == '1':
        limit = 10
        console.print("\n🧪 Тестовый режим: 10 предложений\n")
    elif mode == '2':
        limit = None
        console.print("\n🚀 Полный бенчмарк: все предложения")
        console.print("   ⏱️  Время: ~10-30 минут на CPU\n")
    else:
        console.print("\nОтменено.")
        sys.exit(0)

    # Загрузить pipeline
    pipeline = load_cobald_pipeline()

    # Загрузить тексты
    console.print(f"📂 Загружаю предложения из {SYNTAGRUS_TEST.name}...")
    texts = parse_conllu_for_text(SYNTAGRUS_TEST, limit=limit)
    console.print(f"✅ Загружено {len(texts)} предложений\n")

    # Запустить парсинг
    conllu_output = run_cobald_parsing(pipeline, texts)

    # Сохранить результаты
    output_file = COBALD_OUTPUT if limit is None else RESULTS_DIR / "cobald_test.conllu"
    console.print(f"\n💾 Сохранение в {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(conllu_output)
    console.print(f"✅ Сохранено\n")

    # Оценить (только для полного теста)
    if limit is None:
        metrics = evaluate_cobald_predictions(SYNTAGRUS_TEST, output_file)

        if metrics:
            display_results(metrics)

            # Сохранить JSON
            results_json = RESULTS_DIR / "cobald_metrics.json"
            with open(results_json, 'w', encoding='utf-8') as f:
                json.dump({
                    "parser": "cobald",
                    "dataset": "SynTagRus-test",
                    "metrics": {
                        "UAS": metrics['UAS'],
                        "LAS": metrics['LAS']
                    },
                    "details": metrics
                }, f, indent=2)

            console.print(f"\n💾 Метрики: {results_json.name}")
    else:
        console.print(f"📄 Проверь результат: cat {output_file}")

    console.print("\n" + "=" * 80)
    console.print("\n✨ Готово!\n")


if __name__ == "__main__":
    main()
