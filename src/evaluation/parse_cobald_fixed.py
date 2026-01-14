#!/usr/bin/env python3.11
"""
CoBaLD Parser — правильная версия с gold токенизацией.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

SYNTAGRUS_TEST = SYNTAGRUS_DIR / "ru_syntagrus-ud-test.conllu"
COBALD_OUTPUT = RESULTS_DIR / "cobald_predictions_fixed.conllu"


def load_cobald_model():
    """Загрузить только модель CoBaLD (без pipeline)."""
    console.print("🔍 Загрузка CoBaLD модели...\n")

    try:
        from cobald_parser import CobaldParser

        model_name = "CoBaLD/xlm-roberta-base-cobald-parser-ru"

        console.print(f"📥 Загружаю модель из {model_name}...\n")

        model = CobaldParser.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model.eval()

        console.print("✅ Модель загружена\n")
        return model

    except Exception as e:
        console.print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_conllu_with_tokens(filepath: Path, limit: int = None) -> List[dict]:
    """Загрузить предложения с токенами из gold standard."""
    sentences = []
    current_sent = {"text": "", "tokens": []}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:
                if current_sent["tokens"]:
                    sentences.append(current_sent)
                    current_sent = {"text": "", "tokens": []}

                    if limit and len(sentences) >= limit:
                        break
                continue

            if line.startswith('# text ='):
                current_sent["text"] = line.split('=', 1)[1].strip()
                continue

            if line.startswith('#'):
                continue

            parts = line.split('\t')
            token_id = parts[0]

            # Пропустить range tokens (1-2) и null tokens (1.1)
            if '-' in token_id or '.' in token_id:
                continue

            if len(parts) >= 2:
                current_sent["tokens"].append(parts[1])

    if current_sent["tokens"] and (not limit or len(sentences) < limit):
        sentences.append(current_sent)

    return sentences


def run_cobald_on_gold_tokens(model, sentences: List[dict]) -> List[dict]:
    """
    Запустить CoBaLD на токенах из gold standard.

    ВАЖНО: Используем gold токены, а не ретокенизируем.
    """
    console.print(f"🔍 Парсинг {len(sentences)} предложений (gold токены)...\n")

    results = []
    errors = 0

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Обработка...",
            total=len(sentences)
        )

        for i, sent in enumerate(sentences, 1):
            try:
                # Использовать токены из gold
                words_batch = [sent["tokens"]]

                # Запустить модель напрямую
                output = model(
                    words=words_batch,
                    inference_mode=True
                )

                # Извлечь результаты
                result = {
                    "text": sent["text"],
                    "tokens": sent["tokens"],
                    "output": output
                }
                results.append(result)

            except Exception as e:
                errors += 1
                if errors <= 10:  # Показать только первые 10 ошибок
                    console.print(f"\n⚠️  Ошибка {i}: {str(e)[:100]}")

                results.append({
                    "text": sent["text"],
                    "tokens": sent["tokens"],
                    "error": str(e)
                })

            progress.update(task, advance=1)

    if errors > 0:
        console.print(f"\n⚠️  Всего ошибок: {errors}/{len(sentences)}\n")

    return results


def convert_to_conllu(results: List[dict], model) -> str:
    """Конвертировать результаты в CoNLL-U."""
    conllu_lines = []

    for sent_result in results:
        if "error" in sent_result:
            # Пропустить предложения с ошибками
            continue

        lines = [f"# text = {sent_result['text']}"]

        tokens = sent_result["tokens"]
        output = sent_result["output"]

        # Извлечь deps_ud (синтаксис)
        if "deps_ud" in output and output["deps_ud"] is not None:
            deps_ud = output["deps_ud"]

            # deps_ud имеет формат: [batch_idx, from_idx, to_idx, deprel_id]
            # Фильтруем batch_idx == 0 (первое предложение в батче)
            arcs = deps_ud[deps_ud[:, 0] == 0][:, 1:]  # [from, to, deprel]

            # Создать словарь: token_id -> (head, deprel)
            syntax_dict = {}
            for arc in arcs:
                from_idx = int(arc[0])
                to_idx = int(arc[1])
                deprel_id = int(arc[2])

                # Декодировать deprel
                deprel = model.config.vocabulary.get("ud_deprel", {}).get(deprel_id, "_")

                # to_idx — это индекс токена (начиная с 0)
                # head — это from_idx (0 = root)
                head = from_idx if from_idx != to_idx else 0
                syntax_dict[to_idx] = (head, deprel)
        else:
            syntax_dict = {}

        # Записать токены
        for idx, token in enumerate(tokens):
            token_id = idx + 1
            head, deprel = syntax_dict.get(idx, (0, "root"))

            # Базовый CoNLL-U (без морфологии и семантики пока)
            line = "\t".join([
                str(token_id),  # ID
                token,  # FORM
                "_",  # LEMMA
                "_",  # UPOS
                "_",  # XPOS
                "_",  # FEATS
                str(head),  # HEAD
                deprel,  # DEPREL
                "_",  # DEPS
                "_",  # MISC
                "_",  # DEEPSLOT
                "_"  # SEMCLASS
            ])
            lines.append(line)

        conllu_lines.append("\n".join(lines))

    return "\n\n".join(conllu_lines)


def main():
    console.print("=" * 80)
    console.print("CoBaLD Parser (Fixed - Gold Tokens)".center(80))
    console.print("=" * 80 + "\n")

    console.print("⚠️  Эта версия использует токены из gold standard\n")
    console.print("   для корректного сравнения с бенчмарком.\n")

    # Режим
    mode = input("Режим [1=тест 10, 2=полный 8800]: ").strip()
    limit = 10 if mode == '1' else None

    if limit:
        console.print("\n🧪 Тестовый режим: 10 предложений\n")
    else:
        console.print("\n🚀 Полный бенчмарк\n")

    # Загрузить модель
    model = load_cobald_model()

    # Загрузить предложения с gold токенами
    console.print(f"📂 Загружаю предложения (gold tokens)...")
    sentences = parse_conllu_with_tokens(SYNTAGRUS_TEST, limit=limit)
    console.print(f"✅ Загружено {len(sentences)} предложений\n")

    # Запустить парсинг
    results = run_cobald_on_gold_tokens(model, sentences)

    # Конвертировать в CoNLL-U
    console.print("\n💾 Конвертация в CoNLL-U...")
    conllu_output = convert_to_conllu(results, model)

    # Сохранить
    output_file = COBALD_OUTPUT if not limit else RESULTS_DIR / "cobald_test_fixed.conllu"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(conllu_output)
    console.print(f"✅ Сохранено: {output_file.name}\n")

    console.print("✨ Готово!\n")
    console.print(f"Проверь: head {output_file}\n")


if __name__ == "__main__":
    main()
