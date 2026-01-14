#!/usr/bin/env python3
import sys
from pathlib import Path

# Добавляем путь к корню проекта
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.engines.cobald_engine import CobaldEngine


def test_engine():
    # Текст с семантикой: "Иван купил машину" -> Иван (Agent), машину (Theme/Object)
    text = "Иван купил красную машину."

    # Инициализация (укажите путь, если модель скачана локально, или repo_id)
    # Если модели нет локально, он попытается скачать с HuggingFace Hub
    engine = CobaldEngine(model_path="CoBaLD/xlm-roberta-base-cobald-parser-ru")

    print(f"\nProcessing: '{text}'\n")
    sentences = engine.process(text)

    print(f"{'ID':<3} {'TOKEN':<12} {'POS':<6} {'HEAD':<5} {'REL':<10} {'SEMANTICS (Misc)':<30}")
    print("-" * 80)

    for sent in sentences:
        for t in sent:
            # Красивый вывод словаря misc
            sem_str = ", ".join([f"{k}={v}" for k, v in t.misc.items()])

            print(f"{t.id:<3} {t.text:<12} {t.pos:<6} {t.head_id:<5} {t.rel:<10} {sem_str:<30}")


if __name__ == "__main__":
    test_engine()
