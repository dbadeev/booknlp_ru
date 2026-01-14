#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.engines.deeppavlov_engine import DeepPavlovEngine


def main():
    print("🚀 Запуск теста DeepPavlov Engine...")

    # Флаг install=True запустит скачивание 700Мб моделей, если их нет
    # В продакшене лучше делать это через 'python -m deeppavlov install ...' в Dockerfile
    try:
        engine = DeepPavlovEngine(install=False)
    except Exception:
        print("\n⚠️  Похоже, модель не установлена.")
        print("   Попробуйте запустить: python -m deeppavlov install ru_syntagrus_joint_parsing")
        return

    # Тест на сложном случае с дефисом (проверка detokenization mapping)
    text = "Мы кое-как добрались до Санкт-Петербурга."

    print(f"\nProcessing: '{text}'")
    sentences = engine.process(text)

    print(f"{'ID':<3} {'TEXT':<15} {'POS':<6} {'HEAD':<5} {'REL':<10} {'SPAN':<10}")
    print("-" * 60)

    for sent in sentences:
        for t in sent:
            print(f"{t.id:<3} {t.text:<15} {t.pos:<6} {t.head_id:<5} {t.rel:<10} {t.char_start}-{t.char_end}")


if __name__ == "__main__":
    main()
