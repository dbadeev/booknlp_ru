#!/usr/bin/env python3
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.conllu_reader import load_gold_standard
from src.evaluation.compare_backends import Benchmarker
from src.engines.natasha_engine import NatashaPreprocessor
# Раскомментируйте, когда установите
# from src.engines.deeppavlov_engine import DeepPavlovEngine
# from src.engines.cobald_engine import CobaldEngine

from rich.console import Console

console = Console()

# ПУТЬ К ВАШЕМУ ФАЙЛУ
GOLD_FILE = "ru_syntagrus-ud-test.conllu"


def main():
    if not Path(GOLD_FILE).exists():
        console.print(f"[bold red]❌ Файл {GOLD_FILE} не найден![/]")
        console.print("Пожалуйста, положите файл ru_syntagrus-ud-test.conllu в папку со скриптом.")
        return

    # 1. Загрузка Золотого Стандарта
    console.print(f"[bold green]📂 Чтение {GOLD_FILE}...[/]")
    # Для теста берем первые 50 предложений, чтобы не ждать долго
    gold_data = load_gold_standard(GOLD_FILE, limit=50)
    console.print(f"✅ Загружено {len(gold_data)} эталонных предложений.")

    # 2. Инициализация движков
    engines = {}

    try:
        console.print("🏗️  Инициализация Natasha...")
        engines["Natasha (Baseline)"] = NatashaPreprocessor()
    except Exception as e:
        console.print(f"[red]Ошибка Natasha: {e}[/]")

    # try:
    #     console.print("🏗️  Инициализация DeepPavlov...")
    #     engines["DeepPavlov"] = DeepPavlovEngine()
    # except Exception as e:
    #     console.print(f"[red]Пропуск DeepPavlov (не установлен)[/]")

    # try:
    #     console.print("🏗️  Инициализация CoBaLD...")
    #     engines["CoBaLD"] = CobaldEngine()
    # except Exception as e:
    #      console.print(f"[red]Пропуск CoBaLD (не скачан)[/]")

    if not engines:
        console.print("❌ Нет активных движков для проверки!")
        return

    # 3. Запуск сравнения
    console.print("\n[bold yellow]🚀 Начинаем соревнование движков...[/]")
    bencher = Benchmarker(engines)

    # Этот метод (из предыдущего ответа) прогонит тексты через движки
    # и сравнит их с gold_data посимвольно (Intersection over Union)
    df = bencher.run(gold_data)

    # 4. Результат
    console.print("\n[bold]📊 Итоговая таблица:[/]")
    console.print(df.to_markdown())

    # Сохранить отчет
    df.to_csv("benchmark_results.csv")
    console.print("\n💾 Отчет сохранен в benchmark_results.csv")


if __name__ == "__main__":
    main()
