# #!/usr/bin/env python3.11
# """
# Скачивание SynTagRus корпуса из GitHub.
# UD Russian-SynTagRus: https://github.com/UniversalDependencies/UD_Russian-SynTagRus
# """
#
# import os
# import subprocess
# from pathlib import Path
#
# SYNTAGRUS_URL = "https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git"
# DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# SYNTAGRUS_DIR = DATA_DIR / "syntagrus_old"
#
#
# def download_syntagrus():
#     """Клонировать SynTagRus репозиторий."""
#     print(f"Скачивание SynTagRus в {SYNTAGRUS_DIR}...")
#
#     # if SYNTAGRUS_DIR.exists():
#     #     print(f"⚠️  {SYNTAGRUS_DIR} уже существует. Пропускаем скачивание.")
#     #     return
#
#     try:
#         subprocess.run(
#             ["git", "clone", "--depth", "1", SYNTAGRUS_URL, str(SYNTAGRUS_DIR)],
#             check=True,
#             capture_output=True
#         )
#         print(f"✅ SynTagRus успешно скачан в {SYNTAGRUS_DIR}")
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Ошибка при скачивании: {e.stderr.decode()}")
#         raise
#
#
# def list_syntagrus_files():
#     """Показать доступные файлы."""
#     if not SYNTAGRUS_DIR.exists():
#         print("❌ SynTagRus не скачан. Запусти download_syntagrus() первым.")
#         return
#
#     conllu_files = list(SYNTAGRUS_DIR.glob("*.conllu"))
#     print(f"\n📄 Доступные файлы в {SYNTAGRUS_DIR}:")
#     for f in sorted(conllu_files):
#         size_mb = f.stat().st_size / (1024 * 1024)
#         print(f"  - {f.name} ({size_mb:.1f} MB)")
#
#
# if __name__ == "__main__":
#     download_syntagrus()
#     list_syntagrus_files()

#   ====================================================================
#   ====================================================================
#   ====================================================================

# !/usr/bin/env python3.11
"""
Скачивание SynTagRus корпуса из GitHub.
UD Russian-SynTagRus: https://github.com/UniversalDependencies/UD_Russian-SynTagRus
"""

import os
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

SYNTAGRUS_URL = "https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SYNTAGRUS_DIR = DATA_DIR / "syntagrus"


def download_syntagrus():
    """Клонировать SynTagRus репозиторий."""
    console.print(f"📥 Скачивание SynTagRus в {SYNTAGRUS_DIR}...")

    # if SYNTAGRUS_DIR.exists():
    #     console.print(f"⚠️  {SYNTAGRUS_DIR} уже существует. Пропускаем скачивание.")
    #     list_syntagrus_files()
    #     return

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", SYNTAGRUS_URL, str(SYNTAGRUS_DIR)],
            check=True,
            capture_output=True
        )
        console.print(f"✅ SynTagRus успешно скачан в {SYNTAGRUS_DIR}")
        list_syntagrus_files()
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Ошибка при скачивании: {e.stderr.decode()}")
        raise


def list_syntagrus_files():
    """Показать доступные файлы и их размеры."""
    if not SYNTAGRUS_DIR.exists():
        console.print("❌ SynTagRus не скачан. Запусти download_syntagrus() первым.")
        return

    console.print(f"\n📄 Доступные файлы в {SYNTAGRUS_DIR}:\n")

    conllu_files = sorted(SYNTAGRUS_DIR.glob("*.conllu"))

    if not conllu_files:
        console.print("  (нет .conllu файлов)")
        return

    # Показать файлы по типам
    test_files = [f for f in conllu_files if 'test' in f.name]
    dev_files = [f for f in conllu_files if 'dev' in f.name]
    train_files = [f for f in conllu_files if 'train' in f.name]

    if train_files:
        console.print("[bold]🚂 TRAIN файлы:[/bold]")
        for f in train_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines = len(open(f).readlines())
            console.print(f"  • {f.name:50} ({size_mb:6.1f} MB, {lines:7} lines)")

    if dev_files:
        console.print("\n[bold]📊 DEV файлы:[/bold]")
        for f in dev_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines = len(open(f).readlines())
            console.print(f"  • {f.name:50} ({size_mb:6.1f} MB, {lines:7} lines)")

    if test_files:
        console.print("\n[bold]✅ TEST файлы:[/bold]")
        for f in test_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines = len(open(f).readlines())
            console.print(f"  • {f.name:50} ({size_mb:6.1f} MB, {lines:7} lines)")

    # Рекомендация для бенчмарка
    console.print(f"\n[bold cyan]💡 Для бенчмарка используй:[/bold cyan]")
    if test_files:
        test_file = test_files[0]
        console.print(f"  → {test_file.name}")

    if train_files:
        console.print(f"\n[bold cyan]📚 Для тренировки доступны:[/bold cyan]")
        for f in train_files:
            console.print(f"  → {f.name}")


if __name__ == "__main__":
    download_syntagrus()
