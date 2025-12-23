#!/usr/bin/env python3.11
"""
Точка входа BookNLP-ru Core. INFRA-001 smoke-test.
"""
from rich.console import Console
from rich.panel import Panel

console = Console()


def smoke_test():
    """Проверка работоспособности основного стека."""
    try:
        from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser
        from razdel import sentenize
        from navec import Navec

        console.print("✅ [green]Natasha/Slovnet/Razdel/Navec[/green] импортированы")
        console.print(Panel("INFRA-001: Локальное окружение готово!", border_style="green"))
        return True
    except ImportError as e:
        console.print(f"❌ [red]Ошибка импорта: {e}[/red]")
        return False


if __name__ == "__main__":
    smoke_test()
