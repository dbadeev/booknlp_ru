# src/config.py
import os
from pathlib import Path

# Определение базовых путей относительно корня проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

# Автоматическое создание директорий
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Конфигурация источников данных
# Используем списки файлов для поддержки фрагментированных корпусов (train-a, train-b...)
CORPORA_CONFIG = {
    "SynTagRus": {
        "type": "UD",
        "repo_user": "UniversalDependencies",
        "repo_name": "UD_Russian-SynTagRus",
        "branch": "master", # Рекомендуется зафиксировать commit hash для продакшена
        "files": {
            "train": [
                "ru_syntagrus-ud-train-a.conllu",
                "ru_syntagrus-ud-train-b.conllu",
                "ru_syntagrus-ud-train-c.conllu"
            ],
            "dev": ["ru_syntagrus-ud-dev.conllu"],
            "test": ["ru_syntagrus-ud-test.conllu"]
        },
        "description": "Основной корпус UD для русского языка."
    },
    "Taiga": {
        "type": "UD",
        "repo_user": "UniversalDependencies",
        "repo_name": "UD_Russian-Taiga",
        "branch": "master",
        "files": {
            "train": ["ru_taiga-ud-train.conllu"], # Проверено: в новых версиях файл часто один, но может меняться
            "dev": ["ru_taiga-ud-dev.conllu"],
            "test": ["ru_taiga-ud-test.conllu"]
        },
        "validation_level": "lenient", # Допускаем ошибки из-за специфики соцсетей
        "description": "Корпус Taiga: соцсети, поэзия, нестандартная лексика."
    },
    "CoBaLD": {
        "type": "CoNLL-Plus",
        "repo_user": "CobaldAnnotation",
        "repo_name": "CobaldRus",
        "branch": "main", # Внимание: ветка может называться main или master в зависимости от репо
        "files": {
            "train": ["train.conllu"],
            "dev": ["dev.conllu"]
        },
        # Спецификация дополнительных колонок для CoNLL-U Plus
        # Стандартные 10 + SEM:NE + SEM:COREF (пример)
        "extra_fields": [12],
        "description": "Семантически размеченный корпус с эллипсисом."
    }
}
