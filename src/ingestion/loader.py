# src/ingestion/loader.py
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Generator, Dict, Any
from conllu import parse_incr
import src.config as config
from src.ingestion.utils import download_file, get_raw_github_url
from src.ingestion.validators import DataValidator

logger = logging.getLogger(__name__)


class AbstractCorpusLoader(ABC):
    """Базовый класс для всех загрузчиков корпусов."""

    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.config = cfg
        self.raw_dir = config.RAW_DIR / name
        self.raw_dir.mkdir(exist_ok=True)
        # Определение уровня строгости валидации
        self.strict_validation = cfg.get("validation_level", "strict") == "strict"

    @abstractmethod
    def download(self) -> Dict[str, List[Path]]:
        """Скачивает файлы. Возвращает map: split -> list[paths]."""
        pass

    def load_stream(self, file_paths: List[Path]) -> Generator[Any, None, None]:
        """
        Потоковый генератор валидированных предложений.
        """
        # Извлекаем кастомные поля для CoNLL-Plus, если есть
        extra_fields = self.config.get("extra_fields", None)

        for fp in file_paths:
            logger.info(f"Парсинг файла: {fp.name}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    # parse_incr - ключевой элемент для экономии памяти
                    # Он читает файл лениво.
                    iterator = parse_incr(f, fields=extra_fields)

                    for token_list in iterator:
                        # Валидация "на лету"
                        val_res = DataValidator.validate_sentence(
                            token_list,
                            strict=self.strict_validation
                        )

                        if val_res.is_valid:
                            yield token_list
                        else:
                            # Логируем, но не падаем (Robustness pattern)
                            sid = token_list.metadata.get('sent_id', 'UNKNOWN')
                            logger.warning(f"Skipped invalid sentence {sid} in {fp.name}: {val_res.errors}")

            except Exception as e:
                logger.error(f"Критическая ошибка при чтении {fp}: {e}")
                # Здесь можно решить: падать или пропускать файл.
                # Для Data Engineering лучше упасть, чтобы не потерять данные тихо.
                raise


class UDCorpusLoader(AbstractCorpusLoader):
    """Загрузчик для стандартных UD и CoNLL-Plus репозиториев."""

    def download(self) -> Dict[str, List[Path]]:
        downloaded_map = {}

        files_config = self.config.get("files", {})
        repo_user = self.config["repo_user"]
        repo_name = self.config["repo_name"]
        branch = self.config["branch"]

        for split_name, filenames in files_config.items():
            paths =  [] # Исправлено: инициализация пустым списком
            for fname in filenames:
                url = get_raw_github_url(repo_user, repo_name, branch, fname)
                dest = self.raw_dir / fname
                try:
                    # Скачиваем (с кешированием)
                    p = download_file(url, dest)
                    paths.append(p)
                except Exception as e:
                    logger.warning(f"Не удалось скачать {fname} для {split_name}. Пропуск.")

            if paths:
                downloaded_map[split_name] = paths

        return downloaded_map


class IngressPipeline:
    """Фасад для запуска всего процесса загрузки."""

    def __init__(self):
        self.loaders = {}
        # Инициализация загрузчиков из конфига
        for name, cfg in config.CORPORA_CONFIG.items():
            # В текущей реализации UDCorpusLoader универсален благодаря
            # параметризации extra_fields в AbstractCorpusLoader
            self.loaders[name] = UDCorpusLoader(name, cfg)

    def run(self):
        logger.info("Запуск пайплайна Ingress...")

        for name, loader in self.loaders.items():
            logger.info(f"=== Обработка корпуса: {name} ===")

            # 1. Скачивание
            files_map = loader.download()

            # 2. Обработка и сохранение нормализованных данных
            for split, paths in files_map.items():
                if not paths:
                    continue

                # Путь для объединенного файла (например, SynTagRus_train.conllu)
                output_path = config.INTERIM_DIR / f"{name}_{split}.conllu"
                logger.info(f"Генерация {output_path}...")

                count = 0
                with open(output_path, "w", encoding="utf-8") as out:
                    for sent in loader.load_stream(paths):
                        # Сериализация обратно в текст
                        out.write(sent.serialize())
                        count += 1

                logger.info(f"Сохранено {count} предложений в {output_path.name}")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = IngressPipeline()
    pipeline.run()