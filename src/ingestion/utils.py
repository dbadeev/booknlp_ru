# src/ingestion/utils.py
import requests
import logging
from pathlib import Path
from tqdm import tqdm  # Для отображения прогресса

logger = logging.getLogger(__name__)


def get_raw_github_url(user: str, repo: str, branch: str, filepath: str) -> str:
    """
    Генерирует прямой URL к raw-содержимому файла на GitHub.
    """
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """
    Скачивает файл с отображением прогресса.

    Args:
        url: URL файла.
        dest_path: Локальный путь для сохранения.
        force: Если True, перезаписывает существующий файл.

    Returns:
        Path к скачанному файлу.
    """
    if dest_path.exists() and not force:
        logger.info(f"Файл {dest_path.name} уже существует. Пропуск скачивания.")
        return dest_path

    logger.info(f"Начинаю скачивание: {url}")

    # Имитируем браузер, чтобы избежать блокировок от GitHub
    headers = {'User-Agent': 'BookNLP_Ru_Ingress_Bot/0.1'}

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()

            # Получаем размер файла для прогресс-бара
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB chunk

            with open(dest_path, 'wb') as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    size = f.write(chunk)
                    bar.update(size)

        logger.info(f"Скачивание завершено: {dest_path}")
        return dest_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при скачивании {url}: {e}")
        # Удаляем частично скачанный (битый) файл
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Не удалось скачать {url}") from e
    