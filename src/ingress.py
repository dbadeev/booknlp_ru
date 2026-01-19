import os
import time
import logging
import requests
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


class IngressManager:
    def __init__(self, config):
        self.raw_dir = Path(config['dirs']['raw'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # [cite_start]
        self.chunk_size = 8192  # [cite: 58]

    def download_file(self, url: str, dest_folder: str, filename: str = None) -> Path:
        """
        Скачивает файл потоково с обработкой исключений и ретраями.
        """
        if not filename:
            filename = url.split('/')[-1]

        dest_path = self.raw_dir / dest_folder / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            logger.info(f"File {filename} already exists. Skipping.")
            return dest_path

        logger.info(f"Downloading {filename} from {url}...")

        # [cite_start]Реализация Retry logic (экспоненциальная задержка) [cite: 60]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))

                    with open(dest_path, 'wb') as f, tqdm(
                            desc=filename,
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            size = f.write(chunk)
                            bar.update(size)
                break  # Успех
            except requests.ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download {url} after {max_retries} attempts.")
                    raise e
                time.sleep(2 ** attempt)  # Экспоненциальная задержка

        return dest_path

    def run(self, sources_config):
        """Оркестратор загрузки всех источников"""
        downloaded_files = {}

        # [cite_start]1. SynTagRus [cite: 62]
        sr_conf = sources_config['syntagrus']
        sr_files = {}
        for key, url in sr_conf['urls'].items():
            sr_files[key] = self.download_file(url, "syntagrus")
        downloaded_files['syntagrus'] = sr_files

        # [cite_start]2. Taiga [cite: 64]
        tg_conf = sources_config['taiga']
        tg_files = {}
        for key, url in tg_conf['urls'].items():
            tg_files[key] = self.download_file(url, "taiga")
        downloaded_files['taiga'] = tg_files

        # [cite_start]3. CoBaLD [cite: 66]
        cb_conf = sources_config['cobald']
        cb_files = {}
        for key, url in cb_conf['urls'].items():
            cb_files[key] = self.download_file(url, "cobald")
        downloaded_files['cobald'] = cb_files

        return downloaded_files
