import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def main():
    models_dir = Path("models")

    # 1. Navec (Word Embeddings)
    navec_url = "https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar"
    download_file(navec_url, models_dir / "navec_news_v1_1B_250K_300d_100q.tar")

    # 2. Slovnet (Syntax Parser)
    slovnet_url = "https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar"
    download_file(slovnet_url, models_dir / "slovnet_syntax_news_v1.tar")

    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
