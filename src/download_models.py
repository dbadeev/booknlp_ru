import os
from navec import Navec
from slovnet import NER
from urllib.request import urlretrieve

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Примерные URL, заменить на актуальные под проект:
NAVEC_URL = "https://example.com/navec_model.tar"
SLOVNET_NER_URL = "https://example.com/slovnet_ner.tar"

def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} -> {path}")
        urlretrieve(url, path)
    else:
        print(f"Already exists: {path}")

def main():
    navec_path = os.path.join(MODELS_DIR, "navec_model.tar")
    slovnet_ner_path = os.path.join(MODELS_DIR, "slovnet_ner.tar")

    download(NAVEC_URL, navec_path)
    download(SLOVNET_NER_URL, slovnet_ner_path)

if __name__ == "__main__":
    main()
