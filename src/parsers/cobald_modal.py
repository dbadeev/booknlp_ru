import modal
import logging
import sys
import os

# Пути
LOCAL_COBALD_DIR = "src/cobald_parser"
REMOTE_ROOT = "/root/booknlp_ru"
REMOTE_SRC = f"{REMOTE_ROOT}/src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.0.0", "transformers", "huggingface_hub", "numpy", "razdel")
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_SRC}:$PYTHONPATH"})
    # Копируем исправленные локальные файлы
    .add_local_dir(LOCAL_COBALD_DIR, remote_path=f"{REMOTE_SRC}/cobald_parser", copy=True)
)

app = modal.App("booknlp-ru-cobald")


@app.cls(image=image, gpu="T4", timeout=600)
class CobaldService:
    @modal.enter()
    def setup(self):
        import torch
        import logging
        import sys

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CobaldService")

        if REMOTE_ROOT not in sys.path: sys.path.append(REMOTE_ROOT)
        if REMOTE_SRC not in sys.path: sys.path.append(REMOTE_SRC)

        try:
            from src.cobald_parser.modeling_parser import CobaldParser
            from src.cobald_parser.configuration import CobaldParserConfig
        except ImportError as e:
            self.logger.error(f"Import failed: {e}")
            raise e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "CoBaLD/xlm-roberta-base-cobald-parser-ru"

        try:
            config = CobaldParserConfig.from_pretrained(model_name)
            self.model = CobaldParser.from_pretrained(model_name, config=config)
            self.model.to(self.device)
            self.model.eval()
            self.vocab_deprel = config.vocabulary.get("ud_deprel", {})
            self.logger.info("🚀 CoBaLD loaded with MODELING_PARSER PATCH!")
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise e

    @modal.method()
    def parse(self, tokens: list[str]):
        import torch
        if not tokens: return []

        try:
            with torch.no_grad():
                # Подаем чистые токены. modeling_parser.py сам добавит [CLS]
                output = self.model(words=[tokens], inference_mode=True)

            result = []
            syntax_dict = {}

            if "deps_ud" in output and output["deps_ud"] is not None:
                deps = output["deps_ud"]
                current_deps = deps[deps[:, 0] == 0].cpu().numpy()

                for row in current_deps:
                    head_idx = int(row[1])
                    dep_idx = int(row[2])
                    rel_id = int(row[3])
                    deprel_str = self.vocab_deprel.get(rel_id, "dep")

                    # Логика модели с добавленным [CLS]:
                    # Index 0 = [CLS] (ROOT)
                    # Index 1 = Word 1
                    # ...
                    # dep_idx > 0 пропускает сам [CLS], берем только слова
                    if dep_idx > 0:
                        # dep_idx-1 - это индекс слова в исходном списке tokens
                        # head_idx уже в формате CoNLL-U (0=ROOT, 1=Word1)
                        syntax_dict[dep_idx - 1] = (head_idx, deprel_str)

            for i, token_text in enumerate(tokens):
                head, deprel = syntax_dict.get(i, (0, "root"))

                # Защита от self-loop (ссылка на самого себя = root)
                if head == (i + 1):
                    head = 0

                result.append({
                    "id": i + 1,
                    "form": token_text,
                    "head": head,
                    "deprel": deprel,
                    "start_char": 0, "end_char": 0
                })

            return result

        except Exception as e:
            print(f"Error parsing: {e}")
            return []