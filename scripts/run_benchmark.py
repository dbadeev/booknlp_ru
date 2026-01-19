# import sys
# import os
# import time
# import logging
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# from conllu import parse_incr
#
# # Добавляем корень проекта в путь, чтобы видеть модули src
# sys.path.append(str(Path(__file__).parent.parent))
#
# from src.pipeline import BookNLP
# from src.evaluation.alignment import FuzzyAligner
# from src.evaluation.metrics import MetricsCalculator
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class BenchmarkRunner:
#     def __init__(self, data_path: str, output_file: str = "benchmark_results.csv"):
#         self.data_path = Path(data_path)
#         self.output_file = Path(output_file)
#         self.aligner = FuzzyAligner()
#         self.calculator = MetricsCalculator()
#         self.results = []
#
#     def load_gold_data(self) -> list:
#         """Читает Gold-файл (conllu) и преобразует в список предложений."""
#         sentences = []
#         if not self.data_path.exists():
#             raise FileNotFoundError(f"Gold dataset not found: {self.data_path}")
#
#         with open(self.data_path, "r", encoding="utf-8") as f:
#             for sent in parse_incr(f):
#                 # Извлекаем токены в формате словарей
#                 tokens = []
#                 text = sent.metadata.get("text", "")
#
#                 # Если в conllu нет явных оффсетов, их надо восстановить по тексту.
#                 # Для упрощения используем reconstruct_gold_offsets из aligner'а
#                 # Сначала собираем базовые токены
#                 for t in sent:
#                     tokens.append({
#                         "id": t["id"],
#                         "form": t["form"],
#                         "head_id": t["head"],
#                         "deprel": t["deprel"]
#                     })
#
#                 # Обогащаем оффсетами
#                 rich_tokens = self.aligner.reconstruct_gold_offsets(tokens, text)
#
#                 sentences.append({
#                     "id": sent.metadata.get("sent_id"),
#                     "text": text,
#                     "tokens": rich_tokens
#                 })
#         return sentences
#
#     def run_model(self, model_name: str, model_type: str, gold_sentences: list):
#         """Прогон одной модели по всему датасету."""
#         logger.info(f"--- Starting benchmark for model: {model_name} ({model_type}) ---")
#
#         try:
#             pipeline = BookNLP(model_type=model_type)
#         except Exception as e:
#             logger.error(f"Failed to load {model_name}: {e}")
#             return
#
#         start_time = time.time()
#         processed_count = 0
#         total_uas = 0.0
#         total_las = 0.0
#         total_seg_f1 = 0.0
#
#         # tqdm для прогресса
#         for gold_sent in tqdm(gold_sentences, desc=f"Benchmarking {model_name}"):
#             text = gold_sent["text"]
#             if not text:
#                 continue
#
#             # 1. Инференс
#             try:
#                 # process возвращает список предложений (мы подаем одно)
#                 sys_output = pipeline.process(text)
#                 if not sys_output:
#                     continue
#
#                 # Берем первое предложение (предполагаем 1-к-1 маппинг для теста)
#                 sys_sent = sys_output[0]
#                 sys_tokens = sys_sent["tokens"]
#
#                 # 2. Выравнивание
#                 gold_tokens = gold_sent["tokens"]
#                 alignments = self.aligner.align(sys_tokens, gold_tokens)
#
#                 # 3. Метрики (Синтаксис)
#                 syntax_metrics = self.calculator.calc_soft_metrics(alignments, sys_tokens, gold_tokens)
#
#                 # 4. Метрики (Сегментация)
#                 sys_starts = [t['start_char'] for t in sys_tokens]
#                 gold_starts = [t['start_char'] for t in gold_tokens]
#                 seg_metrics = self.calculator.calc_segmentation_f1(sys_starts, gold_starts)
#
#                 # Накопление (взвешиваем по количеству токенов или просто среднее по предложениям?
#                 # Для простоты считаем Micro-average потом, здесь копим суммы, но сейчас Macro-average по предложениям)
#                 total_uas += syntax_metrics['soft_uas']
#                 total_las += syntax_metrics['soft_las']
#                 total_seg_f1 += seg_metrics.f1
#                 processed_count += 1
#
#             except Exception as e:
#                 logger.warning(f"Error processing sent {gold_sent['id']}: {e}")
#
#         total_time = time.time() - start_time
#         speed = processed_count / total_time if total_time > 0 else 0
#
#         # Усреднение
#         avg_uas = total_uas / processed_count if processed_count > 0 else 0
#         avg_las = total_las / processed_count if processed_count > 0 else 0
#         avg_seg_f1 = total_seg_f1 / processed_count if processed_count > 0 else 0
#
#         logger.info(f"{model_name} Results: LAS={avg_las:.3f}, Speed={speed:.1f} sent/sec")
#
#         self.results.append({
#             "Model": model_name,
#             "Sentences": processed_count,
#             "Speed (sent/sec)": round(speed, 1),
#             "Segmentation F1": round(avg_seg_f1, 3),
#             "Soft UAS": round(avg_uas, 3),
#             "Soft LAS": round(avg_las, 3)
#         })
#
#     def save_report(self):
#         df = pd.DataFrame(self.results)
#         print("\n=== Benchmark Report ===")
#         print(df.to_markdown(index=False))
#         df.to_csv(self.output_file, index=False)
#         logger.info(f"Report saved to {self.output_file}")
#
#
# def main():
#     # Путь к "сложному" датасету из Спринта 1
#     gold_path = "data/processed/val_complex.conllu"
#     # Если его нет, пробуем бейзлайн
#     if not Path(gold_path).exists():
#         gold_path = "data/processed/val_baseline.conllu"
#
#     runner = BenchmarkRunner(gold_path)
#     gold_data = runner.load_gold_data()
#
#     # Чтобы не ждать вечность, для теста можно взять срез (например, 50 предложений)
#     # gold_data = gold_data[:50]
#
#     # 1. Запуск Slovnet
#     runner.run_model("Slovnet (Fast)", "fast", gold_data)
#
#     # 2. Запуск DeepPavlov (если есть доступ к Modal)
#     # Можно закомментировать, если тестируем только локально
#     runner.run_model("DeepPavlov (Accurate)", "accurate", gold_data)
#
#     runner.save_report()
#
#
# if __name__ == "__main__":
#     main()

import modal
import sys
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr

# Ensure src modules are visible
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.alignment import FuzzyAligner
from src.evaluation.metrics import MetricsCalculator
from src.segmentation import RazdelSegmenter

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")

# ==========================================
# 1. MODAL IMAGES (ENVIRONMENTS)
# ==========================================

# Image A: Legacy DeepPavlov (Requires older TF)
dp_image = (
    modal.Image.debian_slim()
    .apt_install("git", "g++")
    .pip_install(
        "deeppavlov",
        "tensorflow==1.15.5",
        "pandas",
        "bert-dp"
    )
    .run_commands("python -m deeppavlov install syntax_ru_syntagrus_bert")
)

# Image B: Modern SOTA (Stanza, Trankit, UDPipe, CoBaLD)
# Added 'swig' and 'build-essential' for ufal.udpipe compilation
modern_image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "build-essential", "swig", "g++")
    .pip_install(
        "torch",
        "stanza",
        "trankit",
        "ufal.udpipe",
        "transformers",
        "huggingface_hub"
    )
    # Pre-download models to speed up runtime
    .run_commands(
        "python -c 'import stanza; stanza.download(\"ru\")'",
        "python -c 'import trankit; trankit.Pipeline(\"russian\", gpu=False, cache_dir=\"/root/.trankit\")'",
        # Download UDPipe model manually
        "curl -L -o /root/russian-syntagrus-ud-2.5.udpipe https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-syntagrus-ud-2.5-191206.udpipe"
    )
)

app = modal.App("booknlp-benchmark-full")


# ==========================================
# 2. REMOTE PARSERS (SERVICES)
# ==========================================

@app.cls(image=dp_image, gpu="T4", timeout=600)
class DeepPavlovService:
    @modal.enter()
    def load(self):
        from deeppavlov import build_model, configs
        self.model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

    @modal.method()
    def parse(self, text: str):
        # DeepPavlov syntax model expects tokenized input
        tokens = [t.text for t in list(RazdelSegmenter().tokenize(text))]
        if not tokens: return []

        try:
            result = self.model([tokens])[0]
            parsed_sent = []
            lines = result.split('\n')
            for line in lines:
                parts = line.split('\t')
                if len(parts) < 10: continue
                parsed_sent.append({
                    "id": int(parts[0]),
                    "form": parts[1],
                    "head": int(parts[6]),
                    "deprel": parts[7]
                })
            return [parsed_sent]
        except Exception as e:
            print(f"DP Error: {e}")
            return []


@app.cls(image=modern_image, gpu="T4", timeout=1200)
class ModernParserService:
    @modal.enter()
    def load(self):
        import stanza
        import trankit
        from ufal.udpipe import Model, Pipeline, ProcessingError

        print("Loading Stanza...")
        self.stanza_nlp = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=True)

        print("Loading Trankit...")
        self.trankit_nlp = trankit.Pipeline('russian', gpu=True, cache_dir='/root/.trankit')

        print("Loading UDPipe...")
        self.udpipe_model = Model.load("/root/russian-syntagrus-ud-2.5.udpipe")
        self.udpipe_pipeline = Pipeline(self.udpipe_model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    @modal.method()
    def run_stanza(self, text: str):
        try:
            doc = self.stanza_nlp(text)
            result = []
            for sent in doc.sentences:
                sent_res = []
                for word in sent.words:
                    sent_res.append({
                        "id": word.id,
                        "form": word.text,
                        "head": word.head,
                        "deprel": word.deprel
                    })
                result.append(sent_res)
            return result
        except:
            return []

    @modal.method()
    def run_trankit(self, text: str):
        try:
            doc = self.trankit_nlp(text)
            result = []
            if 'sentences' in doc:
                for sent in doc['sentences']:
                    sent_res = []
                    for token in sent['tokens']:
                        # Trankit sometimes uses 'id': 1 or 'id': (1, 2) for MWT
                        tid = token['id']
                        if isinstance(tid, list) or isinstance(tid, tuple): tid = tid[0]

                        sent_res.append({
                            "id": int(tid),
                            "form": token['text'],
                            "head": int(token.get('head', 0)),
                            "deprel": token.get('deprel', 'root')
                        })
                    result.append(sent_res)
            return result
        except:
            return []

    @modal.method()
    def run_udpipe(self, text: str):
        try:
            processed = self.udpipe_pipeline.process(text)
            # Parse CoNLL format output manually
            result = []
            current_sent = []
            for line in processed.split('\n'):
                if line.startswith('#'): continue
                if not line.strip():
                    if current_sent:
                        result.append(current_sent)
                        current_sent = []
                    continue
                parts = line.split('\t')
                if len(parts) < 10: continue
                current_sent.append({
                    "id": int(parts[0]),
                    "form": parts[1],
                    "head": int(parts[6]),
                    "deprel": parts[7]
                })
            if current_sent: result.append(current_sent)
            return result
        except:
            return []

    @modal.method()
    def run_cobald(self, text: str):
        # Placeholder: CoBaLD architecture requires complex setup (custom repo clone).
        # Returning empty to indicate it's implemented in interface but skipped in this run
        # to prevent timeouts.
        return []


# ==========================================
# 3. LOCAL BENCHMARK RUNNER
# ==========================================

class BenchmarkRunner:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.aligner = FuzzyAligner()
        self.calculator = MetricsCalculator()
        self.results = []

        # Local Slovnet
        from src.pipeline import BookNLP
        self.slovnet_pipeline = BookNLP(model_type="fast")

    def load_gold(self):
        sentences = []
        if not self.data_path.exists():
            print(f"Warning: {self.data_path} not found. Creating dummy data.")
            return [{"id": "dummy", "text": "Мама мыла раму.",
                     "tokens": [{"id": 1, "form": "Мама", "head": 2, "deprel": "nsubj", "start_char": 0, "end_char": 4},
                                {"id": 2, "form": "мыла", "head": 0, "deprel": "root", "start_char": 5, "end_char": 9},
                                {"id": 3, "form": "раму", "head": 2, "deprel": "obj", "start_char": 10, "end_char": 14},
                                {"id": 4, "form": ".", "head": 2, "deprel": "punct", "start_char": 14,
                                 "end_char": 15}]}]

        with open(self.data_path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                text = sent.metadata.get("text", "")
                if not text: continue
                tokens = []
                for t in sent:
                    tokens.append({"id": t["id"], "form": t["form"], "head_id": t["head"], "deprel": t["deprel"]})

                rich = self.aligner.reconstruct_gold_offsets(tokens, text)
                sentences.append({"text": text, "tokens": rich, "id": sent.metadata.get("sent_id")})
        return sentences

    def evaluate_prediction(self, model_name, sys_batch, gold_sent):
        if not sys_batch: return 0, 0, 0

        # Take first sentence
        sys_tokens = sys_batch[0]

        clean_sys = []
        for t in sys_tokens:
            clean_sys.append({
                "id": t['id'],
                "form": t['form'],
                "start_char": 0,
                "end_char": 0,
                "head_id": t['head'],
                "rel": t['deprel']
            })

        clean_sys = self.aligner.reconstruct_gold_offsets(clean_sys, gold_sent['text'])
        alignments = self.aligner.align(clean_sys, gold_sent['tokens'])
        metrics = self.calculator.calc_soft_metrics(alignments, clean_sys, gold_sent['tokens'])

        return metrics['soft_uas'], metrics['soft_las']

    def run_all(self):
        gold_data = self.load_gold()[:15]  # Benchmark on 15 sentences for speed

        print(f"Loaded {len(gold_data)} sentences for benchmarking.")

        # 1. Slovnet (Local)
        self._run_local_slovnet(gold_data)

        # 2. Remote Models
        print("\nStarting Remote Benchmark (DeepPavlov, Stanza, Trankit, UDPipe)...")
        with app.run():  # Starts Modal context
            dp_service = DeepPavlovService()
            modern_service = ModernParserService()

            self._run_remote(gold_data, dp_service.parse, "DeepPavlov")
            self._run_remote(gold_data, modern_service.run_stanza, "Stanza")
            self._run_remote(gold_data, modern_service.run_trankit, "Trankit")
            self._run_remote(gold_data, modern_service.run_udpipe, "UDPipe 2.5")
            # CoBaLD skipped to save time, but interface is ready:
            # self._run_remote(gold_data, modern_service.run_cobald, "CoBaLD")

    def _run_local_slovnet(self, gold_data):
        logger.info("Benchmarking Slovnet...")
        scores = {"uas": [], "las": [], "times": []}
        for sent in tqdm(gold_data):
            start = time.time()
            res = self.slovnet_pipeline.process(sent['text'])
            dur = time.time() - start

            if res:
                sys_tokens = []
                for t in res[0]['tokens']:
                    sys_tokens.append({"id": t['id'], "head": t['head_id'], "deprel": t['rel'], "form": t['text']})
                uas, las = self.evaluate_prediction("Slovnet", [sys_tokens], sent)
                scores['uas'].append(uas)
                scores['las'].append(las)
                scores['times'].append(dur)

        self._log_result("Slovnet", scores)

    def _run_remote(self, gold_data, remote_method, name):
        logger.info(f"Benchmarking {name}...")
        scores = {"uas": [], "las": [], "times": []}

        # Process sentence by sentence (can be batched for speed)
        for sent in tqdm(gold_data):
            try:
                start = time.time()
                res = remote_method.remote(sent['text'])
                dur = time.time() - start

                uas, las = self.evaluate_prediction(name, res, sent)
                scores['uas'].append(uas)
                scores['las'].append(las)
                scores['times'].append(dur)
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        self._log_result(name, scores)

    def _log_result(self, name, scores):
        if not scores['las']:
            print(f"--> {name}: No results")
            return
        avg_las = sum(scores['las']) / len(scores['las'])
        avg_speed = len(scores['times']) / sum(scores['times'])
        self.results.append({"Model": name, "LAS": avg_las, "Speed": avg_speed})
        print(f"--> {name}: LAS={avg_las:.3f}, Speed={avg_speed:.1f} sent/sec")


if __name__ == "__main__":
    runner = BenchmarkRunner("data/processed/val_complex.conllu")
    # Show logs from Modal image building
    modal.enable_output()
    runner.run_all()

    df = pd.DataFrame(runner.results)
    if not df.empty:
        print("\n=== FINAL BENCHMARK RESULTS ===")
        # Fallback if tabulate missing
        try:
            print(df.to_markdown(index=False))
        except:
            print(df.to_string(index=False))
        df.to_csv("benchmark_results_full.csv", index=False)