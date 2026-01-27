import argparse
import logging
import json
import re
import networkx as nx
from pathlib import Path
from conllu import parse_incr
from conllu.models import TokenList
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedProfiler:
    """
    Вычисляет метрики сложности и лингвистические особенности.
    """

    def __init__(self, is_cobald=False):
        self.is_cobald = is_cobald
        self.speech_verbs = {
            "сказать", "говорить", "ответить", "отвечать", "спросить", "спрашивать",
            "промолвить", "заявить", "подумать", "воскликнуть", "шептать", "пояснить"
        }
        self.abbr_pattern = re.compile(r'\b[А-ЯЁA-Z]\.')
        self.punct_cluster = re.compile(r'[.?!,:;-]{2,}')

    def profile(self, sentence: TokenList) -> dict:
        tokens = [t for t in sentence if isinstance(t['id'], int)]
        text = sentence.metadata.get("text", "")
        forms = [t['form'] for t in tokens]
        length = len(tokens)

        # 1. Синтаксис
        depth = self._get_depth(sentence)
        non_proj = self._is_non_projective(sentence)
        is_dialogue = self._is_dialogue(sentence)

        # 2. Токенизация / Лемматизация
        hyphen_count = sum(1 for w in forms if '-' in w and len(w) > 1)
        abbr_count = len(self.abbr_pattern.findall(text))
        punct_clusters = len(self.punct_cluster.findall(text))

        propn_count = sum(1 for t in tokens if t['upos'] == 'PROPN')
        pronoun_count = sum(1 for t in tokens if t['upos'] == 'PRON')
        max_token_len = max([len(w) for w in forms]) if forms else 0

        # 3. Интегральный скор
        syntax_complexity = depth + (10 if non_proj else 0)
        total_score = (
                syntax_complexity * 1.5 +
                (15.0 if is_dialogue else 0) +
                pronoun_count * 2.0 +
                (20.0 if self.is_cobald else 0)
        )

        return {
            "score": round(total_score, 2),
            "length": length,
            "depth": depth,
            "non_proj": non_proj,
            "syntax_complexity": syntax_complexity,
            "is_dialogue": is_dialogue,
            "pronouns": pronoun_count,
            "propn_ratio": round(propn_count / length, 3) if length else 0,
            "hyphen_density": round(hyphen_count / length, 3) if length else 0,
            "abbr_count": abbr_count,
            "punct_clusters": punct_clusters,
            "max_token_len": max_token_len
        }

    def _get_depth(self, sentence):
        edges = [(t['head'], t['id']) for t in sentence if isinstance(t['id'], int) and t['head']]
        if not edges: return 0
        try:
            return nx.dag_longest_path_length(nx.DiGraph(edges))
        except:
            return 0

    def _is_non_projective(self, sentence):
        arcs = sorted([(min(t['id'], t['head']), max(t['id'], t['head']))
                       for t in sentence if isinstance(t['id'], int) and t['head']])
        for i, (s1, e1) in enumerate(arcs):
            for s2, e2 in arcs[i + 1:]:
                if s1 < s2 < e1 < e2: return True
        return False

    def _is_dialogue(self, sentence):
        text = sentence.metadata.get("text", "").strip()
        has_dash = text.startswith(("-", "—", "–"))
        has_verb = any(t.get("lemma", "").lower() in self.speech_verbs for t in sentence)
        return has_dash and has_verb


class DataEnricher:
    def __init__(self, input_pattern, output_file):
        self.input_pattern = input_pattern
        self.output_file = Path(output_file)
        self.profiler = UnifiedProfiler(is_cobald="cobald" in str(output_file).lower())

    def run(self):
        base_dir = Path(".")
        if "**" in self.input_pattern:
            input_files = sorted(list(base_dir.glob(self.input_pattern)))
        else:
            p = Path(self.input_pattern)
            input_files = sorted(list(Path(p.parent).glob(p.name)))

        if not input_files:
            logger.error(f"Файлы не найдены по паттерну: {self.input_pattern}")
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Объединение {len(input_files)} файлов в {self.output_file}...")

        # Определяем префикс корпуса из имени выходного файла
        # Пример: cobald_full_enriched.conllu -> COBALD
        corpus_prefix = self.output_file.name.split('_')[0].upper()

        global_idx = 1  # Сквозной счетчик

        with open(self.output_file, "w", encoding="utf-8") as f_out:
            for in_file in input_files:
                logger.info(f"Обработка: {in_file.name}")
                with open(in_file, "r", encoding="utf-8") as f_in:
                    for sentence in parse_incr(f_in):
                        # 1. Профилируем
                        metrics = self.profiler.profile(sentence)

                        # 2. Добавляем метрики (сохраняя старые метаданные)
                        sentence.metadata["metrics"] = json.dumps(metrics)

                        # 3. Генерируем новый стандартизированный ID
                        # Формат: CORPUS_000001
                        new_id = f"{corpus_prefix}_{global_idx:07d}"
                        sentence.metadata["sent_id_new"] = new_id

                        # (Опционально) Можно добавить Source File для отладки
                        # sentence.metadata["source_file"] = in_file.name

                        # 4. Сериализация (автоматически сохранит sent_id_new как комментарий)
                        f_out.write(sentence.serialize())
                        global_idx += 1

        logger.info(f"Готово. Сохранено предложений: {global_idx - 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingress + Enrichment + ID Standardization")
    parser.add_argument("--input_pattern", required=True, help="Путь к raw файлам")
    parser.add_argument("--output_file", required=True, help="Путь к interim файлу")
    args = parser.parse_args()

    DataEnricher(args.input_pattern, args.output_file).run()
