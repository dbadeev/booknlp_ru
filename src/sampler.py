import json
import random
import logging
from pathlib import Path
from conllu import parse_incr

logger = logging.getLogger(__name__)


class StratifiedSampler:
    def __init__(self, profile_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        with open(profile_path, 'r', encoding='utf-8') as f:
            self.profiles = json.load(f)

    def _load_sentences(self, filename: str, target_ids: set) -> list:
        """
        Точечно считывает предложения из большого файла по ID.
        """
        found_sentences = []
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"Source file {filepath} not found for sampling.")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            for sentence in parse_incr(f):
                # Наш профилировщик добавлял префикс к ID, но в файле ID старые?
                # Нет, в задаче 1.1 мы уже сохранили файлы с префиксами в data/interim.
                # Поэтому ID должны совпадать полностью.
                sid = sentence.metadata.get('sent_id')
                if sid in target_ids:
                    found_sentences.append(sentence)

        return found_sentences

    def sample_dialogues(self, source_file: str, limit: int = 1000) -> list:
        """
        Стратегия Fiction_Dialogue: Топ предложений с диалогами из SynTagRus.
        """
        # Фильтруем профили: только из этого файла и is_dialogue=True
        candidates = [
            p for p in self.profiles.values()
            if p.get('source_file') == source_file and p.get('is_dialogue')
        ]

        # Если кандидатов больше лимита, берем случайные или самые длинные
        # Возьмем самые длинные для сложности
        candidates.sort(key=lambda x: x['text_len'], reverse=True)
        selected_profiles = candidates[:limit]

        target_ids = {p['id'] for p in selected_profiles}
        logger.info(f"Sampling {len(target_ids)} dialogue sentences from {source_file}...")

        return self._load_sentences(source_file, target_ids)

    def sample_complex(self, source_file: str, limit: int = 500) -> list:
        """
        Стратегия Fiction_Complex: Максимальная глубина дерева + непроективность.
        """
        candidates = [
            p for p in self.profiles.values()
            if p.get('source_file') == source_file
        ]

        # Сортировка: сначала непроективные, потом по глубине дерева
        # (True > False, поэтому non_projectivity идет первым ключом)
        candidates.sort(key=lambda x: (x['non_projectivity'], x['tree_depth']), reverse=True)

        selected_profiles = candidates[:limit]
        target_ids = {p['id'] for p in selected_profiles}

        logger.info(f"Sampling {len(target_ids)} complex sentences from {source_file}...")
        return self._load_sentences(source_file, target_ids)

    def sample_baseline(self, source_file: str, limit: int = 1000) -> list:
        """
        Стратегия General_News: Случайная выборка (Baseline).
        """
        candidates = [
            p for p in self.profiles.values()
            if p.get('source_file') == source_file
        ]

        selected_profiles = random.sample(candidates, k=min(limit, len(candidates)))
        target_ids = {p['id'] for p in selected_profiles}

        logger.info(f"Sampling {len(target_ids)} baseline sentences from {source_file}...")
        return self._load_sentences(source_file, target_ids)

    def save_dataset(self, sentences: list, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for s in sentences:
                f.write(s.serialize())