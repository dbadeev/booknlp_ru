import torch
from typing import List, Dict
from razdel import sentenize, tokenize

from src.core.interfaces import BasePreprocessor
from src.core.data_structures import Token
from src.cobald_parser.modeling_parser import CobaldParser
from src.cobald_parser.configuration import CobaldParserConfig
from transformers import AutoTokenizer


class CobaldPreprocessor(BasePreprocessor):
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"🧠 Загрузка CoBaLD Parser из {model_path}...")
        self.device = device

        # 1. Загрузка конфигурации и модели
        # Используем классы из загруженных вами файлов
        self.config = CobaldParserConfig.from_pretrained(model_path)
        self.model = CobaldParser.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()

        # 2. Словари для декодирования меток (из конфига модели)
        # Формат словарей в конфиге: {"0": "nsubj", "1": "root"...}
        self.vocab = self.config.vocabulary

        print("✅ CoBaLD готов (Синтаксис + Семантика)")

    def process(self, text: str) -> List[List[Token]]:
        output_sentences = []

        # 1. Сегментация на предложения (Razdel)
        # Мы используем razdel, чтобы гарантировать корректные char_start/end
        chunk_sents = list(sentenize(text))

        for sent_span in chunk_sents:
            # 2. Токенизация (Razdel)
            # Собираем токены и их координаты
            razdel_tokens = list(tokenize(sent_span.text))
            if not razdel_tokens:
                continue

            words = [t.text for t in razdel_tokens]

            # Коррекция координат: razdel дает смещение относительно начала строки.
            # Нам нужны абсолютные координаты в text.
            # sent_span.start - начало предложения
            token_metas = []
            for t in razdel_tokens:
                abs_start = sent_span.start + t.start
                abs_end = sent_span.start + t.stop
                token_metas.append((abs_start, abs_end))

            # 3. Инференс CoBaLD
            # Модель принимает батч списков строк
            with torch.no_grad():
                # forward(words=[['word1', ...]], inference_mode=True)
                outputs = self.model(
                    words=[words],
                    inference_mode=True
                )

            # 4. Декодирование результатов
            # Мы берем [0], т.к. обрабатываем по одному предложению для надежности выравнивания
            conll_tokens = self._decode_output(outputs, token_metas, words)
            output_sentences.append(conll_tokens)

        return output_sentences

    def _decode_output(self, outputs: Dict, token_metas: List[tuple], words: List[str]) -> List[Token]:
        """
        Преобразует тензоры модели в объекты Token.
        Логика декодирования адаптирована из pipeline.py
        """
        batch_idx = 0
        n_words = len(words)  # Игнорируем вставленные #NULL для структурного выравнивания

        tokens_result = []

        # --- Извлечение предикатов (ID классов) ---
        # Используем ключи из pipeline.py

        # Леммы
        lemma_rule_ids = outputs["lemma_rules"][batch_idx, :n_words].tolist() if "lemma_rules" in outputs else None

        # POS-теги
        joint_feats_ids = outputs["joint_feats"][batch_idx, :n_words].tolist() if "joint_feats" in outputs else None

        # Синтаксис (UD)
        # deps_ud shape: [n_edges, 3] -> (batch_idx, head, label)
        deps_ud = outputs["deps_ud"]
        current_sent_deps = deps_ud[deps_ud[:, 0] == batch_idx][:, 1:].tolist() if deps_ud is not None else []

        # Семантика (ENG-003)
        deepslot_ids = outputs["deepslots"][batch_idx, :n_words].tolist() if "deepslots" in outputs else None
        semclass_ids = outputs["semclasses"][batch_idx, :n_words].tolist() if "semclasses" in outputs else None

        # --- Сборка ---
        # Сначала создадим маппинг HEAD:REL
        # В output модели индексы 1-based (0 - это root или null).
        head_map = {}  # token_index (0-based) -> (head_index (1-based), rel_str)

        if self.vocab.get("ud_deprel"):
            id2rel = self.vocab["ud_deprel"]
            for arc_from, arc_to, rel_id in current_sent_deps:
                # arc_from: index of HEAD (0..N)
                # arc_to: index of DEPENDENT (1..N)
                # Внимание: модель может выдать arc_to > len(words), если она вставила #NULL.
                # Мы игнорируем связи к несуществующим токенам.
                token_idx = arc_to - 1  # convert to 0-based
                if 0 <= token_idx < n_words:
                    rel_str = id2rel.get(rel_id, "dep")
                    head_map[token_idx] = (arc_from, rel_str)

        for i in range(n_words):
            word_text = words[i]
            char_start, char_end = token_metas[i]

            # 1. Лемматизация
            # CoBaLD предсказывает правило (suffix cut/append)
            lemma = word_text.lower()  # Fallback
            if lemma_rule_ids:
                rule_str = self.vocab["lemma_rule"][lemma_rule_ids[i]]
                # Здесь можно импортировать reconstruct_lemma из lemmatize_helper.py
                # Но для простоты: (в реальном коде нужен импорт)
                lemma = self._apply_lemma_rule(word_text, rule_str)

            # 2. POS
            pos = "X"
            if joint_feats_ids:
                # Format: "UPOS#XPOS#Feats"
                tag_str = self.vocab["joint_feats"][joint_feats_ids[i]]
                pos = tag_str.split('#')[0]

            # 3. Синтаксис
            head_id, rel = head_map.get(i, (0, "root"))

            # 4. Семантика (Misc)
            misc = {}
            if deepslot_ids:
                slot = self.vocab["deepslot"][deepslot_ids[i]]
                if slot != "_": misc["deep_slot"] = slot

            if semclass_ids:
                s_class = self.vocab["semclass"][semclass_ids[i]]
                if s_class != "_": misc["sem_class"] = s_class

            token = Token(
                id=i + 1,
                text=word_text,
                lemma=lemma,
                pos=pos,
                head_id=head_id,
                rel=rel,
                char_start=char_start,
                char_end=char_end,
                misc=misc  # <-- Самое важное для ENG-003
            )
            tokens_result.append(token)

        return tokens_result

    def _apply_lemma_rule(self, word, rule_str):
        # Простая реализация based on lemmatize_helper.py
        try:
            # rule format: cut_prefix|cut_suffix|append_suffix
            parts = rule_str.split('|')
            cut_prefix = int(parts[0].split('=')[1])
            cut_suffix = int(parts[1].split('=')[1])
            append_suffix = parts[2].split('=')[1]

            lemma = word[cut_prefix:]
            if cut_suffix > 0:
                lemma = lemma[:-cut_suffix]
            lemma += append_suffix
            return lemma
        except:
            return word.lower()
        