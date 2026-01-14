import torch
from typing import List, Dict, Optional
from razdel import sentenize, tokenize as razdel_tokenize

# Импорты нашей архитектуры
from src.core.interfaces import BasePreprocessor
from src.core.data_structures import Token

# Импорты CoBaLD (из ваших файлов)
#
from src.cobald_parser.modeling_parser import CobaldParser
from src.cobald_parser.configuration import CobaldParserConfig


class CobaldEngine(BasePreprocessor):
    """
    Адаптер для CoBaLD Parser в архитектуре BookNLP-ru.
    Интегрирует синтаксис и семантику (DeepSlots, SemClasses).
    """

    def __init__(self, model_path: str = "CoBaLD/xlm-roberta-base-cobald-parser-ru", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🧠 Loading CoBaLD form {model_path} on {self.device}...")

        # Загрузка конфигурации и модели
        self.config = CobaldParserConfig.from_pretrained(model_path)
        self.model = CobaldParser.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()

        # Кэшируем словари для декодирования (ID -> Str)
        # Структура словарей описана в
        self.vocab = self.config.vocabulary

        print("✅ CoBaLD Engine Ready")

    def process(self, text: str) -> List[List[Token]]:
        """
        Основной метод пайплайна.
        1. Сегментация (Razdel) -> Получение char_start/end
        2. Инференс CoBaLD
        3. Маппинг тензоров в объекты Token
        """
        doc_sentences = []

        # 1. Сегментация на предложения (Razdel)
        for sent_span in sentenize(text):
            # 2. Токенизация (Razdel)
            # Мы используем Razdel как источник "истины" для координат
            razdel_tokens = list(razdel_tokenize(sent_span.text))
            if not razdel_tokens:
                continue

            # Подготовка слов для модели
            words = [t.text for t in razdel_tokens]

            # Расчет абсолютных координат
            # Razdel дает смещение внутри предложения, нам нужно внутри текста
            token_metas = []
            for t in razdel_tokens:
                abs_start = sent_span.start + t.start
                abs_end = sent_span.start + t.stop
                token_metas.append((abs_start, abs_end))

            # 3. Запуск модели
            # CoBaLD принимает батч списков слов: [["Мама", "мыла"...]]
            #
            with torch.no_grad():
                outputs = self.model(
                    words=[words],
                    inference_mode=True
                )

            # 4. Декодирование (тензоры -> токены)
            tokens = self._decode_sentence_batch(outputs, words, token_metas, batch_idx=0)
            doc_sentences.append(tokens)

        return doc_sentences

    def _decode_sentence_batch(self, outputs: dict, words: List[str], token_metas: List[tuple], batch_idx: int) -> List[
        Token]:
        """
        Превращает сырые выходы модели (Logits/Indices) в объекты Token.
        Логика портирована из
        """
        n_words = len(words)
        result_tokens = []

        # --- Извлечение индексов из тензоров ---

        # 1. Lemma Rules
        lemma_rule_ids = None
        if "lemma_rules" in outputs:
            lemma_rule_ids = outputs["lemma_rules"][batch_idx, :n_words].tolist()

        # 2. POS & Feats (Joint)
        joint_feats_ids = None
        if "joint_feats" in outputs:
            joint_feats_ids = outputs["joint_feats"][batch_idx, :n_words].tolist()

        # 3. Семантика (Misc / DeepSlots / SemClasses) [Задача ENG-003]
        deepslot_ids = outputs["deepslots"][batch_idx, :n_words].tolist() if "deepslots" in outputs else None
        semclass_ids = outputs["semclasses"][batch_idx, :n_words].tolist() if "semclasses" in outputs else None

        # 4. Синтаксис (Deps UD)
        # deps_ud shape: [N_arcs, 4] -> [batch_idx, head, dep, rel_id]
        #
        deps_ud = outputs.get("deps_ud")
        head_map = {}  # dep_idx (0-based) -> (head_idx (1-based), rel_str)

        if deps_ud is not None:
            # Фильтруем дуги только для текущего предложения (batch_idx)
            current_arcs = deps_ud[deps_ud[:, 0] == batch_idx]

            id2rel = self.vocab.get("ud_deprel", {})

            for arc in current_arcs:
                head_idx = int(arc[1])  # 0 is ROOT
                dep_idx = int(arc[2])  # 1-based index of word
                rel_id = int(arc[3])

                rel_str = id2rel.get(rel_id, "dep")

                # dep_idx - 1, т.к. в tokens мы идем с 0, а модель считает с 1 (0=CLS)
                #
                token_idx = dep_idx - 1

                if 0 <= token_idx < n_words:
                    head_map[token_idx] = (head_idx, rel_str)

        # --- Сборка Токенов ---

        for i in range(n_words):
            word_text = words[i]
            char_start, char_end = token_metas[i]

            # Лемматизация
            lemma = word_text.lower()  # Fallback
            if lemma_rule_ids:
                rule_str = self.vocab["lemma_rule"][lemma_rule_ids[i]]
                lemma = self._apply_lemma_rule(word_text, rule_str)

            # POS
            pos = "X"
            feats = {}
            if joint_feats_ids:
                # Format: UPOS#XPOS#Feats
                val = self.vocab["joint_feats"][joint_feats_ids[i]]
                parts = val.split('#')
                pos = parts[0]
                # Можно распарсить feats (parts[2]), если нужно

            # Синтаксис
            head_id, rel = head_map.get(i, (0, "root"))

            # Семантика (Заполняем misc)
            misc = {}

            if deepslot_ids:
                slot = self.vocab["deepslot"][deepslot_ids[i]]
                if slot != "_":
                    misc["deep_slot"] = slot  # Например: "Agent", "Experiencer"

            if semclass_ids:
                s_class = self.vocab["semclass"][semclass_ids[i]]
                if s_class != "_":
                    misc["sem_class"] = s_class  # Например: "Person", "Event"

            token = Token(
                id=i + 1,
                text=word_text,
                lemma=lemma,
                pos=pos,
                head_id=head_id,
                rel=rel,
                char_start=char_start,
                char_end=char_end,
                misc=misc
            )
            result_tokens.append(token)

        return result_tokens

    def _apply_lemma_rule(self, word: str, rule_str: str) -> str:
        """
        Применяет правило лемматизации.
        Адаптировано из
        Format: cut_prefix=0|cut_suffix=1|append_suffix=а
        """
        try:
            # Парсинг строки правила
            # Пример: "0|1|а" или "cut_prefix=0|..." (зависит от версии vocab)
            # В provided file lemmatize_helper.py формат сложный, но vocab обычно хранит уже values
            # Предположим формат из lemmatize_helper.py: keys like "cut_prefix=..."

            params = {}
            for part in rule_str.split('|'):
                key, val = part.split('=')
                params[key] = val

            cut_prefix = int(params.get('cut_prefix', 0))
            cut_suffix = int(params.get('cut_suffix', 0))
            append_suffix = params.get('append_suffix', '')

            # Применение
            lemma = word[cut_prefix:]
            if cut_suffix > 0:
                lemma = lemma[:-cut_suffix]
            lemma += append_suffix

            return lemma
        except Exception:
            # Fallback если формат отличается
            return word.lower()