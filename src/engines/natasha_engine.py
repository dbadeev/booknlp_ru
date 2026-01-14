# src/engines/natasha_engine.py
from typing import List
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
    MorphVocab
)
from src.core.interfaces import BasePreprocessor
from src.core.data_structures import Token


class NatashaPreprocessor(BasePreprocessor):
    def __init__(self):
        print("🏗️ Загрузка моделей Natasha (Slovnet)...")
        # Инициализация компонент (как в вашем скрипте)
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.morph_vocab = MorphVocab()  # Нужен для лемматизации
        print("✅ Natasha готова к работе")

    def process(self, text: str) -> List[List[Token]]:
        doc = Doc(text)

        # 1. Сегментация (Razdel) - здесь появляются координаты start/stop!
        doc.segment(self.segmenter)

        # 2. Морфология
        doc.tag_morph(self.morph_tagger)

        # 3. Лемматизация (Slovnet сам не лемматизирует, нужно прогонять через vocab)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        # 4. Синтаксис
        doc.parse_syntax(self.syntax_parser)

        output_sentences = []

        # Natasha хранит токены плоским списком, но имеет spans для предложений
        # Мы будем итерироваться по предложениям из doc.sents

        for sent in doc.sents:
            converted_sent = []

            # Внутри предложения токены имеют относительные ID, но нам нужны абсолютные координаты
            # Natasha Token API: token.start, token.stop - это смещения относительно начала ТЕКСТА

            for idx, n_token in enumerate(sent.tokens, 1):
                # Безопасный парсинг head_id (из вашего скрипта)
                head_id = self._parse_head_id(n_token.head_id)

                # Обработка Root (Natasha может ставить head_id=id для root или 0)
                # Стандарт UD: head=0 для root.
                # Проверка: если id == head_id, то это ошибка цикла (кроме root),
                # но обычно Slovnet ставит rel='root'

                token = Token(
                    id=idx,
                    text=n_token.text,
                    lemma=n_token.lemma if n_token.lemma else n_token.text.lower(),
                    pos=n_token.pos if n_token.pos else "X",
                    head_id=head_id,
                    rel=n_token.rel if n_token.rel else "root",

                    # ГЛАВНОЕ: Координаты из Razdel
                    char_start=n_token.start,
                    char_end=n_token.stop
                )
                converted_sent.append(token)

            output_sentences.append(converted_sent)

        return output_sentences

    def _parse_head_id(self, head_id) -> int:
        """Адаптировано из вашего parse_syntagrus.py"""
        if not head_id:
            return 0
        try:
            # Natasha может вернуть "1_5" (sent_token), берем только token part
            if '_' in str(head_id):
                parts = str(head_id).split('_')
                return int(parts[-1])
            return int(head_id)
        except ValueError:
            return 0
        