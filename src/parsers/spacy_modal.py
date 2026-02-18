import modal
import logging
from typing import List, Dict, Any, Literal

# ========== ОБРАЗ ДЛЯ SPACY ==========
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "spacy>=3.7.0",
        "pymorphy3>=1.2.0",
        "pymorphy3-dicts-ru>=2.4.0",
        "spacy-conll>=4.0.0",
        "razdel>=0.5.0",
    )
    .run_commands(
        "python -m spacy download ru_core_news_lg"
    )
)

app = modal.App("booknlp-ru-spacy")

TokenizerType = Literal["internal", "razdel"]


@app.cls(image=image, timeout=600, scaledown_window=300)
class SpacyService:
    """
    Сервис для морфо-синтаксического анализа с использованием
    официальной модели ru_core_news_lg (CNN/Tok2Vec архитектура).

    Поддерживает два типа токенизаторов:
    - internal: встроенный токенизатор spaCy
    - razdel:   внешний токенизатор razdel (лучше для русского языка)

    Компоненты pipeline:
    - tok2vec:          векторизация токенов
    - morphologizer:    морфологический анализ (с pymorphy3)
    - parser:           синтаксический анализ (dependency parsing)
    - senter:           сегментация предложений
    - ner:              распознавание именованных сущностей
    - attribute_ruler:  правила для атрибутов
    - lemmatizer:       лемматизация (с pymorphy3)
    - conll_formatter:  экспорт в CoNLL-U
    """

    @modal.enter()
    def setup(self):
        import spacy
        from spacy.tokens import Doc
        from razdel import tokenize as razdel_tokenize

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SpacyService")

        # Загружаем официальную модель
        self.nlp = spacy.load("ru_core_news_lg")

        # Сохраняем оригинальный (internal) токенизатор
        self.original_tokenizer = self.nlp.tokenizer

        # ============================================================
        # Токенизатор на базе razdel
        # ============================================================
        class RazdelTokenizer:
            """Обертка razdel для использования в spaCy."""

            def __init__(self, vocab):
                self.vocab = vocab

            def __call__(self, text):
                razdel_tokens = list(razdel_tokenize(text))
                if not razdel_tokens:
                    return Doc(self.vocab, words=[], spaces=[])

                words = [t.text for t in razdel_tokens]

                # Определяем пробелы между токенами по символьным позициям
                spaces = []
                for i, token in enumerate(razdel_tokens):
                    if i < len(razdel_tokens) - 1:
                        next_token = razdel_tokens[i + 1]
                        spaces.append(token.stop < next_token.start)
                    else:
                        spaces.append(False)

                return Doc(self.vocab, words=words, spaces=spaces)

        self.razdel_tokenizer = RazdelTokenizer(self.nlp.vocab)

        # Добавляем компонент для экспорта в CoNLL-U
        config = {
            "ext_names": {
                "conll_str": "conll_str",
                "conll": "conll",
                "conll_pd": "conll_pd"
            },
            "conversion_maps": {
                "UPOS": {}, "XPOS": {}, "FEATS": {}, "DEPREL": {}
            }
        }
        self.nlp.add_pipe("conll_formatter", config=config, last=True)

        self.logger.info(f"SpaCy loaded (ru_core_news_lg)!")
        self.logger.info(f"Pipeline components: {self.nlp.pipe_names}")
        self.logger.info("Tokenizers available: internal (spaCy), razdel")

    def _set_tokenizer(self, tokenizer_type: TokenizerType):
        """Переключает токенизатор."""
        if tokenizer_type == "razdel":
            self.nlp.tokenizer = self.razdel_tokenizer
        else:
            self.nlp.tokenizer = self.original_tokenizer

    @modal.method()
    def parse(
        self,
        text: str,
        output_format: str = "native",
        tokenizer: TokenizerType = "internal"
    ) -> Any:
        """
        Парсит сырой текст.

        Args:
            text:          Входной текст для анализа
            output_format: 'native' | 'conllu'
            tokenizer:     'internal' | 'razdel'

        Returns:
            native  → List[Dict]  (предложения со всеми полями spaCy)
            conllu  → str         (стандартный формат CoNLL-U)
        """
        self._set_tokenizer(tokenizer)
        doc = self.nlp(text)

        if output_format == "conllu":
            return self._format_conllu(doc)
        else:
            return self._format_native(doc)

    @modal.method()
    def parse_batch(
        self,
        texts: List[str],
        output_format: str = "native",
        tokenizer: TokenizerType = "internal",
        batch_size: int = 32
    ) -> List[Any]:
        """
        Пакетная обработка текстов.

        Args:
            texts:         Список текстов
            output_format: 'native' | 'conllu'
            tokenizer:     'internal' | 'razdel'
            batch_size:    Размер батча
        """
        self._set_tokenizer(tokenizer)
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            if output_format == "conllu":
                results.append(self._format_conllu(doc))
            else:
                results.append(self._format_native(doc))
        return results

    def _format_native(self, doc) -> List[Dict[str, Any]]:
        """
        ПОЛНЫЙ нативный формат spaCy — ВСЕ доступные атрибуты токена.

        Поля уровня предложения:
          text, start_char, end_char, entities

        Поля каждого токена:
          ── Позиция ──────────────────────────────────────────────────
          id            : позиция в предложении (1-indexed)
          start_char    : начало токена в исходном тексте
          end_char      : конец токена в исходном тексте

          ── Форма ────────────────────────────────────────────────────
          form          : оригинальная форма (с сохранением регистра)
          norm          : нормализованная форма (token.norm_)
          lower         : форма в нижнем регистре (token.lower_)
          shape         : орфографическая форма (Xxxx, dddd и т.п.)

          ── Лемма и POS ──────────────────────────────────────────────
          lemma         : лемма (контекстная, от spaCy+pymorphy3)
          upos          : Universal POS tag
          xpos          : язык-специфичный POS tag
          feats         : морфологические признаки (UD формат)

          ── Синтаксис ────────────────────────────────────────────────
          head          : id главного токена (0 = root)
          deprel        : тип синтаксической связи
          n_lefts       : число левых зависимых
          n_rights      : число правых зависимых
          children      : список id всех зависимых токенов

          ── Именованные сущности ─────────────────────────────────────
          ent_type      : тип сущности (PER, LOC, ORG, ...) или None
          ent_iob       : IOB-тег (B/I) или None

          ── Метаданные ───────────────────────────────────────────────
          is_sent_start : начало предложения
          whitespace    : пробел после токена
          misc          : SpaceAfter=No (если нет пробела)

          ── Лексические флаги ────────────────────────────────────────
          is_alpha      : состоит только из букв
          is_digit      : состоит только из цифр
          is_punct      : знак препинания
          is_space      : пробельный символ
          is_stop       : стоп-слово
          is_oov        : вне словаря модели (Out Of Vocabulary)
          like_num      : похоже на число
          like_url      : похоже на URL
          like_email    : похоже на email

          ── Векторные поля ───────────────────────────────────────────
          has_vector    : есть ли вектор у токена
          cluster       : кластер Брауна (int), 0 если не определён

          ── Вероятность ──────────────────────────────────────────────
          prob          : log-вероятность токена в языке (float)
          rank          : ранг по частоте в vocab модели (int)
                            0 = самое частое слово, выше = реже
                            is_oov=True → rank=0 (вне словаря)
        """
        result = []
        for sent in doc.sents:
            sent_data = {
                "text": sent.text,
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "words": []
            }

            for token in sent:
                sent_offset = sent.start

                word_dict = {
                    # ── Позиция ──────────────────────────────────────
                    "id":           token.i - sent_offset + 1,
                    "start_char":   token.idx,
                    "end_char":     token.idx + len(token.text),

                    # ── Форма ────────────────────────────────────────
                    "form":         token.text,
                    "norm":         token.norm_,
                    "lower":        token.lower_,
                    "shape":        token.shape_,

                    # ── Лемма и POS ──────────────────────────────────
                    "lemma":        token.lemma_,
                    "upos":         token.pos_,
                    "xpos":         token.tag_,
                    "feats":        str(token.morph) if token.morph else "_",

                    # ── Синтаксис ────────────────────────────────────
                    "head":         token.head.i - sent_offset + 1
                                    if token.head.i != token.i else 0,
                    "deprel":       token.dep_,
                    "n_lefts":      token.n_lefts,
                    "n_rights":     token.n_rights,
                    "children":     [c.i - sent_offset + 1 for c in token.children],

                    # ── Именованные сущности ─────────────────────────
                    "ent_type":     token.ent_type_ or None,
                    "ent_iob":      token.ent_iob_
                                    if token.ent_iob_ != "O" else None,

                    # ── Метаданные ───────────────────────────────────
                    "is_sent_start": token.is_sent_start,
                    "whitespace":   token.whitespace_,

                    # ── Лексические флаги ────────────────────────────
                    "is_alpha":     token.is_alpha,
                    "is_digit":     token.is_digit,
                    "is_punct":     token.is_punct,
                    "is_space":     token.is_space,
                    "is_stop":      token.is_stop,
                    "is_oov":       token.is_oov,
                    "like_num":     token.like_num,
                    "like_url":     token.like_url,
                    "like_email":   token.like_email,

                    # ── Векторные поля ───────────────────────────────
                    "has_vector":   token.has_vector,
                    "cluster":      token.cluster,  # всегда 0
                    "vector_norm": round(float(token.vector_norm), 6) if token.has_vector else None,

                    # ── Вероятность ──────────────────────────────────
                    # "prob":         token.prob,    ← убрать, всегда -20.0 (нет данных в модели)
                    # "rank":       token.rank,       # ← ранг по частоте в vocab модели (int)
                                                     #   0 = самое частое слово, выше = реже
                                                     #   is_oov=True → rank=0 (вне словаря)
                }

                # misc только если нет пробела
                if not token.whitespace_:
                    word_dict["misc"] = "SpaceAfter=No"

                sent_data["words"].append(word_dict)

            # Именованные сущности на уровне предложения
            ents = [
                {
                    "text":  ent.text,
                    "start": ent.start - sent.start,
                    "end":   ent.end - sent.start,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char":   ent.end_char,
                }
                for ent in sent.ents
            ]
            if ents:
                sent_data["entities"] = ents

            result.append(sent_data)
        return result

    def _format_conllu(self, doc) -> str:
        """
        Формат CoNLL-U с использованием spacy-conll.
        Источник: https://universaldependencies.org/format.html
        """
        return doc._.conll_str


# ========== ТЕСТОВАЯ ТОЧКА ВХОДА ==========
@app.local_entrypoint()
def main():
    """Тестирование SpaCy сервиса с разными токенизаторами."""
    test_text = 'Кружка-термос стоит 500р. Москва-река.'

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ SPACY SERVICE (4 варианта)")
    print("=" * 80)

    service = SpacyService()

    # Тест 1: native + internal
    print("\n1. NATIVE + INTERNAL:")
    result = service.parse.remote(test_text, output_format="native", tokenizer="internal")
    for s in result:
        print(f"  [{len(s['words'])} токенов] {[w['form'] for w in s['words']]}")

    # Тест 2: native + razdel
    print("\n2. NATIVE + RAZDEL:")
    result = service.parse.remote(test_text, output_format="native", tokenizer="razdel")
    for s in result:
        print(f"  [{len(s['words'])} токенов] {[w['form'] for w in s['words']]}")

    # Тест 3: conllu + internal
    print("\n3. CONLL-U + INTERNAL:")
    result = service.parse.remote(test_text, output_format="conllu", tokenizer="internal")
    print(result[:500])

    # Тест 4: conllu + razdel
    print("\n4. CONLL-U + RAZDEL:")
    result = service.parse.remote(test_text, output_format="conllu", tokenizer="razdel")
    print(result[:500])

    print("\n✅ Тестирование завершено!")
