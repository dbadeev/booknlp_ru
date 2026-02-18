import modal
import logging
from typing import List, Dict, Any

# ========== ОБРАЗ ДЛЯ SPACY ==========
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "spacy>=3.7.0",
        "pymorphy3>=1.2.0",
        "pymorphy3-dicts-ru>=2.4.0",
        "spacy-conll>=4.0.0",
    )
    .run_commands(
        "python -m spacy download ru_core_news_lg"
    )
)

app = modal.App("booknlp-ru-spacy")


@app.cls(image=image, timeout=600, scaledown_window=300)
class SpacyService:
    """
    Сервис для морфо-синтаксического анализа с использованием
    официальной модели ru_core_news_lg (CNN/Tok2Vec архитектура).

    Компоненты pipeline:
    - tok2vec: векторизация токенов
    - morphologizer: морфологический анализ (с pymorphy3)
    - parser: синтаксический анализ (dependency parsing)
    - senter: сегментация предложений
    - ner: распознавание именованных сущностей
    - attribute_ruler: правила для атрибутов
    - lemmatizer: лемматизация (с pymorphy3)
    """

    @modal.enter()
    def setup(self):
        import spacy

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SpacyService")

        # Загружаем официальную модель
        self.nlp = spacy.load("ru_core_news_lg")

        # Добавляем компонент для экспорта в CoNLL-U
        config = {
            "ext_names": {
                "conll_str": "conll_str",
                "conll": "conll",
                "conll_pd": "conll_pd"
            },
            "conversion_maps": {
                "UPOS": {},
                "XPOS": {},
                "FEATS": {},
                "DEPREL": {}
            }
        }
        self.nlp.add_pipe("conll_formatter", config=config, last=True)

        self.logger.info(f"SpaCy loaded (ru_core_news_lg)!")
        self.logger.info(f"Pipeline components: {self.nlp.pipe_names}")

    @modal.method()
    def parse(
            self,
            text: str,
            output_format: str = "simplified"
    ) -> List[List[Dict[str, Any]]]:
        """
        Парсит сырой текст.

        Args:
            text: Входной текст для анализа
            output_format: Формат вывода
                - 'simplified': упрощенный формат (совместимость с другими парсерами)
                - 'native': полный нативный формат spaCy
                - 'conllu': стандартный формат CoNLL-U

        Returns:
            Результат парсинга в выбранном формате
        """
        doc = self.nlp(text)

        if output_format == "conllu":
            return self._format_conllu(doc)
        elif output_format == "native":
            return self._format_native(doc)
        else:
            return self._format_simplified(doc)

    @modal.method()
    def parse_batch(
            self,
            texts: List[str],
            output_format: str = "simplified",
            batch_size: int = 32
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Пакетная обработка текстов для повышения производительности.

        Args:
            texts: Список текстов для анализа
            output_format: Формат вывода
            batch_size: Размер батча для обработки

        Returns:
            Список результатов для каждого текста
        """
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            if output_format == "conllu":
                results.append(self._format_conllu(doc))
            elif output_format == "native":
                results.append(self._format_native(doc))
            else:
                results.append(self._format_simplified(doc))

        return results

    def _format_simplified(self, doc) -> List[List[Dict[str, Any]]]:
        """
        Упрощенный формат для совместимости с другими парсерами.

        Возвращает структуру: List[Sentence[List[Token[Dict]]]]

        Поля токена:
        - id: позиция в предложении (1-indexed)
        - form: текстовая форма токена
        - lemma: лемма (начальная форма)
        - upos: универсальная часть речи (Universal POS)
        - xpos: специфичная для языка часть речи
        - feats: морфологические признаки
        - head: индекс главного токена
        - deprel: тип синтаксической связи
        - start_char: позиция начала в тексте
        - end_char: позиция конца в тексте
        """
        result = []
        for sent in doc.sents:
            sent_parsed = []
            for token in sent:
                sent_parsed.append({
                    "id": token.i - sent.start + 1,
                    "form": token.text,
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "feats": str(token.morph) if token.morph else "_",
                    "head": token.head.i - sent.start + 1 if token.head.i != token.i else 0,
                    "deprel": token.dep_,
                    "start_char": token.idx,
                    "end_char": token.idx + len(token.text)
                })
            result.append(sent_parsed)
        return result

    def _format_native(self, doc) -> List[Dict[str, Any]]:
        """
        Полный нативный формат spaCy со всеми доступными атрибутами.

        Включает:
        - Все базовые поля (форма, лемма, POS, морфология, синтаксис)
        - NER информацию (именованные сущности)
        - Метаданные токенов (пробелы, форма, флаги)
        - Информацию о предложении (текст, границы)
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
                word_dict = {
                    "id": token.i - sent.start + 1,
                    "form": token.text,
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "feats": str(token.morph) if token.morph else None,
                    "head": token.head.i - sent.start + 1 if token.head.i != token.i else 0,
                    "deprel": token.dep_,
                    "start_char": token.idx,
                    "end_char": token.idx + len(token.text),
                    # Дополнительные нативные поля spaCy
                    "ent_type": token.ent_type_ if token.ent_type_ else None,
                    "ent_iob": token.ent_iob_ if token.ent_iob_ != "O" else None,
                    "is_sent_start": token.is_sent_start,
                    "whitespace": token.whitespace_,
                    "shape": token.shape_,
                    "is_alpha": token.is_alpha,
                    "is_punct": token.is_punct,
                    "like_num": token.like_num
                }

                # Добавляем misc для SpaceAfter
                if not token.whitespace_:
                    word_dict["misc"] = "SpaceAfter=No"

                sent_data["words"].append(word_dict)

            # Добавляем именованные сущности на уровне предложения
            ents = [
                {
                    "text": ent.text,
                    "start": ent.start - sent.start,
                    "end": ent.end - sent.start,
                    "label": ent.label_
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

        CoNLL-U - стандартный формат для морфо-синтаксической разметки.
        Источник: https://universaldependencies.org/format.html

        Пример:
        # sent_id = 1
        # text = Москва - столица России.
        1	Москва	москва	PROPN	_	Case=Nom|Gender=Fem|Number=Sing	4	nsubj	_	_
        2	-	-	PUNCT	_	_	4	punct	_	_
        3	столица	столица	NOUN	_	Case=Nom|Gender=Fem|Number=Sing	0	root	_	_
        ...
        """
        return doc._.conll_str


# ========== ТЕСТОВАЯ ТОЧКА ВХОДА ==========
@app.local_entrypoint()
def main():
    """Тестирование SpaCy сервиса."""
    test_texts = [
        'Коля сказал:"Привет!"И ушёл.',
        "Москва - столица России."
    ]

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ SPACY SERVICE")
    print("=" * 70)

    service = SpacyService()

    # Тест 1: Simplified format
    print("\n1. SIMPLIFIED FORMAT:")
    print("-" * 70)
    result = service.parse.remote(test_texts[0], output_format="simplified")
    print(f"\nТекст: '{test_texts[0]}'")
    print(f"Предложений: {len(result)}\n")

    for sent_idx, sent in enumerate(result):
        print(f"Предложение {sent_idx + 1}:")
        for tok in sent[:5]:
            print(f"  {tok['id']}\t{tok['form']}\t{tok['lemma']}\t"
                  f"{tok['upos']}\t{tok['head']}\t{tok['deprel']}")
        if len(sent) > 5:
            print(f"  ... (всего {len(sent)} токенов)")

    # Тест 2: Native format
    print("\n2. NATIVE FORMAT:")
    print("-" * 70)
    result_native = service.parse.remote(test_texts[1], output_format="native")
    print(f"\nТекст: '{test_texts[1]}'")

    for sent_data in result_native:
        print(f"\nПредложение: '{sent_data['text']}'")
        print(f"Границы: [{sent_data['start_char']}, {sent_data['end_char']}]")

        if "entities" in sent_data:
            print(f"Именованные сущности:")
            for ent in sent_data["entities"]:
                print(f"  - {ent['text']} [{ent['label']}]")

        print(f"\nПервые 3 токена:")
        for tok in sent_data["words"][:3]:
            print(f"\n  Токен: {tok['form']}")
            print(f"    Lemma: {tok['lemma']}, POS: {tok['upos']}")
            print(f"    Feats: {tok['feats']}")
            print(f"    Head: {tok['head']}, Deprel: {tok['deprel']}")
            if tok.get('ent_type'):
                print(f"    Entity: {tok['ent_type']} ({tok['ent_iob']})")
            if tok.get('misc'):
                print(f"    Misc: {tok['misc']}")

    # Тест 3: CoNLL-U format
    print("\n3. CONLL-U FORMAT:")
    print("-" * 70)
    result_conllu = service.parse.remote(test_texts[0], output_format="conllu")
    print(f"\nТекст: '{test_texts[0]}'")
    print("\nВывод в формате CoNLL-U:")
    print(result_conllu[:600] if len(result_conllu) > 600 else result_conllu)
    if len(result_conllu) > 600:
        print("... (обрезано)")

    # Тест 4: Batch processing
    print("\n4. BATCH PROCESSING:")
    print("-" * 70)
    batch_texts = [
        "Петр купил книгу в магазине.",
        "Она читает интересную газету.",
        "Дети играют в парке."
    ]

    results_batch = service.parse_batch.remote(
        batch_texts,
        output_format="simplified",
        batch_size=32
    )

    print(f"\nОбработано текстов: {len(results_batch)}\n")
    for i, text_result in enumerate(results_batch):
        print(f"Текст {i + 1}: '{batch_texts[i]}'")
        for sent in text_result:
            print(f"  Токенов: {len(sent)}")
            print(f"  Первый токен: {sent[0]['form']} -> {sent[0]['lemma']} ({sent[0]['upos']})")

    print("\n" + "=" * 70)
    print("✅ Тестирование завершено!")
    print("=" * 70)
