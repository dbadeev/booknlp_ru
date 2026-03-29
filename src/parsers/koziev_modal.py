import modal

# Устанавливаем окружение и зависимости напрямую из GitHub автора
koziev_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "git+https://github.com/Koziev/rutokenizer",
        "git+https://github.com/Koziev/rupostagger",
        "git+https://github.com/Koziev/rulemma"
    )
)

app = modal.App("booknlp-ru-koziev-service")


@app.cls(image=koziev_image, timeout=600)
class KozievService:
    @modal.enter()
    def load_models(self):
        """Загрузка моделей в память при старте контейнера."""
        import rutokenizer
        import rupostagger
        import rulemma

        self.tokenizer = rutokenizer.Tokenizer()
        self.tokenizer.load()

        self.tagger = rupostagger.RuPosTagger()
        self.tagger.load()

        self.lemmatizer = rulemma.Lemmatizer()
        self.lemmatizer.load()

    @modal.method()
    def parse_sentence_chunk(self, sentences: list[str], output_format: str = "conllu"):
        """
        Обработка списка предложений.
        Возвращает данные в нативном формате (список словарей) или строкой CoNLL-U.
        """
        results = []

        for sent in sentences:
            # Последовательная обработка токенизатором, теггером и лемматизатором
            tokens = self.tokenizer.tokenize(sent)
            tags = self.tagger.tag(tokens)
            lemmas = self.lemmatizer.lemmatize(tags)

            if output_format == "native":
                sent_data = []
                for word, pos_tags, lemma, *_ in lemmas:
                    sent_data.append({
                        "word": word,
                        "tags": pos_tags,
                        "lemma": lemma
                    })
                results.append(sent_data)

            elif output_format == "conllu":
                conllu_lines = []
                for i, (word, pos_tags, lemma, *_) in enumerate(lemmas, start=1):
                    # Базовый маппинг: выделяем основную часть речи из строки тегов
                    upos = pos_tags.split('|') if pos_tags else "_"
                    feats = pos_tags if pos_tags else "_"

                    # Синтаксические поля (head, deprel) заполняем заглушками, так как синтаксис не строится
                    line = f"{i}\t{word}\t{lemma}\t{upos}\t_\t{feats}\t0\troot\t_\t_"
                    conllu_lines.append(line)

                # Добавляем строку с готовым предложением в формате CoNLL-U
                results.append("\n".join(conllu_lines))

        if output_format == "conllu":
            # Разделяем предложения пустой строкой по стандарту CoNLL-U
            return "\n\n".join(results)

        return results