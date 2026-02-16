import modal
import logging

image = (
    modal.Image.debian_slim()
    .pip_install("stanza", "torch")
    .run_commands("python -c 'import stanza; stanza.download(\"ru\")'")
)

app = modal.App("booknlp-ru-stanza")

@app.cls(image=image, gpu="T4", timeout=600, container_idle_timeout=300)
class StanzaService:

    @modal.enter()
    def setup(self):
        import stanza
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StanzaService")

        # ========== ВАЖНО: ПРОЦЕССОРЫ NER, SENTIMENT, CONSTITUENCY ==========
        # По умолчанию Stanza для русского языка загружает только базовые процессоры:
        # tokenize, pos, lemma, depparse
        #
        # Для получения дополнительных полей нужно явно указать процессоры:
        # - 'ner' - распознавание именованных сущностей (доступен для ru)
        # - 'sentiment' - анализ тональности (НЕ доступен для ru, только en/zh/de)
        # - 'constituency' - дерево составляющих (НЕ доступен для ru)
        # ====================================================================

        # Инициализируем ДВА пайплайна с процессором NER
        self.nlp_raw = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse,ner',
                                       verbose=False, use_gpu=True)
        self.nlp_pretokenized = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse,ner',
                                                verbose=False, use_gpu=True, tokenize_pretokenized=True)

        self.logger.info("Stanza loaded (Dual Mode with NER)!")

    @modal.method()
    def parse(self, text: str, native_format: bool = False) -> list[list[dict]]:
        """Вход: сырой текст. Использует nlp_raw."""
        doc = self.nlp_raw(text)
        # ========== БЛОК ВЫБОРА ФОРМАТА ВЫВОДА ==========
        # Если native_format=True, возвращаем нативный формат Stanza
        # Если native_format=False (по умолчанию), возвращаем текущий упрощенный формат
        if native_format:
            return self._format_output_native(doc)
        else:
            return self._format_output(doc)
        # ================================================

    @modal.method()
    def parse_batch(self, batch_tokens: list[list[str]], native_format: bool = False) -> list[list[dict]]:
        """Вход: готовые токены. Использует nlp_pretokenized."""
        doc = self.nlp_pretokenized(batch_tokens)
        # ========== БЛОК ВЫБОРА ФОРМАТА ВЫВОДА ==========
        # Если native_format=True, возвращаем нативный формат Stanza
        # Если native_format=False (по умолчанию), возвращаем текущий упрощенный формат
        if native_format:
            return self._format_output_native(doc)
        else:
            return self._format_output(doc)
        # ================================================

    def _format_output(self, doc) -> list[list[dict]]:
        """
        Вспомогательный метод (бывший _doc_to_list).
        Конвертирует внутренний объект Stanza Doc в наш формат списка словарей.
        Используется обоими методами выше, чтобы избежать дублирования кода.
        """
        result = []
        for sent in doc.sentences:
            sent_parsed = []
            for word in sent.words:
                sent_parsed.append({
                    "id": int(word.id),
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": int(word.head),
                    "deprel": word.deprel,
                    "start_char": word.start_char,
                    "end_char": word.end_char
                })
            result.append(sent_parsed)
        return result

    # ========== БЛОК ПОДГОТОВКИ НАТИВНОГО ВЫХОДА МОДЕЛИ ==========
    def _format_output_native(self, doc) -> list[list[dict]]:
        """
        Конвертирует объект Stanza Doc в максимально полный нативный формат.

        ВАЖНАЯ ОСОБЕННОСТЬ STANZA:
        - NER хранится на уровне Token (sent.tokens[i].ner), а не Word
        - misc (SpaceAfter) также хранится на уровне Token
        - С версии v1.4+ информация о пробелах в token.spaces_after/spaces_before
        - При экспорте в CoNLL-U SpaceAfter=No добавляется автоматически

        Источники:
        - https://github.com/stanfordnlp/stanza/issues/1315
        - https://stanfordnlp.github.io/stanza/data_objects.html#token
        """
        result = []
        for sent in doc.sentences:
            sent_parsed = []

            # ========== МАППИНГ NER ИЗ TOKENS ==========
            # В Stanza поле ner хранится на уровне Token, а не Word
            word_to_ner = {}
            for token in sent.tokens:
                ner_tag = token.ner if hasattr(token, 'ner') else None
                for word in token.words:
                    word_to_ner[int(word.id)] = ner_tag
            # ===========================================

            # ========== МАППИНГ MISC (SpaceAfter) ИЗ TOKENS ==========
            # Информация о пробелах хранится в token.spaces_after (Stanza v1.4+)
            # Нужно создать маппинг word_id -> misc для корректного вывода
            word_to_misc = {}
            for token in sent.tokens:
                misc_value = None

                if hasattr(token, 'spaces_after'):
                    # С версии v1.4+ есть поле spaces_after
                    if token.spaces_after == '':
                        # Нет пробела после токена
                        misc_value = 'SpaceAfter=No'
                    elif len(token.spaces_after) > 1:
                        # Несколько пробелов (например, "\s\s")
                        misc_value = f'SpacesAfter={repr(token.spaces_after)}'
                else:
                    # Старая версия Stanza - вычисляем вручную
                    # Проверяем, идет ли следующий токен сразу после этого
                    all_tokens = sent.tokens
                    token_idx = all_tokens.index(token)
                    if token_idx < len(all_tokens) - 1:
                        next_token = all_tokens[token_idx + 1]
                        if next_token.start_char == token.end_char:
                            # Нет пробела между токенами
                            misc_value = 'SpaceAfter=No'

                # Присваиваем misc словам этого токена
                # Обычно token содержит только 1 word, но в MWT может быть несколько
                for word in token.words:
                    if len(token.words) == 1:
                        # Обычный токен
                        word_to_misc[int(word.id)] = misc_value
                    else:
                        # MWT - misc присваивается только последнему слову
                        last_word_id = max([int(w.id) for w in token.words])
                        word_to_misc[last_word_id] = misc_value
            # =========================================================

            for word in sent.words:
                # Базовые поля нативного формата
                word_dict = {
                    "id": int(word.id),
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": int(word.head),
                    "deprel": word.deprel,
                    "start_char": word.start_char,
                    "end_char": word.end_char
                }

                # ========== ДОБАВЛЯЕМ MISC ИЗ МАППИНГА ==========
                # misc содержит SpaceAfter=No или SpacesAfter=\s\s
                word_id = int(word.id)
                if word_id in word_to_misc and word_to_misc[word_id] is not None:
                    word_dict["misc"] = word_to_misc[word_id]
                # ================================================

                # ========== ДОБАВЛЯЕМ NER ИЗ МАППИНГА ==========
                # ner содержит теги именованных сущностей (B-PER, I-LOC и т.д.)
                if word_id in word_to_ner and word_to_ner[word_id] is not None:
                    word_dict["ner"] = word_to_ner[word_id]
                # ===============================================

                sent_parsed.append(word_dict)

            # Добавляем метаданные предложения
            sentence_data = {
                "words": sent_parsed
            }

            # ========== SENTIMENT И CONSTITUENCY ==========
            # sentiment - НЕ ДОСТУПЕН для русского (только en, zh, de)
            # constituency - НЕ ДОСТУПЕН для русского
            if hasattr(sent, 'sentiment') and sent.sentiment is not None:
                sentence_data["sentiment"] = sent.sentiment

            if hasattr(sent, 'constituency') and sent.constituency is not None:
                sentence_data["constituency"] = str(sent.constituency)
            # ==============================================

            result.append(sentence_data)

        return result
    # ==============================================================


# ============================================================
# БЛОК: Тестовые примеры использования сервиса
# ============================================================
@app.local_entrypoint()
def main():
    test_texts = [
        'Коля сказал:"Привет!"И ушёл.',
        "Москва,столица России."
    ]

    print("Testing Stanza service with NER and misc...")
    service = StanzaService()

    # ============================================================
    # Демонстрация нативного формата с misc (SpaceAfter)
    # ============================================================
    print("\n" + "=" * 70)
    print("НАТИВНЫЙ ФОРМАТ с NER и misc (SpaceAfter):")
    print("=" * 70)

    results_native = service.parse.remote(test_texts[0], native_format=True)

    print(f"\nТекст: '{test_texts[0]}'\n")
    for sent_idx, sent_data in enumerate(results_native):
        words = sent_data.get("words", [])
        print(f"Предложение {sent_idx + 1}: {len(words)} токенов\n")

        # Показываем токены с NER и misc
        for tok in words:
            ner_info = f" [NER: {tok['ner']}]" if 'ner' in tok else ""
            misc_info = f" [misc: {tok['misc']}]" if 'misc' in tok else ""
            print(f"  {tok['form']} ({tok['upos']}){ner_info}{misc_info}")

    print("\nTest completed!")
