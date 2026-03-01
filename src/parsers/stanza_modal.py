import modal
import logging
from typing import Union

image = (
    modal.Image.debian_slim()
    .pip_install("stanza", "torch")
    .run_commands("python -c 'import stanza; stanza.download(\"ru\")'")
)

app = modal.App("booknlp-ru-stanza")


def _parse_misc_to_dict(misc_str: str | None) -> dict | None:
    """Конвертирует CoNLL-U MISC строку в dict для native-режима."""
    if not misc_str or misc_str == '_':
        return None
    result = {}
    for item in misc_str.split('|'):
        if '=' in item:
            key, val = item.split('=', 1)
            result[key] = val
        else:
            result[item] = True  # булевый флаг без значения
    return result


@app.cls(image=image, gpu="T4", timeout=600, scaledown_window=300)
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
        # - 'ner'          — распознавание именованных сущностей (доступен для ru)
        # - 'sentiment'    — анализ тональности (НЕ доступен для ru, только en/zh/de)
        # - 'constituency' — дерево составляющих (НЕ доступен для ru)
        # ====================================================================

        # Инициализируем ДВА пайплайна с процессором NER
        self.nlp_raw = stanza.Pipeline(
            'ru', processors='tokenize,pos,lemma,depparse,ner',
            verbose=False, use_gpu=True
        )
        self.nlp_pretokenized = stanza.Pipeline(
            'ru', processors='tokenize,pos,lemma,depparse,ner',
            verbose=False, use_gpu=True, tokenize_pretokenized=True
        )
        self.logger.info("Stanza loaded (Dual Mode with NER)!")

    @modal.method()
    def parse(self, text: str, native_format: bool = False) -> Union[list[list[dict]], list[dict]]:
        """Вход: сырой текст. Использует nlp_raw."""
        doc = self.nlp_raw(text)
        if native_format:
            return self._format_output_native(doc)
        else:
            return self._format_output(doc)

    @modal.method()
    def parse_batch(self, batch_tokens: list[list[str]], native_format: bool = False) -> Union[list[list[dict]], list[dict]]:
        """Вход: готовые токены. Использует nlp_pretokenized."""
        doc = self.nlp_pretokenized(batch_tokens)
        if native_format:
            return self._format_output_native(doc)
        else:
            return self._format_output(doc)

    def _format_output(self, doc) -> list[list[dict]]:
        """
        CoNLL-U-совместимый формат вывода.

        Поля соответствуют стандарту CoNLL-U:
          id, form, lemma, upos, xpos, feats, head, deprel, misc

        ВАЖНО:
          - start_char / end_char отсутствуют: в стандарте CoNLL-U этих полей нет;
            используйте native_format=True для получения символьных позиций.
          - spaces_after отсутствует: пробельная информация в CoNLL-U передаётся
            через поле MISC (SpaceAfter=No), которое Stanza заполняет при экспорте;
            для прямого доступа к token.spaces_after используйте native_format=True.
          - misc — строка CoNLL-U вида "SpaceAfter=No|..." или None;
            берётся из word.misc (в русской модели Stanza заполняется редко).
        """
        result = []
        for sent in doc.sentences:
            sent_parsed = []
            for word in sent.words:
                sent_parsed.append({
                    "id":     int(word.id),
                    "form":   word.text,
                    "lemma":  word.lemma,
                    "upos":   word.upos,
                    "xpos":   word.xpos,
                    "feats":  word.feats,   # строка CoNLL-U, напр. "Case=Nom|Number=Sing"
                    "head":   int(word.head),
                    "deprel": word.deprel,
                    "misc":   word.misc,    # строка "SpaceAfter=No|..." или None
                })
            result.append(sent_parsed)
        return result

    # ========== БЛОК ПОДГОТОВКИ НАТИВНОГО ВЫХОДА МОДЕЛИ ==========
    def _format_output_native(self, doc) -> list[dict]:
        """
        Нативный формат вывода — максимально полное представление объекта Stanza Doc.

        ВАЖНЫЕ ОСОБЕННОСТИ STANZA:
          - ner         — хранится на уровне Token (token.ner), а не Word
          - spaces_after — хранится на уровне Token (token.spaces_after, Stanza v1.4+):
                           ''   = нет пробела после токена (≈ SpaceAfter=No в CoNLL-U)
                           ' '  = обычный пробел (норма)
                           '\n', '\t' и др. = нестандартные пробелы
                           None = при tokenize_pretokenized=True (исходная строка недоступна)
          - misc        — прочие MISC-поля CoNLL-U (Translit, LTranslit и др.) из token.misc;
                          НЕ содержит SpaceAfter — он вынесен в отдельное поле spaces_after
          - start_char / end_char — позиции в исходном тексте (word.start_char / word.end_char);
                                    None при tokenize_pretokenized=True

        Поля каждого токена (word_dict):
          id, form, lemma, upos, xpos  — стандарт CoNLL-U
          feats                         — dict {"Case": "Nom", "Number": "Sing", ...}
          head, deprel                  — синтаксис
          start_char, end_char          — word.start_char / word.end_char; None при pretokenized
          spaces_after                  — token.spaces_after (см. выше)
          misc          (опц.)          — dict прочих MISC-полей из token.misc
          ner           (опц.)          — token.ner: тег NER (B-PER, I-LOC, O и т.д.)

        Источники:
          - https://stanfordnlp.github.io/stanza/data_objects.html#token
          - https://github.com/stanfordnlp/stanza/issues/1315
        """
        result = []
        for sent in doc.sentences:
            sent_parsed = []

            # ========== МАППИНГ NER ИЗ TOKENS ==========
            # token.ner хранит тег именованной сущности (B-PER, I-LOC, O и т.д.)
            word_to_ner = {}
            for token in sent.tokens:
                ner_tag = token.ner if hasattr(token, 'ner') else None
                for word in token.words:
                    word_to_ner[int(word.id)] = ner_tag
            # ===========================================

            # ========== МАППИНГ MISC И SPACES_AFTER ИЗ TOKENS ==========
            # spaces_after — token.spaces_after: пробельный суффикс токена
            # misc         — token.misc: прочие CoNLL-U MISC поля (Translit и др.)
            #
            # Для MWT (Multi-Word Token) оба поля присваиваются только последнему
            # слову токена; остальные слова MWT получают None
            word_to_misc = {}
            word_to_spaces_after = {}
            for token in sent.tokens:
                misc_dict = _parse_misc_to_dict(token.misc)  # прочие MISC поля (без SpaceAfter)
                sa = token.spaces_after if hasattr(token, 'spaces_after') else None

                last_word_id = max(int(w.id) for w in token.words) if len(token.words) > 1 else None
                for word in token.words:
                    wid = int(word.id)
                    if last_word_id is None or wid == last_word_id:
                        word_to_misc[wid] = misc_dict
                        word_to_spaces_after[wid] = sa
            # ============================================================

            for word in sent.words:
                word_id = int(word.id)

                word_dict = {
                    "id":     word_id,
                    "form":   word.text,
                    "lemma":  word.lemma,
                    "upos":   word.upos,
                    "xpos":   word.xpos,
                    "feats":  _parse_misc_to_dict(word.feats),  # dict {"Animacy": "Inan", ...}
                    "head":   int(word.head),
                    "deprel": word.deprel,
                    # word.start_char / word.end_char — позиции в исходном тексте
                    # None при tokenize_pretokenized=True (исходная строка недоступна)
                    "start_char": word.start_char,
                    "end_char":   word.end_char,
                }

                # ========== ДОБАВЛЯЕМ SPACES_AFTER ИЗ МАППИНГА ==========
                # token.spaces_after — пробельный суффикс токена (Stanza v1.4+)
                # '' = нет пробела (CoNLL-U: SpaceAfter=No)
                # ' ' = обычный пробел (норма)
                # None = при tokenize_pretokenized=True
                if word_id in word_to_spaces_after:
                    word_dict["spaces_after"] = word_to_spaces_after[word_id]
                # =========================================================

                # ========== ДОБАВЛЯЕМ MISC ИЗ МАППИНГА ==========
                # Прочие CoNLL-U MISC поля из token.misc (Translit, LTranslit и др.)
                # SpaceAfter здесь НЕ присутствует — он вынесен в поле spaces_after
                if word_id in word_to_misc and word_to_misc[word_id] is not None:
                    word_dict["misc"] = word_to_misc[word_id]
                # ================================================

                # ========== ДОБАВЛЯЕМ NER ИЗ МАППИНГА ==========
                # token.ner — тег именованной сущности (B-PER, I-PER, B-LOC, ...)
                if word_id in word_to_ner and word_to_ner[word_id] is not None:
                    word_dict["ner"] = word_to_ner[word_id]
                # ===============================================

                sent_parsed.append(word_dict)

            sentence_data = {"words": sent_parsed}

            # ========== SENTIMENT И CONSTITUENCY ==========
            # sentiment    — НЕ ДОСТУПЕН для русского (только en, zh, de)
            # constituency — НЕ ДОСТУПЕН для русского
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

    print("Testing Stanza service with NER and spaces_after...")
    service = StanzaService()

    print("\n" + "=" * 70)
    print("НАТИВНЫЙ ФОРМАТ с NER и spaces_after:")
    print("=" * 70)
    results_native = service.parse.remote(test_texts[0], native_format=True)
    print(f"\nТекст: '{test_texts[0]}'\n")
    for sent_idx, sent_data in enumerate(results_native):
        words = sent_data.get("words", [])
        print(f"Предложение {sent_idx + 1}: {len(words)} токенов\n")
        for tok in words:
            ner_info = f" [NER: {tok['ner']}]" if 'ner' in tok else ""
            # spaces_after: '' = нет пробела, ' ' = норма (не выводим), None = pretokenized
            sa = tok.get('spaces_after')
            sa_info = f" [spaces_after: {repr(sa)}]" if sa != ' ' and sa is not None else ""
            print(f"  {tok['form']} ({tok['upos']}){ner_info}{sa_info}")

    print("\nTest completed!")
