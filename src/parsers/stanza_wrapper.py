#!/usr/bin/env python3
"""
Обёртка для Stanza (через Modal) с поддержкой нативного формата.
Включает корректную обработку полей spaces_after, misc и ner.
"""

import logging
import modal
from typing import List, Dict, Any, Union


class StanzaParser:
    """
    Клиент для Stanza, запущенного в Modal.
    Stanza выполняет полный морфо-синтаксический анализ + NER для русского языка.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-stanza", "StanzaService")()
            self.logger.info("Connected to Stanza via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(
        self, text: str, native_format: bool = False
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Если native_format=False (по умолчанию): List[List[Dict]]
          CoNLL-U-совместимый формат; список предложений, каждое — список токенов.
          Поля токена: id, form, lemma, upos, xpos, feats (строка), head, deprel, misc (строка)
          Отсутствуют: start_char, end_char, spaces_after — не входят в стандарт CoNLL-U.

        Если native_format=True: List[Dict]
          Нативный формат; список предложений, каждое — dict с ключами:
            "words"        : List[Dict] — токены предложения
            "sentiment"    : str        — (только en/zh/de, для ru отсутствует)
            "constituency" : str        — (только en, для ru отсутствует)

          Каждый токен ("words") содержит:
            id, form, lemma, upos, xpos  — стандарт CoNLL-U
            feats        : dict {"Case": "Nom", "Number": "Sing", ...}
            head, deprel : синтаксис
            start_char   : int  — word.start_char, позиция начала в тексте
            end_char     : int  — word.end_char, позиция конца в тексте
            spaces_after : str  — token.spaces_after (Stanza v1.4+):
                                  '' = нет пробела (≈ SpaceAfter=No в CoNLL-U)
                                  ' ' = обычный пробел (норма)
                                  '\n', '\t' и др. = нестандартные пробелы
            misc (опц.)  : dict — прочие MISC поля CoNLL-U из token.misc
                                  (Translit, LTranslit и др.); НЕ содержит SpaceAfter
            ner  (опц.)  : str  — token.ner: тег NER ("B-PER", "I-LOC", "O" и др.)
        """
        try:
            return self.service.parse.remote(text, native_format=native_format)
        except Exception as e:
            self.logger.error(f"Error during Stanza parsing: {e}")
            raise e

    def parse_batch(
        self,
        batch_tokens: list[list[str]],
        native_format: bool = False,
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Метод принимает готовые токены (список предложений, каждое — список строк).
        Использует nlp_pretokenized на сервере.

        Параметры:
          batch_tokens : list[list[str]]
            Предтокенизированные предложения.
            Пример: [["Москва", "—", "столица"], ["Привет", "!"]]
          native_format : bool, optional (default=False)
            Формат вывода — аналогично parse_text.

        ВАЖНО (при tokenize_pretokenized=True):
          - spaces_after = None: исходная строка недоступна, пробелы неизвестны
          - start_char = None, end_char = None: символьные позиции недоступны
        """
        try:
            return self.service.parse_batch.remote(batch_tokens, native_format=native_format)
        except Exception as e:
            self.logger.error(f"Error during batch parsing: {e}")
            raise


# ============================================================
# БЛОК: Тестовые примеры использования wrapper
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = StanzaParser()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # ============================================================
    # Демонстрация работы в упрощенном формате (CoNLL-U)
    # ============================================================
    print("=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (CoNLL-U):")
    print("=" * 60)
    result = parser.parse_text(test_text, native_format=False)
    print("Stanza Test:")
    for sent in result:
        for tok in sent:
            # print(
            #     f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t"
            #     f"{tok.get('upos')}\t{tok.get('head')}\t{tok.get('deprel')}\t"
            #     f"{tok.get('misc') or '_'}"
            # )
            print(
                f"{tok.get('id')}\t"
                f"{tok.get('form')}\t"
                f"{tok.get('lemma')}\t"
                f"{tok.get('upos')}\t"
                f"{tok.get('xpos') or '_'}\t"  # колонка 5: XPOS
                f"{tok.get('feats') or '_'}\t"  # колонка 6: FEATS (строка CoNLL-U)
                f"{tok.get('head')}\t"
                f"{tok.get('deprel')}\t"
                f"_\t"  # колонка 9: DEPS (enhanced deps — Stanza не выдаёт)
                f"{tok.get('misc') or '_'}"  # колонка 10: MISC
            )

    # ============================================================
    # Демонстрация работы в нативном формате с NER
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native) с NER:")
    print("=" * 60)
    result_native = parser.parse_text(test_text, native_format=True)
    print("Stanza Test (Native):")

    for sent_data in result_native:
        words = sent_data.get("words", [])
        print(f"\nПредложение содержит {len(words)} токенов")

        # ========== ПОЯСНЕНИЕ ПРО SENTIMENT И CONSTITUENCY ==========
        # Для русского языка эти процессоры не доступны
        if "sentiment" in sent_data:
            print(f"Sentiment: {sent_data['sentiment']}")
        else:
            print("Sentiment: не доступен для русского языка (только en/zh/de)")
        if "constituency" in sent_data:
            print(f"Constituency: {sent_data['constituency'][:50]}...")
        else:
            print("Constituency: не доступен для русского языка")

        print("\nТокены с NER:")
        for tok in words:
            print(f"\nText: {tok.get('form')}")
            print(f"  id: {tok.get('id')}")
            print(f"  lemma: {tok.get('lemma')}, upos: {tok.get('upos')}, xpos: {tok.get('xpos')}")
            print(f"  feats: {tok.get('feats')}")
            print(f"  head: {tok.get('head')}, deprel: {tok.get('deprel')}")
            # start_char / end_char: позиции в исходном тексте (word.start_char / word.end_char)
            print(f"  start_char: {tok.get('start_char')}, end_char: {tok.get('end_char')}")
            # spaces_after: token.spaces_after — '' = нет пробела, ' ' = норма, None = pretokenized
            print(f"  spaces_after: {repr(tok.get('spaces_after'))}")
            # misc: прочие CoNLL-U MISC поля (Translit и др.), SpaceAfter здесь отсутствует
            if 'misc' in tok:
                print(f"  misc: {tok.get('misc')}")
            else:
                print("  misc: None")
            # ner: тег именованной сущности из token.ner
            if 'ner' in tok:
                print(f"  ner: {tok.get('ner')}")
            else:
                print("  ner: O")

    # ============================================================
    # Статистика именованных сущностей
    # ============================================================
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ИМЕНОВАННЫХ СУЩНОСТЕЙ:")
    print("=" * 60)
    all_words = [tok for sent_data in result_native for tok in sent_data.get("words", [])]
    ner_tags = [tok.get('ner', 'O') for tok in all_words]
    print(f"Всего токенов: {len(ner_tags)}")
    print(f"Персоны (PER): {sum(1 for t in ner_tags if t and t.endswith('PER'))}")
    print(f"Локации (LOC): {sum(1 for t in ner_tags if t and t.endswith('LOC'))}")
    print(f"Организации (ORG): {sum(1 for t in ner_tags if t and t.endswith('ORG'))}")

    # ============================================================
    # Все ключи первого токена и предложения
    # ============================================================
    print("\n" + "=" * 60)
    print("ВСЕ КЛЮЧИ ПЕРВОГО ТОКЕНА:")
    print("=" * 60)
    if result_native and result_native[0] and result_native[0].get("words"):
        first_token = result_native[0]["words"][0]
        print(f"Ключи: {list(first_token.keys())}")
        print("Значения:")
        for key, value in first_token.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("ВСЕ КЛЮЧИ ПРЕДЛОЖЕНИЯ:")
    print("=" * 60)
    if result_native and result_native[0]:
        print(f"Ключи предложения: {list(result_native[0].keys())}")

    # ============================================================
    # Демонстрация parse_batch (предтокенизированный режим)
    # ============================================================
    print("\n" + "=" * 60)
    print("PARSE_BATCH (предтокенизированный ввод):")
    print("=" * 60)
    batch = [
        ["Зло", ",", "которым", "ты", "меня", "пугаешь"],
        ["Москва", "—", "столица", "России", "."],
    ]
    result_batch = parser.parse_batch(batch, native_format=False)
    for sent in result_batch:
        for tok in sent:
            print(
                f"{tok.get('id')}\t"
                f"{tok.get('form')}\t"
                f"{tok.get('lemma')}\t"
                f"{tok.get('upos')}\t"
                f"{tok.get('xpos') or '_'}\t"  # колонка 5: XPOS
                f"{tok.get('feats') or '_'}\t"  # колонка 6: FEATS (строка CoNLL-U)
                f"{tok.get('head')}\t"
                f"{tok.get('deprel')}\t"
                f"_\t"  # колонка 9: DEPS (enhanced deps — Stanza не выдаёт)
                f"{tok.get('misc') or '_'}"  # колонка 10: MISC
            )
        print()
