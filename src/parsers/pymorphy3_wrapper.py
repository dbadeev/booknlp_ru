#!/usr/bin/env python3

"""
Обёртка для Pymorphy3 (локальный морфологический анализ).
Примитивный синтаксис: первый глагол = root, остальные зависят от него.
Это baseline, не полноценный парсер.
"""
import logging
from typing import List, Dict, Any
from razdel import tokenize as razdel_tokenize

logger = logging.getLogger(__name__)

class Pymorphy3Parser:
    """Локальный парсер на базе pymorphy3 (морфология) + примитивный синтаксис."""
    def __init__(self):
        import pymorphy3

        self.morph = pymorphy3.MorphAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pymorphy3Parser initialized (local).")

    def parse_text(self, text: str, output_format: str = "simplified") -> List[List[Dict[str, Any]]]:
        """
        Парсит текст (одно предложение).

        Аргументы:
            text (str): Входной текст для разбора.
            output_format (str): Формат выхода - "simplified" (текущий формат) или "native" (нативный формат модели).

        Возвращает: List[List[Dict]] - список предложений.

        При output_format="simplified":
            Каждое предложение - список токенов с полями:
            id, form, lemma, upos, xpos, feats, head, deprel.

        При output_format="native":
            Каждое предложение - список токенов с нативными полями Pymorphy3:
            id, word, normal_form, tag, score, methods_stack, lexeme.
        """
        tokens = [t.text for t in razdel_tokenize(text)]
        if not tokens:
            return []

        # ============================================================
        # БЛОК: Выбор формата выхода в зависимости от параметра
        # ============================================================
        if output_format == "native":
            # Нативный формат: возвращаем все данные объекта Parse
            return self._parse_native(tokens)
        else:
            # Упрощенный формат (текущая логика): возвращаем CoNLL-подобную структуру
            return self._parse_simplified(tokens)

    # ============================================================
    # БЛОК: Подготовка нативного выхода модели
    # ============================================================
    def _parse_native(self, tokens: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Подготавливает нативный выход модели Pymorphy3.

        Возвращает полную информацию из объекта Parse для каждого токена:
        - word: исходное слово
        - normal_form: нормальная форма (лемма)
        - tag: полный тег OpenCorpora (строковое представление)
        - score: вероятность данного разбора (float)
        - methods_stack: стек методов разбора (список кортежей)
        - lexeme: парадигма слова - все словоформы (список объектов Parse)
        - is_known: True если форма есть в словаре (bool)
        - normalized: объект Parse для нормальной формы (dict)
        """
        sent: List[Dict[str, Any]] = []

        for i, tok in enumerate(tokens, 1):
            p = self.morph.parse(tok)[0]

            # ============================================================
            # Извлекаем все нативные поля объекта Parse
            # ============================================================
            native_token = {
                "id": i,
                "word": p.word,
                "normal_form": p.normal_form,
                "tag": str(p.tag),
                "score": p.score,
                "methods_stack": p.methods_stack,
                "lexeme": [str(form.tag) for form in p.lexeme],
                "is_known": p.is_known,  # ← ДОБАВЛЕНО!
                "normalized": {  # ← ДОБАВЛЕНО!
                    "word": p.normalized.word,
                    "tag": str(p.normalized.tag),
                    "score": p.normalized.score
                }
            }

            sent.append(native_token)

        return [sent]

    # ============================================================
    # БЛОК: Упрощенный формат (текущая логика без изменений)
    # ============================================================
    def _parse_simplified(self, tokens: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Подготавливает упрощенный выход (текущий формат).

        Возвращает CoNLL-подобную структуру с полями:
        id, form, lemma, upos, xpos, feats, head, deprel.

        Аргументы:
            tokens (List[str]): Список токенов для разбора.

        Возвращает:
            List[List[Dict]]: Список предложений с упрощенными полями.
        """
        sent: List[Dict[str, Any]] = []
        root_idx = None

        for i, tok in enumerate(tokens, 1):
            p = self.morph.parse(tok)[0]
            upos = p.tag.POS or "X"
            lemma = p.normal_form

            # Примитивная эвристика: первый глагол становится root
            if upos in {"VERB", "INFN"} and root_idx is None:
                head = 0
                deprel = "root"
                root_idx = i

            else:
                head = root_idx if root_idx is not None else 0
                deprel = "dep" if head != 0 else "root"

            sent.append(
                {
                    "id": i,
                    "form": tok,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": str(p.tag),
                    "feats": "_",
                    "head": head,
                    "deprel": deprel,
                }
            )

        return [sent]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = Pymorphy3Parser()
    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)
    result_simplified = parser.parse_text(test_text, output_format="simplified")

    for sent in result_simplified:
        for tok in sent:
            print(f"{tok['id']}\t{tok['form']}\t{tok['lemma']}\t{tok['upos']}\t{tok['head']}\t{tok['deprel']}")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 60)
    result_native = parser.parse_text(test_text, output_format="native")

    for sent in result_native:

        for tok in sent:
            print(f"ID: {tok['id']}")
            print(f"  Word: {tok['word']}")
            print(f"  Normal form: {tok['normal_form']}")
            print(f"  Tag: {tok['tag']}")
            print(f"  Score: {tok['score']}")
            print(f"  Lexeme (forms): {tok['lexeme'][:3]}...")  # Показываем первые 3 формы
            print(f"  Methods stack: {tok['methods_stack']}")
            print(f"  Is known: {tok['is_known']}")
            print(f"  Normalized: {tok['normalized']}")
            print()
