#!/usr/bin/env python3

"""
Обёртка для Pymorphy3 (локальный морфологический анализ).
Примитивный синтаксис: первый глагол = root, остальные зависят от него.
Это baseline, не полноценный парсер.
"""
import logging
from typing import List, Dict, Any
from razdel import tokenize as razdel_tokenize, sentenize

# logger = logging.getLogger(__name__)

_OPENCORPORA_TO_UPOS = {
    "NOUN": "NOUN", "ADJF": "ADJ",  "ADJS": "ADJ",
    "COMP": "ADJ",  "VERB": "VERB", "INFN": "VERB",
    "PRTF": "ADJ",  "PRTS": "ADJ",  "GRND": "VERB",
    "NUMR": "NUM",  "ADVB": "ADV",  "NPRO": "PRON",
    "PRED": "ADV",  "PREP": "ADP",  "CONJ": "CCONJ",
    "PRCL": "PART", "INTJ": "INTJ", "LATN": "X",
    "ROMN": "X",    "PNCT": "PUNCT","UNKN": "X",
}

_SCONJ_SET = {
    "что", "чтобы", "как", "когда", "если", "хотя", "пока",
    "потому", "поскольку", "хоть", "будто", "словно", "ибо",
    "раз", "коли", "дабы", "лишь", "едва",
}


def _tag_to_feats(tag) -> str:
    """Конвертирует OpenCorpora тег в CoNLL-U FEATS строку."""
    mapping = {
        # Падеж
        "nomn": "Case=Nom", "gent": "Case=Gen", "datv": "Case=Dat",
        "accs": "Case=Acc", "ablt": "Case=Ins", "loct": "Case=Loc",
        # Число
        "sing": "Number=Sing", "plur": "Number=Plur",
        # Род
        "masc": "Gender=Masc", "femn": "Gender=Fem", "neut": "Gender=Neut",
        # Время
        "past": "Tense=Past", "pres": "Tense=Pres", "futr": "Tense=Fut",
        # Наклонение
        "indc": "Mood=Ind", "impr": "Mood=Imp",
        # Вид
        "perf": "Aspect=Perf", "impf": "Aspect=Imp",
    }
    grammemes = {g.lower() for g in tag.grammemes}
    feats = sorted(v for k, v in mapping.items() if k in grammemes)
    return "|".join(feats) if feats else "_"

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
        all_sentences = []
        for sentence in sentenize(text):
            tokens = [t.text for t in razdel_tokenize(sentence.text)]
            if not tokens:
                continue
            # ============================================================
            # БЛОК: Выбор формата выхода в зависимости от параметра
            # ============================================================
            if output_format == "native":
                # Нативный формат: возвращаем все данные объекта Parse
                all_sentences.extend(self._parse_native(tokens))
            else:
                # Упрощенный формат (текущая логика): возвращаем CoNLL-подобную структуру
                all_sentences.extend(self._parse_simplified(tokens))
        return all_sentences
        # tokens = [t.text for t in razdel_tokenize(text)]
        # if not tokens:
        #     return []
        #
        # # ============================================================
        # # БЛОК: Выбор формата выхода в зависимости от параметра
        # # ============================================================
        # if output_format == "native":
        #     # Нативный формат: возвращаем все данные объекта Parse
        #     return self._parse_native(tokens)
        # else:
        #     # Упрощенный формат (текущая логика): возвращаем CoNLL-подобную структуру
        #     return self._parse_simplified(tokens)

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
                "word": tok,  # ИСПРАВЛЕНО: оригинальный токен, не p.word
                "word_lower": p.word,  # ДОБАВЛЕНО: pymorphy3-версия (lowercase)
                "normal_form": p.normal_form,
                "tag": str(p.tag),
                "score": p.score,
                # конвертируем в строки:
                "methods_stack": [
                    (type(item[0]).__name__,) + tuple(str(v) for v in item[1:])
                    for item in p.methods_stack
                ],
                # "methods_stack": p.methods_stack,
                # "lexeme": [str(form.tag) for form in p.lexeme],
                "lexeme": [(form.word, str(form.tag)) for form in p.lexeme],
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
            # upos = p.tag.POS or "X"
            # сначала берём OpenCorpora POS, потом конвертируем в UPOS:
            oc_pos = p.tag.POS

            if oc_pos is None:
                # PNCT есть в grammemes, но не в PARTS_OF_SPEECH pymorphy3
                oc_pos = "PNCT" if "PNCT" in p.tag.grammemes else "UNKN"
            if oc_pos == "CONJ":
                upos = "SCONJ" if tok.lower() in _SCONJ_SET else "CCONJ"
            else:
                upos = _OPENCORPORA_TO_UPOS.get(oc_pos, "X")

            lemma = p.normal_form

            # # Примитивная эвристика: первый глагол становится root
            # if upos in {"VERB", "INFN"} and root_idx is None:
            #     head = 0
            #     deprel = "root"
            #     root_idx = i
            # else:
            #     # токены до глагола явно помечаются dep с head=0 (pending):
            #     if root_idx is not None:
            #         head, deprel = root_idx, "dep"
            #     else:
            #         # Глагол ещё не найден: временно вешаем на 0, пометим как dep
            #         # (пересчитать после прохода всего предложения — см. N6b)
            #         head, deprel = 0, "dep"
            #     # head = root_idx if root_idx is not None else 0
            #     # deprel = "dep" if head != 0 else "root"

            # все глагольные формы OpenCorpora которые дают UPOS="VERB":
            if oc_pos in {"VERB", "INFN", "GRND"} and root_idx is None:
                head = 0
                deprel = "root"
                root_idx = i
            else:
                if root_idx is not None:
                    head, deprel = root_idx, "dep"
                else:
                    head, deprel = 0, "dep"

            sent.append(
                {
                    "id": i,
                    "form": tok,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": str(p.tag),
                    "feats": _tag_to_feats(p.tag),
                    "head": head,
                    "deprel": deprel,
                }
            )

        # FIX: токены до первого глагола получили head=0, deprel=dep —
        # невалидный CoNLL-U. После прохода переназначаем их на root.
        if root_idx is not None:
            for tok in sent:
                if tok["head"] == 0 and tok["deprel"] == "dep":
                    tok["head"] = root_idx  # теперь dep от root, а не от 0

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
