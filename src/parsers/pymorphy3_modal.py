#!/usr/bin/env python3
"""
Modal-сервис для Pymorphy3.

Вся логика морфологического анализа и форматирования.
Принимает pre-split чанки из wrapper-а.

Поскольку pymorphy3 не имеет встроенного токенизатора,
токенизация слов всегда выполняется через razdel.tokenize.

Производственные методы (вызываются из wrapper):
  parse_sentence_chunk        — razdel path: List[(text, start_char)]
  parse_sentence_chunk_native — native path: List[str]  (без офсетов)

Запуск тестов:
  modal run src/parsers/pymorphy3_modal.py
"""
import modal
import logging
from typing import Any, Dict, List, Literal, Tuple

# ─── Modal image ──────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pymorphy3>=1.2.0",
        "pymorphy3-dicts-ru>=2.4.0",
        "razdel>=0.5.0",
    )
)

app = modal.App("booknlp-ru-pymorphy3")

OutputFormat = Literal["simplified", "native"]

# ─── Константы (перенесены из wrapper) ───────────────────────────────────────

_OPENCORPORA_TO_UPOS: Dict[str, str] = {
    "NOUN": "NOUN", "ADJF": "ADJ",  "ADJS": "ADJ",
    "COMP": "ADJ",  "VERB": "VERB", "INFN": "VERB",
    "PRTF": "ADJ",  "PRTS": "ADJ",  "GRND": "VERB",
    "NUMR": "NUM",  "ADVB": "ADV",  "NPRO": "PRON",
    "PRED": "ADV",  "PREP": "ADP",  "CONJ": "CCONJ",
    "PRCL": "PART", "INTJ": "INTJ", "LATN": "X",
    "ROMN": "X",    "PNCT": "PUNCT","UNKN": "X",
}

_SCONJ_SET: set = {
    "что", "чтобы", "как", "когда", "если", "хотя", "пока",
    "потому", "поскольку", "хоть", "будто", "словно", "едва",
    "раз", "коли", "дабы", "лишь",
}


def _tag_to_feats(tag) -> str:
    """Конвертирует OpenCorpora тег в CoNLL-U FEATS строку."""
    mapping = {
        # Case
        "nomn": "Case=Nom", "gent": "Case=Gen", "datv": "Case=Dat",
        "accs": "Case=Acc", "ablt": "Case=Ins", "loct": "Case=Loc",
        "voct": "Case=Voc",
        # Number
        "sing": "Number=Sing", "plur": "Number=Plur",
        # Gender
        "masc": "Gender=Masc", "femn": "Gender=Fem", "neut": "Gender=Neut",
        # Tense
        "past": "Tense=Past", "pres": "Tense=Pres", "futr": "Tense=Fut",
        # Mood
        "indc": "Mood=Ind", "impr": "Mood=Imp",
        # Aspect
        "perf": "Aspect=Perf", "impf": "Aspect=Imp",
        # Person
        "1per": "Person=1", "2per": "Person=2", "3per": "Person=3",
        # Animacy
        "anim": "Animacy=Anim", "inan": "Animacy=Inan",
        # Voice
        "actv": "Voice=Act", "pssv": "Voice=Pass",
        # Variant
        "Shrt": "Variant=Short",
        # Abbr
        "Abbr": "Abbr=Yes",
    }
    grammemes = set(tag.grammemes)
    feats = sorted(v for k, v in mapping.items() if k in grammemes)
    return "|".join(feats) if feats else "_"


# ─── Service ──────────────────────────────────────────────────────────────────

@app.cls(image=image, timeout=600, scaledown_window=300)
class Pymorphy3Service:
    """
    Modal-сервис: морфологический анализ через pymorphy3.

    Форматы вывода:
      simplified — CoNLL-U-подобный (id, form, lemma, upos, xpos, feats, head, deprel)
      native     — нативный формат pymorphy3 (все морфологические данные)

    Токенизация слов — всегда razdel.tokenize.
    Сентенизация — в wrapper-е (до отправки в Modal).
    """

    @modal.enter()
    def setup(self):
        import pymorphy3
        from razdel import tokenize as razdel_tokenize

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Pymorphy3Service")
        self.morph = pymorphy3.MorphAnalyzer()
        self.razdel_tokenize = razdel_tokenize
        self.logger.info("Pymorphy3Service initialized!")

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _parse_tokens_simplified(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Разбирает список токенов → упрощённый CoNLL-U-подобный формат.

        Эвристика синтаксиса: первый VERB/INFN/GRND = root,
        все остальные → dep от root.

        Returns:
            List[Dict]  — одно предложение
        """
        sent: List[Dict[str, Any]] = []
        root_idx = None

        for i, tok in enumerate(tokens, 1):
            p = self.morph.parse(tok)[0]
            oc_pos = p.tag.POS
            if oc_pos is None:
                oc_pos = "PNCT" if "PNCT" in p.tag.grammemes else "UNKN"
            if oc_pos == "CONJ":
                upos = "SCONJ" if tok.lower() in _SCONJ_SET else "CCONJ"
            else:
                upos = _OPENCORPORA_TO_UPOS.get(oc_pos, "X")

            if oc_pos in {"VERB", "INFN", "GRND"} and root_idx is None:
                head, deprel = 0, "root"
                root_idx = i
            else:
                head, deprel = (root_idx, "dep") if root_idx is not None else (0, "dep")

            sent.append({
                "id":     i,
                "form":   tok,
                "lemma":  p.normal_form,
                "upos":   upos,
                "xpos":   str(p.tag),
                "feats":  _tag_to_feats(p.tag),
                "head":   head,
                "deprel": deprel,
            })

        # FIX: токены до первого глагола получили head=0, deprel=dep;
        # переключаем их на root, если глагол найден.
        if root_idx is not None:
            for token_dict in sent:
                if token_dict["head"] == 0 and token_dict["deprel"] == "dep":
                    token_dict["head"] = root_idx

        return sent

    def _parse_tokens_native(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Разбирает список токенов → нативный формат pymorphy3.

        Returns:
            List[Dict]  — одно предложение
        """
        sent: List[Dict[str, Any]] = []
        for i, tok in enumerate(tokens, 1):
            p = self.morph.parse(tok)[0]
            sent.append({
                "id":          i,
                "word":        tok,
                "word_lower":  p.word,
                "normal_form": p.normal_form,
                "tag":         str(p.tag),
                "score":       p.score,
                "methods_stack": [
                    (type(item[0]).__name__,) + tuple(str(v) for v in item[1:])
                    for item in p.methods_stack
                ],
                "lexeme":     [(form.word, str(form.tag)) for form in p.lexeme],
                "is_known":   p.is_known,
                "normalized": {
                    "word":  p.normalized.word,
                    "tag":   str(p.normalized.tag),
                    "score": p.normalized.score,
                },
            })
        return sent

    # ─── Production methods (called from wrapper) ─────────────────────────────

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: str = "simplified",
    ) -> List[List[Dict[str, Any]]]:
        """
        Razdel path.

        Принимает чанк пар (sentence_text, start_char_in_original).
        start_char — для восстановления позиций в исходном тексте.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format: 'simplified' | 'native'
        Returns:
            List[List[Dict]]  — список предложений, каждое — список токенов
        """
        from typing import Callable

        if output_format not in ("simplified", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")

        parse_fn: Callable[[List[str]], List[Dict[str, Any]]] = (
            self._parse_tokens_native if output_format == "native"
            else self._parse_tokens_simplified
        )
        result: List[List[Dict[str, Any]]] = []
        for sent_text, _start_char in sentences_with_offsets:
            tokens = [t.text for t in self.razdel_tokenize(sent_text)]
            if tokens:
                result.append(parse_fn(tokens))
        return result

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: str = "simplified",
    ) -> List[List[Dict[str, Any]]]:
        """
        Native path (только тексты предложений, без офсетов).

        Args:
            sentences:     List[str]
            output_format: 'simplified' | 'native'
        Returns:
            List[List[Dict]]
        """
        from typing import Callable

        if output_format not in ("simplified", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")

        parse_fn: Callable[[List[str]], List[Dict[str, Any]]] = (
            self._parse_tokens_native if output_format == "native"
            else self._parse_tokens_simplified
        )
        result: List[List[Dict[str, Any]]] = []
        for sent_text in sentences:
            tokens = [t.text for t in self.razdel_tokenize(sent_text)]
            if tokens:
                result.append(parse_fn(tokens))
        return result

    # ─── Backward compat ──────────────────────────────────────────────────────

    @modal.method()
    def parse(
        self,
        text: str,
        output_format: str = "simplified",
    ) -> List[List[Dict[str, Any]]]:
        """Парсит текст целиком. Для local_entrypoint и прямых вызовов."""
        from razdel import sentenize
        from typing import Callable

        if output_format not in ("simplified", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")

        parse_fn: Callable[[List[str]], List[Dict[str, Any]]] = (
            self._parse_tokens_native if output_format == "native"
            else self._parse_tokens_simplified
        )
        result: List[List[Dict[str, Any]]] = []
        for s in sentenize(text):
            tokens = [t.text for t in self.razdel_tokenize(s.text)]
            if tokens:
                result.append(parse_fn(tokens))
        return result

# ─── local_entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    Тестирует Pymorphy3Service напрямую — без wrapper.
    Запуск: modal run src/parsers/pymorphy3_modal.py

    Тест-секции:
      [1] parse_sentence_chunk   — razdel path, simplified
      [2] parse_sentence_chunk   — razdel path, native
      [3] parse_sentence_chunk_native — native path, simplified
      [4] parse_sentence_chunk_native — native path, native
      [5] Проверка офсетов (razdel path)
      [6] Пустые предложения — не падает
      [7] Неверный output_format — ValueError
    """
    from razdel import sentenize

    # ─── Вспомогательные функции вывода ──────────────────────────────────────────

    _HEADER = "ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL"

    def print_simplified(sentence, sentence_text: str = ""):
        if sentence_text:
            print(f"# text = {sentence_text}")
        print(_HEADER)
        for token in sentence:
            fields = [
                str(token["id"]),
                token["form"],
                token["lemma"],
                token["upos"],
                token["xpos"],
                token["feats"],
                str(token["head"]),
                token["deprel"],
            ]
            print("\t".join(fields))

    def print_native(sentence: List[Dict[str, Any]]) -> None:
        for token in sentence:
            print(f"ID: {token['id']}")
            print(f"  Word: {token['word']}")
            print(f"  Normal form: {token['normal_form']}")
            print(f"  Tag: {token['tag']}")
            print(f"  Score: {token['score']}")
            print(f"  Lexeme (forms): {token['lexeme'][:3]}...")
            print(f"  Methods stack: {token['methods_stack']}")
            print(f"  Is known: {token['is_known']}")
            print(f"  Normalized: {token['normalized']}")
            print()

    service = Pymorphy3Service()
    sep = "=" * 72
    passed = 0
    failed = 0

    def ok(name: str):
        nonlocal passed
        passed += 1
        print(f"  ✅  {name}")

    def fail(name: str, err):
        nonlocal failed
        failed += 1
        print(f"  ❌  {name}: {err}")

    text_sample  = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    multi_sample = "Зло пугает. Москва — столица России. Крупнейший город страны."

    # ── [1] parse_sentence_chunk — razdel path, simplified ────────────────────
    print(f"\n{sep}")
    print("[1] parse_sentence_chunk  (razdel path, simplified)")
    print(sep)
    try:
        sentences = list(sentenize(multi_sample))
        chunk = [(s.text, s.start) for s in sentences]
        result = service.parse_sentence_chunk.remote(chunk, output_format="simplified")

        assert isinstance(result, list),               "результат не list"
        assert len(result) == len(sentences),           f"ожидалось {len(sentences)} предл., получено {len(result)}"
        for sent in result:
            assert isinstance(sent, list),             "предложение не list"
            assert len(sent) > 0,                      "пустое предложение"
            for tok in sent:
                for key in ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel"):
                    assert key in tok,                 f"ключ {key!r} отсутствует"
                assert tok["id"] >= 1,                 "id < 1"
            # FIX: нет токенов с head=0 deprel=dep
            roots = [t for t in sent if t["deprel"] == "root"]
            if roots:  # проверяем только если глагол найден
                bad = [t for t in sent if t["head"] == 0 and t["deprel"] == "dep"]
                assert bad == [], f"head=0 deprel=dep при наличии root: ..."

        for sent, (sent_text, _) in zip(result, chunk):
            print_simplified(sent, sent_text)
            print()
        ok("parse_sentence_chunk / simplified — структура и офсеты корректны")
    except Exception as e:
        fail("parse_sentence_chunk / simplified", e)

    # ── [2] parse_sentence_chunk — razdel path, native ────────────────────────
    print(f"\n{sep}")
    print("[2] parse_sentence_chunk  (razdel path, native)")
    print(sep)
    try:
        sentences = list(sentenize(text_sample))
        chunk = [(s.text, s.start) for s in sentences]
        result = service.parse_sentence_chunk.remote(chunk, output_format="native")

        assert isinstance(result, list)
        for sent in result:
            for tok in sent:
                for key in ("id", "word", "normal_form", "tag", "score",
                            "methods_stack", "lexeme", "is_known", "normalized"):
                    assert key in tok, f"ключ {key!r} отсутствует"
                assert 0.0 <= tok["score"] <= 1.0,     "score вне диапазона [0, 1]"
                assert isinstance(tok["lexeme"], list), "lexeme не list"
                assert isinstance(tok["is_known"], bool)

        for sent in result:
            print_native(sent)
        ok("parse_sentence_chunk / native — структура корректна")
    except Exception as e:
        fail("parse_sentence_chunk / native", e)

    # ── [3] parse_sentence_chunk_native — native path, simplified ─────────────
    print(f"\n{sep}")
    print("[3] parse_sentence_chunk_native  (native path, simplified)")
    print(sep)
    try:
        sentences = list(sentenize(multi_sample))
        chunk_texts = [s.text for s in sentences]
        result = service.parse_sentence_chunk_native.remote(
            chunk_texts, output_format="simplified"
        )
        assert len(result) == len(sentences)
        for sent in result:
            for tok in sent:
                assert "form" in tok and "upos" in tok

        for sent, sent_text in zip(result, chunk_texts):
            print_simplified(sent, sent_text)
            print()
        ok("parse_sentence_chunk_native / simplified — структура корректна")
    except Exception as e:
        fail("parse_sentence_chunk_native / simplified", e)

    # ── [4] parse_sentence_chunk_native — native path, native ────────────────
    print(f"\n{sep}")
    print("[4] parse_sentence_chunk_native  (native path, native)")
    print(sep)
    try:
        sentences = list(sentenize(text_sample))
        chunk_texts = [s.text for s in sentences]
        result = service.parse_sentence_chunk_native.remote(
            chunk_texts, output_format="native"
        )
        assert isinstance(result, list)
        for sent in result:
            for tok in sent:
                assert "word" in tok and "tag" in tok

        for sent in result:
            print_native(sent)
        ok("parse_sentence_chunk_native / native — структура корректна")
    except Exception as e:
        fail("parse_sentence_chunk_native / native", e)

    # ── [5] Проверка офсетов ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[5] Проверка символьных офсетов (razdel path)")
    print(sep)
    try:
        sentences = list(sentenize(multi_sample))
        chunk = [(s.text, s.start) for s in sentences]
        for sent_text, start in chunk:
            assert multi_sample[start:start + len(sent_text)] == sent_text, \
                f"Офсет {start} не совпадает с текстом {sent_text!r}"
            print(f"  offset={start:3d}  text={sent_text!r}")
        ok("Символьные офсеты корректны")
    except Exception as e:
        fail("Офсеты", e)

    # ── [6] Пустой чанк — не должен падать ────────────────────────────────────
    print(f"\n{sep}")
    print("[6] Пустой чанк — не падает")
    print(sep)
    try:
        result = service.parse_sentence_chunk.remote([], output_format="simplified")
        assert result == [], f"Ожидался [], получено {result!r}"
        result_n = service.parse_sentence_chunk_native.remote([], output_format="simplified")
        assert result_n == []
        ok("Пустой чанк → []")
    except Exception as e:
        fail("Пустой чанк", e)

    # ── [7] Неверный output_format — ValueError ────────────────────────────────
    print(f"\n{sep}")
    print("[7] Неверный output_format → ValueError")
    print(sep)
    try:
        try:
            service.parse_sentence_chunk.remote([("Текст.", 0)], output_format="conllu")
            fail("ValueError не выброшен", "исключение не возникло")
        except (ValueError, Exception) as exc:
            assert "output_format" in str(exc).lower() or "conllu" in str(exc).lower() or \
                   "unknown" in str(exc).lower(), f"Неожиданное сообщение: {exc}"
            print(f"  Поймано: {exc!r}")
        ok("Неверный output_format → ValueError")
    except Exception as e:
        fail("ValueError", e)

    # ── Итог ──────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    total = passed + failed
    print(f"ИТОГ: {passed}/{total} тестов прошло" + (" ✅" if failed == 0 else f"  ❌ {failed} упало"))
    print(sep)
