#!/usr/bin/env python3
"""
Slovnet Modal Service для booknlp_ru.

Единый публичный метод: parse_text(text, output_format)

  output_format="conllu"  →  List[List[Dict]]
      Ключи: id, form, lemma, upos, xpos,
             feats (Animacy=Anim|Case=Nom|...),   ← CoNLL-U строка
             head, deprel, deps, misc,
             startchar, endchar                    ← символьные смещения

  output_format="native"  →  Dict{"tokens": [...], "spans": [...]}
      Ключи токена: id, text, pos, feats (dict|None),
                    head_id (str), rel,
                    start, stop                    ← символьные смещения
      Ключи span:   start, stop, type,
                    [text, normal, fact]            ← глобальные смещения

Токенизация: razdel sentenize + tokenize (внутри сервиса).
Морфология:  Slovnet Morph.
Синтаксис:   Slovnet Syntax.
NER:         Natasha (только output_format="native").
"""

import logging
import sys
from typing import Union
from typing import List, Tuple, Dict, Any

import modal

# ─────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────

LOCAL_MODELS_DIR = "models"
REMOTE_ROOT      = "/root/booknlp_ru"
REMOTE_SRC       = f"{REMOTE_ROOT}/src"
REMOTE_MODELS    = f"{REMOTE_ROOT}/models"

# ─────────────────────────────────────────────────────────────
# DOCKER ОБРАЗ
# ─────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("slovnet", "navec", "razdel")
    .pip_install("natasha", "pymorphy2-dicts-ru", "numpy")
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_SRC}:$PYTHONPATH"})
    .add_local_dir(LOCAL_MODELS_DIR, remote_path=REMOTE_MODELS, copy=True)
)

app = modal.App("booknlp-ru-slovnet")

# ─────────────────────────────────────────────────────────────
# СЕРВИС
# ─────────────────────────────────────────────────────────────

@app.cls(image=image, timeout=600, cpu=2.0)
class SlovnetService:
    def __init__(self):
        # Атрибуты инициализируются в @modal.enter() (setup).
        # Заглушки для PyCharm — Modal не вызывает __init__.
        self.logger = None
        self.navec = None
        self.syntax = None
        self.morph = None
        self.segmenter = None
        self.morph_vocab = None
        self.emb = None
        self.morph_tagger = None
        self.syntax_parser = None
        self.ner_tagger = None
        self.names_extractor = None
        self.PER = None

    @modal.enter()
    def setup(self):
        from pathlib import Path

        from navec import Navec
        from natasha import (
            MorphVocab, NamesExtractor, NewsEmbedding,
            NewsMorphTagger, NewsNERTagger, NewsSyntaxParser,
            PER, Segmenter,
        )
        from slovnet import Morph, Syntax

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SlovnetService")

        for p in (REMOTE_ROOT, REMOTE_SRC):
            if p not in sys.path:
                sys.path.append(p)

        models_path = Path(REMOTE_MODELS)

        # ── Navec ────────────────────────────────────────────
        self.navec = Navec.load(
            models_path / "navec_news_v1_1B_250K_300d_100q.tar"
        )
        self.logger.info("✓ Navec loaded")

        # ── Slovnet Syntax ───────────────────────────────────
        self.syntax = Syntax.load(models_path / "slovnet_syntax_news_v1.tar")
        self.syntax.navec(self.navec)
        self.logger.info("✓ Slovnet Syntax loaded")

        # ── Slovnet Morph (опционально) ──────────────────────
        morph_path = models_path / "slovnet_morph_news_v1.tar"
        if morph_path.exists():
            self.morph = Morph.load(morph_path)
            self.morph.navec(self.navec)
            self.logger.info("✓ Slovnet Morph loaded")
        else:
            self.morph = None
            self.logger.warning("✗ Morph not found — pos/feats будут '_'")

        # ── Natasha NER ──────────────────────────────────────
        self.segmenter       = Segmenter()
        self.morph_vocab     = MorphVocab()
        self.emb             = NewsEmbedding()
        self.morph_tagger    = NewsMorphTagger(self.emb)
        self.syntax_parser   = NewsSyntaxParser(self.emb)
        self.ner_tagger      = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.PER             = PER
        self.logger.info("✓ Natasha NER loaded")
        self.logger.info("🚀 SlovnetService ready!")

    # ──────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _local_id(raw_id) -> int:
        """
        Извлекает локальный числовой id токена.
        Natasha использует формат "1_2" (предложение_токен) — берём последнюю часть.
        """
        try:
            s = str(raw_id)
            return int(s.split("_")[-1]) if "_" in s else int(s)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _feats_to_conllu(feats_obj) -> str:
        """
        Конвертирует объект feats → строку CoNLL-U формата K=V|K=V.

        Исправляет ошибку str(feats_obj) = "{'Animacy': 'Anim', ...}",
        которая возникала при прямом str()-приведении объекта Slovnet.
        """
        if not feats_obj:
            return "_"
        try:
            return "|".join(f"{k}={v}" for k, v in sorted(feats_obj.items()))
        except AttributeError:
            return str(feats_obj) or "_"

    @staticmethod
    def _fact_to_dict(fact) -> dict:  # ← static, убрать self
        if fact is None:
            return {}
        if hasattr(fact, '_asdict'):
            # noinspection PyProtectedMember
            raw = dict(fact._asdict())  # noqa: SLF001
        elif hasattr(fact, "__dict__"):
            raw = vars(fact)
        else:
            return {}

        # Новый формат Natasha: {"slots": [Slot(key='first', value='Александр'), ...]}
        # Разворачиваем в плоский словарь {"first": "Александр", ...}
        if "slots" in raw and isinstance(raw["slots"], (list, tuple)):
            result = {}
            for slot in raw["slots"]:
                if hasattr(slot, "key") and hasattr(slot, "value") and slot.value:
                    result[slot.key] = slot.value
                elif hasattr(slot, "_asdict"):
                    # noinspection PyProtectedMember
                    sd = dict(slot._asdict())
                    if sd.get("key") and sd.get("value"):
                        result[sd["key"]] = sd["value"]
            return result

        # Старый формат Natasha: {"first": "Александр", "last": "Пушкин", ...}
        return {k: v for k, v in raw.items() if v is not None}


    def _process_sentences_chunk(
            self,
            sentences: List[Tuple[str, int]],
            output_format: str = "conllu",
    ) -> Union[List[List[Dict]], Dict[str, Any]]:

        """
        Общая логика обработки списка предложений с глобальными офсетами.
        Вызывается из parse_text и parse_sentence_chunk.
        """
        from natasha import Doc
        from razdel import tokenize

        results_conllu: list = []
        native_sentences: list = []
        native_spans: list = []

        for sent_text, sent_start in sentences:
            razdel_toks = list(tokenize(sent_text))
            words = [t.text for t in razdel_toks]
            if not words:
                continue

            morph_map: dict = {}
            if self.morph:
                for i, m_tok in enumerate(self.morph(words).tokens, start=1):
                    morph_map[i] = m_tok

            syntax_markup = self.syntax(words)
            sent_conllu: list = []
            sent_native: list = []

            for i, tok in enumerate(syntax_markup.tokens):
                r_tok = razdel_toks[i]
                start_c = sent_start + r_tok.start
                end_c = sent_start + r_tok.stop
                tok_id = self._local_id(tok.id) or (i + 1)
                head_id = self._local_id(tok.head_id)
                if head_id == tok_id:
                    head_id = 0
                rel = getattr(tok, "rel", None) or "_"
                pos = "_"
                feats_obj = None
                if tok_id in morph_map:
                    m = morph_map[tok_id]
                    pos = getattr(m, "pos", None) or "_"
                    feats_obj = getattr(m, "feats", None)

                if output_format == "conllu":
                    sent_conllu.append({
                        "id": tok_id, "form": tok.text, "lemma": "_",
                        "upos": pos, "xpos": "_",
                        "feats": self._feats_to_conllu(feats_obj),
                        "head": head_id, "deprel": rel,
                        "deps": "_", "misc": "_",
                        "startchar": start_c, "endchar": end_c,
                    })
                else:
                    sent_native.append({
                        "id": tok_id, "text": tok.text,
                        "pos": pos if pos != "_" else None,
                        "feats": (dict(feats_obj.items())
                                  if feats_obj and hasattr(feats_obj, "items") else None),
                        "head_id": tok.head_id,
                        "rel": rel if rel != "_" else None,
                        "start": start_c, "stop": end_c,
                    })

            if output_format == "conllu":
                results_conllu.append(sent_conllu)
            else:
                native_sentences.append(sent_native)  # ← список предложений, не плоский список

            # NER только для native
            if output_format == "native":
                try:
                    doc = Doc(sent_text)
                    doc.segment(self.segmenter)
                    doc.tag_morph(self.morph_tagger)
                    doc.parse_syntax(self.syntax_parser)
                    doc.tag_ner(self.ner_tagger)
                    for span in doc.spans:
                        span.normalize(self.morph_vocab)
                    for span in doc.spans:
                        if span.type == self.PER:
                            span.extract_fact(self.names_extractor)
                    for span in doc.spans:
                        sp: dict = {
                            "start": sent_start + span.start,
                            "stop": sent_start + span.stop,
                            "type": span.type,
                        }
                        if getattr(span, "text", None): sp["text"] = span.text
                        if getattr(span, "normal", None): sp["normal"] = span.normal
                        if getattr(span, "fact", None): sp["fact"] = SlovnetService._fact_to_dict(span.fact)
                        native_spans.append(sp)
                except Exception as e:
                    self.logger.warning(f"NER failed for '{sent_text[:30]}': {e}")

        return (results_conllu if output_format == "conllu"
                else {"sentences": native_sentences, "spans": native_spans})

    # ──────────────────────────────────────────────────────────
    # Публичный Modal-метод
    # ──────────────────────────────────────────────────────────

    @modal.method()
    def parse_text(
        self,
        text: str,
        output_format: str = "conllu",
    ) -> Union[list, dict]:
        """
        Парсинг текста.

        Parameters
        ----------
        text : str
            Входной текст (любое число предложений).
        output_format : str, default "conllu"
            "conllu" или "native" — см. docstring модуля.

        Returns
        -------
        conllu → List[List[Dict]]
        native → Dict{"sentences": [[...], [...]], "spans": [...]}
        """
        from razdel import sentenize

        if not text or not text.strip():
            return [] if output_format == "conllu" else {"sentences": [], "spans": []}

        sentences = [(s.text, s.start) for s in sentenize(text)]
        return self._process_sentences_chunk(sentences, output_format)

    @modal.method()
    def parse_sentence_chunk(
            self,
            sentences: List[Tuple[str, int]],
            output_format: str = "conllu",
    ) -> Union[List[List[Dict]], Dict[str, Any]]:
        """
        Обрабатывает один чанк предложений с глобальными офсетами.
        Вызывается из SlovnetParser.parse_text через .map().
        sentences: [(sent_text, start_char_in_original_text), ...]
        """
        return self._process_sentences_chunk(sentences, output_format)


# ─────────────────────────────────────────────────────────────
# ЛОКАЛЬНЫЙ ТЕСТ  (modal run slovnet_modal.py)
# ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    import json
    logging.basicConfig(level=logging.INFO)

    test_text = "Мама Мария без мыла мыла раму. Александр Пушкин родился в Москве."
    sep = "=" * 70
    service = SlovnetService()

    # ════════════════════════════════════════════
    # 1. CoNLL-U
    # ════════════════════════════════════════════
    print(f"\n{sep}\nРЕЖИМ: conllu  →  List[List[Dict]]\n{sep}")
    result_conllu = service.parse_text.remote(test_text, output_format="conllu")
    print(f"Предложений: {len(result_conllu)}\n")
    for s_idx, sent in enumerate(result_conllu, 1):
        print(f"  Предложение {s_idx}:")
        print(f"  {'ID':>4} {'FORM':<16} {'LEMMA':<12} {'UPOS':<8} {'XPOS':<6} "
              f"{'HEAD':>5} {'DEPREL':<14} {'DEPS':<6} {'MISC':<10} START  END")
        print("  " + "-" * 105)
        for t in sent:
            print(f"  {t['id']:>4} {t['form']:<16} {t['lemma']:<12} "
                  f"{t['upos']:<8} {(t['xpos'] or '_'):<6} "
                  f"{t['head']:>5} {t['deprel']:<14} "
                  f"{(t['deps'] or '_'):<6} {(t['misc'] or '_'):<10} "
                  f"{t['startchar']}  {t['endchar']}")
            print(f"       feats: {t['feats'] or '_'}")
        print()

    print(f"\nКлючи conllu-токена: {list(result_conllu[0][0].keys())}")
    print("\nJSON первого токена:")
    print(json.dumps(result_conllu[0][0], ensure_ascii=False, indent=2))

    # ════════════════════════════════════════════
    # 2. Native
    # ════════════════════════════════════════════
    print(f"\n{sep}\nРЕЖИМ: native → Dict{{'sentences': [...], 'spans': [...]}}\n{sep}")
    result_native = service.parse_text.remote(test_text, output_format="native")
    sentences = result_native["sentences"]  # список предложений
    spans = result_native["spans"]
    tokens = [t for sent in sentences for t in sent]  # плоский — только для display
    print(f"Предложений: {len(sentences)},  Токенов: {len(tokens)},  Spans (NER): {len(spans)}\n")
    print(f"  {'ID':<4} {'TEXT':<14} {'POS':<7} {'HEAD_ID':<8} {'REL':<10} START  STOP")
    print("  " + "-" * 66)
    for t in tokens:
        print(f"  {t['id']:<4} {t['text']:<14} {str(t['pos']):<7} "
              f"{str(t['head_id']):<8} {str(t['rel']):<10} "
              f"{t['start']}  {t['stop']}")

    print(f"\nКлючи native-токена: {list(sentences[0][0].keys())}")
    print("\nJSON первого токена:")
    print(json.dumps(sentences[0][0], ensure_ascii=False, indent=2, default=str))

    if spans:
        print(f"\nSpans ({len(spans)}):")
        for sp in spans:
            print(f"  [{sp['start']}:{sp['stop']}] {sp['type']:6} "
                  f"'{sp.get('text','')}' → '{sp.get('normal','')}'")
            if sp.get("fact"):
                for k, v in sp["fact"].items():
                    if v:
                        print(f"    {k}: '{v}'")
    else:
        print("\nSpans: []")

    # ════════════════════════════════════════════
    # 3. Сравнение ключей и feats
    # ════════════════════════════════════════════
    print(f"\n{sep}\nСРАВНЕНИЕ КЛЮЧЕЙ И ФОРМАТА FEATS\n{sep}")
    ck = set(result_conllu[0][0].keys())
    nk = set(sentences[0][0].keys())
    print(f"  Только в conllu: {sorted(ck - nk)}")
    print(f"  Только в native: {sorted(nk - ck)}")
    print(f"\n  conllu feats (строка CoNLL-U): {repr(result_conllu[0][0]['feats'])}")
    print(f"  native feats (dict|None):       {repr(tokens[0]['feats'])}")
