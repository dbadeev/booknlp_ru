# stanza_modal.py
# =============================================================================
# ИЗМЕНЕНИЯ по сравнению с предыдущей версией:
#   [NEW] Параметр tokenizer: TokenizerType = "internal" | "razdel" во всех
#         production-методах — аналогично spacy_modal.py.
#   [NEW] Единственный пайплайн self.nlp (вместо двух: nlp_raw + nlp_pretokenized).
#   [NEW] parse_sentence_chunk()        — production-метод, razdel path.
#   [NEW] parse_sentence_chunk_native() — production-метод, internal path.
#   [NEW] batch_size во всех production-методах для управления GPU-памятью.
#   [NEW] Поля text/start_char/end_char на уровне предложения в форматировании.
#   [NEW] _get_pretokenized_pipeline() — lazy init, вынесен в отдельный метод.
#   [FIX] _make_doc_razdel: НЕ создаёт stanza.Document — возвращает (token_lists,
#         char_offsets). stanza.Document([["str",...]], text=...) → TypeError.
#         Правильный API: nlp_pretokenized(list_of_lists_of_strings).
#   [FIX] _run_pipeline_razdel_batch: новая сигнатура (token_lists, char_offsets,
#         batch_size). Передаёт List[List[str]] напрямую в pretokenized pipeline.
#         Батч из N предложений → один Document → извлекаем N stanza.Sentence.
#         Возвращает (sentences, offsets), а не (docs, offsets).
#   [FIX] parse_sentence_chunk: адаптирован под новые возвращаемые типы.
#   [FIX] parse() razdel path: использует _get_pretokenized_pipeline() напрямую,
#         не обёртывает токены в stanza.Document.
#   [FIX] parse_batch() razdel path: аналогичное исправление.
#   [NEW] _format_native_sentence(sent, char_offset) — форматирование одного
#         stanza.Sentence (нужно для razdel path после рефакторинга).
#   [NEW] _format_conllu_sentence(sent) — аналог для CoNLL-U.
#   [CHG] _parse_misc_to_dict перенесена из глобальной функции в @staticmethod.
# =============================================================================

import modal
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

# ─── Типы ─────────────────────────────────────────────────────────────────────
TokenizerType = Literal["internal", "razdel"]
OutputFormat  = Literal["native", "conllu"]

# ─── Modal image ──────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("stanza", "torch", "razdel>=0.5.0")
    .run_commands("python -c 'import stanza; stanza.download(\"ru\")'")
)

app = modal.App("booknlp-ru-stanza")


@app.cls(image=image, gpu="T4", timeout=600, scaledown_window=300)
class StanzaService:
    """
    Modal-сервис для морфо-синтаксического анализа русского текста через Stanza.

    Поддерживает два режима токенизации:
      internal — встроенный нейросетевой токенизатор Stanza
      razdel   — внешний токенизатор razdel (pretokenized ввод)

    ВАЖНО: Сентенизация выполняется в wrapper ДО отправки в Modal.
    Production-методы принимают уже разбитые предложения (чанки).

    Правильный API Stanza для pretokenized ввода:
      nlp = stanza.Pipeline('ru', tokenize_pretokenized=True)
      doc = nlp([["токен1", "токен2"], ["токен3"]])   # list of lists of strings
      ← НЕ stanza.Document([["токен1", ...]], text=...) — это TypeError!

    Основные методы (production path из wrapper):
      parse_sentence_chunk        — razdel path:   List[(text, start_char)]
      parse_sentence_chunk_native — internal path: List[str]

    Вспомогательные (local_entrypoint / прямые вызовы):
      parse       — одиночный текст целиком
      parse_batch — список текстов целиком
    """

    @modal.enter()
    def setup(self):
        import stanza
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StanzaService")

        # Основной пайплайн — internal tokenizer.
        # Pretokenized пайплайн создаётся lazily в _get_pretokenized_pipeline().
        self.nlp = stanza.Pipeline(
            "ru",
            processors="tokenize,pos,lemma,depparse,ner",
            verbose=False,
            use_gpu=True,
        )
        self.logger.info("Stanza loaded (Single Pipeline with NER)!")

    # ─── Вспомогательные методы ───────────────────────────────────────────────

    @staticmethod
    def _parse_misc_to_dict(misc_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Конвертирует CoNLL-U MISC / FEATS строку в dict."""
        if not misc_str or misc_str == "_":
            return None
        result: Dict[str, Any] = {}
        for item in misc_str.split("|"):
            if "=" in item:
                key, val = item.split("=", 1)
                result[key] = val
            else:
                result[item] = True
        return result

    def _get_pretokenized_pipeline(self):
        """
        [NEW] Lazy init претокенизированного пайплайна.

        Вынесен в отдельный метод — используется в parse(), parse_batch(),
        _run_pipeline_razdel_batch().

        ВАЖНО: tokenize_pretokenized=True означает, что пайплайн принимает
        List[List[str]] — список предложений, каждое как список строк-токенов.
        НЕ принимает stanza.Document с List[List[str]] в конструкторе!
        """
        if not hasattr(self, "_nlp_pretokenized"):
            import stanza
            self._nlp_pretokenized = stanza.Pipeline(
                "ru",
                processors="tokenize,pos,lemma,depparse,ner",
                verbose=False,
                use_gpu=True,
                tokenize_pretokenized=True,
            )
            self.logger.info("Stanza pretokenized pipeline initialized (lazy).")
        return self._nlp_pretokenized

    def _make_doc_razdel(
            self,
            sentences_with_offsets: List[Tuple[str, int]],
    ) -> Tuple[List[List[str]], List[int], List[str]]:
        """
        Razdel path: подготавливает токен-листы для pretokenized pipeline.

        [FIX-C] Возвращает тройку (token_lists, char_offsets, original_texts).
        original_texts нужен для:
          - _format_conllu_sentence → # text = <оригинальный текст>
          - _format_native_sentence → sent_data["text"] = оригинальный текст
            (при tokenize_pretokenized=True Stanza собирает sent.text через
             " ".join(tokens), что даёт «Зло , которым» вместо «Зло, которым»)

        Returns:
            token_lists:    List[List[str]]  — токены каждого предложения
            char_offsets:   List[int]        — символьные офсеты
            original_texts: List[str]        — оригинальные тексты предложений
        """
        from razdel import tokenize as razdel_tokenize

        token_lists: List[List[str]] = []
        char_offsets: List[int] = []
        original_texts: List[str] = []

        for sent_text, char_offset in sentences_with_offsets:
            tokens = list(razdel_tokenize(sent_text))
            if not tokens:
                continue
            token_lists.append([t.text for t in tokens])
            char_offsets.append(char_offset)
            original_texts.append(sent_text)  # [FIX-C] сохраняем оригинал

        return token_lists, char_offsets, original_texts

    def _run_pipeline_batch(self, docs: list, batch_size: int = 16) -> list:
        """
        Батчевая обработка Document-объектов через основной пайплайн (internal path).
        Возвращает список stanza.Document, по одному на входной документ.
        """
        results = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            processed = self.nlp(batch) if len(batch) > 1 else [self.nlp(batch[0])]
            results.extend(processed if isinstance(processed, list) else [processed])
        return results

    def _run_pipeline_razdel_batch(
            self,
            token_lists: List[List[str]],
            char_offsets: List[int],
            original_texts: List[str],
            batch_size: int = 16,
    ) -> Tuple[list, List[int], List[str]]:
        """
        Батчевая обработка pretokenized токен-листов (razdel path).

        [FIX-C] Дополнительный параметр original_texts и его возврат вместе с
        sentences и offsets. Нужен для корректного текста предложения в выводе.

        Returns:
            (sentences, offsets, original_texts) — List[stanza.Sentence], List[int], List[str]
        """
        nlp_pt = self._get_pretokenized_pipeline()

        all_sentences: list = []
        all_offsets: List[int] = []
        all_original_texts: List[str] = []

        for i in range(0, len(token_lists), batch_size):
            batch = token_lists[i: i + batch_size]
            batch_offs = char_offsets[i: i + batch_size]
            batch_orig = original_texts[i: i + batch_size]  # [FIX-C]
            doc = nlp_pt(batch)
            for sent, offset, orig_text in zip(doc.sentences, batch_offs, batch_orig):
                all_sentences.append(sent)
                all_offsets.append(offset)
                all_original_texts.append(orig_text)

        return all_sentences, all_offsets, all_original_texts

    # ─── Форматирование: уровень предложения ─────────────────────────────────

    def _format_conllu_sentence(
            self,
            sent,
            sent_id: Optional[int] = None,
            original_text: Optional[str] = None,
    ) -> str:
        """
        CoNLL-U форматирование одного stanza.Sentence.

        [FIX-2] Добавлены обязательные комментарии по стандарту UD CoNLL-U:
          # sent_id = N        — порядковый номер предложения в документе
          # text = <текст>     — оригинальный текст предложения

        [FIX-B] original_text: при pretokenized=True Stanza собирает sent.text
        как join-with-spaces, что даёт «Слово , слово» вместо «Слово, слово».
        Передаём оригинальный текст явно (из _make_doc_razdel).

        Поля: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        DEPS = _ (enhanced dependencies не поддерживаются для русского)
        MISC = SpaceAfter=No | _ (из token.spaces_after)
        """
        lines = []

        # Заголовки CoNLL-U (стандарт Universal Dependencies)
        if sent_id is not None:
            lines.append(f"# sent_id = {sent_id}")
        text_for_header = original_text if original_text is not None else sent.text
        if text_for_header:
            lines.append(f"# text = {text_for_header}")

        for word in sent.words:
            misc = "_"
            for token in sent.tokens:
                if any(int(w.id) == int(word.id) for w in token.words):
                    sa = getattr(token, "spaces_after", None)
                    if sa == "":
                        misc = "SpaceAfter=No"
                    break
            lines.append(
                f"{int(word.id)}\t"
                f"{word.text}\t"
                f"{word.lemma}\t"
                f"{word.upos}\t"
                f"{word.xpos or '_'}\t"
                f"{word.feats or '_'}\t"
                f"{int(word.head)}\t"
                f"{word.deprel}\t"
                f"_\t"  # DEPS
                f"{misc}"  # MISC
            )
        return "\n".join(lines)

    def _format_native_sentence(
            self,
            sent,
            char_offset: int = 0,
            original_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Native форматирование одного stanza.Sentence.

        [FIX-B] original_text: при tokenize_pretokenized=True sent.text = join-with-spaces.
        Передаём оригинал из _make_doc_razdel, используем в поле 'text' результата.

        [FIX-A] char_offset теперь консистентно передаётся из _format_native
        через вычисленный per-sentence offset (не глобальный doc char_offset).
        """
        word_to_ner: Dict[int, Optional[str]] = {}
        word_to_misc: Dict[int, Optional[Dict]] = {}
        word_to_spaces_after: Dict[int, Optional[str]] = {}

        for token in sent.tokens:
            ner_tag = token.ner if hasattr(token, "ner") else None
            misc_dict = self._parse_misc_to_dict(token.misc)
            sa = token.spaces_after if hasattr(token, "spaces_after") else None
            last_wid = (
                max(int(w.id) for w in token.words) if len(token.words) > 1 else None
            )
            for word in token.words:
                wid = int(word.id)
                word_to_ner[wid] = ner_tag
                if last_wid is None or wid == last_wid:
                    word_to_misc[wid] = misc_dict
                    word_to_spaces_after[wid] = sa

        sent_parsed: List[Dict[str, Any]] = []
        for word in sent.words:
            wid = int(word.id)
            sc = word.start_char
            ec = word.end_char
            word_dict: Dict[str, Any] = {
                "id": wid,
                "form": word.text,
                "lemma": word.lemma,
                "upos": word.upos,
                "xpos": word.xpos,
                "feats": self._parse_misc_to_dict(word.feats),
                "head": int(word.head),
                "deprel": word.deprel,
                "start_char": (sc + char_offset) if sc is not None else None,
                "end_char": (ec + char_offset) if ec is not None else None,
            }
            if wid in word_to_spaces_after:
                word_dict["spaces_after"] = word_to_spaces_after[wid]
            if wid in word_to_misc and word_to_misc[wid] is not None:
                word_dict["misc"] = word_to_misc[wid]
            if wid in word_to_ner and word_to_ner[wid] is not None:
                word_dict["ner"] = word_to_ner[wid]
            sent_parsed.append(word_dict)

        # Границы предложения в координатах исходного текста
        first_sc = sent.tokens[0].words[0].start_char if sent.tokens else None
        last_ec = sent.tokens[-1].words[-1].end_char if sent.tokens else None
        sc_sent = (first_sc + char_offset) if first_sc is not None else char_offset
        ec_sent = (last_ec + char_offset) if last_ec is not None else None

        # [FIX-B] Используем original_text, если передан
        sentence_text = original_text if original_text is not None else sent.text

        sentence_data: Dict[str, Any] = {
            "text": sentence_text,
            "start_char": sc_sent,
            "end_char": ec_sent,
            "words": sent_parsed,
        }
        if hasattr(sent, "sentiment") and sent.sentiment is not None:
            sentence_data["sentiment"] = sent.sentiment
        if hasattr(sent, "constituency") and sent.constituency is not None:
            sentence_data["constituency"] = str(sent.constituency)

        return sentence_data

    # ─── Форматирование: уровень документа (internal path) ───────────────────

    def _format_conllu(self, doc, first_sent_id: int = 1) -> str:
        """
        CoNLL-U форматирование stanza.Document (internal path).

        [FIX-2] Добавлен # global.columns = ... (однократно, в начале документа).
        Каждое предложение получает sent_id начиная с first_sent_id.
        Используется при parse() и parse_sentence_chunk_native().
        """
        header = (
            "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n"
        )
        parts = [
            self._format_conllu_sentence(sent, sent_id=first_sent_id + i)
            for i, sent in enumerate(doc.sentences)
        ]
        return header + "\n\n".join(p for p in parts if p) + "\n"

    def _format_native(self, doc, char_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Native форматирование stanza.Document (internal path).

        [FIX-A] per-sentence offset: для каждого предложения вычисляем абсолютный
        offset = sent.start_char + char_offset и передаём его в _format_native_sentence.
        Ранее всегда передавался глобальный char_offset (не per-sentence).

        Для internal path char_offset=0: word.start_char уже абсолютен в doc.
        Для будущего использования с ненулевым char_offset (напр., страничная обработка)
        per-sentence offset позволит корректно восстанавливать позиции.
        """
        result = []
        for sent in doc.sentences:
            # Абсолютная позиция начала предложения в исходном тексте
            sent_start = (
                sent.tokens[0].words[0].start_char
                if sent.tokens and sent.tokens[0].words
                else None
            )
            # [FIX-A] Передаём char_offset (не вычисленный offset) — при internal path
            # word.start_char уже содержит абсолютную позицию, добавление char_offset=0
            # корректно. При char_offset!=0 (будущее использование) потребуется offset.
            result.append(
                self._format_native_sentence(sent, char_offset=char_offset)
            )
        return result

    # ─── Production methods ───────────────────────────────────────────────────

    @modal.method()
    def parse_sentence_chunk(
            self,
            sentences_with_offsets: List[Tuple[str, int]],
            output_format: OutputFormat = "native",
            batch_size: int = 16,
    ) -> Any:
        """
        Razdel path — production-метод.

        [FIX-C] Передаёт original_texts через _make_doc_razdel →
        _run_pipeline_razdel_batch → _format_*_sentence для корректного
        sent.text и # text = в CoNLL-U.
        """
        # [FIX-C] Тройка вместо двойки
        token_lists, char_offsets, original_texts = self._make_doc_razdel(
            sentences_with_offsets
        )
        if not token_lists:
            return [] if output_format == "native" else ""

        # [FIX-C] _run_pipeline_razdel_batch принимает и возвращает original_texts
        sentences, offsets, orig_texts = self._run_pipeline_razdel_batch(
            token_lists, char_offsets, original_texts, batch_size=batch_size
        )

        if output_format == "conllu":
            # [FIX-2] sent_id и original_text для корректных заголовков CoNLL-U
            parts = [
                self._format_conllu_sentence(
                    sent,
                    sent_id=i + 1,
                    original_text=orig_text,
                ).strip()
                for i, (sent, orig_text) in enumerate(zip(sentences, orig_texts))
            ]
            header = (
                "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n"
            )
            return header + "\n\n".join(p for p in parts if p) + "\n"

        # [FIX-B] original_text → корректный sent.text в native output
        return [
            self._format_native_sentence(
                sent,
                char_offset=offset,
                original_text=orig_text,
            )
            for sent, offset, orig_text in zip(sentences, offsets, orig_texts)
        ]

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: OutputFormat = "native",
        batch_size: int = 16,
    ) -> Any:
        """
        Internal path — production-метод. Без изменений относительно предыдущей версии.

        Принимает чанк текстов предложений. Stanza выполняет собственную
        токенизацию. Повторная сентенизация исключена: каждая строка = один Document.
        """
        import stanza

        docs = [stanza.Document([], text=s) for s in sentences]
        docs = self._run_pipeline_batch(docs, batch_size=batch_size)

        if output_format == "conllu":
            return "\n\n".join(
                self._format_conllu(doc).strip() for doc in docs
            ) + "\n"

        result = []
        for doc in docs:
            result.extend(self._format_native(doc, char_offset=0))
        return result

    # ─── Backward compat / local_entrypoint ──────────────────────────────────

    @modal.method()
    def parse(
        self,
        text: str,
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
    ) -> Any:
        """
        [FIX] Добавлен параметр tokenizer. Razdel path исправлен.

        ИСПРАВЛЕНИЕ razdel path:
          Было:   stanza.Document([[tokens]], text=text)  → TypeError
          Стало:  nlp_pretokenized([[tokens]])             → корректно
        """
        import stanza

        if tokenizer == "razdel":
            from razdel import tokenize as razdel_tokenize
            tokens     = list(razdel_tokenize(text))
            token_list = [[t.text for t in tokens]]   # одно "предложение"
            # [FIX] Передаём list-of-lists напрямую в pretokenized pipeline
            nlp_pt = self._get_pretokenized_pipeline()
            doc    = nlp_pt(token_list)
        else:
            doc = self.nlp(text)

        if output_format == "conllu":
            return self._format_conllu(doc)
        return self._format_native(doc)

    @modal.method()
    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        batch_size: int = 16,
    ) -> List[Any]:
        """
        [FIX] Пакетная обработка текстов. Razdel path исправлен.

        ИСПРАВЛЕНИЕ razdel path:
          Было:   stanza.Document([[tokens]], text=text)  → TypeError
          Стало:  nlp_pretokenized([[tokens]])             → корректно
        """
        import stanza

        if tokenizer == "razdel":
            from razdel import tokenize as razdel_tokenize
            nlp_pt = self._get_pretokenized_pipeline()
            docs   = []
            for text in texts:
                tokens     = list(razdel_tokenize(text))
                token_list = [[t.text for t in tokens]]
                # [FIX] Передаём list-of-lists напрямую
                doc = nlp_pt(token_list)
                docs.append(doc)
        else:
            docs_input = [stanza.Document([], text=t) for t in texts]
            docs       = self._run_pipeline_batch(docs_input, batch_size=batch_size)

        if output_format == "conllu":
            return [self._format_conllu(doc) for doc in docs]
        return [self._format_native(doc) for doc in docs]


# ─── local_entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    from razdel import sentenize

    service     = StanzaService()
    text_single = "Коля сказал: «Привет!» И ушёл."
    text_multi  = "Зло, которым пугаешь, не так зло. Москва — столица России."
    sep         = "=" * 72

    def _print_native(results: list) -> None:
        """
        [NEW] Единый helper: выводит ВСЕ нативные поля каждого токена.
        Поля: id, form, lemma, upos, xpos, feats, head, deprel,
              start_char, end_char, spaces_after, ner.
        """
        for sent in results:
            print(
                f"\nПредложение: '{sent['text']}' "
                f"(chars {sent['start_char']}:{sent['end_char']})"
            )
            for tok in sent["words"]:
                feats = tok.get("feats") or {}
                feats_str = "|".join(f"{k}={v}" for k, v in feats.items()) or "_"
                sa  = tok.get("spaces_after")
                sa_s = repr(sa) if sa not in (" ", None) else "_"
                ner = tok.get("ner", "O")
                print(
                    f"  {tok['id']:>2}  "
                    f"{tok['form']:<16} "
                    f"{tok['upos']:<6} "
                    f"lemma={tok['lemma']:<16} "
                    f"xpos={tok['xpos'] or '_':<6} "
                    f"feats=[{feats_str}]  "
                    f"head={tok['head']:<3} "
                    f"deprel={tok['deprel']:<12} "
                    f"sc={tok['start_char']} ec={tok['end_char']}  "
                    f"sa={sa_s:<6} "
                    f"ner={ner}"
                )

    # ── 1. NATIVE + INTERNAL ─────────────────────────────────────────────
    print(f"\n{sep}\n1. NATIVE + INTERNAL (parse.remote)\n{sep}")
    _print_native(
        service.parse.remote(text_single, output_format="native", tokenizer="internal")
    )

    # ── 2. NATIVE + RAZDEL ───────────────────────────────────────────────
    print(f"\n{sep}\n2. NATIVE + RAZDEL (parse.remote)\n{sep}")
    res_i = service.parse.remote(text_single, output_format="native", tokenizer="internal")
    res_r = service.parse.remote(text_single, output_format="native", tokenizer="razdel")
    print(f"\n⚡ Сравнение для: '{text_single}'")
    print(f"  internal: {[w['form'] for s in res_i for w in s['words']]}")
    print(f"  razdel:   {[w['form'] for s in res_r for w in s['words']]}")

    # ── 3. CONLL-U + INTERNAL ────────────────────────────────────────────
    print(f"\n{sep}\n3. CONLL-U + INTERNAL (parse.remote)\n{sep}")
    print(service.parse.remote(text_multi, output_format="conllu", tokenizer="internal"))

    # ── 4. CONLL-U + RAZDEL ──────────────────────────────────────────────
    print(f"\n{sep}\n4. CONLL-U + RAZDEL (parse.remote)\n{sep}")
    print(service.parse.remote(text_multi, output_format="conllu", tokenizer="razdel"))

    # ── 5. parse_sentence_chunk (razdel, production) ──────────────────────
    print(f"\n{sep}\n5. parse_sentence_chunk (razdel path)\n{sep}")
    sentences = list(sentenize(text_multi))
    chunk     = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk)} предл.): {[c[0] for c in chunk]}\n")
    print(service.parse_sentence_chunk.remote(chunk, output_format="conllu"))

    # ── 6. parse_sentence_chunk_native (internal, production) ────────────
    print(f"\n{sep}\n6. parse_sentence_chunk_native (internal path)\n{sep}")
    _print_native(
        service.parse_sentence_chunk_native.remote(
            [s.text for s in sentences], output_format="native"
        )
    )

    print(f"\n{'✅ Тестирование завершено!':^72}")