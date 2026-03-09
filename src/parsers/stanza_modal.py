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
    ) -> Tuple[List[List[str]], List[int]]:
        """
        [FIX] Razdel path: подготавливает токен-листы для pretokenized pipeline.

        ИСПРАВЛЕНИЕ: НЕ создаёт stanza.Document.
        stanza.Document([["слово1", "слово2"]], text=...) вызывает TypeError:
          File "doc.py", line 592, in _process_tokens
            entry[ID] = (i+1,)  ← entry — это str, не dict!

        Правильный API:
          nlp_pretokenized([["слово1", "слово2"]])  ← list of lists напрямую в pipeline

        Args:
            sentences_with_offsets: List[(sentence_text, start_char_in_original)]

        Returns:
            (token_lists, char_offsets)
              token_lists:  List[List[str]]  — токены каждого предложения
              char_offsets: List[int]        — символьный офсет каждого предложения
        """
        from razdel import tokenize as razdel_tokenize

        token_lists: List[List[str]] = []
        char_offsets: List[int] = []

        for sent_text, char_offset in sentences_with_offsets:
            tokens = list(razdel_tokenize(sent_text))
            if not tokens:
                continue
            token_lists.append([t.text for t in tokens])
            char_offsets.append(char_offset)

        return token_lists, char_offsets

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
        batch_size: int = 16,
    ) -> Tuple[list, List[int]]:
        """
        [FIX] Батчевая обработка pretokenized токен-листов (razdel path).

        ИСПРАВЛЕНИЕ: принимает List[List[str]] и char_offsets, а НЕ список
        stanza.Document. Передаёт батчи напрямую в pretokenized pipeline.

        Как работает батчинг:
          batch = [["слово1","слово2"], ["слово3","слово4"]]  # N предложений
          doc = nlp_pretokenized(batch)     # → один Document с N sentences
          doc.sentences[i]                  # → i-е обработанное предложение

        Возвращает:
          (sentences, offsets) — List[stanza.Sentence] + List[int]
          Индексы соответствуют входным предложениям.

        Args:
            token_lists:  List[List[str]] — токены каждого предложения
            char_offsets: List[int]       — символьные офсеты (из _make_doc_razdel)
            batch_size:   размер батча (подбирается под GPU, OOM-защита)
        """
        nlp_pt = self._get_pretokenized_pipeline()

        all_sentences = []
        all_offsets = []

        for i in range(0, len(token_lists), batch_size):
            batch       = token_lists[i : i + batch_size]
            batch_offs  = char_offsets[i : i + batch_size]
            # Передаём List[List[str]] напрямую — Stanza возвращает один Document
            doc = nlp_pt(batch)
            # Каждое предложение в doc.sentences соответствует одному элементу batch
            for sent, offset in zip(doc.sentences, batch_offs):
                all_sentences.append(sent)
                all_offsets.append(offset)

        return all_sentences, all_offsets

    # ─── Форматирование: уровень предложения ─────────────────────────────────

    def _format_conllu_sentence(self, sent) -> str:
        """
        [NEW] CoNLL-U форматирование одного stanza.Sentence.

        Используется в razdel path (parse_sentence_chunk), где после
        _run_pipeline_razdel_batch возвращаются отдельные Sentence, а не Document.
        """
        lines = []
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
                f"_\t"
                f"{misc}"
            )
        return "\n".join(lines)

    def _format_native_sentence(
        self, sent, char_offset: int = 0
    ) -> Dict[str, Any]:
        """
        [NEW] Native форматирование одного stanza.Sentence.

        Используется в razdel path. char_offset — смещение начала предложения
        в исходном тексте (передаётся из _run_pipeline_razdel_batch).

        Примечание по символьным позициям в razdel path:
          word.start_char / word.end_char — None при pretokenized=True
          (Stanza не сохраняет позиции для pretokenized токенов).
          Позиция предложения: start_char = char_offset.
        """
        word_to_ner: Dict[int, Optional[str]] = {}
        word_to_misc: Dict[int, Optional[Dict]] = {}
        word_to_spaces_after: Dict[int, Optional[str]] = {}

        for token in sent.tokens:
            ner_tag   = token.ner if hasattr(token, "ner") else None
            misc_dict = self._parse_misc_to_dict(token.misc)
            sa        = token.spaces_after if hasattr(token, "spaces_after") else None
            last_wid  = (
                max(int(w.id) for w in token.words)
                if len(token.words) > 1 else None
            )
            for word in token.words:
                wid = int(word.id)
                word_to_ner[wid] = ner_tag
                if last_wid is None or wid == last_wid:
                    word_to_misc[wid]         = misc_dict
                    word_to_spaces_after[wid] = sa

        sent_parsed: List[Dict[str, Any]] = []
        for word in sent.words:
            wid = int(word.id)
            sc  = word.start_char
            ec  = word.end_char
            word_dict: Dict[str, Any] = {
                "id":         wid,
                "form":       word.text,
                "lemma":      word.lemma,
                "upos":       word.upos,
                "xpos":       word.xpos,
                "feats":      self._parse_misc_to_dict(word.feats),
                "head":       int(word.head),
                "deprel":     word.deprel,
                # При pretokenized=True start_char/end_char = None
                "start_char": (sc + char_offset) if sc is not None else None,
                "end_char":   (ec + char_offset) if ec is not None else None,
            }
            if wid in word_to_spaces_after:
                word_dict["spaces_after"] = word_to_spaces_after[wid]
            if wid in word_to_misc and word_to_misc[wid] is not None:
                word_dict["misc"] = word_to_misc[wid]
            if wid in word_to_ner and word_to_ner[wid] is not None:
                word_dict["ner"] = word_to_ner[wid]
            sent_parsed.append(word_dict)

        # Позиция предложения: start_char известен (из wrapper),
        # end_char = char_offset + len(sent.text) при наличии текста
        ec_sent = (
            char_offset + len(sent.text)
            if sent.text else None
        )
        sentence_data: Dict[str, Any] = {
            "text":       sent.text,
            "start_char": char_offset,
            "end_char":   ec_sent,
            "words":      sent_parsed,
        }
        if hasattr(sent, "sentiment") and sent.sentiment is not None:
            sentence_data["sentiment"] = sent.sentiment
        if hasattr(sent, "constituency") and sent.constituency is not None:
            sentence_data["constituency"] = str(sent.constituency)

        return sentence_data

    # ─── Форматирование: уровень документа (internal path) ───────────────────

    def _format_conllu(self, doc) -> str:
        """CoNLL-U форматирование stanza.Document (internal path)."""
        parts = [
            self._format_conllu_sentence(sent)
            for sent in doc.sentences
        ]
        return "\n\n".join(p for p in parts if p) + "\n"

    def _format_native(self, doc, char_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Native форматирование stanza.Document (internal path).
        Делегирует форматирование каждого предложения в _format_native_sentence.
        """
        result = []
        for sent in doc.sentences:
            # Для internal path char_offset = 0 (позиции — от начала переданного текста)
            sent_start = sent.tokens[0].words[0].start_char if sent.tokens else None
            offset = (
                (sent_start + char_offset) if sent_start is not None else char_offset
            )
            result.append(self._format_native_sentence(sent, char_offset=char_offset))
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
        [FIX] Razdel path — production-метод.

        Принимает чанк пар (sentence_text, start_char_in_original).
        Использует исправленные _make_doc_razdel / _run_pipeline_razdel_batch.

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
            output_format:          'native' | 'conllu'
            batch_size:             размер батча (OOM-защита)
        """
        token_lists, char_offsets = self._make_doc_razdel(sentences_with_offsets)
        if not token_lists:
            return [] if output_format == "native" else ""

        # [FIX] _run_pipeline_razdel_batch возвращает (sentences, offsets)
        sentences, offsets = self._run_pipeline_razdel_batch(
            token_lists, char_offsets, batch_size=batch_size
        )

        if output_format == "conllu":
            parts = [
                self._format_conllu_sentence(sent).strip()
                for sent in sentences
            ]
            return "\n\n".join(p for p in parts if p) + "\n"

        return [
            self._format_native_sentence(sent, char_offset=offset)
            for sent, offset in zip(sentences, offsets)
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
    """
    Тестирует StanzaService напрямую — без wrapper, без chunking.
    Проверяет оба токенизатора, оба формата, все production-методы.
    """
    from razdel import sentenize

    service     = StanzaService()
    text_single = "Коля сказал: «Привет!» И ушёл."
    text_multi  = "Зло, которым пугаешь, не так зло. Москва — столица России."
    sep         = "=" * 72

    # ── 1. NATIVE + INTERNAL ─────────────────────────────────────────────
    print(f"\n{sep}\n1. NATIVE + INTERNAL (parse.remote)\n{sep}")
    result = service.parse.remote(text_single, output_format="native", tokenizer="internal")
    for sent in result:
        print(f"\nПредложение: '{sent['text']}' "
              f"(chars {sent['start_char']}:{sent['end_char']})")
        for tok in sent["words"]:
            ner  = f" [NER: {tok['ner']}]" if "ner" in tok else ""
            sa   = tok.get("spaces_after")
            sa_s = f" [sa: {repr(sa)}]" if sa not in (" ", None) else ""
            print(f"  {tok['id']:>2}  {tok['form']:<15} {tok['upos']:<6}"
                  f"  lemma={tok['lemma']}{ner}{sa_s}")

    # ── 2. NATIVE + RAZDEL ───────────────────────────────────────────────
    print(f"\n{sep}\n2. NATIVE + RAZDEL (parse.remote)\n{sep}")
    result_r = service.parse.remote(text_single, output_format="native", tokenizer="razdel")
    print(f"\n⚡ Сравнение токенизаторов для: '{text_single}'")
    print(f"  internal: {[w['form'] for s in result   for w in s['words']]}")
    print(f"  razdel:   {[w['form'] for s in result_r for w in s['words']]}")

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
    print(f"Чанк ({len(chunk)} предл.): {[c[0] for c in chunk]}")
    print(service.parse_sentence_chunk.remote(chunk, output_format="conllu"))

    # ── 6. parse_sentence_chunk_native (internal, production) ─────────────
    print(f"\n{sep}\n6. parse_sentence_chunk_native (internal path)\n{sep}")
    chunk_texts = [s.text for s in sentences]
    result_cn   = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    for sent in result_cn:
        print(f"\nПредложение: '{sent['text']}' "
              f"(chars {sent['start_char']}:{sent['end_char']})")
        for tok in sent["words"]:
            ner = f" [NER: {tok['ner']}]" if "ner" in tok else ""
            print(f"  {tok['id']:>2}  {tok['form']:<15} {tok['upos']:<6}"
                  f"  lemma={tok['lemma']}{ner}")

    print(f"\n{'✅ Тестирование завершено!':^72}")


