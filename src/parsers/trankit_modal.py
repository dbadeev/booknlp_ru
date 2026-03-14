#!/usr/bin/env python3
"""
trankit_modal.py — Trankit NLP-сервис на Modal.

Поддерживает два формата вывода:
  simplified — CoNLL-U совместимый набор полей:
               id, form, lemma, upos, xpos, feats, head, deprel,
               deps ("_"), misc ("_"), start_char, end_char
  native     — полный нативный формат Trankit:
               id, text, lemma, upos, xpos, feats, head, deprel,
               span, dspan, ner, expanded, lang

Два пути обработки (оба используют razdel.sentenize в wrapper):

  razdel path — parse_sentence_chunk():
      принимает List[Tuple[str, int]] (текст предложения + символьный офсет).
      Вызывает self.nlp(sent_text, is_sent=True) — Trankit пропускает
      внутреннюю сентенизацию.
      dspan / start_char корректируются: span + char_offset (razdel).

  native path — parse_sentence_chunk_native():
      принимает List[str] (только тексты предложений).
      Вызывает self.nlp(sent_text, is_sent=True).
      Офсеты токенов относительны каждого предложения (char_offset=0).

Сентенизация ВСЕГДА выполняется в wrapper (razdel.sentenize) до вызова Modal.

Вспомогательные методы (backward compat / local_entrypoint):
  parse()       — парсит текст целиком (Trankit сентенизирует сам)
  parse_batch() — пакетная обработка текстов целиком
"""

import logging
import os
from typing import Any, Dict, List, Literal, Tuple

import modal

# ─── Константы ────────────────────────────────────────────────────────────────

LOCAL_MODEL_PATH = "/root/local_models/xlm-roberta-large"
LANG = "russian"
TITLE = "xlm-roberta-large"

# [НОВОЕ] Литеральные типы форматов — для аннотаций и документации
OutputFormat = Literal["simplified", "native"]

# ─── Modal image ──────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "wget", "unzip")
    .pip_install(
        "six",
        "torch==2.0.0",
        "numpy<2",
        "trankit==1.1.1",
        "transformers==4.39.0",
        "huggingface-hub",
        "razdel>=0.5.0",       # [НОВОЕ] нужен в образе для local_entrypoint
    )
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}")
    .run_commands(
        f"huggingface-cli download {TITLE} "
        f"--local-dir {LOCAL_MODEL_PATH} "
        f"--local-dir-use-symlinks False"
    )
    .run_commands(f"mkdir -p {LOCAL_MODEL_PATH}/{LANG}")
    .run_commands(
        f"wget https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/"
        f"{TITLE}/{LANG}.zip -O /tmp/russian.zip"
    )
    .run_commands(f"unzip -j /tmp/russian.zip -d {LOCAL_MODEL_PATH}/{LANG}")
    .run_commands("rm /tmp/russian.zip")
    .run_commands(f"touch {LOCAL_MODEL_PATH}/{LANG}/.downloaded")
)

app = modal.App("booknlp-ru-trankit")


# ─── Вспомогательные функции вывода (используются в local_entrypoint) ─────────

def _print_token_simplified(tok: Dict[str, Any]) -> None:
    """Выводит токен в simplified-формате (CoNLL-U колонки)."""
    print(
        f"  {tok['id']:<4} {tok['form']:<14} {tok['lemma']:<14} "
        f"{tok['upos']:<7} {tok['xpos']:<5} "
        f"{tok['head']:<5} {tok['deprel']:<12} "
        f"{tok['deps']:<5} {tok['misc']:<5} "
        f"{tok['start_char']} {tok['end_char']}"
    )
    if tok.get("feats", "_") != "_":
        print(f"       feats: {tok['feats']}")


def _print_token_native(tok: Dict[str, Any]) -> None:
    """Выводит токен в native-формате (все поля Trankit)."""
    print(f"\n  Text: {tok.get('text')}")
    print(f"    id: {tok.get('id')}")
    print(f"    lemma: {tok.get('lemma')}, upos: {tok.get('upos')}, xpos: {tok.get('xpos')}")
    print(f"    feats: {tok.get('feats')}")
    print(f"    head: {tok.get('head')}, deprel: {tok.get('deprel')}")
    print(f"    span: {tok.get('span')}, dspan: {tok.get('dspan')}")
    print(f"    ner: {tok.get('ner')}")
    print(f"    lang: {tok.get('lang')}")
    expanded = tok.get("expanded")
    print(f"    expanded: {expanded if expanded else '[]'}")


# ─── TrankitService ───────────────────────────────────────────────────────────

@app.cls(image=image, gpu="T4", timeout=600)
class TrankitService:
    """Trankit NLP-сервис с поддержкой CUDA и chunked-обработки."""

    logger: logging.Logger
    nlp: Any

    @modal.enter()
    def setup(self):
        import trankit

        # [ИЗМЕНЕНО] TRANSFORMERS_OFFLINE выставляется ДО обращения к
        # trankit.supported_embeddings — гарантирует офлайн-режим до
        # любой инициализации трансформеров.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TrankitService")
        self.logger.info(
            f"Setting up Trankit with local model at {LOCAL_MODEL_PATH} ({TITLE})"
        )

        # Белый список: разрешает использовать локальный путь вместо
        # HuggingFace-имени модели
        if LOCAL_MODEL_PATH not in trankit.supported_embeddings:
            trankit.supported_embeddings.append(LOCAL_MODEL_PATH)
            self.logger.info(f"Added {LOCAL_MODEL_PATH} to whitelist.")

        try:
            self.nlp = trankit.Pipeline(
                LANG,
                embedding=LOCAL_MODEL_PATH,
                gpu=True,
                cache_dir=LOCAL_MODEL_PATH,
            )
            self.logger.info("Trankit loaded successfully from local files!")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to initialize Trankit: {e}")
            try:
                import glob
                self.logger.info(f"Files in {LOCAL_MODEL_PATH}: "
                                 f"{glob.glob(f'{LOCAL_MODEL_PATH}/**')}")
                self.logger.info(f"Files in {LOCAL_MODEL_PATH}/{LANG}: "
                                 f"{glob.glob(f'{LOCAL_MODEL_PATH}/{LANG}/**')}")
            except Exception:  # noqa: BLE001
                pass
            self.nlp = None

    # ─── Вспомогательные методы форматирования ───────────────────────────────

    @staticmethod
    def _extract_span(token: dict) -> Tuple[int, int]:
        """
        Извлекает sentence-local символьные офсеты токена.

        Порядок приоритета: span → dspan → (0, 0).

        При is_sent=True оба поля sentence-local (относительны начала
        предложения), поэтому используем span как основной источник.
        dspan оставлен как fallback для совместимости с вызовами без
        is_sent=True (backward compat через parse()).

        [ИЗМЕНЕНО] Было: только dspan. Стало: span → dspan → (0, 0).
        """
        for key in ("span", "dspan"):
            val = token.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return int(val[0]), int(val[1])
        return 0, 0

    @staticmethod
    def _process_simplified(
        doc: dict,
        char_offset: int = 0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Преобразует нативный вывод Trankit в simplified (CoNLL-U) формат.

        Полный набор CoNLL-U полей:
          id       — номер токена в предложении (1-based)
          form     — текстовая форма токена
          lemma    — лемма
          upos     — Universal POS tag
          xpos     — language-specific POS (для русского — всегда "_")
          feats    — морфологические признаки
          head     — индекс головы (0 = root)
          deprel   — тип синтаксической связи
          deps     — Enhanced Dependencies: "_" (Trankit не предсказывает)
          misc     — SpaceAfter и пр.: "_" (Trankit не предсказывает)
          start_char — начало токена в исходном документе
          end_char   — конец токена в исходном документе

        [НОВОЕ] Параметр char_offset: смещение предложения в исходном тексте,
        добавляется к span для получения глобальных позиций.
        [НОВОЕ] Поля deps и misc (всегда "_") — для полного CoNLL-U.
        [ИЗМЕНЕНО] Fallback: span → dspan → (0, 0) через _extract_span.
        [ИСПРАВЛЕНО] Обрабатывает оба формата вывода Trankit:
            - doc-level:      {"sentences": [...]}  ← nlp(text) и nlp(list, is_sent=True)
            - sentence-level: {"tokens": [...]}     ← nlp(str, is_sent=True)

        Args:
            doc: нативный вывод trankit.Pipeline()
            char_offset: смещение предложения в исходном документе
        Returns:
            List[List[Dict]]: список предложений → список токенов
        """
        # [ИСПРАВЛЕНО] Нормализация входного формата:
        # nlp(str, is_sent=True) возвращает sentence-level dict без "sentences"
        if "sentences" in doc:
            sentences = doc["sentences"]
        elif "tokens" in doc:
            # Оборачиваем одиночное предложение в стандартный формат
            sentences = [doc]
        else:
            return []

        result = []
        for sent in sentences:
            sent_tokens = []
            for t in sent["tokens"]:
                # ID токена: для MWT может быть списком — берём первый элемент
                tid = t.get("id", 0)
                if isinstance(tid, list):
                    tid = tid[0] if tid else 0

                # Sentence-local офсеты + глобальное смещение из razdel
                local_start, local_end = TrankitService._extract_span(t)
                start_char = local_start + char_offset
                end_char = local_end + char_offset

                # Нормализация полей: None → "_"
                upos = t.get("upos") or t.get("pos", "_") or "_"
                xpos = t.get("xpos") or "_"
                feats = t.get("feats") or "_"
                lemma = t.get("lemma") or t.get("text", "") or ""

                sent_tokens.append({
                    "id": int(tid) if str(tid).isdigit() else 0,
                    "form": t.get("text", ""),
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": feats,
                    "head": int(t.get("head", 0)),
                    "deprel": t.get("deprel", "_") or "_",
                    # Enhanced Dependencies — не поддерживается Trankit
                    "deps": "_",
                    # SpaceAfter и пр. — не поддерживается Trankit
                    "misc": "_",
                    "start_char": start_char,
                    "end_char": end_char,
                })
            if sent_tokens:
                result.append(sent_tokens)
        return result

    @staticmethod
    def _process_native(
        doc: dict,
        char_offset: int = 0,
        lang_fallback: str = LANG,
    ) -> List[List[Dict[str, Any]]]:
        """
        Возвращает полный нативный формат Trankit со всеми полями.

        Поля токена:
          id       — int или list (для MWT)
          text     — текстовая форма
          lemma    — лемма
          upos     — Universal POS tag
          xpos     — language-specific POS (для русского — "_")
          feats    — морфологические признаки
          head     — индекс головы
          deprel   — тип синтаксической связи
          span     — (start, end) относительно начала предложения (sentence-local)
          dspan    — (start, end) в исходном документе = span + char_offset
          ner      — NER-тег в формате BIO/BIOES
          expanded — список словарей для MWT (Multi-Word Tokens)
          lang     — язык предложения

        [НОВОЕ] Параметр char_offset: используется для вычисления dspan.
        [ИЗМЕНЕНО] dspan = span + char_offset (а не берётся из Trankit-вывода).
        При is_sent=True Trankit возвращает dspan=span (sentence-local),
        поэтому глобальный dspan вычисляется здесь.
        [ИСПРАВЛЕНО] Обрабатывает оба формата вывода Trankit (аналогично _process_simplified).

        Args:
            doc: нативный вывод trankit.Pipeline()
            char_offset: смещение предложения в исходном документе
            lang_fallback: язык по умолчанию, если sent['lang'] отсутствует
        Returns:
            List[List[Dict]]: список предложений → список токенов
        """
        # Нормализация входного формата
        if "sentences" in doc:
            sentences = doc["sentences"]
        elif "tokens" in doc:
            sentences = [doc]
        else:
            return []

        result = []
        for sent in doc["sentences"]:
            # Язык предложения — из поля Trankit или fallback
            sent_lang = sent.get("lang") or lang_fallback

            sent_tokens = []
            for t in sent["tokens"]:
                # span — sentence-local, всегда корректен независимо от пути
                raw_span = t.get("span")
                if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                    span = (int(raw_span[0]), int(raw_span[1]))
                else:
                    # Fallback: пробуем dspan как sentence-local
                    raw_dspan = t.get("dspan")
                    if isinstance(raw_dspan, (list, tuple)) and len(raw_dspan) == 2:
                        span = (int(raw_dspan[0]), int(raw_dspan[1]))
                    else:
                        span = (0, 0)

                # [ИЗМЕНЕНО] dspan вычисляется явно: span + char_offset.
                # Нельзя доверять t['dspan'] при is_sent=True — Trankit
                # возвращает там sentence-local значения, не глобальные.
                dspan = (span[0] + char_offset, span[1] + char_offset)

                sent_tokens.append({
                    "id": t.get("id"),
                    "text": t.get("text", ""),
                    "lemma": t.get("lemma", "") or "",
                    "upos": t.get("upos", "_") or "_",
                    "xpos": t.get("xpos", "_") or "_",
                    "feats": t.get("feats", "_") or "_",
                    "head": t.get("head", 0),
                    "deprel": t.get("deprel", "_") or "_",
                    "span": span,
                    "dspan": dspan,      # [ИЗМЕНЕНО] скорректирован char_offset
                    "ner": t.get("ner", "O"),
                    "expanded": t.get("expanded", []),
                    "lang": sent_lang,
                })
            if sent_tokens:
                result.append(sent_tokens)
        return result

    # ─── Production methods: принимают pre-split чанки из wrapper ────────────

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences_with_offsets: List[Tuple[str, int]],
        output_format: str = "simplified",
    ) -> List[List[Dict[str, Any]]]:
        """
        Razdel path. [НОВЫЙ МЕТОД]

        Принимает чанк пар (sentence_text, start_char_in_original).
        Каждая пара — одно предложение из razdel.sentenize() в wrapper
        + его символьный офсет в исходном документе.

        Для каждого предложения вызывает:
            self.nlp(sent_text, is_sent=True)
        is_sent=True — Trankit пропускает внутреннюю сентенизацию,
        обрабатывает строку как одно готовое предложение.

        [ИСПРАВЛЕНО] Trankit вызывается ОДИН РАЗ для всего чанка через список строк.
        nlp(List[str], is_sent=True) → {"sentences": [...]} — стандартный формат.
        Предыдущая версия вызывала nlp(str, is_sent=True) в цикле →
        возвращало {"tokens": [...]} без ключа "sentences" → _process_* возвращал [].
        Офсеты применяются поэлементно: zip(doc["sentences"], char_offsets).

        Глобальные офсеты токенов:
            start_char = span[0] + char_offset   (simplified)
            dspan      = span + char_offset       (native)

        Args:
            sentences_with_offsets: List[(sentence_text, start_char)]
                sentence_text — текст предложения
                start_char    — позиция начала предложения в исходном документе
                                (s.start из razdel.sentenize)
            output_format: "simplified" | "native"
        Returns:
            List[List[Dict]] — список предложений чанка, каждое — список токенов
        """
        if self.nlp is None or not sentences_with_offsets:
            return []

        # Фильтруем пустые предложения, сохраняя соответствие офсетов
        filtered = [
            (text, offset)
            for text, offset in sentences_with_offsets
            if text.strip()
        ]
        if not filtered:
            return []

        sent_texts = [text for text, _ in filtered]
        char_offsets = [offset for _, offset in filtered]

        try:
            # [ИСПРАВЛЕНО] Передаём список → Trankit возвращает {"sentences": [...]}
            # is_sent=True: Trankit пропускает внутреннюю сентенизацию
            doc = self.nlp(sent_texts, is_sent=True)

            result: List[List[Dict[str, Any]]] = []
            # Каждое предложение из doc["sentences"] обрабатываем со своим char_offset
            for sent_dict, char_offset in zip(doc.get("sentences", []), char_offsets):
                # Оборачиваем в стандартный формат для _process_*
                wrapped = {"sentences": [sent_dict]}
                if output_format == "native":
                    processed = self._process_native(wrapped, char_offset=char_offset)
                else:
                    processed = self._process_simplified(wrapped, char_offset=char_offset)
                result.extend(processed)
            return result

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"parse_sentence_chunk error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    @modal.method()
    def parse_sentence_chunk_native(
        self,
        sentences: List[str],
        output_format: str = "simplified",
    ) -> List[List[Dict[str, Any]]]:
        """
        Native path. [НОВЫЙ МЕТОД]

        Принимает чанк текстов предложений без символьных офсетов.
        Предложения получены через razdel.sentenize() в wrapper — только тексты.

        Для каждого предложения вызывает:
            self.nlp(sent_text, is_sent=True)

        Офсеты токенов (span, dspan, start_char, end_char) — относительны
        начала каждого предложения (char_offset=0). Это осознанное ограничение:
        для получения глобальных позиций используйте razdel path
        (parse_sentence_chunk).
        [ИСПРАВЛЕНО] передаём список строк, а не итерируемся с nlp(str, is_sent=True).

        Args:
            sentences: List[str] — тексты предложений чанка
            output_format: "simplified" | "native"
        Returns:
            List[List[Dict]] — список предложений чанка
        """
        if self.nlp is None or not sentences:
            return []

        filtered = [s for s in sentences if s.strip()]
        if not filtered:
            return []

        try:
            # [ИСПРАВЛЕНО] Один вызов для всего чанка
            doc = self.nlp(filtered, is_sent=True)

            result: List[List[Dict[str, Any]]] = []
            for sent_dict in doc.get("sentences", []):
                wrapped = {"sentences": [sent_dict]}
                if output_format == "native":
                    processed = self._process_native(wrapped, char_offset=0)
                else:
                    processed = self._process_simplified(wrapped, char_offset=0)
                result.extend(processed)
            return result

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"parse_sentence_chunk_native error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    # ─── Backward compat / local_entrypoint ──────────────────────────────────

    def _parse_text_internal(
        self, text: str, output_format: str = "simplified"
    ) -> List[List[Dict[str, Any]]]:
        """
        Парсит текст целиком: Trankit выполняет внутреннюю сентенизацию.
        Используется только в backward-compat методах parse() и parse_batch().
        В production использовать parse_sentence_chunk / parse_sentence_chunk_native.
        """
        if self.nlp is None or not text.strip():
            return []
        try:
            doc = self.nlp(text)
            if output_format == "native":
                return self._process_native(doc, char_offset=0)
            return self._process_simplified(doc, char_offset=0)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    @modal.method()
    def parse(
        self, text: str, output_format: str = "simplified"
    ) -> List[List[Dict[str, Any]]]:
        """
        Backward compat: парсит текст целиком (Trankit сентенизирует сам).
        Для local_entrypoint и прямых вызовов в тестах.
        В production использовать parse_sentence_chunk / parse_sentence_chunk_native.
        """
        return self._parse_text_internal(text, output_format)

    @modal.method()
    def parse_batch(
        self, texts: List[str], output_format: str = "simplified"
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Backward compat: последовательная пакетная обработка текстов целиком.
        В production использовать parse_sentence_chunk.map() из wrapper.
        """
        return [self._parse_text_internal(t, output_format) for t in texts]


# ─── local_entrypoint: прямое тестирование Modal-сервиса ─────────────────────

@app.local_entrypoint()
def main():
    """
    Тестирует TrankitService напрямую — без wrapper, без chunking.

    Блоки тестов:
    1. parse.remote() — simplified, весь текст (backward compat)
    2. parse.remote() — native, весь текст (backward compat)
    3. Сравнение ключей форматов simplified / native
    4. parse_sentence_chunk.remote() — razdel path, pre-split чанк
    5. parse_sentence_chunk_native.remote() — native path, pre-split чанк
    6. Проверка корректности офсетов (razdel path)
    """
    import json
    from razdel import sentenize

    logging.basicConfig(level=logging.INFO)

    test_text = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься."
    )
    text_multi = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься. "
        "Москва — столица России."
    )
    sep = "=" * 70
    service = TrankitService()  # type: ignore[call-arg]

    # ── 1. Simplified (backward compat) ──────────────────────────────────────
    print(f"\n{sep}")
    print("1. РЕЖИМ: simplified (parse.remote — backward compat)")
    print(sep)
    result_s = service.parse.remote(test_text, output_format="simplified")
    print(f"Предложений: {len(result_s)}\n")
    print(
        f"  {'ID':<4} {'FORM':<14} {'LEMMA':<14} {'UPOS':<7} {'XPOS':<5} "
        f"{'HEAD':<5} {'DEPREL':<12} {'DEPS':<5} {'MISC':<5} START END"
    )
    print("  " + "-" * 110)
    for sent in result_s:
        for t in sent:
            _print_token_simplified(t)
    print(f"\nКлючи simplified-токена: {list(result_s[0][0].keys())}")
    print("\nJSON первого токена:")
    print(json.dumps(result_s[0][0], ensure_ascii=False, indent=2))

    # ── 2. Native (backward compat) ──────────────────────────────────────────
    print(f"\n{sep}")
    print("2. РЕЖИМ: native (parse.remote — backward compat)")
    print(sep)
    result_n = service.parse.remote(test_text, output_format="native")
    for sent in result_n:
        print(
            f"\n  {'ID':<4} {'TEXT':<14} {'LEMMA':<14} {'UPOS':<7} "
            f"{'HEAD':<5} {'DEPREL':<12} {'NER':<8} LANG"
        )
        print("  " + "-" * 106)
        for t in sent:
            print(
                f"  {t['id']:<4} {t['text']:<14} {t['lemma']:<14} {t['upos']:<7} "
                f"{t['head']:<5} {t['deprel']:<12} {t.get('ner', 'O'):<8} "
                f"{t.get('lang', '')}"
            )
            if t.get("feats", "_") != "_":
                print(f"       feats: {t['feats']}")
    print(f"\nКлючи native-токена: {list(result_n[0][0].keys())}")
    print("\nJSON первого токена:")
    print(json.dumps(result_n[0][0], ensure_ascii=False, indent=2, default=str))

    # ── 3. Сравнение ключей форматов ─────────────────────────────────────────
    print(f"\n{sep}")
    print("3. СРАВНЕНИЕ КЛЮЧЕЙ ФОРМАТОВ")
    print(sep)
    sk = set(result_s[0][0].keys())
    nk = set(result_n[0][0].keys())
    print(f"  Только в simplified: {sorted(sk - nk)}")
    print(f"  Только в native:     {sorted(nk - sk)}")
    print(f"  Общие ключи:         {sorted(sk & nk)}")

    # ── 4. parse_sentence_chunk — razdel path ────────────────────────────────
    print(f"\n{sep}")
    print("4. parse_sentence_chunk (razdel path, pre-split чанк)")
    print(sep)
    sentences = list(sentenize(text_multi))
    chunk_razdel = [(s.text, s.start) for s in sentences]
    print(f"Чанк ({len(chunk_razdel)} предложений):")
    for s_text, s_offset in chunk_razdel:
        print(f"  offset={s_offset:>3}: '{s_text}'")

    result_chunk_s = service.parse_sentence_chunk.remote(
        chunk_razdel, output_format="simplified"
    )
    print(f"\nresult: {len(result_chunk_s)} предложений")
    for i, sent in enumerate(result_chunk_s):
        print(
            f"\n  Предложение {i + 1} "
            f"(razdel start={chunk_razdel[i][1]}):"
        )
        print(
            f"  {'ID':<4} {'FORM':<14} {'LEMMA':<14} {'UPOS':<7} "
            f"{'HEAD':<5} {'DEPREL':<12} {'DEPS':<5} {'MISC':<5} START END"
        )
        print("  " + "-" * 100)
        for t in sent:
            _print_token_simplified(t)

    # ── 5. parse_sentence_chunk_native — native path ──────────────────────────
    print(f"\n{sep}")
    print("5. parse_sentence_chunk_native (native path, pre-split чанк)")
    print(sep)
    chunk_texts = [s.text for s in sentences]
    print(f"Чанк ({len(chunk_texts)} предложений): {chunk_texts}")

    result_chunk_n = service.parse_sentence_chunk_native.remote(
        chunk_texts, output_format="native"
    )
    print(f"\nresult: {len(result_chunk_n)} предложений")
    for i, sent in enumerate(result_chunk_n):
        print(f"\n  Предложение {i + 1}:")
        print(
            f"  {'ID':<4} {'TEXT':<14} {'UPOS':<7} "
            f"span          dspan         {'NER':<8} LANG"
        )
        print("  " + "-" * 90)
        for t in sent:
            print(
                f"  {str(t['id']):<4} {t['text']:<14} {t['upos']:<7} "
                f"{str(t['span']):<14} {str(t['dspan']):<14} "
                f"{t.get('ner', 'O'):<8} {t.get('lang', '')}"
            )

    # ── 6. Проверка корректности офсетов (razdel path) ────────────────────────
    print(f"\n{sep}")
    print("6. ПРОВЕРКА ОФСЕТОВ (razdel path)")
    print(sep)
    print(f"Исходный текст: '{text_multi}'")
    print(f"\nОжидаемые офсеты razdel.sentenize:")
    for s in sentences:
        print(f"  start={s.start:>3}, stop={s.stop:>3}: '{s.text}'")
    print(f"\nПервый токен каждого предложения из parse_sentence_chunk:")
    for i, sent in enumerate(result_chunk_s):
        if not sent:
            continue
        t0 = sent[0]
        expected = chunk_razdel[i][1]
        actual = t0["start_char"]
        # Первый токен может начинаться не с позиции 0 предложения
        # (если предложение начинается с пробела), допустимо >= expected
        ok = "✅" if actual >= expected else "❌"
        print(
            f"  {ok} Предложение {i + 1}: "
            f"razdel_start={expected}, "
            f"token[0].start_char={actual}, "
            f"form='{t0['form']}'"
        )

    print(f"\n{'✅ Тестирование завершено!':^70}")
