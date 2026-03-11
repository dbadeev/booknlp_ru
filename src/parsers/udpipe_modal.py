# udpipe_modal.py
# ─────────────────────────────────────────────────────────────────────────────
# Modal-сервис UDPipe для морфосинтаксического анализа русского текста.
#
# ИЗМЕНЕНИЯ по сравнению с исходной версией помечены: # ← НОВОЕ / # ← ИЗМЕНЕНО
# ─────────────────────────────────────────────────────────────────────────────

import modal
import logging
from typing import Any, Dict, List, Literal

# ─── Modal image ──────────────────────────────────────────────────────────────
# Образ контейнера: Python 3.11, ufal.udpipe + razdel.
# razdel используется в parse() для backward-compat раздела local_entrypoint.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ufal.udpipe>=1.3.0",
        "razdel>=0.5.0",
    )
    .run_commands(
        # Скачиваем модель russian-syntagrus-ud-2.5 при сборке образа
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "\'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/"
        "11234/1-3131/russian-syntagrus-ud-2.5-191206.udpipe\', "
        "\'/root/russian-syntagrus-ud-2.5-191206.udpipe\')\"",
    )
)

app = modal.App("booknlp-ru-udpipe")

# ← ИЗМЕНЕНО: добавлен TokenizerType; OutputFormat переименован для ясности
TokenizerType = Literal["native", "razdel"]
OutputFormat  = Literal["dict", "native"]


# ─── Service ──────────────────────────────────────────────────────────────────

@app.cls(image=image, timeout=600, scaledown_window=300)
class UDPipeService:
    """
    Modal-сервис для морфосинтаксического анализа текста через UDPipe
    (модель russian-syntagrus-ud-2.5).

    Поддерживает два режима токенизации:
      native  — встроенный токенизатор UDPipe (Pipeline input_format="tokenize")
      razdel  — внешняя токенизация razdel   (Pipeline input_format="horizontal")

    Два формата поля misc:
      dict    — misc как строка CoNLL-U ("SpaceAfter=No")
      native  — misc как словарь Python  ({'SpaceAfter': 'No'})

    Производственные методы (вызываются из wrapper через .map()):
      parse_sentence_chunk         — native-путь: List[str] (тексты предложений)
      parse_sentence_chunk_razdel  — razdel-путь: List[List[str]] (токены)

    Вспомогательные методы (local_entrypoint / прямые вызовы):
      parse        — один текст целиком (backward compat)
      parse_batch  — пакет текстов     (backward compat)
    """

    @modal.enter()
    def load(self):
        """Загружает модель UDPipe при старте контейнера (один раз на instance)."""
        import ufal.udpipe

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UDPipeService")

        model_path = "/root/russian-syntagrus-ud-2.5-191206.udpipe"
        self.logger.info("Loading UDPipe model...")
        # noinspection PyTypeChecker
        self.model = ufal.udpipe.Model.load(model_path)
        if not self.model:
            raise RuntimeError(f"Не удалось загрузить модель UDPipe: {model_path}")
        self.logger.info("UDPipe loaded!")

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _get_pipeline(self, input_format: str = "tokenize"):
        """
        ← ИЗМЕНЕНО: принимает input_format: str вместо tokenizer_options: dict.

        Создаёт ufal.udpipe.Pipeline с заданным форматом входных данных.

        Args:
            input_format:
                "tokenize"   — встроенный токенизатор UDPipe (native-путь);
                               принимает обычный текст на входе.
                "horizontal" — предтокенизированный ввод (razdel-путь);
                               одна строка = одно предложение, токены через пробел.

        Returns:
            ufal.udpipe.Pipeline с выходным форматом CoNLL-U.

        Raises:
            ValueError: если input_format не входит в допустимые значения.
        """
        import ufal.udpipe

        # ← ИЗМЕНЕНО: убран блок tokenizer_options={'ranges': True} с NotImplementedError
        if input_format not in ("tokenize", "horizontal"):
            raise ValueError(
                f"Неизвестный input_format: {input_format!r}. "
                "Допустимые значения: 'tokenize', 'horizontal'."
            )

        return ufal.udpipe.Pipeline(
            self.model,
            input_format,                      # режим ввода
            ufal.udpipe.Pipeline.DEFAULT,      # tagger (POS + lemma + feats)
            ufal.udpipe.Pipeline.DEFAULT,      # parser (head + deprel)
            "conllu",                          # выходной формат — всегда CoNLL-U
        )

    @staticmethod
    def _build_horizontal_input(token_lists: List[List[str]]) -> str:
        """
        ← НОВОЕ: Формирует строку в horizontal-формате UDPipe.

        UDPipe horizontal-формат:
          - одна строка = одно предложение
          - токены разделены пробелами
          - строки разделены символом \n

        В этом режиме UDPipe пропускает шаг токенизации и выполняет
        только тегирование и синтаксический разбор.

        Args:
            token_lists: список предложений; каждое предложение —
                         список строк-токенов (из razdel.tokenize)

        Returns:
            Строка для подачи в Pipeline("horizontal").
        """
        # ← фильтруем пустые предложения до построения строки
        clean = [tl for tl in token_lists if tl]
        return "\n".join(" ".join(tokens) for tokens in clean) + "\n"

    @staticmethod
    def _parse_misc(misc_str: str, output_format: str) -> Any:
        """
        Разбирает поле MISC из CoNLL-U.

        Args:
            misc_str:      строка вида "SpaceAfter=No|Translit=Zlo" или "_".
            output_format: "dict"   → возвращает строку как есть (сырой CoNLL-U).
                           "native" → возвращает словарь {'SpaceAfter': 'No', ...}.

        Returns:
            str (output_format="dict") | dict (output_format="native").
        """
        if misc_str == "_":
            return "_" if output_format == "dict" else {}
        if output_format == "native":
            return dict(kv.split("=", 1) for kv in misc_str.split("|") if "=" in kv)
        return misc_str  # ← dict: возвращаем raw CoNLL-U строку без изменений

        # if output_format == "dict":
        #     # Сырая строка CoNLL-U: "_" возвращаем как есть
        #     return misc_str
        #
        # # native: разбираем "Key=Val|Key2=Val2" → словарь
        # if misc_str == "_":
        #     return {}
        #
        # result: Dict[str, Any] = {}
        # for part in misc_str.split("|"):
        #     if "=" in part:
        #         key, _, val = part.partition("=")
        #         # Декодируем экранированные символы (\n → реальный newline и т.д.)
        #         val = val.replace("\\n", "\n").replace("\\t", "\t")
        #         result[key] = val
        #     else:
        #         # Одиночный флаг без значения → True
        #         result[part] = True
        # return result

    def _parse_conllu_output(
        self,
        conllu_text: str,
        output_format: str,
    ) -> List[List[Dict[str, Any]]]:
        """
        Преобразует строку CoNLL-U от Pipeline в список предложений с токенами.

        Каждый токен содержит ровно 10 стандартных полей CoNLL-U:
          id, form, lemma, upos, xpos, feats, head, deprel, deps, misc.

        Args:
            conllu_text:   строка в формате CoNLL-U.
            output_format: "dict" | "native" — формат поля misc.

        Returns:
            List[List[Dict]] — список предложений, каждое — список токенов.
        """
        sentences: List[List[Dict[str, Any]]] = []
        current:   List[Dict[str, Any]]       = []

        for line in conllu_text.splitlines():
            line = line.strip()

            # Граница предложения — пустая строка
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # Пропускаем комментарии CoNLL-U (# sent_id, # text, ...)
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            # Пропускаем multi-word tokens (1-2) и пустые узлы (1.1)
            if "-" in parts[0] or "." in parts[0]:
                continue

            # ← ИЗМЕНЕНО: безопасный fallback для HEAD с предупреждением в лог
            if parts[6].isdigit():
                head_val = int(parts[6])
            else:
                self.logger.warning(
                    f"Неожиданное значение HEAD: {parts[6]!r} для токена {parts[1]!r}; "
                    f"подставляем 0."
                )
                head_val = 0

            token: Dict[str, Any] = {
                "id":     int(parts[0]),
                "form":   parts[1],
                "lemma":  parts[2],
                "upos":   parts[3],
                "xpos":   parts[4],
                "feats":  parts[5],
                "head":   head_val,
                "deprel": parts[7],
                "deps":   parts[8],
                "misc":   self._parse_misc(parts[9], output_format),
            }
            current.append(token)

        # Последнее предложение (если файл не заканчивается пустой строкой)
        if current:
            sentences.append(current)

        return sentences

    def _process_text(
        self,
        text: str,
        input_format: str,
        output_format: str,
    ) -> List[List[Dict[str, Any]]]:
        """
        Внутренний метод: прогоняет текст через Pipeline и возвращает разбор.

        Используется обоими производственными методами и методами backward compat.

        Args:
            text:          входная строка (plain text или horizontal-формат).
            input_format:  "tokenize" | "horizontal".
            output_format: "dict" | "native".

        Returns:
            List[List[Dict]] — список предложений с токенами (10 полей).

        Raises:
            RuntimeError: если UDPipe сообщил об ошибке.
        """
        import ufal.udpipe

        # ← ИСПРАВЛЕНО: пустой ввод → всегда [], убрана ветка return ""
        if not text or not text.strip():
            return []

        pipeline = self._get_pipeline(input_format)
        error    = ufal.udpipe.ProcessingError()
        processed = pipeline.process(text, error)

        if error.occurred():
            raise RuntimeError(f"UDPipe ProcessingError: {error.message}")

        # ← ИСПРАВЛЕНО: безопасная проверка ошибки
        # (было: "error" in processed.lower() — ложные срабатывания на слово "terror")
        if processed.startswith("error") or "\nerror" in processed:
            raise RuntimeError(
                f"UDPipe вернул ошибку в выводе: {processed[:200]!r}"
            )

        return self._parse_conllu_output(processed, output_format)

    # ─── Production methods ───────────────────────────────────────────────────
    # Вызываются из udpipe_wrapper.py через .map() / .remote().
    # Принимают предварительно разбитые чанки; повторная сентенизация исключена.

    @modal.method()
    def parse_sentence_chunk(
        self,
        sentences: List[str],
        output_format: str = "dict",
    ) -> List[List[Dict[str, Any]]]:
        """
        ← НОВОЕ: Native-путь (встроенный токенизатор UDPipe).

        Принимает список текстов предложений, предварительно разбитых
        razdel.sentenize в wrapper. UDPipe получает одно предложение за раз
        → повторная сентенизация физически невозможна.

        Тегирование (POS, lemma, feats) и синтаксический разбор (head, deprel)
        выполняются штатным образом.

        Args:
            sentences:     List[str] — тексты предложений (из razdel.sentenize).
            output_format: "dict"   → misc как строка CoNLL-U.
                           "native" → misc как словарь Python.

        Returns:
            List[List[Dict]] — все предложения чанка, 10 полей на токен.
        """
        result: List[List[Dict[str, Any]]] = []
        for sent_text in sentences:
            if not sent_text or not sent_text.strip():
                continue
            # Одно предложение → UDPipe не может его разбить на несколько
            parsed = self._process_text(sent_text, "tokenize", output_format)
            result.extend(parsed)
        return result

    @modal.method()
    def parse_sentence_chunk_razdel(
        self,
        token_lists: List[List[str]],
        output_format: str = "dict",
    ) -> List[List[Dict[str, Any]]]:
        """
        ← НОВОЕ: Razdel-путь (внешний токенизатор razdel).

        Принимает список предложений, уже токенизированных razdel.tokenize
        в wrapper. Передаёт в UDPipe в horizontal-формате:
          одна строка = одно предложение, токены разделены пробелами.

        UDPipe пропускает шаг токенизации и выполняет только тегирование
        (POS, lemma, feats) и синтаксический разбор (head, deprel).

        Возможное расхождение: razdel и UDPipe могут по-разному сегментировать
        отдельные токены (кавычки, тире). Это влияет на качество разбора,
        но не на корректность обработки предложения как единицы.

        Args:
            token_lists:   List[List[str]] — список предложений; каждое —
                           список строк-токенов (из razdel.tokenize).
            output_format: "dict" | "native".

        Returns:
            List[List[Dict]] — все предложения чанка, 10 полей на токен.

        Ограничение: в horizontal-режиме UDPipe не заполняет MISC (SpaceAfter и др.),
        т.к. не имеет доступа к оригинальному тексту. Используйте native-путь,
        если MISC необходим. Офсеты для постобработки доступны через wrapper.
        """
        clean = [tl for tl in token_lists if tl]  # ← добавить
        if not clean:
            return []
        # Строим horizontal-ввод для UDPipe
        horizontal_text = self._build_horizontal_input(token_lists)
        return self._process_text(horizontal_text, "horizontal", output_format)

    # ─── Backward compat / local_entrypoint ───────────────────────────────────

    @modal.method()
    def parse(
        self,
        text: str,
        output_format: str = "dict",
        tokenizer: TokenizerType = "native",
    ) -> List[List[Dict[str, Any]]]:
        """
        Парсит один текст целиком.
        Для local_entrypoint и прямых вызовов (backward compat).

        При tokenizer="razdel" выполняет сентенизацию и токенизацию внутри
        контейнера, затем передаёт в horizontal-формат.

        Args:
            text:          входной текст.
            output_format: "dict" | "native".
            tokenizer:     "native" | "razdel".

        Returns:
            List[List[Dict]]
        """
        if tokenizer == "razdel":
            # ← НОВОЕ: razdel-путь через horizontal-формат
            from razdel import sentenize, tokenize as razdel_tokenize
            sentences   = list(sentenize(text))
            token_lists = [
                [tok.text for tok in razdel_tokenize(s.text)]
                for s in sentences
            ]
            if not token_lists:
                return []
            horizontal_text = self._build_horizontal_input(token_lists)
            return self._process_text(horizontal_text, "horizontal", output_format)

        # native: UDPipe сам токенизирует
        return self._process_text(text, "tokenize", output_format)

    @modal.method()
    def parse_batch(
        self,
        texts: List[str],
        output_format: str = "dict",
        tokenizer: TokenizerType = "native",
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        ← ИЗМЕНЕНО: добавлен параметр tokenizer (был только output_format).

        Пакетная обработка текстов. Backward compat.
        В production используйте wrapper.parse_batch(), который передаёт
        все чанки через Modal .map() параллельно.

        Args:
            texts:         список входных текстов.
            output_format: "dict" | "native".
            tokenizer:     "native" | "razdel".

        Returns:
            List[List[List[Dict]]] — для каждого текста: список предложений.

        """

        return [
            self.parse.local(text, output_format=output_format, tokenizer=tokenizer)
            for text in texts
        ]


# ─── Вспомогательные функции вывода для local_entrypoint ─────────────────────

# Заголовок таблицы CoNLL-U (10 стандартных полей)
_CONLLU_HEADER = (
    f"  {'ID':>4}  "
    f"{'FORM':<14} "
    f"{'LEMMA':<14} "
    f"{'UPOS':<7} "
    f"{'XPOS':<12} "
    f"{'FEATS':<35} "
    f"{'HEAD':>4}  "
    f"{'DEPREL':<12} "
    f"{'DEPS':<6}  "
    f"MISC"
)
_CONLLU_SEP = "  " + "─" * (len(_CONLLU_HEADER) - 2)
_CONLLU_HEADER = (
    f"  {'ID':>4}  {'FORM':<14} {'LEMMA':<14} "
    f"{'UPOS':<7} {'XPOS':<12} {'HEAD':>4}  "
    f"{'DEPREL':<12} {'DEPS':<6}  MISC"
)

def _print_sentence_table(
    sent_idx: int,
    tokens: List[Dict[str, Any]],
    sent_text: str = "",
) -> None:
    """
    Выводит предложение в виде таблицы CoNLL-U со всеми 10 полями.

    Формат вывода:
        Предложение N:  # text = <текст>
          ID   FORM   LEMMA   UPOS   XPOS   FEATS   HEAD   DEPREL   DEPS   MISC
          ───────────────────────────────────────────────────────────────────────
           1   ...
    """
    if sent_text:
        # Если sent_text уже содержит лейбл (начинается с '#') — используем как есть,
        # иначе добавляем стандартный '# text ='
        if sent_text.startswith("#"):
            print(f"\n  Предложение {sent_idx}:  {sent_text}")
        else:
            print(f"\n  Предложение {sent_idx}:  # text = {sent_text}")
    else:
        print(f"\n  Предложение {sent_idx}:")

    print(_CONLLU_HEADER)
    print(_CONLLU_SEP)

    for tok in tokens:
        print(
            f"  {tok['id']:>4}  {tok['form']:<14} {tok['lemma']:<14} "
            f"{tok['upos']:<7} {tok['xpos']:<12} {tok['head']:>4}  "
            f"{tok['deprel']:<12} {tok['deps']:<6}  {tok['misc']}"
        )
        # FEATS выводится полностью отдельной строкой
        if tok["feats"] != "_":
            print(f"        ↳ feats: {tok['feats']}")


def _print_token_full(tok: Dict[str, Any]) -> None:
    """
    Выводит все 10 CoNLL-U полей токена в подробном вертикальном формате.
    Используется для отладки и демонстрации структуры данных.
    """
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' " + "─" * 32)
    print(f"     [CoNLL-U поля — 10 стандартных полей]")
    print(f"     id:     {tok['id']}")
    print(f"     form:   {tok['form']}")
    print(f"     lemma:  {tok['lemma']}")
    print(f"     upos:   {tok['upos']}")
    print(f"     xpos:   {tok['xpos']}")
    print(f"     feats:  {tok['feats']}")
    print(f"     head:   {tok['head']}")
    print(f"     deprel: {tok['deprel']}")
    print(f"     deps:   {tok['deps']}")
    print(f"     misc:   {tok['misc']!r}")


def _print_misc_summary(results: List[List[Dict[str, Any]]], label: str) -> None:
    """
    Сводная таблица всех непустых MISC-значений в разборе.
    Помогает верифицировать корректность _parse_misc.
    """
    print(f"\n  Уникальные MISC-значения ({label}):")
    seen = {}
    for sent in results:
        for tok in sent:
            m = tok["misc"]
            key = repr(m)
            if m and m != "_" and m != {} and key not in seen:
                seen[key] = tok["form"]
    if seen:
        for val, form in seen.items():
            print(f"    {form:<16} → {val}")
    else:
        print("    (нет непустых MISC)")


# ─── local_entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    Интеграционный тест UDPipeService напрямую (без wrapper).

    Проверяет:
      1. parse_sentence_chunk  (native, output_format='dict')
      2. parse_sentence_chunk  (native, output_format='native')
      3. parse_sentence_chunk_razdel  (razdel, output_format='dict')
      4. parse_sentence_chunk_razdel  (razdel, output_format='native')
      5. Сравнение токенизации: native vs razdel
      6. parse / parse_batch (backward compat)
      7. Пустой ввод → []
      8. Подробный вывод первого токена (_print_token_full)
    """
    import json
    from razdel import sentenize, tokenize as razdel_tokenize

    service = UDPipeService()

    text_single = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься."
    )
    text_multi = "Нет!\nЭто невозможно,— сказал он.\n«Правда?» — спросила она."

    sep = "=" * 72

    # ─── 1. Native-путь, формат dict ──────────────────────────────────────────
    print(f"\n{sep}")
    print("РЕЖИМ: parse_sentence_chunk  →  native (misc как строка CoNLL-U)")
    print(sep)
    sents_single = [s.text for s in sentenize(text_single)]
    result_nd = service.parse_sentence_chunk.remote(
        sents_single, output_format="dict"
    )
    print(f"Предложений: {len(result_nd)}")
    for i, tokens in enumerate(result_nd, 1):
        # → показываем sents_single[i-1] (original)
        _print_sentence_table(i, tokens, sents_single[i - 1])
    _print_misc_summary(result_nd, "dict")
    if result_nd:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_nd[0][0], ensure_ascii=False, indent=2))

    # ─── 2. Native-путь, формат native ────────────────────────────────────────
    print(f"\n{sep}")
    print("РЕЖИМ: parse_sentence_chunk  →  native (misc как словарь Python)")
    print(sep)
    result_nn = service.parse_sentence_chunk.remote(
        sents_single, output_format="native"
    )
    print(f"Предложений: {len(result_nn)}")
    for i, tokens in enumerate(result_nn, 1):
        # → показываем sents_single[i-1] (original)
        _print_sentence_table(i, tokens, sents_single[i - 1])
    _print_misc_summary(result_nn, "native")
    if result_nn:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_nn[0][0], ensure_ascii=False, indent=2))

    # ─── 3. Razdel-путь, формат dict ──────────────────────────────────────────
    print(f"\n{sep}")
    print("РЕЖИМ: parse_sentence_chunk_razdel  →  dict (misc как строка CoNLL-U)")
    print(sep)
    sents_multi  = list(sentenize(text_multi))
    token_lists  = [
        [tok.text for tok in razdel_tokenize(s.text)]
        for s in sents_multi
    ]
    print(f"Передаём {len(token_lists)} предложений в horizontal-формате:")
    for i, tl in enumerate(token_lists, 1):
        print(f"  [{i}] {tl}")
    result_rd = service.parse_sentence_chunk_razdel.remote(
        token_lists, output_format="dict"
    )
    print(f"\nПредложений получено: {len(result_rd)}")
    for i, tokens in enumerate(result_rd, 1):
        # Не "# text", а явный лейбл для horizontal-ввода
        udpipe_header = "# udpipe_input = " + " ".join(t["form"] for t in tokens)
        _print_sentence_table(i, tokens, udpipe_header)
        # _print_sentence_table(i, tokens, " ".join(t["form"] for t in tokens))
    _print_misc_summary(result_rd, "dict")
    if result_rd:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_rd[0][0], ensure_ascii=False, indent=2))

    # ─── 4. Razdel-путь, формат native ────────────────────────────────────────
    print(f"\n{sep}")
    print("РЕЖИМ: parse_sentence_chunk_razdel  →  native (misc как словарь Python)")
    print(sep)
    result_rn = service.parse_sentence_chunk_razdel.remote(
        token_lists, output_format="native"
    )
    print(f"Предложений получено: {len(result_rn)}")
    for i, tokens in enumerate(result_rn, 1):
        # Не "# text", а явный лейбл для horizontal-ввода
        udpipe_header = "# udpipe_input = " + " ".join(t["form"] for t in tokens)
        _print_sentence_table(i, tokens, udpipe_header)
        # _print_sentence_table(i, tokens, " ".join(t["form"] for t in tokens))
    _print_misc_summary(result_rn, "native")
    if result_rn:
        print(f"\n  JSON первого токена:")
        print(json.dumps(result_rn[0][0], ensure_ascii=False, indent=2))

    # ─── 5. Подробный вывод одного токена (все 10 полей) ─────────────────────
    print(f"\n{sep}")
    print("ПОДРОБНЫЙ ВЫВОД: первый токен native-пути (output_format='native')")
    print(sep)
    if result_nn:
        _print_token_full(result_nn[0][0])

    # ─── 6. Сравнение форматов misc ───────────────────────────────────────────
    print(f"\n{sep}")
    print("СРАВНЕНИЕ ФОРМАТОВ misc  (native path, предложение 1)")
    print(sep)
    if result_nd and result_nn:
        toks_d = result_nd[0]
        toks_n = result_nn[0]
        print(f"  Ключи одинаковы: {[t['id'] for t in toks_d] == [t['id'] for t in toks_n]}")
        print()
        print(f"  {'FORM':<14}  {'dict misc':<30}  native misc")
        print(f"  {'─' * 14}  {'─' * 30}  {'─' * 30}")
        for td, tn in zip(toks_d, toks_n):
            if td["misc"] != "_" or tn["misc"]:
                print(
                    f"  {td['form']:<14}  "
                    f"{repr(td['misc']):<30}  "
                    f"{repr(tn['misc'])}"
                )

    # ─── 7. Сравнение токенизации: native vs razdel ───────────────────────────
    print(f"\n{sep}")
    print("СРАВНЕНИЕ ТОКЕНИЗАЦИИ: native vs razdel")
    print(sep)
    sents_cmp  = list(sentenize(text_multi))
    tl_cmp     = [[t.text for t in razdel_tokenize(s.text)] for s in sents_cmp]
    r_n = service.parse_sentence_chunk.remote(
        [s.text for s in sents_cmp], output_format="dict"
    )
    r_r = service.parse_sentence_chunk_razdel.remote(tl_cmp, output_format="dict")
    print(f"  {'№':<3}  {'native':<45}  {'razdel':<45}  match")
    print(f"  {'─' * 3}  {'─' * 45}  {'─' * 45}  {'─' * 5}")
    for i, (sn, sr) in enumerate(zip(r_n, r_r), 1):
        fn = [t["form"] for t in sn]
        fr = [t["form"] for t in sr]
        match = "✅" if fn == fr else "⚠️ "
        print(f"  {i:<3}  {str(fn):<45}  {str(fr):<45}  {match}")

    # ─── 8. parse / parse_batch (backward compat) ────────────────────────────
    print(f"\n{sep}")
    print("BACKWARD COMPAT: parse_batch (native + razdel)")
    print(sep)
    batch = [
        "Он думал о море.",
        "Кот лежал на диване.",
        text_multi,
    ]
    for tok_type in ("native", "razdel"):
        results_b = service.parse_batch.remote(
            batch, output_format="dict", tokenizer=tok_type
        )
        print(f"\n  tokenizer='{tok_type}':")
        for i, (txt, sents) in enumerate(zip(batch, results_b)):
            n_s = len(sents)
            n_t = sum(len(s) for s in sents)
            print(f"    [{i}] '{txt[:35]}...' → {n_s} предл., {n_t} токенов")

    # ─── 9. Пустой ввод ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("ПУСТОЙ ВВОД → ожидается []")
    print(sep)
    e1 = service.parse_sentence_chunk.remote([],        output_format="dict")
    e2 = service.parse_sentence_chunk.remote(["", " "], output_format="dict")
    e3 = service.parse_sentence_chunk_razdel.remote([],  output_format="dict")
    print(f"  parse_sentence_chunk([]):          {e1}  {'✅' if e1 == [] else '❌'}")
    print(f"  parse_sentence_chunk(['', ' ']):   {e2}  {'✅' if e2 == [] else '❌'}")
    print(f"  parse_sentence_chunk_razdel([]):   {e3}  {'✅' if e3 == [] else '❌'}")

    print(f"\n{'=' * 72}")
    print(f"{'✅ Все тесты завершены':^72}")
    print(f"{'=' * 72}")
