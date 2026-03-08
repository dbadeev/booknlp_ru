import modal
import logging
from typing import List, Dict, Any
import sys

# ─────────────────────────── ПУТИ ────────────────────────────
LOCALCOBALDDIR = "src/cobald_parser"
REMOTEROOT = "/root/booknlp_ru"
REMOTESRC = f"{REMOTEROOT}/src"

# ─────────────────────────── ОБРАЗ ───────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub",
        "numpy",
        "razdel",
        "torch==2.10.0",
        "transformers==4.35.2",
    )
    .env({
        "PYTHONPATH": f"{REMOTEROOT}:{REMOTESRC}",
        "ACCELERATE_DISABLE_MAPPING": "1",
        "ACCELERATE_USE_CPU": "0",
    })
    # copy=True — файлы копируются в образ, шаги после .env() разрешены
    .add_local_dir(LOCALCOBALDDIR, remote_path=f"{REMOTESRC}/cobald_parser", copy=True)
)

app    = modal.App("booknlp-ru-cobald")

# ──────────────────────────────────────────────────────────────────────────
# CoNLL-U утилиты (используются внутри сервиса для output_format='conllu').
# Зеркало на стороне клиента: cobald_wrapper.py :: _to_conllu_str
# ──────────────────────────────────────────────────────────────────────────

def _dep_tuple_to_str(dep: Any) -> str:
    """
    Конвертирует deps_eud в строку CoNLL-U формата head:deprel.
      - tuple 3: ('head', 'self_id', 'deprel') → 'head:deprel'
      - tuple 2: ('head', 'deprel')             → 'head:deprel'
      - str                                     → как есть
      - None / '_'                              → '_'
    """
    if dep is None:
        return "_"
    if isinstance(dep, str):
        return dep.strip() or "_"
    if isinstance(dep, (list, tuple)):
        if len(dep) == 3:
            return f"{dep[0]}:{dep[2]}"
        if len(dep) == 2:
            return f"{dep[0]}:{dep[1]}"
    return "_"


def _to_conllu_str(sentences: List[List[Dict[str, Any]]]) -> str:
    """
    Конвертирует список предложений в native-формате в строку CoNLL-U.

    Поля CoNLL-U (10 колонок, разделитель TAB):
        ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS_EUD  MISC

    CoBaLD-поля deepslot и semclass добавляются в MISC:
        SpaceAfter=No|Deepslot=Agent|Semclass=BEING

    Требует native-формата (lemma, upos, xpos, feats, deps_eud).

    Зеркало на стороне клиента: cobald_wrapper.py :: _to_conllu_str.
    При изменении логики — синхронизировать оба файла.
    """
    lines = []
    for sent_idx, snt in enumerate(sentences, 1):
        if not snt:
            continue
        lines.append(f"# sent_id = {sent_idx}")
        lines.append(f"# text = {' '.join(t.get('form', '') for t in snt)}")
        for tok in snt:
            misc_parts = []
            raw_misc = (tok.get("misc") or "").strip()
            if raw_misc and raw_misc != "_":
                misc_parts.append(raw_misc)
            deepslot = (tok.get("deepslot") or "").strip()
            semclass  = (tok.get("semclass")  or "").strip()
            if deepslot and deepslot != "_":
                misc_parts.append(f"Deepslot={deepslot}")
            if semclass and semclass != "_":
                misc_parts.append(f"Semclass={semclass}")
            misc_str = "|".join(misc_parts) if misc_parts else "_"
            deps_eud = _dep_tuple_to_str(tok.get("deps_eud"))

            tok_id = tok.get("id", "_")
            line = "\t".join([
                str(tok_id),
                tok.get("form",   "_"),
                tok.get("lemma",  "_") or "_",
                tok.get("upos",   "_") or "_",
                tok.get("xpos",   "_") or "_",
                tok.get("feats",  "_") or "_",
                str(tok.get("head", 0)),
                tok.get("deprel", "_") or "_",
                deps_eud,
                misc_str,
            ])
            lines.append(line)
        lines.append("")
    return "\n".join(lines)

@app.cls(image=image, gpu="T4", timeout=600)
class CobaldService:
    """
    Сервис синтаксического разбора на основе CoBaLD-парсера.

    Принимает чанки предложений (List[str]), уже нарезанных на стороне
    wrapper (cobald_wrapper.py) с помощью razdel.sentenize.
    Сентенизация и chunking выполняются исключительно в wrapper —
    сервис получает готовые предложения и не занимается разбивкой текста.

    Форматы вывода
    --------------
    'dict'   : CoNLL-U поля (id, form, head, deprel, misc) + CoBaLD-поля
               (deepslot, semclass).
    'native' : Полный набор полей: lemma, upos, xpos, feats, deps_eud,
               is_null — плюс все поля 'dict'.
    'conllu' : Строка в формате CoNLL-U (10 TAB-колонок, sent_id, text).
               Формируется из native-результата внутри сервиса.
               Зеркало: cobald_wrapper.py :: _to_conllu_str.
    """


    @modal.enter()
    def setup(self):
        import torch
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CobaldService")

        if REMOTEROOT not in sys.path:
            sys.path.append(REMOTEROOT)
        if REMOTESRC not in sys.path:
            sys.path.append(REMOTESRC)

        # Оригинальные импорты из cobald_parser
        from src.cobald_parser.modeling_parser import CobaldParser
        from src.cobald_parser.configuration import CobaldParserConfig
        from src.cobald_parser.pipeline import ConlluTokenClassificationPipeline
        from razdel import tokenize as razdel_tokenize, sentenize

        self.sentenize = sentenize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Модель грузится с HuggingFace Hub, не из volume
        model_name = "CoBaLD/xlm-roberta-base-cobald-parser-ru"
        config = CobaldParserConfig.from_pretrained(model_name)
        model = CobaldParser.from_pretrained(model_name, config=config)
        model.to(self.device)
        model.eval()

        self.pipeline = ConlluTokenClassificationPipeline(
            model=model,
            tokenizer=lambda text: [tok.text for tok in razdel_tokenize(text)],
            sentenizer=lambda text: [sent.text for sent in self.sentenize(text)],
        )
        self.vocab = config.vocabulary
        self.logger.info(f"CoBaLD pipeline loaded on {self.device}!")

    # ─────────────────────────────────────────────────────────
    # FIX P7: построение id_mapping вынесено в общий приватный метод.
    # Оригинал дублировал идентичный блок в _format_native_output
    # и в dict-ветке parse_batch (~15 строк кода дважды).
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _build_id_mapping(sentence_data: dict) -> Dict[str, int]:
        """
        Строит маппинг {внутренний_id_модели → порядковый_id_CoNLL-U}.

        Returns
        -------
        Dict[str, int]
            Строковый ключ (исходный id) → int-значение (CoNLL-U 1-based id).
            [CLS] → 0, #NULL-узлы → не включаются.
        """
        id_mapping: Dict[str, int] = {}
        conllu_counter = 0
        for word, word_id in zip(sentence_data["words"], sentence_data["ids"]):
            str_id = str(word_id)
            if word == "[CLS]":
                # FIX P6: в оригинале dict-ветка использовала хардкод
                #   id_mapping['1'] = '0'
                # что предполагало: [CLS] всегда имеет word_id == "1".
                # В _format_native_output тот же код был написан правильно:
                #   id_mapping[str(word_id)] = 0
                # Теперь оба места унифицированы — используем реальный word_id.
                id_mapping[str_id] = 0
            elif "#NULL" not in str_id:
                conllu_counter += 1
                id_mapping[str_id] = conllu_counter
        return id_mapping

    # ─────────────────────────────────────────────────────────
    # FIX P2: логика разбора вынесена в _parse_batch_impl.
    # Оригинал: parse() вызывал self.parse_batch.remote([tokens]) —
    # полноценный network round-trip через Modal (сериализация,
    # потенциальный спавн нового контейнера, лишняя задержка).
    # Теперь оба публичных метода вызывают _parse_batch_impl напрямую.
    # ─────────────────────────────────────────────────────────
    def _parse_batch_impl(
            self,
            texts: List[str],
            output_format: str = "dict",
    ) -> List[List[Any]]:
        """
        Внутренняя реализация пакетного разбора списка текстов.

        Зеркало логики wrapper-а: сентенизация → по одному предложению в pipeline.
        Сентенизация выполняется здесь только для методов parse() и parse_batch(),
        вызываемых напрямую (минуя wrapper). При вызове через wrapper сентенизация
        уже выполнена на его стороне, и используется parse_sentence_chunk().

        Parameters
        ----------
        texts : List[str]
            Список сырых текстов.
        output_format : str
            'dict' | 'native' | 'conllu'

        Returns
        -------
        List[List[sentence]]
            Для каждого входного текста — список предложений.
            При output_format='conllu' каждый текст возвращается как
            List с одним элементом — CoNLL-U строкой всего текста.
        """
        if output_format not in ("dict", "native", "conllu"):
            raise ValueError(
                f"Неизвестный output_format={output_format!r}. "
                "Допустимые значения: 'dict', 'native', 'conllu'."
            )

        all_results = []

        for text in texts:
            if not text or not text.strip():
                all_results.append([])
                continue

            # Зеркало wrapper: sentenize → по одному предложению в pipeline.
            # Подача полного текста в pipeline приводит к OOM на больших текстах
            # и к ошибкам zip() при наличии #NULL-узлов в multi-sentence входе.
            sentences = [s.text for s in self.sentenize(text)]
            text_results = []

            for sent_text in sentences:
                if not sent_text or not sent_text.strip():
                    continue
                decoded = self.pipeline(sent_text, output_format="list")
                if not decoded:
                    continue
                sd = decoded[0]

                if output_format in ("native", "conllu"):
                    # conllu строится поверх native — накапливаем как native
                    text_results.append(self._format_native_output(sd))
                else:
                    text_results.append(self._build_dict(sd))

            if output_format == "conllu":
                # Конвертируем весь текст одним вызовом → одна CoNLL-U строка
                all_results.append([_to_conllu_str(text_results)])
            else:
                all_results.append(text_results)

        return all_results

    @modal.method()
    def parse_batch(
        self,
        # FIX P3: был List[List[str]] (списки токенов), теперь List[str] (тексты)
        texts: List[str],
        output_format: str = "dict",
    ) -> List[List[Any]]:
        """
        Пакетный разбор списка текстов.

        Parameters
        ----------
        texts : List[str]
            Сырые тексты. Токенизация выполняется внутри pipeline.
        output_format : str
            'dict' | 'native'

        Returns
        -------
        List[List[sentence]]
            Для каждого текста — список предложений.
        """
        return self._parse_batch_impl(texts, output_format)

    @modal.method()
    def parse_sentence_chunk(
            self,
            sentences: List[str],
            output_format: str = "dict",
    ) -> List[Any]:
        """
        Разбирает чанк предложений.

        Предложения передаются уже готовыми — сентенизация текста
        выполняется на стороне wrapper до вызова этого метода.
        Каждое предложение подаётся в pipeline как отдельный текст,
        pipeline возвращает ровно один результат (decoded[0]).

        Parameters
        ----------
        sentences : List[str]
            Чанк предложений от wrapper-а (не более SENTENCE_CHUNK_SIZE).
        output_format : str
            'dict' | 'native' | 'conllu'
        ...
        """
        if output_format not in ("dict", "native", "conllu"):
            raise ValueError(
                f"Неизвестный output_format={output_format!r}. "
                "Допустимые значения: 'dict', 'native', 'conllu'."
            )
        result = []
        for sent_text in sentences:
            if not sent_text or not sent_text.strip():
                continue
            decoded = self.pipeline(sent_text, output_format="list")
            if not decoded:
                continue
            # pipeline получает одно предложение → берём только первый элемент
            sd = decoded[0]
            if output_format == "native":
                result.append(self._format_native_output(sd))
            elif output_format == "conllu":
                # conllu строится поверх native
                native_sent = self._format_native_output(sd)
                result.append(native_sent)  # накапливаем как native...
            else:
                result.append(self._build_dict(sd))

        if output_format == "conllu":
            # ...и конвертируем весь чанк одним вызовом → одна строка на чанк
            return [_to_conllu_str(result)]
        return result

    @modal.method()
    def parse(
        self,
        # FIX P3: был List[str] (токены), теперь str (сырой текст)
        text: str,
        output_format: str = "dict",
    ) -> List[Any]:
        """
        Разбор одного текста.

        Returns
        -------
        List[sentence]
            Список предложений в тексте.
        """
        # FIX P2: прямой вызов _parse_batch_impl без .remote() round-trip
        result = self._parse_batch_impl([text], output_format)
        return result[0] if result else []

    # ─────────────────────────────────────────────────────────
    # Форматирование результатов
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _extract_dep(dep_ud_item) -> tuple:
        """
        Извлекает (head_id_str, deprel) из одного элемента deps_ud.

        Реальный формат pipeline: кортеж из 3 элементов (head_id, self_id, deprel)
        Пример: ('3', '1', 'obl')  →  head='3', deprel='obl'
        """
        if isinstance(dep_ud_item, (list, tuple)):
            if len(dep_ud_item) == 3:
                # ('head_id', 'self_id', 'deprel')  ← реальный формат CoBaLD
                return str(dep_ud_item[0]), str(dep_ud_item[2])
            if len(dep_ud_item) == 2:
                # ('head_id', 'deprel')  ← fallback
                return str(dep_ud_item[0]), str(dep_ud_item[1])
            if len(dep_ud_item) == 0:
                logging.warning(f"_extract_dep: пустой deps_ud, подставляем ('0','_')")
                return "0", "_"
        if isinstance(dep_ud_item, dict):
            return str(dep_ud_item.get("head", "0")), str(dep_ud_item.get("deprel", "_"))
        if isinstance(dep_ud_item, str) and ":" in dep_ud_item:
            head_str, deprel = dep_ud_item.split(":", 1)
            return head_str.strip(), deprel.strip()
        return "0", "_"

    def _build_dict(self, sentence_data: dict) -> List[Dict[str, Any]]:
        """Токены в dict-формате (CoNLL-U + CoBaLD-специфичные поля)."""
        id_mapping = self._build_id_mapping(sentence_data)
        result = []

        # Defensive: pipeline иногда возвращает deps_ud короче остальных массивов
        n_ref = len(sentence_data["words"])
        deps_ud = list(sentence_data["deps_ud"])
        if len(deps_ud) < n_ref:
            self.logger.warning(
                f"deps_ud короче words ({len(deps_ud)} vs {n_ref}), дополняем defaults"
            )
            while len(deps_ud) < n_ref:
                # ('0', self_id, '_') — безопасный fallback в формате CoBaLD
                deps_ud.append(("0", str(len(deps_ud) + 1), "_"))

        lengths = {k: len(sentence_data[k]) for k in
                   ("words", "ids", "miscs", "deepslots", "semclasses")}
        lengths["deps_ud"] = len(deps_ud)  # проверяем уже padded
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Длины массивов расходятся: {lengths}")

        for word, word_id, dep_ud, misc, deepslot, semclass in zip(
                sentence_data["words"],
                sentence_data["ids"],
                deps_ud,  # ← padded
                sentence_data["miscs"],
                sentence_data["deepslots"],
                sentence_data["semclasses"],
        ):
            str_id = str(word_id)
            if word == "[CLS]" or "#NULL" in str_id:
                continue
            head_orig, deprel = self._extract_dep(dep_ud)
            token: Dict[str, Any] = {
                "id": int(id_mapping.get(str_id, 0)),
                "form": word,
                "head": int(id_mapping.get(head_orig, 0)),
                "deprel": deprel,
                "misc": misc,
                "deepslot": deepslot,
                "semclass": semclass,
            }
            result.append(token)
        return result

    def _format_native_output(self, sentence_data: dict) -> List[Dict[str, Any]]:
        """Токены в native-формате — все поля включая lemma, upos, feats, eud."""
        id_mapping = self._build_id_mapping(sentence_data)
        result = []

        n_ref = len(sentence_data["words"])

        deps_ud = list(sentence_data["deps_ud"])
        if len(deps_ud) < n_ref:
            self.logger.warning(
                f"deps_ud короче words ({len(deps_ud)} vs {n_ref}), дополняем defaults"
            )
        while len(deps_ud) < n_ref:
            deps_ud.append(("0", str(len(deps_ud) + 1), "_"))

        deps_eud = list(sentence_data["deps_eud"])
        while len(deps_eud) < n_ref:
            deps_eud.append(("0", str(len(deps_eud) + 1), "_"))

        lengths = {k: len(sentence_data[k]) for k in (
            "words", "ids", "lemmas", "upos", "xpos",
            "feats", "miscs", "deepslots", "semclasses",
        )}
        lengths["deps_ud"] = len(deps_ud)
        lengths["deps_eud"] = len(deps_eud)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Длины массивов расходятся: {lengths}")

        for word, word_id, dep_ud, lemma, upos, xpos, feats, deps_eud_item, misc, deepslot, semclass in zip(
                sentence_data["words"], sentence_data["ids"],
                deps_ud,  # ← padded
                sentence_data["lemmas"], sentence_data["upos"], sentence_data["xpos"],
                sentence_data["feats"],
                deps_eud,  # ← padded
                sentence_data["miscs"], sentence_data["deepslots"], sentence_data["semclasses"],
        ):
            str_id = str(word_id)
            if word == "[CLS]" or "#NULL" in str_id:
                continue
            head_orig, deprel = self._extract_dep(dep_ud)
            token: Dict[str, Any] = {
                "id": int(id_mapping.get(str_id, 0)),
                "form": word,
                "lemma": lemma,
                "upos": upos,
                "xpos": xpos,
                "feats": feats,
                "head": int(id_mapping.get(head_orig, 0)),
                "deprel": deprel,
                "deps_eud": deps_eud_item,
                "misc": misc,
                "deepslot": deepslot,
                "semclass": semclass,
                "is_null": False,
            }
            result.append(token)
        return result


# ─────────────────────── LOCAL ENTRYPOINT ────────────────────
# @app.local_entrypoint()
# def main():
#     """Тестирование CoBaLD сервиса (4 комбинации)."""
#     test_single = "Мама мыла раму. Папа читал газету."
#     test_batch  = ["Он думал о море.", "Кот лежал на диване."]
#
#     sep = "=" * 70
#     print(f"{sep}\nТЕСТИРОВАНИЕ COBALD SERVICE\n{sep}")
#
#     service = CobaldService()
#
#     # 1. parse → dict
#     print("\n1. parse (dict):")
#     result = service.parse.remote(test_single, output_format="dict")
#     print(f"   Предложений: {len(result)}")
#     for s_idx, sent in enumerate(result, 1):
#         forms = [t["form"] for t in sent]
#         print(f"   [{s_idx}] {forms}")
#         for tok in sent:
#             print(f"       id={tok['id']} head={tok['head']} "
#                   f"deprel={tok['deprel']:<12} "
#                   f"deepslot={tok['deepslot']} semclass={tok['semclass']}")
#
#     # 2. parse → native
#     print("\n2. parse (native):")
#     result = service.parse.remote(test_single, output_format="native")
#     print(f"   Предложений: {len(result)}")
#     for s_idx, sent in enumerate(result, 1):
#         print(f"   [{s_idx}] ключи токена: {list(sent[0].keys()) if sent else '—'}")
#
#     # 3. parse_batch → dict
#     print("\n3. parse_batch (dict):")
#     result = service.parse_batch.remote(test_batch, output_format="dict")
#     for t_idx, text_sents in enumerate(result):
#         total = sum(len(s) for s in text_sents)
#         print(f"   [{t_idx}] '{test_batch[t_idx]}' "
#               f"→ {len(text_sents)} предл., {total} токенов")
#
#     print(f"\n{'=' * 70}\n✅ Тестирование завершено\n{'=' * 70}")

@app.local_entrypoint()
def main():
    """
    Тестирует CobaldService напрямую — без wrapper.
    Запуск: modal run src/parsers/cobald_modal.py

    Тест-секции:
    [1] parse_sentence_chunk — dict
    [2] parse_sentence_chunk — native
    [3] Пустой чанк → []
    [4] Неверный output_format → ValueError
    [5] parse (backward compat, dict)
    """
    from razdel import sentenize

    sep = "=" * 72
    passed = 0
    failed = 0

    def ok(name: str):
        nonlocal passed
        passed += 1
        print(f"  ✅ {name}")

    def fail(name: str, err):
        nonlocal failed
        failed += 1
        print(f"  ❌ {name}: {err}")

    text_sample = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    multi_sample = "Мама мыла раму. Папа читал газету. Кот лежал на диване."

    service = CobaldService()

    # ── [1] parse_sentence_chunk — dict ─────────────────────────────────────
    print(f"\n{sep}")
    print("[1] parse_sentence_chunk (dict)")
    print(sep)
    try:
        sentences = [s.text for s in sentenize(multi_sample)]
        result = service.parse_sentence_chunk.remote(sentences, output_format="dict")
        assert isinstance(result, list), "результат не list"
        assert len(result) == len(sentences), \
            f"ожидалось {len(sentences)} предл., получено {len(result)}"
        for sent in result:
            assert isinstance(sent, list) and len(sent) > 0, "предложение пустое"
            for tok in sent:
                for key in ("id", "form", "head", "deprel", "misc", "deepslot", "semclass"):
                    assert key in tok, f"ключ {key!r} отсутствует"
                assert isinstance(tok["id"], int), "id не int"
                assert isinstance(tok["head"], int), "head не int"
        # Вывод для визуальной проверки
        for i, (sent, sent_text) in enumerate(zip(result, sentences), 1):
            forms = [t["form"] for t in sent]
            print(f"  [{i}] {sent_text!r} → {len(sent)} токенов: {forms}")
        ok("[1] parse_sentence_chunk / dict — структура корректна")
    except Exception as e:
        fail("[1] parse_sentence_chunk / dict", e)

    # ── [2] parse_sentence_chunk — native ───────────────────────────────────
    print(f"\n{sep}")
    print("[2] parse_sentence_chunk (native)")
    print(sep)
    try:
        sentences = [s.text for s in sentenize(text_sample)]
        result = service.parse_sentence_chunk.remote(sentences, output_format="native")
        assert isinstance(result, list)
        assert len(result) == len(sentences)
        for sent in result:
            for tok in sent:
                for key in ("id", "form", "lemma", "upos", "xpos", "feats",
                            "head", "deprel", "deps_eud", "misc",
                            "deepslot", "semclass", "is_null"):
                    assert key in tok, f"ключ {key!r} отсутствует в native"
                assert tok["is_null"] is False
        sent0 = result[0]
        print(f"  Предложение 1 ({len(sent0)} токенов):")
        print(f"  {'ID':<5} {'FORM':<16} {'LEMMA':<16} {'UPOS':<10} "
              f"{'HEAD':<6} {'DEPREL':<14} {'DEEPSLOT':<22} {'SEMCLASS':<30} MISC")
        print(f"  {'-' * 120}")
        for tok in sent0:
            print(
                f"  {tok['id']:<5} {tok['form']:<16} {tok['lemma']:<16} "
                f"{tok['upos']:<10} {tok['head']:<6} {tok['deprel']:<14} "
                f"{tok['deepslot']:<22} {tok['semclass']:<30} "
                f"{tok.get('misc') or '_'}"
            )
        print(f"  Ключи токена: {list(result[0][0].keys())}")
        ok("[2] parse_sentence_chunk / native — структура корректна")
    except Exception as e:
        fail("[2] parse_sentence_chunk / native", e)

    # ── [2b] parse_sentence_chunk → conllu ───────────────────────────────────
    print(f"\n{sep}")
    print("[2b] parse_sentence_chunk (conllu)")
    print(sep)
    try:
        sentences = [s.text for s in sentenize(text_sample)]
        result_conllu = service.parse_sentence_chunk.remote(
            sentences, output_format="conllu"
        )
        # conllu-режим возвращает List с одной строкой на чанк
        assert isinstance(result_conllu, list) and len(result_conllu) == 1
        conllu_str = result_conllu[0]
        assert isinstance(conllu_str, str)
        assert "# sent_id = 1" in conllu_str, "отсутствует sent_id"
        assert "# text = " in conllu_str, "отсутствует text"
        assert "\t" in conllu_str, "отсутствует TAB-разделитель"
        print("  # Колонки: ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS_EUD  MISC")
        print(conllu_str)
        ok("[2b] parse_sentence_chunk / conllu — sent_id, text, TAB присутствуют")
    except Exception as e:
        fail("[2b] parse_sentence_chunk / conllu", e)

    # ── [3] Пустой чанк → [] ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[3] Пустой чанк → []")
    print(sep)
    try:
        result = service.parse_sentence_chunk.remote([], output_format="dict")
        assert result == [], f"ожидался [], получено {result!r}"
        ok("[3] Пустой чанк → []")
    except Exception as e:
        fail("[3] Пустой чанк", e)

    # ── [4] Неверный output_format → ValueError ──────────────────────────────
    print(f"\n{sep}")
    print("[4] Неверный output_format → ValueError")
    print(sep)
    try:
        try:
            service.parse_sentence_chunk.remote(["Текст."], output_format="some_invalid_format")
            fail("[4] ValueError не выброшен", "исключение не возникло")
        except ValueError as exc:
            assert "output_format" in str(exc).lower() or "some_invalid_format" in str(exc).lower() \
                   or "unknown" in str(exc).lower(), f"Неожиданное сообщение: {exc}"
            print(f"  Поймано: {exc!r}")
            ok("[4] Неверный output_format → ValueError")
    except Exception as e:
        fail("[4]  ", f"{type(e).__name__}: {e}")

    # ── [5] parse (backward compat) ──────────────────────────────────────────
    print(f"\n{sep}")
    print("[5] parse (backward compat, dict)")
    print(sep)
    try:
        result = service.parse.remote(multi_sample, output_format="dict")
        assert isinstance(result, list) and len(result) == 3, \
            f"ожидалось 3 предл., получено {len(result)}"
        print(f"  Предложений: {len(result)}")
        ok("[5] parse / backward compat")
    except Exception as e:
        fail("[5] parse / backward compat", e)

    # ── Итог ─────────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{sep}")
    print(f"ИТОГ: {passed}/{total} тестов прошло" + (" ✅" if failed == 0 else f" ❌ {failed} упало"))
    print(sep)
