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
        "PYTHONPATH": f"{REMOTEROOT}:{REMOTESRC}:$PYTHONPATH",
        "ACCELERATE_DISABLE_MAPPING": "1",
        "ACCELERATE_USE_CPU": "0",
    })
    # copy=True — файлы копируются в образ, шаги после .env() разрешены
    .add_local_dir(LOCALCOBALDDIR, remote_path=f"{REMOTESRC}/cobald_parser", copy=True)
)

app    = modal.App("booknlp-ru-cobald")


@app.cls(image=image, gpu="T4", timeout=600)
class CobaldService:
    """
    Сервис синтаксического разбора на основе CoBaLD-парсера.

    Принимает сырые тексты (str), токенизация выполняется внутри pipeline.
    Два формата вывода: 'dict' (CoNLL-U + CoBaLD поля) и 'native' (полный).
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
            sentenizer=lambda text: [sent.text for sent in sentenize(text)],
        )
        self.vocab = config.vocabulary
        self.logger.info(f"CoBaLD pipeline loaded on {self.device}!")

    # ─────────────────────────────────────────────────────────
    # FIX P7: построение id_mapping вынесено в общий приватный метод.
    # Оригинал дублировал идентичный блок в _format_native_output
    # и в dict-ветке parse_batch (~15 строк кода дважды).
    # ─────────────────────────────────────────────────────────
    def _build_id_mapping(self, sentence_data: dict) -> Dict[str, int]:
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
        Внутренняя реализация: разбирает список текстов.

        Returns
        -------
        List[List[sentence]]
            Для каждого входного текста — список предложений.
        """
        all_results = []
        for text in texts:
            # FIX P3: pipeline получает сырой текст и сам токенизирует razdel-ом.
            # Оригинал принимал List[str] (токены) и делал " ".join(tokens),
            # что создавало тройную токенизацию:
            #   1. wrapper: razdel_tokenize(text) → tokens
            #   2. modal:   " ".join(tokens) → text (с потерей границ!)
            #   3. pipeline: razdel внутри preprocess() → новые токены
            # Пример потери: ["Кружка-термос"] → join → pipeline razdel
            #   → ["Кружка", "-", "термос"] (другой результат!)
            decoded_sentences = self.pipeline(text, output_format="list")

            # FIX P4: обрабатываем ВСЕ предложения текста, не только [0].
            # Оригинал: sentence_data = decoded_sentences[0]
            # При нескольких предложениях в тексте остальные молча терялись.
            text_results = []
            for sentence_data in decoded_sentences:
                if output_format == "native":
                    text_results.append(self._format_native_output(sentence_data))
                else:
                    text_results.append(self._build_dict(sentence_data))
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
        for i, (word, word_id, dep_ud) in enumerate(zip(
                sentence_data["words"],
                sentence_data["ids"],
                sentence_data["deps_ud"],  # (head_id, self_id, deprel)
        )):
            str_id = str(word_id)
            if word == "[CLS]" or "#NULL" in str_id:
                continue
            head_orig, deprel = self._extract_dep(dep_ud)
            new_id = id_mapping.get(str_id, 0)
            new_head = id_mapping.get(head_orig, 0)
            token: Dict[str, Any] = {
                "id": int(new_id),
                "form": word,
                "head": int(new_head),
                "deprel": deprel,
                "misc": sentence_data["miscs"][i],
                "deepslot": sentence_data["deepslots"][i],
                "semclass": sentence_data["semclasses"][i],
            }
            result.append(token)
        return result

    def _format_native_output(self, sentence_data: dict) -> List[Dict[str, Any]]:
        """Токены в native-формате — все поля включая lemma, upos, feats, eud."""
        id_mapping = self._build_id_mapping(sentence_data)
        result = []
        for i, (word, word_id, dep_ud) in enumerate(zip(
                sentence_data["words"],
                sentence_data["ids"],
                sentence_data["deps_ud"],  # (head_id, self_id, deprel)
        )):
            str_id = str(word_id)
            if word == "[CLS]" or "#NULL" in str_id:
                continue
            head_orig, deprel = self._extract_dep(dep_ud)
            new_id = id_mapping.get(str_id, 0)
            new_head = id_mapping.get(head_orig, 0)
            token: Dict[str, Any] = {
                "id": int(new_id),
                "form": word,
                "lemma": sentence_data["lemmas"][i],
                "upos": sentence_data["upos"][i],
                "xpos": sentence_data["xpos"][i],
                "feats": sentence_data["feats"][i],
                "head": int(new_head),
                "deprel": deprel,
                "deps_eud": sentence_data["deps_eud"][i],
                "misc": sentence_data["miscs"][i],
                "deepslot": sentence_data["deepslots"][i],
                "semclass": sentence_data["semclasses"][i],
                "is_null": False,
            }
            result.append(token)
        return result


# ─────────────────────── LOCAL ENTRYPOINT ────────────────────
@app.local_entrypoint()
def main():
    """Тестирование CoBaLD сервиса (4 комбинации)."""
    test_single = "Мама мыла раму. Папа читал газету."
    test_batch  = ["Он думал о море.", "Кот лежал на диване."]

    SEP = "=" * 70
    print(f"{SEP}\nТЕСТИРОВАНИЕ COBALD SERVICE\n{SEP}")

    service = CobaldService()

    # 1. parse → dict
    print("\n1. parse (dict):")
    result = service.parse.remote(test_single, output_format="dict")
    print(f"   Предложений: {len(result)}")
    for s_idx, sent in enumerate(result, 1):
        forms = [t["form"] for t in sent]
        print(f"   [{s_idx}] {forms}")
        for tok in sent:
            print(f"       id={tok['id']} head={tok['head']} "
                  f"deprel={tok['deprel']:<12} "
                  f"deepslot={tok['deepslot']} semclass={tok['semclass']}")

    # 2. parse → native
    print("\n2. parse (native):")
    result = service.parse.remote(test_single, output_format="native")
    print(f"   Предложений: {len(result)}")
    for s_idx, sent in enumerate(result, 1):
        print(f"   [{s_idx}] ключи токена: {list(sent[0].keys()) if sent else '—'}")

    # 3. parse_batch → dict
    print("\n3. parse_batch (dict):")
    result = service.parse_batch.remote(test_batch, output_format="dict")
    for t_idx, text_sents in enumerate(result):
        total = sum(len(s) for s in text_sents)
        print(f"   [{t_idx}] '{test_batch[t_idx]}' "
              f"→ {len(text_sents)} предл., {total} токенов")

    print(f"\n{'=' * 70}\n✅ Тестирование завершено\n{'=' * 70}")
