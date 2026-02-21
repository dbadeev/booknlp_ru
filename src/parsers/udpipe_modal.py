#!/usr/bin/env python3
"""
UDPipe Modal Service для booknlp_ru.

Публичный метод: parse(text, output_format)

  output_format="dict"    →  List[List[Dict]]
      Стандартные поля CoNLL-U:
      id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
      misc — raw строка CoNLL-U: "SpaceAfter=No|TokenRange=0:4"

  output_format="native"  →  List[List[Dict]]
      Те же поля CoNLL-U, но misc разобран в словарь:
      misc — dict: {"SpaceAfter": "No", "TokenRange": "0:4"}
      Удобен для downstream-задач (атрибуция цитат, кореференция).

Примечание: UDPipe нативно работает с CoNLL-U, поэтому оба формата
возвращают List[List[Dict]]. Различие — только в представлении поля misc.

Модель: russian-syntagrus-ud-2.5 (Universal Dependencies 2.5)
Токенизация: встроенная в UDPipe
"""

import logging
import re

import modal

# ─────────────────────────────────────────────────────────────
# DOCKER ОБРАЗ
# ─────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "build-essential", "swig", "g++")
    .pip_install("ufal.udpipe")
    .run_commands(
        "curl -L -o /root/russian-syntagrus.udpipe "
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/"
        "russian-syntagrus-ud-2.5-191206.udpipe"
    )
)

app = modal.App("booknlp-ru-udpipe")


# ─────────────────────────────────────────────────────────────
# СЕРВИС
# ─────────────────────────────────────────────────────────────

@app.cls(image=image, timeout=600)
class UDPipeService:

    @modal.enter()
    def setup(self):
        from ufal.udpipe import Model, Pipeline

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UDPipeService")
        self.logger.info("Loading UDPipe model...")

        self.model = Model.load("/root/russian-syntagrus.udpipe")
        if not self.model:
            raise RuntimeError("Cannot load UDPipe model!")

        self.pipeline = Pipeline(
            self.model,
            "tokenize",
            Pipeline.DEFAULT,
            Pipeline.DEFAULT,
            "conllu",
        )
        self.logger.info("UDPipe loaded!")

    # ──────────────────────────────────────────────────────────
    # Вспомогательный метод: парсинг строки MISC → dict
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_misc(misc_str: str) -> dict:
        """
        Разбирает строку MISC в словарь.

        Примеры:
            "SpaceAfter=No|TokenRange=0:4"  →  {"SpaceAfter": "No", "TokenRange": "0:4"}
            "SpaceAfter=No"                 →  {"SpaceAfter": "No"}
            "_"                             →  {}
            "Translit"                      →  {"Translit": True}  # флаг без значения
        """
        if not misc_str or misc_str == "_":
            return {}
        result = {}
        for item in misc_str.split("|"):
            if "=" in item:
                k, _, v = item.partition("=")
                result[k] = v
            else:
                result[item] = True  # флаг без значения
        return result

    # ──────────────────────────────────────────────────────────
    # Внутренний метод: CoNLL-U строка → List[List[Dict]]
    # ──────────────────────────────────────────────────────────

    def _conllu_to_dict(self, conllu_str: str, parse_misc: bool = False) -> list:
        """
        Парсит CoNLL-U строку в список предложений.

        Parameters
        ----------
        conllu_str : str
            Вывод pipeline.process().
        parse_misc : bool, default False
            False → misc остаётся raw строкой CoNLL-U (dict-формат)
            True  → misc разбирается в словарь             (native-формат)

        Мультитокены (1-2) и пустые узлы (1.1) пропускаются.

        Ключи токена:
            id, form, lemma, upos, xpos, feats,
            head, deprel, deps, misc
        """
        result       = []
        current_sent = []

        for line in conllu_str.split("\n"):
            line = line.strip()

            if not line or line.startswith("#"):
                if current_sent:
                    result.append(current_sent)
                    current_sent = []
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            raw_id = parts[0]
            if "-" in raw_id or "." in raw_id:
                continue

            misc_raw = parts[9]
            misc = self._parse_misc(misc_raw) if parse_misc else misc_raw

            token = {
                "id":     int(raw_id),
                "form":   parts[1],
                "lemma":  parts[2],
                "upos":   parts[3],
                "xpos":   parts[4],
                "feats":  parts[5],
                "head":   int(parts[6]) if parts[6].isdigit() else 0,
                "deprel": parts[7],
                "deps":   parts[8],
                "misc":   misc,
            }
            current_sent.append(token)

        if current_sent:
            result.append(current_sent)

        return result

    # ──────────────────────────────────────────────────────────
    # Внутренний метод: парсинг текста
    # ──────────────────────────────────────────────────────────

    def parse_text(self, text: str, output_format: str = "dict") -> list:
        """
        Parameters
        ----------
        text : str
            Входной текст.
        output_format : str, default "dict"
            "dict"   → List[List[Dict]], misc — raw CoNLL-U строка
            "native" → List[List[Dict]], misc — разобранный словарь

        Returns
        -------
        List[List[Dict]]
        """
        if not text or not text.strip():
            return []

        try:
            processed = self.pipeline.process(text)

            if not processed or not processed.strip():
                self.logger.error("UDPipe returned empty output.")
                return []

            parse_misc = (output_format == "native")
            return self._conllu_to_dict(processed, parse_misc=parse_misc)

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    # ──────────────────────────────────────────────────────────
    # Публичные Modal-методы
    # ──────────────────────────────────────────────────────────

    @modal.method()
    def parse(self, text: str, output_format: str = "dict") -> list:
        """Парсинг одного текста. output_format: 'dict' или 'native'."""
        return self.parse_text(text, output_format=output_format)

    @modal.method()
    def parse_batch(self, texts: list, output_format: str = "dict") -> list:
        """Батч-обработка списка текстов."""
        return [self.parse_text(t, output_format=output_format) for t in texts]


# ─────────────────────────────────────────────────────────────
# ЛОКАЛЬНЫЙ ТЕСТ  (modal run src/parsers/udpipe_modal.py)
# ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import json
    logging.basicConfig(level=logging.INFO)

    TEST_TEXT = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, "
        "как ты зло ухмыляешься."
    )
    SEP = "=" * 70
    service = UDPipeService()

    # ════════════════════════════════════════════
    # 1. Dict формат (misc — raw строка)
    # ════════════════════════════════════════════
    print(f"\n{SEP}\nРЕЖИМ: dict  →  misc как raw CoNLL-U строка\n{SEP}")
    result_dict = service.parse.remote(TEST_TEXT, output_format="dict")

    if not result_dict:
        print("⚠ Результат пустой.")
    else:
        print(f"Предложений: {len(result_dict)}\n")
        for s_idx, sent in enumerate(result_dict, 1):
            print(f"  Предложение {s_idx}:")
            print(f"  {'ID':<4} {'FORM':<14} {'LEMMA':<14} {'UPOS':<7} "
                  f"{'HEAD':<5} {'DEPREL':<12} MISC")
            print("  " + "-" * 90)
            for t in sent:
                print(f"  {t['id']:<4} {t['form']:<14} {t['lemma']:<14} "
                      f"{t['upos']:<7} {t['head']:<5} {t['deprel']:<12} "
                      f"{t['misc']}")

        print(f"\nКлючи dict-токена: {list(result_dict[0][0].keys())}")
        print(f"Тип misc:          {type(result_dict[0][0]['misc']).__name__}")
        print("\nJSON первого токена:")
        print(json.dumps(result_dict[0][0], ensure_ascii=False, indent=2))

    # ════════════════════════════════════════════
    # 2. Native формат (misc — dict)
    # ════════════════════════════════════════════
    print(f"\n{SEP}\nРЕЖИМ: native  →  misc как словарь\n{SEP}")
    result_native = service.parse.remote(TEST_TEXT, output_format="native")

    if not result_native:
        print("⚠ Результат пустой.")
    else:
        print(f"Предложений: {len(result_native)}\n")
        for s_idx, sent in enumerate(result_native, 1):
            print(f"  Предложение {s_idx}:")
            print(f"  {'ID':<4} {'FORM':<14} {'LEMMA':<14} {'UPOS':<7} "
                  f"{'HEAD':<5} {'DEPREL':<12} MISC (dict)")
            print("  " + "-" * 90)
            for t in sent:
                print(f"  {t['id']:<4} {t['form']:<14} {t['lemma']:<14} "
                      f"{t['upos']:<7} {t['head']:<5} {t['deprel']:<12} "
                      f"{t['misc']}")

        print(f"\nКлючи native-токена: {list(result_native[0][0].keys())}")
        print(f"Тип misc:            {type(result_native[0][0]['misc']).__name__}")
        print("\nJSON первого токена:")
        print(json.dumps(result_native[0][0], ensure_ascii=False, indent=2))

    # ════════════════════════════════════════════
    # 3. Сравнение форматов
    # ════════════════════════════════════════════
    if result_dict and result_native:
        print(f"\n{SEP}\nСРАВНЕНИЕ ФОРМАТОВ\n{SEP}")
        print(f"  Ключи одинаковы:  "
              f"{list(result_dict[0][0].keys()) == list(result_native[0][0].keys())}")
        print(f"\n  dict   misc: {repr(result_dict[0][0]['misc'])}")
        print(f"  native misc: {repr(result_native[0][0]['misc'])}")

        # Показываем токены с непустым misc
        print("\n  Токены с непустым misc:")
        for t_d, t_n in zip(result_dict[0], result_native[0]):
            if t_d["misc"] != "_":
                print(f"    [{t_d['form']}]")
                print(f"      dict:   {repr(t_d['misc'])}")
                print(f"      native: {repr(t_n['misc'])}")
