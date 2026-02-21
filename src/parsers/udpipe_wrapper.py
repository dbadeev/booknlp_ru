#!/usr/bin/env python3
"""
UDPipe Wrapper — клиент для UDPipeService на Modal.

Запуск:
    python src/parsers/udpipe_wrapper.py
"""

import json
import logging
from typing import Any, Dict, List

import modal

logger = logging.getLogger(__name__)

APP_NAME = "booknlp-ru-udpipe"


class UDPipeParser:
    """
    Клиентская обёртка над задеплоенным Modal-сервисом UDPipeService.

    Пример
    ------
        parser = UDPipeParser()
        sents = parser.parse_text("Мама мыла раму.", output_format="native")
        for sent in sents:
            for tok in sent:
                print(tok["id"], tok["form"], tok["misc"])
    """

    def __init__(self):
        self._service = modal.Cls.from_name(APP_NAME, "UDPipeService")()
        logger.info(f"UDPipeParser подключён к Modal-приложению '{APP_NAME}'.")

    def parse_text(
        self,
        text: str,
        output_format: str = "dict",
    ) -> List[List[Dict[str, Any]]]:
        """
        Parameters
        ----------
        text : str
            Входной текст.
        output_format : str, default "dict"
            "dict"   → List[List[Dict]], misc — raw CoNLL-U строка
            "native" → List[List[Dict]], misc — разобранный словарь
        """
        return self._service.parse.remote(text, output_format=output_format)


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции вывода
# ─────────────────────────────────────────────────────────────

def _fmt_misc(misc: Any, is_dict: bool) -> str:
    """Форматирует поле MISC для отображения."""
    if is_dict:
        if not misc:
            return "—"
        return " | ".join(
            f"{k}={v}" if v is not True else k
            for k, v in misc.items()
        )
    return misc if misc not in ("_", "", None) else "—"


def _print_table(sent: list, misc_is_dict: bool) -> None:
    """Печатает токены предложения в виде выровненной таблицы."""
    col = dict(id=4, form=16, lemma=16, upos=7, feats=30, head=5, deprel=12)
    hdr = (f"  {'ID':<{col['id']}} {'FORM':<{col['form']}} "
           f"{'LEMMA':<{col['lemma']}} {'UPOS':<{col['upos']}} "
           f"{'FEATS':<{col['feats']}} {'HEAD':<{col['head']}} "
           f"{'DEPREL':<{col['deprel']}} MISC")
    print(hdr)
    print("  " + "-" * (len(hdr) + 10))
    for t in sent:
        feats = t["feats"]
        feats_s = (feats[:28] + "..") if len(feats) > 30 else feats
        misc_s = _fmt_misc(t["misc"], misc_is_dict)
        print(f"  {str(t['id']):<{col['id']}} {t['form']:<{col['form']}} "
              f"{t['lemma']:<{col['lemma']}} {t['upos']:<{col['upos']}} "
              f"{feats_s:<{col['feats']}} {str(t['head']):<{col['head']}} "
              f"{t['deprel']:<{col['deprel']}} {misc_s}")


def _print_misc_spotlight(sent: list, misc_is_dict: bool) -> None:
    """Выводит только токены с непустым MISC, с repr для наглядности."""
    tokens = [
        t for t in sent
        if t["misc"] not in ("_", {}, "", None) and t["misc"] is not None
    ]
    if not tokens:
        print("    (нет токенов с заполненным MISC)")
        return
    for t in tokens:
        misc_repr = repr(t["misc"]) if misc_is_dict else repr(t["misc"])
        print(f"    [{str(t['id']):>2}] {t['form']:<16}  {misc_repr}")


# ─────────────────────────────────────────────────────────────
# Тестовые тексты — максимум MISC-полей от UDPipe
# ─────────────────────────────────────────────────────────────

# UDPipe реально заполняет эти MISC-поля:
#   SpaceAfter=No        — нет пробела перед следующим токеном
#   SpacesBefore=\n      — нестандартный пробел перед токеном (перенос строки)
#   SpacesAfter=\n       — нестандартный пробел после токена
#   SpacesInToken=...    — токен содержит пробел/таб внутри
#   TokenRange=start:end — только с опцией tokenizer=ranges
#
# Стратегия:
#   1. Знаки препинания вплотную к словам → SpaceAfter=No
#   2. Переносы строк между предложениями → SpacesBefore=\n / SpacesAfter=\n
#   3. Пробел внутри токена (имя+отчество без разрыва) → SpacesInToken
#
# Остальные поля (Translit, Gloss, Entity, CorrectForm и т.д.) —
# аннотации деревобанков UD, UDPipe их НЕ генерирует автоматически.

# Текст 1: пунктуация вплотную → SpaceAfter=No, переносы → SpacesBefore/SpacesAfter
TEXT_SPACES = (
    "Нет!\n"          # \n после ! → SpacesAfter=\n на токене !
    "Это невозможно,— сказал он.\n"  # запятая+тире вплотную
    "«Правда?» — спросила она."
)

# Текст 2: тот же текст, но парсим с опцией ranges → TokenRange в каждом токене
# (передаётся через сервис, если он поддерживает параметр tokenizer_options)
TEXT_RANGES = "Мама мыла раму. Папа читал газету."


# ─────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = UDPipeParser()
    SEP  = "=" * 72
    SEP2 = "-" * 72

    # ════════════════════════════════════════════════════════
    # 1. dict-формат: misc как raw CoNLL-U строка
    # ════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("РЕЖИМ: dict  (misc = raw CoNLL-U строка)")
    print(SEP)

    result_dict = parser.parse_text(TEXT_SPACES, output_format="dict")
    print(f"Предложений: {len(result_dict)}\n")

    for s_idx, sent in enumerate(result_dict, 1):
        print(f"  Предложение {s_idx}  ({len(sent)} токенов):")
        _print_table(sent, misc_is_dict=False)
        print(f"\n  ↳ Токены с MISC:")
        _print_misc_spotlight(sent, misc_is_dict=False)
        print()

    # Мета-информация
    if result_dict:
        tok0 = result_dict[0][0]
        print(f"  Ключи токена : {list(tok0.keys())}")
        print(f"  Тип misc     : {type(tok0['misc']).__name__}")
        print(f"\n  JSON первого токена:")
        print("  " + json.dumps(tok0, ensure_ascii=False, indent=2)
              .replace("\n", "\n  "))

    # ════════════════════════════════════════════════════════
    # 2. native-формат: misc как словарь
    # ════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("РЕЖИМ: native  (misc = словарь Python)")
    print(SEP)

    result_native = parser.parse_text(TEXT_SPACES, output_format="native")
    print(f"Предложений: {len(result_native)}\n")

    for s_idx, sent in enumerate(result_native, 1):
        print(f"  Предложение {s_idx}  ({len(sent)} токенов):")
        _print_table(sent, misc_is_dict=True)
        print(f"\n  ↳ Токены с MISC:")
        _print_misc_spotlight(sent, misc_is_dict=True)
        print()

    if result_native:
        tok0 = result_native[0][0]
        print(f"  Ключи токена : {list(tok0.keys())}")
        print(f"  Тип misc     : {type(tok0['misc']).__name__}")
        print(f"\n  JSON первого токена:")
        print("  " + json.dumps(tok0, ensure_ascii=False, indent=2)
              .replace("\n", "\n  "))

    # ════════════════════════════════════════════════════════
    # 3. Сводка: все MISC-ключи по корпусу текста
    # ════════════════════════════════════════════════════════
    if result_dict and result_native:
        print(f"\n{SEP}")
        print("СВОДКА: все MISC-ключи в разборе текста")
        print(SEP)

        # 3a. Уникальные raw-значения (dict)
        all_raw = set()
        for sent in result_dict:
            for t in sent:
                if t["misc"] not in ("_", None, ""):
                    all_raw.add(t["misc"])
        print(f"\n  Уникальные raw MISC (dict-формат):")
        for v in sorted(all_raw):
            print(f"    {repr(v)}")

        # 3b. Уникальные ключи словарей (native)
        all_keys: dict[str, set] = {}
        for sent in result_native:
            for t in sent:
                if isinstance(t["misc"], dict):
                    for k, v in t["misc"].items():
                        all_keys.setdefault(k, set()).add(
                            v if v is not True else "<flag>"
                        )
        print(f"\n  Уникальные MISC-ключи (native-формат):")
        if all_keys:
            for k in sorted(all_keys):
                vals = sorted(str(v) for v in all_keys[k])
                print(f"    {k:20s} → {', '.join(vals)}")
        else:
            print("    (ни одного — модель вернула только SpaceAfter или пусто)")

        # 3c. Попарное сравнение форматов (первое предложение)
        print(f"\n  Попарное сравнение misc, предложение 1:")
        print(f"  {'FORM':<16} {'dict misc':<35} native misc")
        print(f"  {SEP2}")
        for td, tn in zip(result_dict[0], result_native[0]):
            if td["misc"] != "_" or tn["misc"]:
                print(f"  {td['form']:<16} "
                      f"{repr(td['misc']):<35} "
                      f"{repr(tn['misc'])}")
