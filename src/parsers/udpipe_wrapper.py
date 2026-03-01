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
        try:
            self._service = modal.Cls.from_name(APP_NAME, "UDPipeService")()
            logger.info(f"UDPipeParser подключён к Modal-приложению '{APP_NAME}'.")
        except Exception as e:
            logger.error(f"❌ Не удалось подключиться к Modal: {e}")
            raise

    def parse_text(
            self,
            text: str,
            output_format: str = "dict",
            # ДОБАВЛЕНО: поддержка дополнительных опций токенизатора UDPipe.
            # Например: {"ranges": True} → TokenRange=start:end в каждом токене MISC.
            # Требует поддержки параметра tokenizer_options в UDPipeService.parse().
            tokenizer_options: dict | None = None,
    ) -> List[List[Dict[str, Any]]]:
        try:
            kwargs = dict(output_format=output_format)
            if tokenizer_options is not None:
                kwargs["tokenizer_options"] = tokenizer_options
            result = self._service.parse.remote(text, **kwargs)
            return result or []
        except NotImplementedError:
            raise  # просто пробрасываем — тест в __main__ сам напечатает предупреждение
        except Exception as e:
            logger.error(f"❌ Ошибка при разборе: {e}")
            raise

    def parse_batch(
            self,
            texts: List[str],
            output_format: str = "dict",
            batch_size: int = 32,
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Пакетная обработка списка текстов.

        Returns
        -------
        List[List[List[Dict]]]
            Для каждого текста — список предложений.
        """
        try:
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                batch_result = self._service.parse_batch.remote(
                    batch, output_format=output_format
                )
                results.extend(batch_result or [])
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка пакетной обработки: {e}")
            raise

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
        # misc_repr = repr(t["misc"]) if misc_is_dict else repr(t["misc"])
        # СТАЛО — для строк раскрываем CoNLL-U эскейпы:
        if misc_is_dict:
            # repr() на dict корректен, но значения внутри нужно раскрыть
            # для читаемого отображения:
            readable = {
                k: v.replace("\\n", "↵").replace("\\t", "→") if isinstance(v, str) else v
                for k, v in t["misc"].items()
            }
            misc_repr = repr(readable)
        else:
            misc_repr = repr(t["misc"])
        print(f"  [{str(t['id']):>2}] {t['form']:<16} {misc_repr}")


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
            for k in sorted(all_keys.keys()):
                vals = sorted(
                    v.replace("\\n", "↵").replace("\\t", "→")
                    if isinstance(v, str) else str(v)
                    for v in all_keys[k]
                )
                print(f"  {k:<20} → {', '.join(vals)}")
        else:
            print("    (ни одного — модель вернула только SpaceAfter или пусто)")

        # 3c. Попарное сравнение форматов (первое предложение)
        print(f"\n  Попарное сравнение misc, предложение 1:")
        print(f"  {'FORM':<16} {'dict misc':<35} native misc")
        print(f"  {SEP2}")
        for td, tn in zip(result_dict[0], result_native[0]):
            if td["misc"] != "_" or tn["misc"]:
                # dict — repr raw строки; native — ↵ для читаемости
                if isinstance(tn["misc"], dict):
                    tn_display = repr({
                        k: v.replace("\\n", "↵").replace("\\t", "→")
                        if isinstance(v, str) else v
                        for k, v in tn["misc"].items()
                    })
                else:
                    tn_display = repr(tn["misc"])
                print(f"  {td['form']:<16} "
                      f"{repr(td['misc']):<35} "
                      f"{tn_display}")

        # ════════════════════════════════════════════════════════
        # 4. parse_batch — пакетная обработка нескольких текстов
        # ════════════════════════════════════════════════════════
        print(f"\n{SEP}")
        print("РЕЖИМ: parse_batch (несколько текстов за один вызов)")
        print(SEP)

        TEST_BATCH = [
            "Он думал о море.",
            "Кот лежал на диване.",
            TEXT_SPACES,  # сложный текст с пунктуацией и переносами
        ]

        result_batch = parser.parse_batch(TEST_BATCH, output_format="native")
        print(f"Текстов подано   : {len(TEST_BATCH)}")
        print(f"Результатов получено: {len(result_batch)}\n")

        for t_idx, text_sents in enumerate(result_batch):
            total_tok = sum(len(s) for s in text_sents)
            preview = TEST_BATCH[t_idx][:40].replace("\n", "↵")
            print(f"  [{t_idx}] '{preview}'"
                  f" → {len(text_sents)} предл., {total_tok} токенов")

        # Проверяем что структура совпадает с parse_text
        single_result = parser.parse_text(TEST_BATCH[0], output_format="native")
        batch_first   = result_batch[0]
        match = (
            len(single_result) == len(batch_first)
            and all(
                s["form"] == b["form"]
                for ss, bs in zip(single_result, batch_first)
                for s, b in zip(ss, bs)
            )
        )
        print(f"\n  Совпадение parse_text vs parse_batch[0]: "
              f"{'✅ да' if match else '❌ расхождение!'}")

        # ════════════════════════════════════════════════════════
        # 5. TEXT_RANGES — опция tokenizer ranges → TokenRange в MISC
        # ════════════════════════════════════════════════════════
        print(f"\n{SEP}")
        print("РЕЖИМ: ranges (tokenizer_options={'ranges': True})")
        print("Ожидается: TokenRange=start:end в поле MISC каждого токена")
        print(SEP)

        try:
            result_ranges = parser.parse_text(
                TEXT_RANGES,
                output_format="native",
                tokenizer_options={"ranges": True},
            )
            print(f"Предложений: {len(result_ranges)}\n")

            for s_idx, sent in enumerate(result_ranges, 1):
                print(f"  Предложение {s_idx} ({len(sent)} токенов):")
                print(f"  {'ID':<4} {'FORM':<16} {'TokenRange':<18} прочие MISC-ключи")
                print("  " + "-" * 60)
                has_ranges = False
                for tok in sent:
                    misc = tok.get("misc") or {}
                    # misc может быть dict (native) или str (dict-формат)
                    if isinstance(misc, dict):
                        token_range = misc.get("TokenRange", "—")
                        other = {k: v for k, v in misc.items() if k != "TokenRange"}
                        has_ranges = has_ranges or token_range != "—"
                    else:
                        token_range = "—"
                        other = misc
                    print(f"  {tok['id']:<4} {tok['form']:<16} {str(token_range):<18} {other}")
                print()

                if not has_ranges:
                    print("  ⚠️  TokenRange отсутствует — сервис может не поддерживать"
                          " tokenizer_options.\n"
                          "     Проверьте: UDPipeService.parse() принимает tokenizer_options?")

        except TypeError as e:
            # Modal выбросит TypeError если parse() не принимает tokenizer_options
            print(f"  ⚠️  Сервис не поддерживает tokenizer_options: {e}")
            print(f"  Добавьте параметр в UDPipeService.parse() в udpipe_modal.py:")
            print(f"      def parse(self, text, output_format='dict', tokenizer_options=None)")
        except NotImplementedError as e:
            print(f"  ⚠️  Функция не поддерживается: {e}")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

        print(f"\n{SEP}\n✅ Все тесты завершены\n{SEP}")