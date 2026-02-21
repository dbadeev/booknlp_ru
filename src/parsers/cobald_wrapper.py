#!/usr/bin/env python3
"""
Локальная обёртка для CoBaLD-парсера (Modal-сервис).

Требует предварительного деплоя:
    modal deploy src/parsers/cobald_modal.py

Использование:
    parser = CobaldParser()

    # Все предложения текста, dict-формат:
    sentences = parser.parse_text("Мама мыла раму. Папа читал газету.")
    for sent in sentences:
        for tok in sent:
            print(tok["id"], tok["form"], tok["deprel"])

    # Native-формат (расширенные поля):
    sentences = parser.parse_text(text, output_format="native")

    # CoNLL-U строка (из native):
    native = parser.parse_text(text, output_format="native")
    print(_to_conllu_str(native))

    # Пакетная обработка:
    results = parser.parse_batch(["Текст 1.", "Текст 2."])
"""

import logging
import sys
from typing import Any, Dict, List, Literal

import modal

logger = logging.getLogger(__name__)

OutputFormat = Literal["dict", "native"]


class CobaldParser:
    """Клиент CoBaLD-парсера, запущенного в Modal."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-cobald", "CobaldService"
            )()
            self.logger.info("✓ Connected to CoBaLD via Modal.")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Modal: {e}")
            raise

    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "dict",
    ) -> List[List[Dict[str, Any]]]:
        """
        Разбирает текст, возвращает все предложения.

        Parameters
        ----------
        text : str
            Сырой текст.
        output_format : str
            'dict'   → id, form, head, deprel, misc, deepslot, semclass
            'native' → то же + lemma, upos, xpos, feats, deps_eud, is_null

        Returns
        -------
        List[List[Dict]]
            Список предложений; каждое — список токенов.
        """
        try:
            result = self.service.parse.remote(text, output_format=output_format)
            if not result:
                self.logger.warning("Сервис вернул пустой результат.")
                return []
            return result
        except Exception as e:
            self.logger.error(f"❌ Ошибка при разборе: {e}")
            raise

    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "dict",
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
                batch        = texts[i : i + batch_size]
                batch_result = self.service.parse_batch.remote(
                    batch, output_format=output_format
                )
                results.extend(batch_result or [])
            return results
        except Exception as e:
            self.logger.error(f"❌ Ошибка при пакетной обработке: {e}")
            raise


# ─────────────────────── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────
def _dep_tuple_to_str(dep: Any) -> str:
    """
    Конвертирует deps_ud / deps_eud в строку CoNLL-U формата head:deprel.

    Поддерживает все форматы, которые возвращает pipeline:
      - tuple 3: ('head', 'self_id', 'deprel')  → 'head:deprel'
      - tuple 2: ('head', 'deprel')              → 'head:deprel'
      - str:     уже строка                      → возвращаем как есть
      - None/_:                                  → '_'
    """
    if dep is None:
        return "_"
    if isinstance(dep, str):
        return dep.strip() or "_"
    if isinstance(dep, (list, tuple)):
        if len(dep) == 3:
            # ('head_id', 'self_id', 'deprel') — реальный формат CoBaLD
            return f"{dep[0]}:{dep[2]}"
        if len(dep) == 2:
            return f"{dep[0]}:{dep[1]}"
    return "_"


def _to_conllu_str(sentences: List[List[Dict[str, Any]]]) -> str:
    """
    Конвертирует список предложений в native-формате в строку CoNLL-U.

    Поля CoNLL-U (10 колонок, разделитель TAB):
        ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

    CoBaLD-специфичные поля deepslot и semclass добавляются в MISC:
        SpaceAfter=No|Deepslot=Agent|Semclass=BEING

    Требует native-формата (нужны lemma, upos, xpos, feats, deps_eud).
    При dict-формате LEMMA/UPOS/XPOS/FEATS/DEPS будут '_'.
    """
    lines = []
    for sent in sentences:
        if not sent:
            continue
        for tok in sent:
            # ── MISC: объединяем оригинальный misc с CoBaLD-полями ──────────
            misc_parts = []
            raw_misc = (tok.get("misc") or "").strip()
            if raw_misc and raw_misc != "_":
                misc_parts.append(raw_misc)
            deepslot = (tok.get("deepslot") or "").strip()
            semclass = (tok.get("semclass") or "").strip()
            if deepslot and deepslot != "_":
                misc_parts.append(f"Deepslot={deepslot}")
            if semclass and semclass != "_":
                misc_parts.append(f"Semclass={semclass}")
            misc_str = "|".join(misc_parts) if misc_parts else "_"

            # ── Enhanced UD (DEPS, 9-я колонка) ─────────────────────────────
            deps_eud = _dep_tuple_to_str(tok.get("deps_eud"))

            line = "\t".join([
                str(tok["id"]),
                tok.get("form", "_"),
                tok.get("lemma", "_") or "_",   # только в native
                tok.get("upos",  "_") or "_",   # только в native
                tok.get("xpos",  "_") or "_",   # только в native
                tok.get("feats", "_") or "_",   # только в native
                str(tok.get("head", 0)),
                tok.get("deprel", "_") or "_",
                deps_eud,                       # только в native
                misc_str,
            ])
            lines.append(line)
        lines.append("")  # пустая строка между предложениями

    return "\n".join(lines)


def _print_sentence_table(sent: List[Dict]) -> None:
    """Выводит токены предложения в виде таблицы."""
    print(f"  {'ID':<4} {'FORM':<16} {'HEAD':<5} {'DEPREL':<14} "
          f"{'DEEPSLOT':<14} {'SEMCLASS':<12} MISC")
    print("  " + "-" * 78)
    for tok in sent:
        print(f"  {tok['id']:<4} {tok['form']:<16} {tok['head']:<5} "
              f"{tok['deprel']:<14} {tok.get('deepslot', '—'):<14} "
              f"{tok.get('semclass', '—'):<12} {tok.get('misc', '—')}")


# ─────────────────────────────── ТЕСТЫ ───────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    SEP = "=" * 70

    # ── Проверка доступности сервиса ─────────────────────────────────────────
    print(f"{SEP}\nПРОВЕРКА MODAL-СЕРВИСА\n{SEP}")
    try:
        parser = CobaldParser()
    except Exception as e:
        print(f"⚠️ Modal-сервис недоступен: {e}")
        print("Запустите: modal deploy src/parsers/cobald_modal.py")
        sys.exit(1)

    test_text  = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    test_batch = ["Он думал о море.", "Кот лежал на диване."]

    # ── 1. dict-формат ───────────────────────────────────────────────────────
    print(f"\n{SEP}\nФОРМАТ: dict\n{SEP}")
    result_dict = parser.parse_text(test_text, output_format="dict")
    print(f"Текст: '{test_text}'")
    print(f"Предложений: {len(result_dict)}\n")
    for s_idx, sent in enumerate(result_dict, 1):
        print(f"  Предложение {s_idx} ({len(sent)} токенов):")
        _print_sentence_table(sent)
        print()

    if result_dict and result_dict[0]:
        tok0 = result_dict[0][0]
        print(f"  Ключи токена : {list(tok0.keys())}")
        print(f"  Тип id       : {type(tok0['id']).__name__}  (ожидается int)")
        print(f"\n  CoBaLD-поля первого токена:")
        print(f"    misc      : {tok0.get('misc', '—')}")
        print(f"    deepslot  : {tok0.get('deepslot', '—')}")
        print(f"    semclass  : {tok0.get('semclass', '—')}")

    # ── 2. native-формат ─────────────────────────────────────────────────────
    print(f"\n{SEP}\nФОРМАТ: native\n{SEP}")
    result_native = parser.parse_text(test_text, output_format="native")
    print(f"Предложений: {len(result_native)}\n")
    for s_idx, sent in enumerate(result_native, 1):
        print(f"  Предложение {s_idx} ({len(sent)} токенов):")
        _print_sentence_table(sent)
        if sent:
            extra_keys = [k for k in sent[0]
                          if k not in ("id", "form", "head", "deprel",
                                       "misc", "deepslot", "semclass")]
            if extra_keys:
                print(f"  Доп. поля native: {extra_keys}")
        print()

    # ── 3. CoNLL-U формат (из native) ────────────────────────────────────────
    # ДОБАВЛЕНО: вывод в стандартном CoNLL-U формате.
    # Данные берутся из result_native — только он содержит lemma/upos/feats/eud.
    # CoBaLD-специфичные поля (deepslot, semclass) добавляются в MISC-колонку.
    print(f"\n{SEP}\nФОРМАТ: CoNLL-U (из native)\n{SEP}")
    conllu_str = _to_conllu_str(result_native)
    print(conllu_str)

    # ── 4. Проверка типа возврата ─────────────────────────────────────────────
    print(f"\n{SEP}\nПРОВЕРКА ТИПА ВОЗВРАТА\n{SEP}")
    for fmt in ("dict", "native"):
        r = parser.parse_text("Тест.", output_format=fmt)
        status = "✅" if r is not None else "❌ None!"
        print(f"  parse_text(format={fmt!r})"
              f" → type={type(r).__name__}"
              f" is_none={r is None}"
              f" {status}")

    # ── 5. Пакетная обработка ─────────────────────────────────────────────────
    print(f"\n{SEP}\nПАКЕТНАЯ ОБРАБОТКА\n{SEP}")
    result_batch = parser.parse_batch(test_batch, output_format="dict")
    print(f"Текстов: {len(test_batch)}, результатов: {len(result_batch)}\n")
    for t_idx, text_sents in enumerate(result_batch):
        total = sum(len(s) for s in text_sents)
        print(f"  [{t_idx}] '{test_batch[t_idx]}'"
              f" → {len(text_sents)} предл., {total} токенов")

    print(f"\n{SEP}\n✅ Все тесты завершены\n{SEP}")
