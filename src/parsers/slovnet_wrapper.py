#!/usr/bin/env python3
"""
Slovnet Wrapper — клиент для SlovnetService на Modal.

Требует предварительного деплоя:
    modal deploy src/parsers/slovnet_modal.py

Запуск тестов:
    python src/parsers/slovnet_wrapper.py
"""

import logging
from typing import Any, Dict, List, Union

import modal

logger = logging.getLogger(__name__)

APP_NAME      = "booknlp-ru-slovnet"
METHOD_NAME   = "SlovnetService.parse_text"


class SlovnetParser:
    """
    Клиентская обёртка над задеплоенным Modal-сервисом SlovnetService.

    Требует: modal deploy slovnet_modal.py
    """

    def __init__(self):
        # Cls.from_name — ленивый метод, не обращается к серверу до первого вызова.
        # Требует задеплоенного приложения: modal deploy slovnet_modal.py
        SlovnetService = modal.Cls.from_name(APP_NAME, "SlovnetService")
        self._service = SlovnetService()
        logger.info(
            f"SlovnetParser подключён к Modal-приложению '{APP_NAME}'."
        )

    def parse_text(
        self,
        text: str,
        output_format: str = "conllu",
    ) -> Union[List[List[Dict[str, Any]]], Dict[str, Any]]:
        return self._service.parse_text.remote(text, output_format=output_format)

# ─────────────────────────────────────────────────────────────
# Точка входа — тестовые примеры
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = SlovnetParser()
    TEST_TEXT = "Александр Сергеевич Пушкин родился в Москве в 1799 году."
    SEP = "=" * 70

    # ════════════════════════════════════════════
    # 1. CoNLL-U
    # ════════════════════════════════════════════
    print(f"\n{SEP}\nРЕЖИМ: conllu  →  List[List[Dict]]\n{SEP}")
    result_conllu = parser.parse_text(TEST_TEXT, output_format="conllu")
    print(f"Предложений: {len(result_conllu)}\n")
    for s_idx, sent in enumerate(result_conllu, 1):
        print(f"  Предложение {s_idx}:")
        print(f"  {'ID':<4} {'FORM':<14} {'UPOS':<7} {'FEATS':<36} "
              f"{'HEAD':<5} {'DEPREL':<10} START  END")
        print("  " + "-" * 92)
        for t in sent:
            feats_d = (t["feats"][:34] + "..") if len(t["feats"]) > 36 else t["feats"]
            print(f"  {t['id']:<4} {t['form']:<14} {t['upos']:<7} {feats_d:<36} "
                  f"{t['head']:<5} {t['deprel']:<10} {t['startchar']}  {t['endchar']}")

    print(f"\nКлючи conllu-токена: {list(result_conllu[0][0].keys())}")
    print("\nJSON первого токена:")
    print(json.dumps(result_conllu[0][0], ensure_ascii=False, indent=2))
    # ════════════════════════════════════════════
    # 2. Native
    # ════════════════════════════════════════════
    print(f"\n{SEP}\nРЕЖИМ: native  →  Dict{{'tokens': [...], 'spans': [...]}}\n{SEP}")
    result_native = parser.parse_text(TEST_TEXT, output_format="native")
    tokens = result_native["tokens"]
    spans  = result_native["spans"]

    print(f"Токенов: {len(tokens)},  Spans (NER): {len(spans)}\n")

    # ── Таблица токенов: все поля ──────────────────────────────
    print(f"  {'ID':<4} {'TEXT':<14} {'POS':<7} {'FEATS':<46} "
          f"{'HEAD_ID':<8} {'REL':<12} {'START':<6} STOP")
    print("  " + "-" * 110)
    for t in tokens:
        # feats — словарь или None → строка K=V|K=V
        if isinstance(t["feats"], dict):
            feats_s = "|".join(f"{k}={v}" for k, v in sorted(t["feats"].items()))
        else:
            feats_s = str(t["feats"]) if t["feats"] else "None"
        feats_d = (feats_s[:44] + "..") if len(feats_s) > 46 else feats_s

        print(f"  {t['id']:<4} {t['text']:<14} {str(t['pos']):<7} "
              f"{feats_d:<46} {str(t['head_id']):<8} {str(t['rel']):<12} "
              f"{t['start']:<6} {t['stop']}")

    # ── JSON-дамп первых двух токенов целиком ─────────────────
    print(f"\nКлючи native-токена: {list(tokens[0].keys())}")
    print("\nJSON первых двух токенов (все поля):")
    print(json.dumps(tokens[:2], ensure_ascii=False, indent=2, default=str))

    # ── Spans: все поля ───────────────────────────────────────
    if spans:
        print(f"\nSpans ({len(spans)}):")
        for sp in spans:
            print(f"\n  [{sp['start']}:{sp['stop']}]  type={sp['type']}")
            print(f"    text   = '{sp.get('text', '')}'")
            print(f"    normal = '{sp.get('normal', '')}'")
            if sp.get("fact"):
                print(f"    fact:")
                for k, v in sp["fact"].items():
                    print(f"      {k:<8} = '{v}'")
            else:
                print(f"    fact   = None")
        # JSON-дамп первого span целиком
        print(f"\nJSON первого span (все поля):")
        print(json.dumps(spans[0], ensure_ascii=False, indent=2, default=str))
    else:
        print("\nSpans: []")

    # ════════════════════════════════════════════
    # 3. Сравнение ключей и feats
    # ════════════════════════════════════════════
    print(f"\n{SEP}\nСРАВНЕНИЕ КЛЮЧЕЙ И ФОРМАТА FEATS\n{SEP}")
    ck = set(result_conllu[0][0].keys())
    nk = set(tokens[0].keys())
    print(f"  Только в conllu: {sorted(ck - nk)}")
    print(f"  Только в native: {sorted(nk - ck)}")
    print(f"\n  conllu feats (строка CoNLL-U): {repr(result_conllu[0][0]['feats'])}")
    print(f"  native feats (dict|None):       {repr(tokens[0]['feats'])}")
