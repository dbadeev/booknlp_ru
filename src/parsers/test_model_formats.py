#!/usr/bin/env python3
"""
Тестовый скрипт для определения фактической структуры вывода всех моделей
в двух форматах: CoNLL-U (упрощённый/dict) и нативном.

Выходные файлы:
  model_formats_conllu.txt  — результаты всех моделей в CoNLL-U / dict формате
  model_formats_native.txt  — результаты всех моделей в нативном формате модели

Модели: pymorphy3, stanza, cobald, udpipe, deeppavlov, trankit, mystem, slovnet, spacy

Примечание по токенизации:
  Модели, поддерживающие внешнюю токенизацию, используют razdel:
    pymorphy3, cobald, deeppavlov, slovnet — razdel внутри враппера
    spacy — razdel через параметр tokenizer="razdel"
  Остальные модели выполняют токенизацию самостоятельно:
    stanza, udpipe, trankit, mystem
"""

import json
import logging
from typing import Any, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MODEL_FORMAT_TEST")

TEST_TEXT = "Мама мыла раму."

OUTPUT_CONLLU = "model_formats_conllu.txt"
OUTPUT_NATIVE = "model_formats_native.txt"


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

class _SafeEncoder(json.JSONEncoder):
    """
    JSONEncoder с безопасным fallback для не-сериализуемых объектов.

    Необходим для pymorphy3 native-формата, где поле methods_stack
    содержит кортежи с объектами DictionaryAnalyzer (не JSON-сериализуемы).
    Все нераспознанные типы конвертируются через repr() для сохранения
    максимума диагностической информации.
    """
    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def _first_token_keys(result: Any) -> list:
    """
    Извлекает ключи первого токена из результата любого формата.

    Поддерживаемые форматы:
      List[List[Dict]]  — стандартный CoNLL-U-like (большинство моделей)
      List[Dict]        — spaCy/Stanza native (предложения с ключом "words")
      List[str]         — CoBaLD native (список CoNLL-Plus строк)
      Dict              — DeepPavlov full ("sentences"), Slovnet native ("tokens")
      str               — UDPipe/SpaCy CoNLL-U (текстовый формат, ключи неприменимы)
    """
    try:
        if isinstance(result, list) and result:
            first = result[0]
            # List[List[Dict]]: [[{id, form, ...}, ...], ...]
            if isinstance(first, list) and first and isinstance(first[0], dict):
                return list(first[0].keys())
            # List[Dict]: spaCy/Stanza native [{text, words:[...]}, ...]
            if isinstance(first, dict):
                words = first.get("words")
                if isinstance(words, list) and words:
                    return list(words[0].keys())
                return list(first.keys())
            # List[str]: строковый формат — ключи неприменимы
        elif isinstance(result, dict):
            # DeepPavlov full: {"format": ..., "conllu": ..., "sentences": [[...]]}
            sentences = result.get("sentences")
            if isinstance(sentences, list) and sentences and sentences[0]:
                return list(sentences[0][0].keys())
            # Slovnet native: {"tokens": [...], "spans": [...]}
            tokens = result.get("tokens")
            if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
                return list(tokens[0].keys())
            return list(result.keys())
    except Exception:
        pass
    return []


def _write_model_result(f, model_name: str, result: Any) -> None:
    """
    Записывает результат модели в файл в формате,
    аналогичном выводу test_model_formats.py:
      --- {model} результат ---
      {JSON или текст}
      --- Поля первого токена ---
      Доступные ключи: [...]

    Использует _SafeEncoder для корректной сериализации объектов,
    не поддерживаемых стандартным json (напр. DictionaryAnalyzer из pymorphy3).
    """
    f.write(f"\n--- {model_name} результат ---\n")

    # Строковый формат: UDPipe native, SpaCy conllu (str)
    if isinstance(result, str):
        f.write(result)
    # Список строк: CoBaLD native (List[str])
    elif isinstance(result, list) and result and isinstance(result[0], str):
        for sent_str in result:
            f.write(sent_str)
            f.write("\n")
    # Все остальные форматы (dict, List[List[Dict]], List[Dict])
    # _SafeEncoder конвертирует нераспознанные типы через repr()
    else:
        f.write(json.dumps(result, ensure_ascii=False, indent=2, cls=_SafeEncoder))

    f.write("\n")

    keys = _first_token_keys(result)
    if keys:
        f.write(f"\n--- Поля первого токена ---\n")
        f.write(f"Доступные ключи: {keys}\n")


# ============================================================
# ФУНКЦИИ ТЕСТИРОВАНИЯ МОДЕЛЕЙ
# ============================================================

def test_pymorphy3() -> Tuple[Any, Any]:
    """
    Pymorphy3 (локально).
    CoNLL-U: output_format="simplified"
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel
    Native:  output_format="native"
        → List[List[Dict]]: id, word, normal_form, tag, score,
                             methods_stack, lexeme, is_known, normalized
    Токенизация: razdel (внутри враппера).
    """
    from pymorphy3_wrapper import Pymorphy3Parser

    logger.info("=== Тестирование Pymorphy3 ===")
    parser = Pymorphy3Parser()
    conllu = parser.parse_text(TEST_TEXT, output_format="simplified")
    native = parser.parse_text(TEST_TEXT, output_format="native")
    return conllu, native


def test_stanza() -> Tuple[Any, Any]:
    """
    Stanza (Modal).
    CoNLL-U: native_format=False
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel
    Native:  native_format=True
        → List[Dict]: {"words": [{id, form, lemma, upos, xpos, feats, head, deprel,
                                   start_char, end_char, misc, ner}, ...]}
    Токенизация: Stanza выполняет сегментацию самостоятельно.
    """
    from stanza_wrapper import StanzaParser

    logger.info("=== Тестирование Stanza ===")
    parser = StanzaParser()
    conllu = parser.parse_text(TEST_TEXT, native_format=False)
    native = parser.parse_text(TEST_TEXT, native_format=True)
    return conllu, native


def test_cobald() -> Tuple[Any, Any]:
    """
    CoBaLD (Modal).
    CoNLL-U: output_format='dict'
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel,
                             deps, misc, semclass, deepslot
    Native:  output_format='native'
        → List[str]: список строк в формате CoNLL-Plus (12 колонок:
                     ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, SC, DS)
    Токенизация: razdel (внутри враппера).
    """
    from cobald_wrapper import CobaldParser

    logger.info("=== Тестирование CoBaLD ===")
    parser = CobaldParser()
    conllu = parser.parse_text(TEST_TEXT, output_format='dict')
    native = parser.parse_text(TEST_TEXT, output_format='native')
    return conllu, native


def test_udpipe() -> Tuple[Any, Any]:
    """
    UDPipe (Modal).
    CoNLL-U: output_format='dict'
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel,
                             deps, misc, startchar, endchar
    Native:  output_format='native'
        → str: строка в стандартном формате CoNLL-U (10 колонок,
               предложения разделены пустой строкой)
    Токенизация: UDPipe выполняет токенизацию самостоятельно.
    """
    from udpipe_wrapper import UDPipeParser

    logger.info("=== Тестирование UDPipe ===")
    parser = UDPipeParser()
    conllu = parser.parse_text(TEST_TEXT, output_format='dict')
    native = parser.parse_text(TEST_TEXT, output_format='native')
    return conllu, native


def test_deeppavlov() -> Tuple[Any, Any]:
    """
    DeepPavlov (Modal, токенизатор: razdel).
    CoNLL-U: output_format='dict'
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel
    Native:  output_format='full'
        → Dict: {"format": ..., "conllu": ..., "sentences": [[...]]}
                 каждый токен: id, form, lemma, upos, upos_proba,
                               head, deprel, heads_proba, deps_proba
    Токенизация: razdel (задаётся в конструкторе через tokenizer='razdel').
    """
    from deeppavlov_wrapper import DeepPavlovParser

    logger.info("=== Тестирование DeepPavlov ===")
    parser = DeepPavlovParser(tokenizer='razdel')
    conllu = parser.parse_text(TEST_TEXT, output_format='dict')
    native = parser.parse_text(TEST_TEXT, output_format='full')
    return conllu, native


def test_trankit() -> Tuple[Any, Any]:
    """
    Trankit (Modal).
    CoNLL-U: output_format="simplified"
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats,
                             head, deprel, start_char, end_char
    Native:  output_format="native"
        → List[List[Dict]]: id, text, lemma, upos, xpos, feats, head, deprel,
                             span, dspan, ner, expanded
    Токенизация: Trankit выполняет токенизацию самостоятельно.
    """
    from trankit_wrapper import TrankitParser

    logger.info("=== Тестирование Trankit ===")
    parser = TrankitParser()
    conllu = parser.parse_text(TEST_TEXT, output_format="simplified")
    native = parser.parse_text(TEST_TEXT, output_format="native")
    return conllu, native


def test_mystem() -> Tuple[Any, Any]:
    """
    Mystem (Modal) — только морфология, синтаксический разбор отсутствует.
    CoNLL-U: output_format="simplified"
        → List[List[Dict]]: id, form, lemma, upos
    Native:  output_format="native"
        → List[List[Dict]]: id, text, analysis
                             (analysis: список гипотез разбора с полями lex, gr, wt, qual)
    Токенизация: Mystem выполняет токенизацию самостоятельно.
    """
    from mystem_wrapper import MystemParser

    logger.info("=== Тестирование Mystem ===")
    parser = MystemParser()
    conllu = parser.parse_text(TEST_TEXT, output_format="simplified")
    native = parser.parse_text(TEST_TEXT, output_format="native")
    return conllu, native


def test_slovnet() -> Tuple[Any, Any]:
    """
    Slovnet (Modal).
    CoNLL-U: parse_text(include_ner=False)
        → List[List[Dict]]: id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
    Native:  parse_text_with_ner()
        → Dict: {"tokens": [{id, form, lemma, upos, xpos, feats, head, deprel, deps, misc}, ...],
                 "spans":  [{start, stop, type, text, normal, fact}, ...]}
    Токенизация: razdel (внутри враппера, в обоих методах).
    """
    from slovnet_wrapper import SlovnetParser

    logger.info("=== Тестирование Slovnet ===")
    parser = SlovnetParser()
    conllu = parser.parse_text(TEST_TEXT, include_ner=False)
    native = parser.parse_text_with_ner(TEST_TEXT)
    return conllu, native


def test_spacy() -> Tuple[Any, Any]:
    """
    SpaCy (Modal).
    CoNLL-U: output_format="conllu", tokenizer="razdel"
        → str: строка в стандартном формате CoNLL-U
    Native:  output_format="native", tokenizer="razdel"
        → List[Dict]: [{"text": ...,
                         "words": [{id, start_char, end_char, form, norm, lower, shape,
                                    lemma, upos, xpos, feats, head, deprel, n_lefts,
                                    n_rights, children, ent_type, ent_iob, is_sent_start,
                                    whitespace, misc, is_alpha, is_digit, is_punct,
                                    is_space, is_stop, is_oov, like_num, like_url,
                                    like_email, has_vector, cluster}, ...],
                         "entities": [...]}, ...]
    Токенизация: razdel (внешняя, через параметр tokenizer="razdel").
    """
    from spacy_wrapper import SpacyParser

    logger.info("=== Тестирование SpaCy ===")
    parser = SpacyParser()
    conllu = parser.parse_text(TEST_TEXT, output_format="conllu", tokenizer="razdel")
    native = parser.parse_text(TEST_TEXT, output_format="native", tokenizer="razdel")
    return conllu, native


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main() -> None:
    """Запускает все тесты и сохраняет результаты в два выходных файла."""

    models: Dict[str, Any] = {
        "pymorphy3":  test_pymorphy3,
        "stanza":     test_stanza,
        "cobald":     test_cobald,
        "udpipe":     test_udpipe,
        "deeppavlov": test_deeppavlov,
        "trankit":    test_trankit,
        "mystem":     test_mystem,
        "slovnet":    test_slovnet,
        "spacy":      test_spacy,
    }

    results_conllu: Dict[str, Any] = {}
    results_native: Dict[str, Any] = {}

    for model_name, test_func in models.items():
        try:
            logger.info(f"Запуск теста для {model_name}...")
            conllu_result, native_result = test_func()
            results_conllu[model_name] = conllu_result
            results_native[model_name] = native_result
            logger.info(f"✓ {model_name} успешно протестирован")
        except Exception as e:
            logger.error(f"✗ {model_name} завершился с ошибкой: {e}")
            import traceback
            traceback.print_exc()
            results_conllu[model_name] = None
            results_native[model_name] = None

    # ── Запись файла CoNLL-U форматов ────────────────────────────────────────
    with open(OUTPUT_CONLLU, "w", encoding="utf-8") as f:
        f.write("# Результаты парсинга в CoNLL-U / dict формате\n")
        f.write(f"# Тестовый текст: {TEST_TEXT!r}\n")
        for model_name, result in results_conllu.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"=== Тестирование {model_name.upper()} ===\n")
            f.write(f"{'=' * 70}\n")
            if result is not None:
                _write_model_result(f, model_name, result)
            else:
                f.write("\n[ОШИБКА: модель завершилась с ошибкой]\n")

    # ── Запись файла нативных форматов ───────────────────────────────────────
    with open(OUTPUT_NATIVE, "w", encoding="utf-8") as f:
        f.write("# Результаты парсинга в нативном формате модели\n")
        f.write(f"# Тестовый текст: {TEST_TEXT!r}\n")
        for model_name, result in results_native.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"=== Тестирование {model_name.upper()} ===\n")
            f.write(f"{'=' * 70}\n")
            if result is not None:
                _write_model_result(f, model_name, result)
            else:
                f.write("\n[ОШИБКА: модель завершилась с ошибкой]\n")

    # ── Итоги ────────────────────────────────────────────────────────────────
    logger.info("\n=== Итоги тестирования ===")
    for model_name in models:
        conllu_ok = results_conllu.get(model_name) is not None
        native_ok = results_native.get(model_name) is not None
        status = "✓ OK" if (conllu_ok and native_ok) else "✗ FAIL"
        logger.info(f"{model_name}: {status}")

    logger.info(f"\nCoNLL-U результаты сохранены в {OUTPUT_CONLLU}")
    logger.info(f"Нативные результаты сохранены в {OUTPUT_NATIVE}")


if __name__ == "__main__":
    main()
