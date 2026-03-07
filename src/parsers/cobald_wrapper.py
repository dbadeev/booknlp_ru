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
from razdel import sentenize

import modal

OutputFormat = Literal["dict", "native"]


class CobaldParser:
    """Клиент CoBaLD-парсера, запущенного в Modal."""
    SENTENCE_CHUNK_SIZE: int = 32  # предложений на чанк; подбирается под GPU/тип текстов

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-cobald", "CobaldService"
            )()
            self.logger.info("✓ Connected to CoBaLD via Modal.")
        except Exception as e2:
            self.logger.error(f"❌ Failed to connect to Modal: {e2}")
            raise

    # ─────────────────────── ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ─────────────────────────────
    @staticmethod
    def _split_to_sentence_chunks(
            text: str,
            chunk_size: int,
    ) -> List[List[str]]:
        """
        Разбивает текст на предложения (razdel.sentenize) и нарезает на чанки.

        Returns
        -------
        List[List[str]]
            Список чанков; каждый чанк — список текстов предложений.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size должен быть > 0, получено: {chunk_size}")
        sentences = [s.text for s in sentenize(text)]
        return [
            sentences[i: i + chunk_size]
            for i in range(0, len(sentences), chunk_size)
        ]

    @staticmethod
    def _merge_chunks(
            chunk_results: List[List[List[Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Собирает результаты чанков в плоский список предложений."""
        return [sent for chunk in chunk_results for sent in chunk]

    def parse_text(
            self,
            text: str,
            output_format: OutputFormat = "dict",
            chunk_size: int = SENTENCE_CHUNK_SIZE,
    ) -> List[List[Dict[str, Any]]]:
        """
        Разбирает текст, возвращает все предложения.

        Sentenize и chunking выполняются локально (razdel), чанки отправляются
        в Modal параллельно через .map(). Это предотвращает OOM при больших текстах.

        Parameters
        ----------
        text : str
            Сырой текст.
        output_format : str
            'dict' | 'native'
        chunk_size : int
            Число предложений в одном чанке. Уменьшайте при OOM.

        Returns
        -------
        List[List[Dict]]
            Список предложений; каждое — список токенов.
        """
        if output_format not in ("dict", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")
        if not text or not text.strip():
            return []

        chunks = self._split_to_sentence_chunks(text, chunk_size)
        if not chunks:
            return []

        try:
            chunk_results = list(
                self.service.parse_sentence_chunk.map(
                    chunks,
                    kwargs={"output_format": output_format},
                )
            )
            return self._merge_chunks(chunk_results)
        except Exception as e:
            self.logger.error(f"❌ Ошибка при разборе текста: {e}")
            raise

    def parse_batch(
            self,
            texts: List[str],
            output_format: OutputFormat = "dict",
            chunk_size: int = SENTENCE_CHUNK_SIZE,
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Пакетная обработка списка текстов.

        Все чанки всех текстов собираются и отправляются в Modal одним .map() —
        Modal распределяет по контейнерам параллельно.

        Returns
        -------
        List[List[List[Dict]]]
            Для каждого текста — список предложений.
        """
        if output_format not in ("dict", "native"):
            raise ValueError(f"Unknown output_format: {output_format!r}")
        if not texts:
            return []

        # Нарезаем все тексты на чанки, запоминаем сколько чанков у каждого текста
        chunks_per_text: List[int] = []
        all_chunks: List[List[str]] = []
        for text in texts:
            if not text or not text.strip():
                chunks_per_text.append(0)
                continue
            text_chunks = self._split_to_sentence_chunks(text, chunk_size)
            chunks_per_text.append(len(text_chunks))
            all_chunks.extend(text_chunks)

        if not all_chunks:
            return [[] for _ in texts]

        try:
            # Один .map() → Modal параллелит все чанки
            all_results = list(
                self.service.parse_sentence_chunk.map(
                    all_chunks,
                    kwargs={"output_format": output_format},
                )
            )
        except Exception as e:
            self.logger.error(f"❌ Ошибка при пакетной обработке: {e}")
            raise

        # Собираем обратно: каждому тексту возвращаем его предложения
        results: List[List[List[Dict[str, Any]]]] = []
        offset = 0
        for n_chunks in chunks_per_text:
            if n_chunks == 0:
                results.append([])
            else:
                results.append(
                    self._merge_chunks(all_results[offset: offset + n_chunks])
                )
                offset += n_chunks
        return results


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
    for sent_idx, snt in enumerate(sentences, 1):
        if not snt:
            continue
        lines.append(f"# sent_id = {sent_idx}")
        lines.append(f"# text = {' '.join(t.get('form', '') for t in snt)}")

        for tok in snt:
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


def _print_sentence_table(sentence: List[Dict]) -> None:
    """Выводит токены предложения в виде таблицы."""
    print(f"  {'ID':<4} {'FORM':<16} {'HEAD':<5} {'DEPREL':<14} "
          f"{'DEEPSLOT':<14} {'SEMCLASS':<12} MISC")
    print("  " + "-" * 78)
    for tok in sentence:
        print(f"  {tok['id']:<4} {tok['form']:<16} {tok['head']:<5} "
              f"{tok['deprel']:<14} {tok.get('deepslot', '—'):<14} "
              f"{tok.get('semclass', '—'):<12} {tok.get('misc', '—')}")


# ─────────────────────────────── ТЕСТЫ ───────────────────────────────────────

# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(message)s",
#     )
#
#     SEP = "=" * 70
#
#     # ── Проверка доступности сервиса ─────────────────────────────────────────
#     print(f"{SEP}\nПРОВЕРКА MODAL-СЕРВИСА\n{SEP}")
#     try:
#         parser = CobaldParser()
#     except Exception as e:
#         print(f"⚠️ Modal-сервис недоступен: {e}")
#         print("Запустите: modal deploy src/parsers/cobald_modal.py")
#         sys.exit(1)
#
#     test_text  = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
#     test_batch = ["Он думал о море.", "Кот лежал на диване."]
#
#     # ── 1. dict-формат ───────────────────────────────────────────────────────
#     print(f"\n{SEP}\nФОРМАТ: dict\n{SEP}")
#     result_dict = parser.parse_text(test_text, output_format="dict")
#     print(f"Текст: '{test_text}'")
#     print(f"Предложений: {len(result_dict)}\n")
#     for s_idx, sent in enumerate(result_dict, 1):
#         print(f"  Предложение {s_idx} ({len(sent)} токенов):")
#         _print_sentence_table(sent)
#         print()
#
#     if result_dict and result_dict[0]:
#         tok0 = result_dict[0][0]
#         print(f"  Ключи токена : {list(tok0.keys())}")
#         print(f"  Тип id       : {type(tok0['id']).__name__}  (ожидается int)")
#         print(f"\n  CoBaLD-поля первого токена:")
#         print(f"    misc      : {tok0.get('misc', '—')}")
#         print(f"    deepslot  : {tok0.get('deepslot', '—')}")
#         print(f"    semclass  : {tok0.get('semclass', '—')}")
#
#     # ── 2. native-формат ─────────────────────────────────────────────────────
#     print(f"\n{SEP}\nФОРМАТ: native\n{SEP}")
#     result_native = parser.parse_text(test_text, output_format="native")
#     print(f"Предложений: {len(result_native)}\n")
#     for s_idx, sent in enumerate(result_native, 1):
#         print(f"  Предложение {s_idx} ({len(sent)} токенов):")
#         _print_sentence_table(sent)
#         if sent:
#             extra_keys = [k for k in sent[0]
#                           if k not in ("id", "form", "head", "deprel",
#                                        "misc", "deepslot", "semclass")]
#             if extra_keys:
#                 print(f"  Доп. поля native: {extra_keys}")
#         print()
#
#     # ── 3. CoNLL-U формат (из native) ────────────────────────────────────────
#     # ДОБАВЛЕНО: вывод в стандартном CoNLL-U формате.
#     # Данные берутся из result_native — только он содержит lemma/upos/feats/eud.
#     # CoBaLD-специфичные поля (deepslot, semclass) добавляются в MISC-колонку.
#     print(f"\n{SEP}\nФОРМАТ: CoNLL-U (из native)\n{SEP}")
#     conllu_str = _to_conllu_str(result_native)
#     print(conllu_str)
#
#     # ── 4. Проверка типа возврата ─────────────────────────────────────────────
#     print(f"\n{SEP}\nПРОВЕРКА ТИПА ВОЗВРАТА\n{SEP}")
#     for fmt in ("dict", "native"):
#         r = parser.parse_text("Тест.", output_format=fmt)
#         status = "✅" if r is not None else "❌ None!"
#         print(f"  parse_text(format={fmt!r})"
#               f" → type={type(r).__name__}"
#               f" is_none={r is None}"
#               f" {status}")
#
#     # ── 5. Пакетная обработка ─────────────────────────────────────────────────
#     print(f"\n{SEP}\nПАКЕТНАЯ ОБРАБОТКА\n{SEP}")
#     result_batch = parser.parse_batch(test_batch, output_format="dict")
#     print(f"Текстов: {len(test_batch)}, результатов: {len(result_batch)}\n")
#     for t_idx, text_sents in enumerate(result_batch):
#         total = sum(len(s) for s in text_sents)
#         print(f"  [{t_idx}] '{test_batch[t_idx]}'"
#               f" → {len(text_sents)} предл., {total} токенов")
#
#     print(f"\n{SEP}\n✅ Все тесты завершены\n{SEP}")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="CoBaLD wrapper тест")
    ap.add_argument("--output-format", choices=["dict", "native"],
                    default="dict", dest="output_format")
    ap.add_argument("--chunk-size", type=int,
                    default=CobaldParser.SENTENCE_CHUNK_SIZE, dest="chunk_size")

    def _run_tests(args) -> None:
        """
        Тест-секции:
        [1]   Chunking (локально, без Modal)
        [1.1] _split_to_sentence_chunks: правильное число предложений
        [1.2] _split_to_sentence_chunks: chunk_size=1 → каждое предл. отдельно
        [1.3] _merge_chunks: склейка корректна
        [1.4] chunk_size=0 → ValueError
        [1.5] output_format неверный → ValueError
        [2]   parse_text — dict
        [3]   parse_text — native
        [4]   parse_text — CoNLL-U (из native)
        [5]   chunk_size=1 результат совпадает с chunk_size=32
        [6]   parse_text — пустой текст → []
        [7]   parse_batch — dict
        [8]   parse_batch ≡ parse_text × N
        [9]   parse_batch со смешанными пустыми/непустыми текстами
        """
        sep = "=" * 72
        passed = 0
        failed = 0

        def ok(name):
            nonlocal passed; passed += 1; print(f"  ✅ {name}")

        def fail(name, err):
            nonlocal failed; failed += 1; print(f"  ❌ {name}: {err}")

        text_sample  = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
        multi_sample = "Мама мыла раму. Папа читал газету. Кот лежал на диване."
        batch        = [text_sample, "Москва — столица.", "Лиса прыгает через забор."]

        # ── [1] Chunking (без Modal) ─────────────────────────────────────────
        print(f"\n{sep}\n[1] Chunking (локально, без Modal)\n{sep}")

        # [1.1]
        try:
            chunks = CobaldParser._split_to_sentence_chunks(multi_sample, chunk_size=10)
            total_sents = sum(len(c) for c in chunks)
            assert total_sents == 3, f"ожидалось 3 предл., получено {total_sents}"
            ok(f"[1.1] _split_to_sentence_chunks — {total_sents} предл., chunk_size=10")
        except Exception as e:
            fail("[1.1] _split_to_sentence_chunks", e)

        # [1.2]
        try:
            chunks = CobaldParser._split_to_sentence_chunks(multi_sample, chunk_size=1)
            assert len(chunks) == 3, f"ожидалось 3 чанка, получено {len(chunks)}"
            assert all(len(c) == 1 for c in chunks), "каждый чанк должен содержать 1 предл."
            ok("[1.2] _split_to_sentence_chunks chunk_size=1 — по одному предл. в чанке")
        except Exception as e:
            fail("[1.2] chunk_size=1", e)

        # [1.3]
        try:
            fake = [{"id": 1, "form": "слово"}]
            merged = CobaldParser._merge_chunks([[fake, fake], [fake]])
            assert len(merged) == 3, f"ожидалось 3, получено {len(merged)}"
            ok("[1.3] _merge_chunks — склейка корректна")
        except Exception as e:
            fail("[1.3] _merge_chunks", e)

        # [1.4]
        try:
            try:
                CobaldParser._split_to_sentence_chunks("Текст.", chunk_size=0)
                fail("[1.4] chunk_size=0", "ValueError не выброшен")
            except ValueError as exc:
                print(f"  Поймано: {exc!r}")
                ok("[1.4] chunk_size=0 → ValueError")
        except Exception as e:
            fail("[1.4] ValueError chunk_size", e)

        # [1.5] — проверяем валидацию output_format до Modal
        parser_stub = object.__new__(CobaldParser)
        parser_stub.logger = logging.getLogger("test")
        parser_stub.service = None
        try:
            try:
                CobaldParser.parse_text(parser_stub, "Текст.", output_format="conllu")  # type: ignore
                fail("[1.5] output_format=conllu", "ValueError не выброшен")
            except ValueError as exc:
                print(f"  Поймано: {exc!r}")
                ok("[1.5] output_format неверный → ValueError")
        except Exception as e:
            fail("[1.5] output_format ValueError", e)

        # ── Инициализация parser (Modal) ──────────────────────────────────────
        print(f"\n{sep}\nПодключение к Modal...\n{sep}")
        try:
            parser = CobaldParser()
        except Exception as e:
            print(f"\n⚠️ Modal-сервис недоступен: {e}")
            print("Запустите: modal deploy src/parsers/cobald_modal.py")
            total = passed + failed
            print(f"\n── Локальные тесты: {passed} ✅  Modal-тесты: пропущены")
            sys.exit(1)

        # ── [2] parse_text — dict ────────────────────────────────────────────
        print(f"\n{sep}\n[2] parse_text (dict, chunk_size={args.chunk_size})\n{sep}")
        try:
            result = parser.parse_text(text_sample, output_format="dict",
                                        chunk_size=args.chunk_size)
            assert isinstance(result, list) and len(result) > 0
            for sent in result:
                assert len(sent) > 0
                for tok in sent:
                    for k in ("id", "form", "head", "deprel", "misc", "deepslot", "semclass"):
                        assert k in tok, f"ключ {k!r} отсутствует"
                    assert isinstance(tok["id"], int)
            print(f"  Предложений: {len(result)}, токенов в первом: {len(result[0])}")
            _print_sentence_table(result[0])
            tok0 = result[0][0]
            print(f"\n  Ключи токена : {list(tok0.keys())}")
            print(f"  Тип id       : {type(tok0['id']).__name__}  (ожидается int)")
            ok(f"[2] parse_text / dict — {len(result)} предл.")
        except Exception as e:
            fail("[2] parse_text / dict", e)

        # ── [3] parse_text — native ──────────────────────────────────────────
        print(f"\n{sep}\n[3] parse_text (native, chunk_size={args.chunk_size})\n{sep}")
        try:
            result_native = parser.parse_text(text_sample, output_format="native",
                                               chunk_size=args.chunk_size)
            assert isinstance(result_native, list) and len(result_native) > 0
            for sent in result_native:
                for tok in sent:
                    for k in ("id", "form", "lemma", "upos", "xpos", "feats",
                              "head", "deprel", "deps_eud", "misc",
                              "deepslot", "semclass", "is_null"):
                        assert k in tok, f"ключ {k!r} отсутствует в native"
            extra = [k for k in result_native[0][0]
                     if k not in ("id", "form", "head", "deprel", "misc", "deepslot", "semclass")]
            print(f"  Доп. поля native: {extra}")
            ok(f"[3] parse_text / native — {len(result_native)} предл.")
        except Exception as e:
            fail("[3] parse_text / native", e)

        # ── [4] parse_text — CoNLL-U ─────────────────────────────────────────
        print(f"\n{sep}\n[4] parse_text → CoNLL-U\n{sep}")
        try:
            conllu = _to_conllu_str(result_native)
            assert "# sent_id = 1" in conllu, "отсутствует sent_id"
            assert "# text = "    in conllu, "отсутствует text"
            assert "\t" in conllu,           "отсутствует TAB-разделитель"
            print(conllu)
            ok("[4] CoNLL-U — sent_id, text, TAB-колонки присутствуют")
        except Exception as e:
            fail("[4] CoNLL-U", e)

        # ── [5] chunk_size=1 ≡ chunk_size=32 ────────────────────────────────
        print(f"\n{sep}\n[5] parse_text — chunk_size=1 совпадает с chunk_size=32\n{sep}")
        try:
            r1  = parser.parse_text(multi_sample, output_format="dict", chunk_size=1)
            r32 = parser.parse_text(multi_sample, output_format="dict", chunk_size=32)
            assert len(r1) == len(r32), f"len: chunk=1→{len(r1)}, chunk=32→{len(r32)}"
            for s1, s32 in zip(r1, r32):
                f1  = [t["form"] for t in s1]
                f32 = [t["form"] for t in s32]
                assert f1 == f32, f"forms differ: {f1} vs {f32}"
            ok(f"[5] chunk_size=1 ({len(r1)} предл.) ≡ chunk_size=32")
        except Exception as e:
            fail("[5] chunk_size совместимость", e)

        # ── [6] Пустой текст → [] ────────────────────────────────────────────
        print(f"\n{sep}\n[6] parse_text — пустой текст → []\n{sep}")
        try:
            for txt in ("", "   "):
                r = parser.parse_text(txt, output_format="dict")
                assert r == [], f"ожидался [], получено {r!r} для {txt!r}"
            ok("[6] Пустой текст → []")
        except Exception as e:
            fail("[6] Пустой текст", e)

        # ── [7] parse_batch — dict ────────────────────────────────────────────
        print(f"\n{sep}\n[7] parse_batch (dict, {len(batch)} текста)\n{sep}")
        try:
            results = parser.parse_batch(batch, output_format="dict",
                                          chunk_size=args.chunk_size)
            assert len(results) == len(batch), \
                f"ожидалось {len(batch)}, получено {len(results)}"
            for idx, (text, res) in enumerate(zip(batch, results)):
                assert len(res) > 0, f"текст {idx}: пустой результат"
                print(f"  [{idx}] {text!r} → {len(res)} предл., "
                      f"{sum(len(s) for s in res)} токенов")
            ok(f"[7] parse_batch / dict — {len(batch)} текста")
        except Exception as e:
            fail("[7] parse_batch / dict", e)

        # ── [8] parse_batch ≡ parse_text × N ─────────────────────────────────
        print(f"\n{sep}\n[8] parse_batch ≡ parse_text × N (chunk_size=1)\n{sep}")
        try:
            batch_res = parser.parse_batch(batch, output_format="dict", chunk_size=1)
            for i, text in enumerate(batch):
                single = parser.parse_text(text, output_format="dict", chunk_size=1)
                assert len(batch_res[i]) == len(single), \
                    f"текст {i+1}: batch={len(batch_res[i])} vs single={len(single)}"
                for sb, ss in zip(batch_res[i], single):
                    fb = [t["form"] for t in sb]
                    fs = [t["form"] for t in ss]
                    assert fb == fs, f"текст {i+1}: forms differ: {fb} vs {fs}"
            ok(f"[8] parse_batch ≡ parse_text × {len(batch)}")
        except Exception as e:
            fail("[8] parse_batch vs parse_text", e)

        # ── [9] parse_batch со смешанными пустыми текстами ───────────────────
        print(f"\n{sep}\n[9] parse_batch — смешанные пустые/непустые тексты\n{sep}")
        try:
            mixed = ["", "Зло пугает.", ""]
            res = parser.parse_batch(mixed, output_format="dict",
                                      chunk_size=args.chunk_size)
            assert len(res) == 3
            assert res[0] == [], f"текст 0: ожидался [], получено {res[0]!r}"
            assert len(res[1]) > 0, "текст 1: ожидался непустой результат"
            assert res[2] == [], f"текст 2: ожидался [], получено {res[2]!r}"
            ok("[9] parse_batch с пустыми текстами в батче")
        except Exception as e:
            fail("[9] parse_batch пустые тексты", e)

        # ── Итог ──────────────────────────────────────────────────────────────
        total = passed + failed
        print(f"\n{sep}")
        print(f"ИТОГ: {passed}/{total} тестов прошло" +
              (" ✅" if failed == 0 else f" ❌ {failed} упало"))
        print(sep)
        sys.exit(0 if failed == 0 else 1)

    _run_tests(ap.parse_args())

