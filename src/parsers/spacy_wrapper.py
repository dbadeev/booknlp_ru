#!/usr/bin/env python3
"""
Локальная обертка для spaCy парсера.

Использование:
    from spacy_wrapper import SpacyParser, SpacyPymorphy3Parser

    # Базовый парсер (только spaCy)
    parser = SpacyParser()

    # 4 варианта вывода (2 формата × 2 токенизатора):
    result = parser.parse_text(text, output_format="native",  tokenizer="internal")
    result = parser.parse_text(text, output_format="native",  tokenizer="razdel")
    result = parser.parse_text(text, output_format="conllu",  tokenizer="internal")
    result = parser.parse_text(text, output_format="conllu",  tokenizer="razdel")

    # Обогащённый парсер (spaCy + pymorphy3), только native формат
    parser_enriched = SpacyPymorphy3Parser()
    result = parser_enriched.parse_text_enriched(
        text,
        tokenizer="razdel",
        include_lexeme=True,
        include_all_parses=True
    )
"""
import logging
import sys
import modal
from typing import List, Dict, Any, Union, Literal

logger = logging.getLogger(__name__)

OutputFormat = Literal["native", "conllu"]
TokenizerType = Literal["internal", "razdel"]


class SpacyParser:
    """
    Клиент для spaCy парсера, запущенного в Modal.

    Поддерживает:
    - 2 формата вывода: native (полный), conllu (стандарт UD)
    - 2 токенизатора:   internal (spaCy), razdel (внешний)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-spacy",
                "SpacyService"
            )()
            self.logger.info("✓ Connected to SpaCy via Modal.")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Modal: {e}")
            raise

    def parse_text(
        self,
        text: str,
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal"
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Парсит текст с использованием spaCy.

        Args:
            text:          Входной текст для анализа
            output_format: Формат вывода
                - 'native':  полный нативный формат spaCy (ВСЕ поля токена)
                - 'conllu':  строка в стандартном формате CoNLL-U
            tokenizer: Тип токенизатора
                - 'internal': встроенный токенизатор spaCy
                - 'razdel':   внешний токенизатор razdel

        Returns:
            native  → List[Dict]  (предложения со всеми полями spaCy)
            conllu  → str

        Поля токена в native формате:
        ── Позиция ──────────────────────────────────────────
          id, start_char, end_char
        ── Форма ────────────────────────────────────────────
          form, norm, lower, shape
        ── Лемма и POS ──────────────────────────────────────
          lemma, upos, xpos, feats
        ── Синтаксис ────────────────────────────────────────
          head, deprel, n_lefts, n_rights, children
        ── Именованные сущности ─────────────────────────────
          ent_type, ent_iob
        ── Метаданные ───────────────────────────────────────
          is_sent_start, whitespace, misc (SpaceAfter=No)
        ── Лексические флаги ────────────────────────────────
          is_alpha, is_digit, is_punct, is_space,
          is_stop, is_oov, like_num, like_url, like_email
        ── Векторные поля ───────────────────────────────────
          has_vector, cluster
        ── Вероятность ──────────────────────────────────────
        ИСКЛЮЧЕНО ИЗ-ЗА НЕИНФОРМАТИВНОСТИ
        #   prob
        # ── Обогащение pymorphy3 (только SpacyPymorphy3Parser)
        #   pymorphy3_word, pymorphy3_lemma, pymorphy3_tag,
        #   pymorphy3_score, pymorphy3_is_known,
        #   pymorphy3_methods_stack, pymorphy3_normalized,
        #   pymorphy3_lexeme / pymorphy3_lexeme_count,
        #   pymorphy3_all_parses / pymorphy3_parses_count
        """
        try:
            return self.service.parse.remote(
                text,
                output_format=output_format,
                tokenizer=tokenizer
            )
        except Exception as e:
            self.logger.error(f"❌ Error during spaCy parsing: {e}")
            raise

    def parse_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = "native",
        tokenizer: TokenizerType = "internal",
        batch_size: int = 32
    ) -> List[Union[List[Dict[str, Any]], str]]:
        """
        Пакетная обработка текстов.

        Args:
            texts:         Список текстов для анализа
            output_format: Формат вывода
            tokenizer:     Тип токенизатора
            batch_size:    Размер батча (по умолчанию 32)
        """
        try:
            return self.service.parse_batch.remote(
                texts,
                output_format=output_format,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
        except Exception as e:
            self.logger.error(f"❌ Error during batch parsing: {e}")
            raise


# ============================================================
# ИНТЕГРАЦИЯ С PYMORPHY3
# ============================================================
class SpacyPymorphy3Parser:
    """
    Расширенная версия парсера: spaCy (ВСЕ поля) + pymorphy3 (обогащение).

    Только формат native — обогащение не применимо к conllu.

    Поля обогащения pymorphy3:
    ── Базовые (всегда) ─────────────────────────────────────
      pymorphy3_word:         форма в lowercase (p.word)
      pymorphy3_lemma:        нормальная форма (p.normal_form)
      pymorphy3_tag:          полный тег OpenCorpora (str(p.tag))
      pymorphy3_score:        вероятность разбора (p.score)
      pymorphy3_is_known:     слово в словаре (p.is_known)
    ── Расширенные (всегда) ─────────────────────────────────
      pymorphy3_methods_stack: методы разбора (p.methods_stack)
      pymorphy3_normalized:   {word, tag, score} нормализованной формы
    ── Опциональные ─────────────────────────────────────────
      pymorphy3_lexeme:       полная парадигма [{word, tag}, ...]
                              (если include_lexeme=True)
      pymorphy3_lexeme_count: кол-во форм парадигмы
                              (если include_lexeme=False)
      pymorphy3_all_parses:   все разборы [{normal_form, tag, score, is_known}, ...]
                              (если include_all_parses=True и кол-во разборов > 1)
      pymorphy3_parses_count: кол-во возможных разборов
                              (если include_all_parses=True)

    Ключевые отличия spaCy vs pymorphy3:
      form   (spaCy)        ≠  pymorphy3_word   — регистр vs lowercase
      lemma  (spaCy)        ≠  pymorphy3_lemma  — контекст vs первый разбор
      feats  (spaCy/UD)     ≠  pymorphy3_tag    — UD формат vs OpenCorpora
    """

    def __init__(self, spacy_parser: "SpacyParser | None" = None):
        import pymorphy3
        # Принимаем уже созданный SpacyParser или создаём новый.
        # Это предотвращает двойное подключение к Modal при совместном использовании.
        self.spacy_parser = spacy_parser if spacy_parser is not None else SpacyParser()
        self.morph = pymorphy3.MorphAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.logger.info("✓ SpaCy+Pymorphy3 parser initialized.")

    def parse_text_enriched(
        self,
        text: str,
        tokenizer: TokenizerType = "internal",
        include_lexeme: bool = False,
        include_all_parses: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Парсит текст и обогащает каждый токен данными из pymorphy3.

        Args:
            text:               Входной текст
            tokenizer:          Тип токенизатора ('internal' | 'razdel')
            include_lexeme:     Включать полную парадигму слова
            include_all_parses: Включать все варианты разбора (для омонимов)

        Returns:
            List[Dict] — предложения в native формате со всеми полями spaCy
            и дополнительными полями pymorphy3_*
        """
        # Получаем полный native результат от spaCy
        spacy_result = self.spacy_parser.parse_text(
            text,
            output_format="native",
            tokenizer=tokenizer
        )

        # Обогащаем каждый токен данными pymorphy3
        enriched_result = []
        for sent in spacy_result:
            enriched_sent = sent.copy()
            enriched_words = []

            for token_dict in sent.get("words", []):
                form = token_dict.get("form", "")
                all_parses = self.morph.parse(form)
                p = all_parses[0]  # Наиболее вероятный разбор

                token_enriched = token_dict.copy()

                # ── Базовые поля ──────────────────────────────────
                token_enriched.update({
                    "pymorphy3_word":     p.word,          # всегда lowercase!
                    "pymorphy3_lemma":    p.normal_form,
                    "pymorphy3_tag":      str(p.tag),
                    "pymorphy3_score":    p.score,
                    "pymorphy3_is_known": p.is_known,
                })

                # ── Расширенные поля ──────────────────────────────
                token_enriched["pymorphy3_methods_stack"] = p.methods_stack
                token_enriched["pymorphy3_normalized"] = {
                    "word":  p.normalized.word,
                    "tag":   str(p.normalized.tag),
                    "score": p.normalized.score,
                }

                # ── Опциональные поля ─────────────────────────────
                if include_lexeme:
                    token_enriched["pymorphy3_lexeme"] = [
                        {"word": lf.word, "tag": str(lf.tag)}
                        for lf in p.lexeme
                    ]
                else:
                    token_enriched["pymorphy3_lexeme_count"] = len(p.lexeme)

                if include_all_parses:
                    token_enriched["pymorphy3_parses_count"] = len(all_parses)
                    if len(all_parses) > 1:
                        token_enriched["pymorphy3_all_parses"] = [
                            {
                                "normal_form": parse.normal_form,
                                "tag":         str(parse.tag),
                                "score":       parse.score,
                                "is_known":    parse.is_known,
                            }
                            for parse in all_parses
                        ]

                enriched_words.append(token_enriched)

            enriched_sent["words"] = enriched_words
            enriched_result.append(enriched_sent)

        return enriched_result


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЫВОДА
# ============================================================
def _print_token_full(tok: Dict[str, Any], with_pymorphy3: bool = False):
    """Выводит ВСЕ поля токена в читаемом виде."""
    print(f"\n  ── Токен #{tok['id']}: '{tok['form']}' ──────────────────────")

    print(f"  ПОЗИЦИЯ:")
    print(f"    start_char:    {tok['start_char']}")
    print(f"    end_char:      {tok['end_char']}")

    print(f"  ФОРМА:")
    print(f"    form:          {tok['form']}")
    print(f"    norm:          {tok.get('norm', '—')}")
    print(f"    lower:         {tok.get('lower', '—')}")
    print(f"    shape:         {tok.get('shape', '—')}")

    print(f"  ЛЕММА И POS:")
    print(f"    lemma:         {tok['lemma']}")
    print(f"    upos:          {tok['upos']}")
    print(f"    xpos:          {tok['xpos']}")
    print(f"    feats:         {tok['feats']}")

    print(f"  СИНТАКСИС:")
    print(f"    head:          {tok['head']}")
    print(f"    deprel:        {tok['deprel']}")
    print(f"    n_lefts:       {tok.get('n_lefts', '—')}")
    print(f"    n_rights:      {tok.get('n_rights', '—')}")
    print(f"    children:      {tok.get('children', [])}")

    print(f"  СУЩНОСТИ:")
    print(f"    ent_type:      {tok.get('ent_type') or '—'}")
    print(f"    ent_iob:       {tok.get('ent_iob') or '—'}")

    print(f"  МЕТАДАННЫЕ:")
    print(f"    is_sent_start: {tok.get('is_sent_start')}")
    print(f"    whitespace:    '{tok.get('whitespace', '')}'")
    print(f"    misc:          {tok.get('misc', '—')}")

    print(f"  ФЛАГИ:")
    print(f"    is_alpha:      {tok.get('is_alpha')}")
    print(f"    is_digit:      {tok.get('is_digit')}")
    print(f"    is_punct:      {tok.get('is_punct')}")
    print(f"    is_space:      {tok.get('is_space')}")
    print(f"    is_stop:       {tok.get('is_stop')}")
    print(f"    is_oov:        {tok.get('is_oov')}")
    print(f"    like_num:      {tok.get('like_num')}")
    print(f"    like_url:      {tok.get('like_url')}")
    print(f"    like_email:    {tok.get('like_email')}")

    print(f"  ВЕКТОР:")
    print(f"    has_vector:    {tok.get('has_vector')}")
    vn = tok.get('vector_norm')
    print(f"    vector_norm:   {vn if vn is not None else '—'}")

    if with_pymorphy3:
        print(f"\n  🔍 PYMORPHY3:")
        print(f"    word (lower):   {tok.get('pymorphy3_word', '—')}")
        print(f"    lemma:          {tok.get('pymorphy3_lemma', '—')}")
        print(f"    tag (OpenCorp): {tok.get('pymorphy3_tag', '—')}")
        print(f"    score:          {tok.get('pymorphy3_score', 0):.4f}")
        print(f"    is_known:       {tok.get('pymorphy3_is_known', '—')}")
        print(f"    methods_stack:  {tok.get('pymorphy3_methods_stack', '—')}")

        normalized = tok.get('pymorphy3_normalized', {})
        if normalized:
            print(f"    normalized:")
            print(f"      word:  {normalized.get('word')}")
            print(f"      tag:   {normalized.get('tag')}")
            print(f"      score: {normalized.get('score', 0):.4f}")

        if 'pymorphy3_lexeme' in tok:
            lexeme = tok['pymorphy3_lexeme']
            print(f"    lexeme ({len(lexeme)} форм):")
            for lf in lexeme[:5]:
                print(f"      {lf['word']:<15} [{lf['tag']}]")
            if len(lexeme) > 5:
                print(f"      ... (еще {len(lexeme)-5} форм)")
        else:
            print(f"    lexeme_count:   {tok.get('pymorphy3_lexeme_count', '—')}")

        if 'pymorphy3_all_parses' in tok:
            parses = tok['pymorphy3_all_parses']
            print(f"    all_parses ({tok.get('pymorphy3_parses_count')}):")
            for i, parse in enumerate(parses, 1):
                print(f"      {i}. {parse['normal_form']:<12} "
                      f"[{parse['tag']:<30}] score={parse['score']:.4f}")
        elif 'pymorphy3_parses_count' in tok:
            print(f"    parses_count:   {tok.get('pymorphy3_parses_count')}")

        # Ключевые расхождения
        if tok.get('form') != tok.get('pymorphy3_word'):
            print(f"\n    ⚠️  РЕГИСТР: '{tok['form']}' vs '{tok['pymorphy3_word']}'")
        if tok.get('lemma') != tok.get('pymorphy3_lemma'):
            print(f"    ⚠️  ЛЕММА:   spaCy='{tok['lemma']}' vs "
                  f"pymorphy3='{tok['pymorphy3_lemma']}'")


# ============================================================
# ТЕСТЫ
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ── Проверка доступности Modal ──────────────────────────────────────────
    print("=" * 80)
    print("ПРОВЕРКА ДОСТУПНОСТИ MODAL-СЕРВИСА")
    print("=" * 80)
    try:
        parser = SpacyParser()
    except Exception as e:
        print(f"⚠️  Modal-сервис недоступен: {e}")
        print("\nЗапустите сервис командой:")
        print("  modal deploy src/parsers/spacy_modal.py")
        print("или для разработки:")
        print("  modal serve src/parsers/spacy_modal.py")
        sys.exit(1)

    # parser_enriched = SpacyPymorphy3Parser()
    parser_enriched = SpacyPymorphy3Parser(spacy_parser=parser)  # ← переиспользует

    # Тестовые тексты
    test_short   = "Кружка-термос стоит 500р."
    test_ner     = "Москва - столица России."
    test_ambig   = "мой брат"
    test_complex = "Зло, которым пугаешь, не так зло."

    # ========================================================================
    # ВАРИАНТ 1: native + internal tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("ВАРИАНТ 1: NATIVE + INTERNAL TOKENIZER")
    print("=" * 80)

    result_1 = parser.parse_text(
        test_short,
        output_format="native",
        tokenizer="internal"
    )

    print(f"\nТекст: '{test_short}'")
    for sent_data in result_1:
        print(f"\nПредложение: '{sent_data['text']}'")
        if sent_data.get("entities"):
            print(f"Сущности: {[(e['text'], e['label']) for e in sent_data['entities']]}")
        for tok in sent_data["words"]:
            _print_token_full(tok, with_pymorphy3=False)

    # ========================================================================
    # ВАРИАНТ 2: native + razdel tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("ВАРИАНТ 2: NATIVE + RAZDEL TOKENIZER")
    print("=" * 80)

    result_2 = parser.parse_text(
        test_short,
        output_format="native",
        tokenizer="razdel"
    )

    print(f"\nТекст: '{test_short}'")
    print("\n⚡ Сравнение токенизаторов:")
    internal_toks = [w['form'] for s in result_1 for w in s['words']]
    razdel_toks   = [w['form'] for s in result_2 for w in s['words']]
    print(f"  Internal: {internal_toks}")
    print(f"  Razdel:   {razdel_toks}")
    if internal_toks != razdel_toks:
        print("  ⚠️  ТОКЕНИЗАТОРЫ ДАЮТ РАЗНЫЕ РЕЗУЛЬТАТЫ!")

    print(f"\nВсе поля токенов (razdel):")
    for sent_data in result_2:
        for tok in sent_data["words"]:
            _print_token_full(tok, with_pymorphy3=False)

    # ========================================================================
    # ВАРИАНТ 3: conllu + internal tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("ВАРИАНТ 3: CONLL-U + INTERNAL TOKENIZER")
    print("=" * 80)

    result_3 = parser.parse_text(
        test_complex,
        output_format="conllu",
        tokenizer="internal"
    )
    print(f"\nТекст: '{test_complex}'")
    print(result_3)

    # ========================================================================
    # ВАРИАНТ 4: conllu + razdel tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("ВАРИАНТ 4: CONLL-U + RAZDEL TOKENIZER")
    print("=" * 80)

    result_4 = parser.parse_text(
        test_complex,
        output_format="conllu",
        tokenizer="razdel"
    )
    print(f"\nТекст: '{test_complex}'")
    print(result_4)

    # ========================================================================
    # ВАРИАНТ 1+2 С ОБОГАЩЕНИЕМ PYMORPHY3 (internal tokenizer)
    # ========================================================================
    print("\n" + "=" * 80)
    print("NATIVE + INTERNAL + PYMORPHY3 ОБОГАЩЕНИЕ")
    print("=" * 80)

    enriched_internal = parser_enriched.parse_text_enriched(
        test_ambig,
        tokenizer="internal",
        include_lexeme=False,
        include_all_parses=True
    )

    print(f"\nТекст: '{test_ambig}' (тест на омонимы)")
    for sent_data in enriched_internal:
        for tok in sent_data["words"]:
            _print_token_full(tok, with_pymorphy3=True)

    # ========================================================================
    # ВАРИАНТ 1+2 С ОБОГАЩЕНИЕМ PYMORPHY3 (razdel tokenizer)
    # ========================================================================
    print("\n" + "=" * 80)
    print("NATIVE + RAZDEL + PYMORPHY3 ОБОГАЩЕНИЕ")
    print("=" * 80)

    enriched_razdel = parser_enriched.parse_text_enriched(
        test_ner,
        tokenizer="razdel",
        include_lexeme=True,
        include_all_parses=False
    )

    print(f"\nТекст: '{test_ner}' (тест NER + парадигма)")
    for sent_data in enriched_razdel:
        if sent_data.get("entities"):
            print(f"\nСущности:")
            for ent in sent_data["entities"]:
                print(f"  - '{ent['text']}' [{ent['label']}] "
                      f"chars:[{ent['start_char']},{ent['end_char']}]")
        for tok in sent_data["words"]:
            _print_token_full(tok, with_pymorphy3=True)

    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)
    print("\n4 варианта вывода:")
    print("  1. native  + internal  — все поля spaCy")
    print("  2. native  + razdel    — все поля spaCy, лучший токенизатор")
    print("  3. conllu  + internal  — стандартный CoNLL-U")
    print("  4. conllu  + razdel    — стандартный CoNLL-U, лучший токенизатор")
    print("\n+ обогащение pymorphy3 (только к native):")
    print("  - базовые:    word, lemma, tag, score, is_known")
    print("  - расширенные: methods_stack, normalized")
    print("  - опциональные: lexeme / lexeme_count, all_parses / parses_count")
