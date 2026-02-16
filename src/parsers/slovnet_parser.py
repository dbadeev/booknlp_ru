import logging
from pathlib import Path
from navec import Navec
from slovnet import Syntax
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class SlovnetParser:

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.navec_path = self.models_dir / "navec_news_v1_1B_250K_300d_100q.tar"
        self.slovnet_path = self.models_dir / "slovnet_syntax_news_v1.tar"
        self._load_models()

    def _load_models(self):
        if not self.navec_path.exists() or not self.slovnet_path.exists():
            raise FileNotFoundError(
                f"Models not found in {self.models_dir}. "
                "Run 'python scripts/download_models.py' first."
            )

        logger.info("Loading Navec embeddings...")
        self.navec = Navec.load(self.navec_path)

        logger.info("Loading Slovnet syntax model...")
        self.syntax = Syntax.load(self.slovnet_path)
        self.syntax.navec(self.navec)

        logger.info("Slovnet loaded successfully.")

    # ========== ДОПОЛНЕНИЕ: ПАРАМЕТР native_format ==========
    def parse(self, tokens: List[str], native_format: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Парсинг одного предложения.

        Параметры:
        ----------
        tokens : List[str]
            Список токенов предложения
        native_format : bool, optional (default=False)
            Если False - возвращает текущий упрощенный формат (список словарей)
            Если True - возвращает нативный формат Slovnet со структурой Doc

        Возвращает:
        -----------
        Если native_format=False: List[Dict[str, Any]]
            Текущий формат - список словарей с полями {id, form, lemma, upos, xpos,
                                                       feats, head, deprel, deps, misc}

        Если native_format=True: Dict[str, Any]
            Нативный формат Slovnet - структура Doc с полями:
            - "tokens": список токенов с полями {id, text, pos, feats, head_id, rel}
            - "spans": список именованных сущностей и фактов (если NER был включен)
              каждый span содержит: {start, stop, type, text, normal, fact}

        ВАЖНО:
        ------
        Согласно анализу, нативный вывод Slovnet (Natasha) - это объект Doc,
        содержащий списки tokens и spans. Spans хранят NER информацию и факты,
        которые теряются при конвертации в пословный формат.
        """
        if not tokens:
            return [] if not native_format else {"tokens": [], "spans": []}

        # Инференс: передаем токены напрямую
        markup = self.syntax(tokens)

        # ========== БЛОК ВЫБОРА ФОРМАТА ВЫВОДА ==========
        # Если native_format=True, возвращаем нативный формат Slovnet
        # Если native_format=False (по умолчанию), возвращаем текущий упрощенный формат
        if native_format:
            return self._format_output_native(markup)
        else:
            return self._format_output_simplified(markup)
        # ================================================

    # ========== ДОПОЛНЕНИЕ: ПАРАМЕТР native_format ДЛЯ БАТЧА ==========
    def parse_batch(self, batch_tokens: List[List[str]], native_format: bool = False) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Парсинг батча предложений.

        Параметры:
        ----------
        batch_tokens : List[List[str]]
            Список предложений, каждое предложение - список токенов
        native_format : bool, optional (default=False)
            Если False - возвращает текущий упрощенный формат
            Если True - возвращает нативный формат Slovnet

        Возвращает:
        -----------
        Если native_format=False: List[List[Dict[str, Any]]]
            Список предложений, каждое - список словарей токенов

        Если native_format=True: List[Dict[str, Any]]
            Список предложений в нативном формате Doc
        """
        if not batch_tokens:
            return []

        batch_results = []

        # Используем .map() для обработки списка списков (батча)
        for sent_markup in self.syntax.map(batch_tokens):
            # ========== БЛОК ВЫБОРА ФОРМАТА ВЫВОДА ==========
            if native_format:
                batch_results.append(self._format_output_native(sent_markup))
            else:
                batch_results.append(self._format_output_simplified(sent_markup))
            # ================================================

        return batch_results
    # ==================================================================

    def _format_output_simplified(self, markup) -> List[Dict[str, Any]]:
        """
        Вспомогательный метод для конвертации в текущий упрощенный формат.
        Используется обоими методами выше, чтобы избежать дублирования кода.
        """
        parsed_tokens = []

        for token in markup.tokens:
            # Безопасное преобразование head_id
            head_id = int(token.head_id) if token.head_id and token.head_id.isdigit() else 0

            # Безопасное извлечение атрибутов (pos и feats отсутствуют в синтаксической модели)
            # Используем getattr с дефолтным значением "_"
            pos = getattr(token, "pos", "_")
            feats = getattr(token, "feats", "_")
            rel = getattr(token, "rel", "_")

            token_data = {
                "id": int(token.id),
                "form": token.text,
                "lemma": "_",
                "upos": pos if pos else "_",
                "xpos": "_",
                "feats": feats if feats else "_",
                "head": head_id,
                "deprel": rel if rel else "_",
                "deps": "_",
                "misc": "_"
            }
            parsed_tokens.append(token_data)

        return parsed_tokens

    # ========== БЛОК ПОДГОТОВКИ НАТИВНОГО ВЫХОДА МОДЕЛИ ==========
    def _format_output_native(self, markup) -> Dict[str, Any]:
        """
        Конвертирует объект Slovnet Markup в максимально полный нативный формат.

        Согласно анализу, нативный вывод Slovnet (Natasha) - это объект Doc,
        содержащий:
        - tokens: список токенов с атрибутами (id, text, pos, feats, head_id, rel)
        - spans: список именованных сущностей и фактов (start, stop, type, text, normal, fact)

        ВАЖНАЯ ПОТЕРЯ при конвертации в упрощенный формат:
        - doc.spans содержит NER информацию (персоны, локации, даты)
        - span.fact содержит структурированные данные (Name(first='...', last='...'))
        - span.normal содержит нормализованную форму сущности (лемму фразы)

        Эти данные полностью теряются при конвертации в пословный формат,
        так как spans привязаны к диапазонам токенов, а не к отдельным словам.

        Источник: Analiz-vykhodnykh-dannykh-modelei-NLP.docx, раздел "5. Slovnet (Natasha)"
        """
        native_doc = {
            "tokens": [],
            "spans": []
        }

        # ========== ИЗВЛЕЧЕНИЕ ТОКЕНОВ ==========
        # Сохраняем все нативные поля токенов
        for token in markup.tokens:
            token_data = {
                "id": int(token.id),
                "text": token.text,  # В нативном формате используется "text", а не "form"
            }

            # Извлекаем все доступные атрибуты токена
            # pos - часть речи (может отсутствовать в синтаксической модели)
            if hasattr(token, "pos") and token.pos:
                token_data["pos"] = token.pos

            # feats - морфологические признаки
            if hasattr(token, "feats") and token.feats:
                token_data["feats"] = token.feats

            # head_id - индекс главного слова
            if hasattr(token, "head_id") and token.head_id:
                # В нативном формате head_id это строка, сохраняем как есть
                token_data["head_id"] = token.head_id

            # rel - тип зависимости (deprel)
            if hasattr(token, "rel") and token.rel:
                token_data["rel"] = token.rel

            native_doc["tokens"].append(token_data)

        # ========== ИЗВЛЕЧЕНИЕ SPANS (NER И ФАКТЫ) ==========
        # Spans содержат именованные сущности и извлеченные факты
        # ВАЖНО: Spans доступны только если использовался NER процессор Slovnet
        # В данном случае используется только Syntax модель, поэтому spans будет пустым
        # Но структура предусмотрена для совместимости с полным функционалом Slovnet
        if hasattr(markup, "spans") and markup.spans:
            for span in markup.spans:
                span_data = {
                    "start": span.start,  # Индекс начального токена
                    "stop": span.stop,    # Индекс конечного токена
                    "type": span.type,    # Тип сущности (PER, LOC, ORG и т.д.)
                }

                # text - исходный текст сущности
                if hasattr(span, "text") and span.text:
                    span_data["text"] = span.text

                # normal - нормализованная форма сущности (лемма фразы)
                if hasattr(span, "normal") and span.normal:
                    span_data["normal"] = span.normal

                # fact - структурированные данные факта
                # Например: Name(first='Александр', last='Пушкин')
                if hasattr(span, "fact") and span.fact:
                    # Конвертируем объект fact в словарь
                    span_data["fact"] = self._fact_to_dict(span.fact)

                native_doc["spans"].append(span_data)

        return native_doc
    # ==============================================================

    def _fact_to_dict(self, fact) -> Dict[str, Any]:
        """
        Вспомогательный метод для конвертации объекта Fact в словарь.
        Факты в Slovnet представлены как именованные кортежи с атрибутами.
        """
        if fact is None:
            return {}

        # Преобразуем все атрибуты факта в словарь
        fact_dict = {}
        if hasattr(fact, "_asdict"):
            # Если это именованный кортеж (namedtuple)
            fact_dict = fact._asdict()
        else:
            # Иначе пытаемся извлечь атрибуты через __dict__
            fact_dict = vars(fact) if hasattr(fact, "__dict__") else {}

        return fact_dict


# ============================================================
# БЛОК: Тестовые примеры использования парсера
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Инициализируем парсер
    parser = SlovnetParser(models_dir="models")

    # Тестовые данные
    test_tokens = ["Александр", "Сергеевич", "Пушкин", "родился", "в", "Москве", "."]

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("=" * 70)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 70)

    result = parser.parse(test_tokens, native_format=False)

    print("\nSlovnet Test:")
    for tok in result:
        print(f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t"
              f"{tok.get('upos')}\t{tok.get('head')}\t{tok.get('deprel')}")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 70)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 70)

    result_native = parser.parse(test_tokens, native_format=True)

    print("\nSlovnet Test (Native):")
    print(f"\nКоличество токенов: {len(result_native['tokens'])}")
    print(f"Количество spans (NER): {len(result_native['spans'])}")

    print("\nТокены в нативном формате:")
    for tok in result_native["tokens"]:
        print(f"\nToken: {tok.get('text')}")
        print(f"  id: {tok.get('id')}")

        # Показываем только присутствующие поля
        if 'pos' in tok:
            print(f"  pos: {tok.get('pos')}")
        else:
            print(f"  pos: не доступен (синтаксическая модель)")

        if 'feats' in tok:
            print(f"  feats: {tok.get('feats')}")
        else:
            print(f"  feats: не доступен (синтаксическая модель)")

        if 'head_id' in tok:
            print(f"  head_id: {tok.get('head_id')}")

        if 'rel' in tok:
            print(f"  rel: {tok.get('rel')}")

    # ============================================================
    # Проверка всех ключей в нативном формате
    # ============================================================
    print("\n" + "=" * 70)
    print("СТРУКТУРА НАТИВНОГО ФОРМАТА:")
    print("=" * 70)

    print(f"\nКлючи Doc: {list(result_native.keys())}")

    if result_native["tokens"]:
        first_token = result_native["tokens"][0]
        print(f"\nКлючи первого токена: {list(first_token.keys())}")
        print(f"Значения:")
        for key, value in first_token.items():
            print(f"  {key}: {value}")

    if result_native["spans"]:
        print(f"\nПервый span:")
        first_span = result_native["spans"][0]
        print(f"Ключи span: {list(first_span.keys())}")
        for key, value in first_span.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nSpans: пусто (NER не использовался)")
        print("Примечание: для получения spans нужно использовать NER модель Slovnet")
        print("Текущая синтаксическая модель не извлекает именованные сущности")

    # ============================================================
    # Тест батч-обработки
    # ============================================================
    print("\n" + "=" * 70)
    print("ТЕСТ БАТЧ-ОБРАБОТКИ:")
    print("=" * 70)

    batch_tokens = [
        ["Москва", "-", "столица", "России", "."],
        ["Пушкин", "великий", "поэт", "."]
    ]

    batch_result = parser.parse_batch(batch_tokens, native_format=False)
    print(f"\nУпрощенный формат - обработано предложений: {len(batch_result)}")

    batch_result_native = parser.parse_batch(batch_tokens, native_format=True)
    print(f"Нативный формат - обработано предложений: {len(batch_result_native)}")
    print(f"Первое предложение содержит токенов: {len(batch_result_native[0]['tokens'])}")
    print(f"Второе предложение содержит токенов: {len(batch_result_native[1]['tokens'])}")

