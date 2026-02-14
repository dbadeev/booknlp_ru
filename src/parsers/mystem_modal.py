import modal
import logging
import re

# Образ: Python + pymystem3
image = (
    modal.Image.debian_slim()
    .pip_install("pymystem3")
    # Предзагрузка бинарника Mystem
    .run_commands("python -c 'from pymystem3 import Mystem; Mystem()'")
)

app = modal.App("booknlp-ru-mystem")

# Маппинг Mystem POS -> Universal Dependencies UPOS
MYSTEM_TO_UPOS = {
    'S': 'NOUN', 'A': 'ADJ', 'V': 'VERB', 'ADV': 'ADV',
    'SPRO': 'PRON', 'PR': 'ADP', 'CONJ': 'CCONJ',
    'PART': 'PART', 'INTJ': 'INTJ', 'NUM': 'NUM',
    'COM': 'X', 'APRO': 'DET', 'ANUM': 'ADJ', 'ADVPRO': 'ADV'
}


@app.cls(image=image, timeout=600)
class MystemService:

    @modal.enter()
    def setup(self):
        from pymystem3 import Mystem
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MystemService")
        # entire_input=False убирает лишние пробелы из вывода
        # С disambiguation=False возвращает все варианты БЕЗ учета контекста, отсортированные по частотности в корпусе
        # С disambiguation=True возвращает все варианты С учетом контекста
        self.mystem = Mystem(entire_input=False, disambiguation=True)
        self.logger.info("Mystem initialized!")

    @modal.method()
    def parse_batch(self, batch_texts: list, output_format: str = "simplified"):
        """
        batch_texts: Список предложений (list[str] или list[list[str]]).
        output_format: Формат выхода - "simplified" (текущий) или "native" (нативный формат Mystem).
        Возвращает список документов: List[List[List[Dict]]].
        """
        results = []

        for text_obj in batch_texts:
            try:
                # Нормализация входа
                if isinstance(text_obj, list):
                    text = " ".join([str(t) for t in text_obj])
                elif isinstance(text_obj, str):
                    text = text_obj
                else:
                    text = str(text_obj) if text_obj else ""

                if not isinstance(text, str):
                    self.logger.error(f"Text is not string: {type(text)}")
                    results.append([[]])
                    continue

                if not text.strip():
                    results.append([[]])
                    continue

                # Mystem анализ
                analysis = self.mystem.analyze(text)

                # ============================================================
                # БЛОК: Выбор формата выхода в зависимости от параметра
                # ============================================================
                if output_format == "native":
                    # Нативный формат: возвращаем полную структуру JSON от Mystem
                    sent_res = self._process_native(analysis)
                else:
                    # Упрощенный формат (текущая логика): возвращаем токены с базовыми полями
                    sent_res = self._process_simplified(analysis)

                # Возвращаем [sent_res] - список предложений
                results.append([sent_res] if sent_res else [[]])

            except Exception as e:
                self.logger.error(f"Mystem error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                results.append([[]])

        return results

    # ============================================================
    # БЛОК: Подготовка нативного выхода модели Mystem
    # ============================================================
    def _process_native(self, analysis: list) -> list:
        """
    Подготавливает нативный выход модели Mystem.

    Возвращает полную структуру JSON, которую отдает Mystem:
    - text: исходный токен
    - analysis: список омонимов (гипотез разбора)
      - lex: лемма
      - gr: полная грамматическая строка (например, "S,жен,неод=вин,ед")
      - wt: вес (вероятность) гипотезы
      - qual: маркер качества (ОПЦИОНАЛЬНОЕ поле, появляется ТОЛЬКО для несловарных слов)
              Возможные значения: "bastard" (неизвестное слово), "sob", "prefixoid"
              Для обычных словарных слов поле отсутствует

        Аргументы:
            analysis (list): Нативный вывод от mystem.analyze()

        Возвращает:
            list: Список токенов с полной нативной структурой
        """
        sent_res = []

        for i, item in enumerate(analysis):
            token_text = item.get('text', '')

            # Пропускаем пустые токены и чистые пробелы
            if not token_text:
                continue

            token_clean = token_text.strip()
            if not token_clean and token_text:
                # Это пробел или whitespace - пропускаем
                continue

            # Используем очищенную версию
            token_text = token_clean

            # ============================================================
            # Сохраняем полную нативную структуру Mystem
            # ============================================================
            native_token = {
                "id": len(sent_res) + 1,  # ID добавляем для удобства (не является нативным полем)
                "text": token_text,  # Исходный токен
                "analysis": item.get('analysis', [])  # Список всех гипотез разбора (омонимов)
            }

            sent_res.append(native_token)

        return sent_res

    # ============================================================
    # БЛОК: Упрощенный формат (текущая логика без изменений)
    # ============================================================
    def _process_simplified(self, analysis: list) -> list:
        """
        Подготавливает упрощенный выход (текущий формат).

        Возвращает токены с базовыми полями: id, form, lemma, upos.
        Берется только первая гипотеза из analysis.

        Аргументы:
            analysis (list): Нативный вывод от mystem.analyze()

        Возвращает:
            list: Список токенов с упрощенными полями
        """
        sent_res = []

        # ===== ИСПРАВЛЕНО: ОБРАБОТКА ПУНКТУАЦИИ =====
        for i, item in enumerate(analysis):
            token_text = item.get('text', '')

            # НОВОЕ: НЕ пропускаем пустые строки сразу
            # Проверяем, что это не просто пробел
            if not token_text:
                continue

            # ИСПРАВЛЕНО: strip() может удалить значимую пунктуацию
            # Сохраняем оригинальный текст, если он не пустой после strip
            token_clean = token_text.strip()
            if not token_clean and token_text:
                # Это пробел или whitespace - пропускаем
                continue

            # Используем очищенную версию
            token_text = token_clean
            # ===== КОНЕЦ ИСПРАВЛЕНИЯ =====

            lemma = token_text.lower()
            upos = "X"

            # Морфологический анализ
            if 'analysis' in item and item['analysis']:
                lex_entry = item['analysis'][0]
                lemma = lex_entry.get('lex', token_text.lower())
                gr_full = lex_entry.get('gr', '')

                # Извлекаем POS из грамматики
                gr_pos = re.split('[,=]', gr_full)[0]
                upos = MYSTEM_TO_UPOS.get(gr_pos, 'X')

            # НОВОЕ: Специальная обработка пунктуации
            if token_text in '.!?,;:—–-"«»()[]{}':
                upos = 'PUNCT'

            sent_res.append({
                "id": len(sent_res) + 1,  # ИСПРАВЛЕНО: нумерация с 1
                "form": token_text,
                "lemma": lemma,
                "upos": upos
            })

        return sent_res


@app.local_entrypoint()
def main():
    test_texts = [
        "Это тестовое предложение.",
        ["Список", "токенов", "для", "теста"],
        "Мама мыла раму."
    ]

    print("🚀 Testing Mystem service...")
    service = MystemService()

    # ============================================================
    # Демонстрация работы в упрощенном формате (по умолчанию)
    # ============================================================
    print("\n" + "=" * 60)
    print("УПРОЩЕННЫЙ ФОРМАТ (simplified):")
    print("=" * 60)
    results = service.parse_batch.remote(test_texts, output_format="simplified")

    for i, doc in enumerate(results):
        print(f"\n📄 Document {i + 1}: {test_texts[i]}")
        if not doc or not doc[0]:
            print("  [Empty result]")
            continue

        sent = doc[0]
        print(f"  Tokens: {len(sent)}")
        for tok in sent:
            print(f"  {tok['id']}\t{tok['form']} -> {tok['lemma']} ({tok.get('upos', 'X')})")

    # ============================================================
    # Демонстрация работы в нативном формате
    # ============================================================
    print("\n" + "=" * 60)
    print("НАТИВНЫЙ ФОРМАТ (native):")
    print("=" * 60)
    results_native = service.parse_batch.remote(test_texts[:1], output_format="native")

    for i, doc in enumerate(results_native):
        print(f"\n📄 Document {i + 1}: {test_texts[i]}")
        if not doc or not doc[0]:
            print("  [Empty result]")
            continue

        sent = doc[0]
        print(f"  Tokens: {len(sent)}")
        for tok in sent[:3]:  # Показываем первые 3 токена
            print(f"  Token: {tok['text']}")
            print(f"    Analysis variants: {len(tok['analysis'])}")
            for j, variant in enumerate(tok['analysis'][:2]):  # Первые 2 гипотезы
                print(f"      [{j+1}] lex={variant.get('lex')}, gr={variant.get('gr')}, wt={variant.get('wt')}")

    print("\n✅ Test completed!")


