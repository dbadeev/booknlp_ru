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
        self.mystem = Mystem(entire_input=False)
        self.logger.info("Mystem initialized!")

    @modal.method()
    def parse_batch(self, batch_texts: list):
        """
        batch_texts: Список предложений (list[str] или list[list[str]]).
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

                # Возвращаем [sent_res] - список предложений
                results.append([sent_res] if sent_res else [[]])

            except Exception as e:
                self.logger.error(f"Mystem error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                results.append([[]])

        return results


@app.local_entrypoint()
def main():
    test_texts = [
        "Это тестовое предложение.",
        ["Список", "токенов", "для", "теста"],
        "Мама мыла раму."
    ]

    print("🚀 Testing Mystem service...")
    service = MystemService()
    results = service.parse_batch.remote(test_texts)

    for i, doc in enumerate(results):
        print(f"\n📄 Document {i + 1}: {test_texts[i]}")
        if not doc or not doc[0]:
            print("  [Empty result]")
            continue

        sent = doc[0]
        print(f"  Tokens: {len(sent)}")
        for tok in sent:
            print(f"    {tok['id']}\t{tok['form']} -> {tok['lemma']} ({tok.get('upos', 'X')})")

    print("\n✅ Test completed!")

