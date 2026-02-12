import modal
from typing import List, Dict, Any


# Создаём Volume для кеширования моделей DeepPavlov
cache_volume = modal.Volume.from_name("deeppavlov-cache", create_if_missing=True)

# Образ с поддержкой GPU и необходимых лингвистических библиотек
dp_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "transformers",
        "deeppavlov",
        "razdel",
        "pandas",
        "nltk",
        "tqdm")

    .run_commands(
        "python -m deeppavlov install ru_syntagrus_joint_parsing",
        "python -c \"from deeppavlov import build_model;"
        "build_model('ru_syntagrus_joint_parsing', download=True)\""
    )

    .run_commands(
        "python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\""
    )

    .env({
        # Отключаем повторные проверки хешей
        "DEEPPAVLOV_DOWNLOAD_PROGRESSIVE": "0",
    })
)

app = modal.App("booknlp-ru-deeppavlov")


@app.cls(image=dp_image, gpu="T4", timeout=1200)
class DeepPavlovService:

    @modal.enter()
    def enter(self):
        from deeppavlov import build_model, configs
        # Конфигурация joint_parsing обеспечивает SOTA-точность (LAS ~93.4%)
        # и полную разметку морфологических признаков
        self.model = build_model(
            configs.morpho_syntax_parser.ru_syntagrus_joint_parsing,
            download=True
        )

    @modal.method()
    def parse_text(self, text: str) -> List:
        from razdel import tokenize, sentenize

        # 1. Сегментация (Razdel)
        sentences = list(sentenize(text))

        tokenized_sentences = []
        token_spans = []  # Для сохранения символьных смещений

        for sent in sentences:
            tokens = list(tokenize(sent.text))
            tokenized_sentences.append([t.text for t in tokens])
            # Смещения считаем глобально относительно начала исходного текста
            token_spans.append([
                (sent.start + t.start, sent.start + t.stop)
                for t in tokens
            ])

        # 2. Выполнение разбора
        # DeepPavlov возвращает список строк в формате CoNLL-U (10 полей)
        parsed_batch = self.model(tokenized_sentences)

        results = []
        for i, sent_conllu in enumerate(parsed_batch):
            sent_res = []
            # Разбираем CoNLL-U вывод
            lines = [
                l for l in sent_conllu.split('\n')
                if l and not l.startswith('#')
            ]

            for j, line in enumerate(lines):
                fields = line.split('\t')

                # Проверяем, что это не multi-word token
                if '-' in fields[0]:
                    continue

                start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                # ПОЛНЫЙ CoNLL-U формат (10 полей)
                sent_res.append({
                    'id': int(fields[0]),  # ID (1-based)
                    'form': fields[1],  # Словоформа
                    'lemma': fields[2],  # Лемма
                    'upos': fields[3],  # Universal POS
                    'xpos': fields[4],  # Language-specific POS (может быть "_")
                    'feats': fields[5],  # Морфологические признаки
                    'head': int(fields[6]),  # Главное слово
                    'deprel': fields[7],  # Тип связи
                    'deps': fields[8],  # Enhanced dependencies (обычно "_")
                    'misc': fields[9],  # MISC (обычно "_")
                    'startchar': start_c,  # Дополнительно: позиция начала
                    'endchar': end_c  # Дополнительно: позиция конца
                })

            results.append(sent_res)

        return results

    @modal.method()
    def parse_batch(self, texts: List[str]) -> List:
        """Обработка списка документов для повышения эффективности GPU"""
        return [self.parse_text(t) for t in texts]

    @modal.method()
    def parse_text_native(self, text: str) -> List:
        """
        Версия с встроенной токенизацией DeepPavlov (не рекомендуется).
        """
        # DeepPavlov сам токенизирует
        parsed_batch = self.model([text])

        results = []
        for sent_conllu in parsed_batch:
            sent_res = []
            lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

            for line in lines:
                fields = line.split('\t')
                sent_res.append({
                    'id': int(fields[0]),
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': int(fields[6]),
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9]
                })

            results.append(sent_res)

        return results


# Для локального тестирования
@app.local_entrypoint()
def main():
    test_text = "Мама мыла раму."
    print("🚀 Testing DeepPavlov service...")

    service = DeepPavlovService()
    result = service.parse_text.remote(test_text)

    print(f"\nReceived {len(result)} sentence(s)")
    for s_id, sent in enumerate(result, 1):
        print(f"\n--- Sentence {s_id} ---")
        print("ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL")
        for tok in sent:
            print(
                f"{tok['id']}\t{tok['form']}\t{tok['lemma']}\t"
                f"{tok['upos']}\t{tok['xpos']}\t{tok['feats']}\t"
                f"{tok['head']}\t{tok['deprel']}"
            )

    print("\n✅ Test completed!")

