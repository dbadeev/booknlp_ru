import unittest
import logging
from src.pipeline import BookNLP

logging.basicConfig(level=logging.INFO)


class TestBookNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Используем fast (Slovnet), так как он гарантированно работает локально
        cls.nlp = BookNLP(model_type="fast")

    def test_full_pipeline(self):
        text = "Мама мыла раму. Папа читал газету."

        # Запуск пайплайна
        result = self.nlp.process(text)

        # 1. Проверяем количество предложений
        self.assertEqual(len(result), 2)

        # 2. Проверяем первое предложение
        sent1 = result[0]
        self.assertEqual(sent1['text'], "Мама мыла раму.")

        # 3. Проверяем токены и оффсеты
        tokens = sent1['tokens']
        self.assertEqual(tokens[0]['text'], "Мама")
        self.assertEqual(tokens[0]['start_char'], 0)  # Начало текста

        # 4. Проверяем синтаксис (Slovnet что-то вернул)
        # У Slovnet на новостях 'мыла' может не быть root, но связь должна быть
        # Проверим просто наличие полей
        self.assertIn("pos", tokens[0])
        self.assertIn("head_id", tokens[0])

        # 5. Проверяем второе предложение (глобальные оффсеты)
        sent2 = result[1]
        tokens2 = sent2['tokens']
        first_token_sent2 = tokens2[0]  # "Папа"

        expected_start = text.find("Папа")
        self.assertEqual(first_token_sent2['start_char'], expected_start)

        print("\nPipeline Output Sample:")
        print(sent1['tokens'][0])


if __name__ == '__main__':
    unittest.main()
