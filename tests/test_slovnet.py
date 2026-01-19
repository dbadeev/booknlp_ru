import unittest
import logging
from src.parsers.slovnet_parser import SlovnetParser

# Отключаем лишние логи для теста
logging.basicConfig(level=logging.INFO)


class TestSlovnetParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Initializing Slovnet model (this may take a few seconds)...")
        cls.parser = SlovnetParser()

    def test_simple_dependency(self):
        # Тест: Мама мыла раму.
        tokens = ["Мама", "мыла", "раму", "."]
        result = self.parser.parse(tokens)

        print("\nParsed Result:")
        for t in result:
            print(f"{t['id']}: {t['form']} --({t['deprel']})--> {t['head']}")

        # Проверки
        self.assertEqual(len(result), 4)

        # Находим "мыла" (должен быть root или predicate)
        myla = next(t for t in result if t['form'] == "мыла")

        # Находим "раму" (должен быть obj / comp у "мыла")
        ramu = next(t for t in result if t['form'] == "раму")

        # Проверяем связь: head у "раму" должен указывать на id "мыла"
        self.assertEqual(ramu['head'], myla['id'])

    def test_batch_processing(self):
        batch = [
            ["Привет", "мир"],
            ["Как", "дела", "?"]
        ]
        results = self.parser.parse_batch(batch)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[1]), 3)


if __name__ == '__main__':
    unittest.main()
