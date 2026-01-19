import unittest
from src.evaluation.alignment import FuzzyAligner


class TestFuzzyAligner(unittest.TestCase):
    def setUp(self):
        self.aligner = FuzzyAligner()

    def test_exact_match(self):
        # Идеальное совпадение
        sys = [{'id': 1, 'start_char': 0, 'end_char': 4}]  # "Мама"
        gold = [{'id': 10, 'start_char': 0, 'end_char': 4}]

        res = self.aligner.align(sys, gold)
        self.assertEqual(res[0].gold_id, 10)
        self.assertEqual(res[0].type, 'exact')

    def test_merge_error(self):
        # Система: "потомучто" (0-9)
        # Gold: "потому" (0-6), "что" (6-9)
        # Это "Merge Error" со стороны системы (она склеила)

        sys = [{'id': 1, 'start_char': 0, 'end_char': 9}]
        gold = [
            {'id': 10, 'start_char': 0, 'end_char': 6},
            {'id': 11, 'start_char': 6, 'end_char': 9}
        ]

        res = self.aligner.align(sys, gold)
        # Система (1) должна найти пересечение с обоими, но align вернет топ-1 (самое большое перекрытие)
        # "потому" (6 символов) > "что" (3 символа)

        self.assertEqual(res[0].gold_id, 10)  # Должен привязаться к большей части

    def test_gold_offset_reconstruction(self):
        text = "Мама мыла раму"
        tokens = [{'id': 1, 'form': "Мама"}, {'id': 2, 'form': "раму"}]  # "мыла" пропущено намеренно для теста

        rich = self.aligner.reconstruct_gold_offsets(tokens, text)

        self.assertEqual(rich[0]['start_char'], 0)
        self.assertEqual(rich[1]['start_char'], 10)  # "раму" начинается с 10-го индекса


if __name__ == '__main__':
    unittest.main()
