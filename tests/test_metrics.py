import unittest
from src.evaluation.metrics import MetricsCalculator, MetricsResult
from src.evaluation.alignment import Alignment


class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = MetricsCalculator()

    def test_segmentation_f1(self):
        # Система нашла: [0, 10, 20]
        # Золото: [0, 10, 25] (одно совпадение, одно мимо, одно пропущено)
        sys = [0, 10, 20]
        gold = [0, 10, 25]

        res = self.calc.calc_segmentation_f1(sys, gold)

        # TP=2 (0, 10), FP=1 (20), FN=1 (25)
        # Precision = 2/3, Recall = 2/3, F1 = 0.66...
        self.assertAlmostEqual(res.f1, 0.666, places=2)

    def test_soft_las_calculation(self):
        # Пример:
        # Sys:  Мама(1) <- мыла(2) [nsubj]
        # Gold: Мама(10) <- мыла(20) [nsubj]
        # Выравнивание: 1->10, 2->20

        alignments = [
            Alignment(sys_id=1, gold_id=10, iou=1.0, type='exact'),
            Alignment(sys_id=2, gold_id=20, iou=1.0, type='exact')
        ]

        sys_tokens = [
            {'id': 1, 'head_id': 2, 'rel': 'nsubj'},
            {'id': 2, 'head_id': 0, 'rel': 'root'}
        ]

        # В Gold используем ключи 'head_id' и 'deprel' как в нашем парсере/conllu
        gold_tokens = [
            {'id': 10, 'head_id': 20, 'deprel': 'nsubj'},
            {'id': 20, 'head_id': 0, 'deprel': 'root'}
        ]

        metrics = self.calc.calc_soft_metrics(alignments, sys_tokens, gold_tokens)

        self.assertEqual(metrics['soft_las'], 1.0)
        self.assertEqual(metrics['total_aligned_tokens'], 2)

    def test_soft_las_mismatch(self):
        # Ошибка в голове (Head Mismatch)
        # Sys: Мама(1) <- ROOT(0) (Ошибка! должно быть к 2)
        alignments = [
            Alignment(sys_id=1, gold_id=10, iou=1.0, type='exact'),
            Alignment(sys_id=2, gold_id=20, iou=1.0, type='exact')
        ]
        sys_tokens = [{'id': 1, 'head_id': 0, 'rel': 'root'}]  # Ошибка
        gold_tokens = [{'id': 10, 'head_id': 20, 'deprel': 'nsubj'}]

        # Добавляем инфо про токен 2, чтобы sys_to_gold map сработал,
        # хотя в этом тесте мы проверяем только токен 1
        sys_tokens.append({'id': 2, 'head_id': 0, 'rel': 'root'})
        gold_tokens.append({'id': 20, 'head_id': 0, 'deprel': 'root'})

        metrics = self.calc.calc_soft_metrics(alignments, sys_tokens, gold_tokens)

        # Токен 1: Sys Head=0, Gold Head=20. Mismatch.
        # Токен 2: Sys Head=0, Gold Head=0. Match.
        # Итог: 1 из 2 верно.
        self.assertEqual(metrics['soft_uas'], 0.5)


if __name__ == '__main__':
    unittest.main()