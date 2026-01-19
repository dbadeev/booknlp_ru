import logging
from typing import List, Set, Tuple, Dict, Any
from dataclasses import dataclass
from src.evaluation.alignment import Alignment

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    precision: float
    recall: float
    f1: float


class MetricsCalculator:
    """
    Вычисляет метрики качества сегментации и синтаксического парсинга
    с учетом нечеткого выравнивания (Fuzzy Alignment).
    """

    @staticmethod
    def calc_segmentation_f1(sys_offsets: List[int], gold_offsets: List[int]) -> MetricsResult:
        """
        Считает F1 по начальным позициям (start_char) предложений или токенов.
        """
        sys_set = set(sys_offsets)
        gold_set = set(gold_offsets)

        tp = len(sys_set & gold_set)  # True Positives (совпали)
        fp = len(sys_set - gold_set)  # False Positives (лишние в системе)
        fn = len(gold_set - sys_set)  # False Negatives (пропущенные в системе)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricsResult(precision, recall, f1)

    def calc_soft_metrics(
            self,
            alignments: List[Alignment],
            sys_tokens: List[Dict[str, Any]],
            gold_tokens: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Считает Soft UAS (Unlabeled) и Soft LAS (Labeled Attachment Score).
        Использует карту выравнивания, чтобы понять, соответствуют ли Head-ы друг другу.
        """
        # 1. Индексируем токены и выравнивания
        # map: sys_id -> gold_id (только для 'exact' или 'partial' совпадений)
        sys_to_gold_map = {}
        for alg in alignments:
            if alg.sys_id is not None and alg.gold_id is not None:
                # В MVP считаем совпадением любое выравнивание с IoU > 0
                sys_to_gold_map[alg.sys_id] = alg.gold_id

        # Индекс токенов для быстрого доступа к head и deprel
        sys_dict = {t['id']: t for t in sys_tokens}
        gold_dict = {t['id']: t for t in gold_tokens}

        correct_uas = 0
        correct_las = 0
        total_aligned = 0

        for alg in alignments:
            # Считаем метрики только для токенов, которые удалось выровнять
            if alg.sys_id is None or alg.gold_id is None:
                continue

            total_aligned += 1

            s_tok = sys_dict[alg.sys_id]
            g_tok = gold_dict[alg.gold_id]

            # --- Проверка Head (UAS) ---
            s_head_id = s_tok['head_id']  # ID родителя в системе
            g_head_id = g_tok['head_id']  # ID родителя в золоте (conllu использует 'head')

            # Если head=0 (Root), проверяем, что в золоте тоже Root
            head_match = False

            if s_head_id == 0:
                if g_head_id == 0:
                    head_match = True
            else:
                # Если не Root, то родитель системы должен быть выровнен с родителем золота
                # Ищем, с кем выровнен s_head_id
                aligned_g_head = sys_to_gold_map.get(s_head_id)
                if aligned_g_head == g_head_id:
                    head_match = True

            if head_match:
                correct_uas += 1

                # --- Проверка Deprel (LAS) ---
                # Сравниваем строки (nsubj == nsubj)
                # Можно добавить нормализацию (убрать :pass и т.д.), но пока строгое сравнение
                if s_tok.get('rel') == g_tok.get('deprel'):
                    correct_las += 1

        # Расчет итоговых цифр
        uas = correct_uas / total_aligned if total_aligned > 0 else 0.0
        las = correct_las / total_aligned if total_aligned > 0 else 0.0

        return {
            "total_aligned_tokens": total_aligned,
            "soft_uas": uas,
            "soft_las": las
        }
    