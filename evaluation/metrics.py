from .aligner import FuzzyAligner


class SoftMetricCalculator:
    """
    Рассчитывает метрики Soft UAS/LAS на основе карты выравнивания.
    """

    def __init__(self):
        pass

    def compute_metrics(self, alignment_map: list, sys_tokens: list, gold_tokens: list):
        """
        Вычисляет Soft Precision, Recall, UAS, LAS.
        """
        correct_uas = 0
        correct_las = 0
        total_gold_relations = 0

        # Создаем быстрый поиск по карте выравнивания: Gold Index -> Sys Indices
        gold_to_sys_map = {}
        for item in alignment_map:
            for g_idx in item['gold_indices']:
                gold_to_sys_map[g_idx] = item['sys_indices']

        # Проходим по всем золотым токенам для расчета Recall-based метрик
        for g_idx, g_token in enumerate(gold_tokens):
            # Игнорируем корни (если они технические) или пунктуацию, если нужно
            # Но по UD метрики считаются по всем токенам
            total_gold_relations += 1

            # 1. Находим выровненные системные токены [cite: 93]
            sys_indices = gold_to_sys_map.get(g_idx, [])

            if not sys_indices:
                # Omission (токен пропущен системой) -> Ошибка
                continue

            # Эвристика: если выровнено несколько токенов (Merge),
            # берем "основной" (например, первый или корень поддерева).
            # Для простоты MVP берем первый.
            s_child_idx = sys_indices[0]
            s_token = sys_tokens[s_child_idx]

            # 2. Получаем ID родителей (Head)
            # Важно: в CONLL-U id начинаются с 1, а индексы списка с 0.
            # Нужно приведение типов. Предполагаем, что .head возвращает int ID (1-based).
            g_head_id = int(g_token.head)
            s_head_id = int(s_token.head)

            # 3. Проверка Structural Alignment (Soft UAS) [cite: 96]
            is_structure_correct = False

            if g_head_id == 0:  # Root
                if s_head_id == 0:
                    is_structure_correct = True
            else:
                # Находим системный эквивалент золотого родителя
                g_parent_idx = g_head_id - 1
                sys_parent_indices = gold_to_sys_map.get(g_parent_idx, [])

                # Проверяем, указывает ли системный токен на кого-то из группы
                # системных токенов, соответствующих золотому родителю.
                # S_head (ID) -> S_head_idx (0-based)
                if s_head_id > 0:
                    s_parent_idx = s_head_id - 1
                    if s_parent_idx in sys_parent_indices:
                        is_structure_correct = True

            if is_structure_correct:
                correct_uas += 1

                # 4. Проверка Label Alignment (Soft LAS) [cite: 100]
                # С нормализацией меток (игнорируем регистр и подтипы, если нужно)
                if s_token.deprel.lower() == g_token.deprel.lower():
                    correct_las += 1

        return {
            "soft_uas": correct_uas / total_gold_relations if total_gold_relations > 0 else 0,
            "soft_las": correct_las / total_gold_relations if total_gold_relations > 0 else 0,
            "total_tokens": total_gold_relations
        }
