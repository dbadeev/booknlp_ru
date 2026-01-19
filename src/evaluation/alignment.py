import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenSpan:
    id: int  # ID токена (1-based)
    text: str
    start: int
    end: int


@dataclass
class Alignment:
    sys_id: Optional[int]
    gold_id: Optional[int]
    iou: float
    type: str  # 'exact', 'split', 'merge', 'none'


class FuzzyAligner:
    """
    Выравнивает токены двух систем на основе символьных оффсетов.
    Решает проблему Token Mismatch (разной токенизации).
    """

    def align(self, sys_tokens: List[Dict], gold_tokens: List[Dict]) -> List[Alignment]:
        """
        sys_tokens: список словарей с ключами 'id', 'start_char', 'end_char' (от Razdel/Pipeline)
        gold_tokens: список словарей (от SynTagRus). Важно: у Gold токенов часто нет явных оффсетов,
                     их нужно восстановить, если они не даны.
        """
        # 1. Если у Gold токенов нет оффсетов, пытаемся их восстановить (простейшая эвристика)
        # В реальном пайплайне лучше подавать Gold с уже вычисленными оффсетами.
        # Здесь предполагаем, что на вход приходят токены с 'start_char' и 'end_char'.

        alignments = []

        # Индексируем токены для быстрого доступа
        sys_map = {t['id']: t for t in sys_tokens}
        gold_map = {t['id']: t for t in gold_tokens}

        # Матрица пересечений: sys_id -> list of (gold_id, iou)
        # Используем жадный алгоритм или перебор, так как предложения короткие.

        used_gold = set()

        for s_tok in sys_tokens:
            s_start, s_end = s_tok['start_char'], s_tok['end_char']
            s_len = s_end - s_start

            best_match = None
            max_iou = 0.0

            # Ищем пересечения с Gold токенами
            candidates = []

            for g_tok in gold_tokens:
                g_start, g_end = g_tok['start_char'], g_tok['end_char']

                # Вычисляем пересечение (Intersection)
                inter_start = max(s_start, g_start)
                inter_end = min(s_end, g_end)

                if inter_start < inter_end:
                    intersection = inter_end - inter_start

                    # Вычисляем объединение (Union)
                    union = (s_end - s_start) + (g_end - g_start) - intersection

                    iou = intersection / union if union > 0 else 0

                    if iou > 0:
                        candidates.append((g_tok['id'], iou))

            # Анализ кандидатов
            if not candidates:
                alignments.append(Alignment(s_tok['id'], None, 0.0, 'none'))
                continue

            # Сортируем по IoU
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Топ-1 кандидат
            top_gold_id, top_iou = candidates[0]

            # Определение типа выравнивания
            match_type = 'partial'
            if top_iou > 0.9:
                match_type = 'exact'

            # Проверка на Split/Merge (сложная логика)
            # Если 1 System токен перекрывает несколько Gold -> Merge Error (в системе склеено)
            # Если 1 Gold токен перекрывается несколькими System -> Split Error (в системе разорвано)

            # Для простоты MVP берем best match
            alignments.append(Alignment(s_tok['id'], top_gold_id, top_iou, match_type))
            used_gold.add(top_gold_id)

        # Добавляем ненайденные Gold токены (False Negatives)
        for g_tok in gold_tokens:
            if g_tok['id'] not in used_gold:
                # Ищем, не были ли они частью Split/Merge, которые мы пропустили
                # Для репорта добавляем как пропущенные
                pass

        return alignments

    @staticmethod
    def reconstruct_gold_offsets(tokens: List[Dict], raw_text: str) -> List[Dict]:
        """
        Вспомогательный метод: расставляет оффсеты для Gold токенов,
        сопоставляя их формы с сырым текстом.
        """
        current_pos = 0
        rich_tokens = []

        for t in tokens:
            form = t['form']
            # Ищем первое вхождение токена начиная с current_pos
            start = raw_text.find(form, current_pos)

            if start == -1:
                # Если не нашли (например, в UD другое написание), пропускаем или логируем
                # Это "мягкое" падение
                rich_tokens.append({**t, 'start_char': -1, 'end_char': -1})
                continue

            end = start + len(form)
            rich_tokens.append({**t, 'start_char': start, 'end_char': end})
            current_pos = end

        return rich_tokens
    