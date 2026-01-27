from typing import List, Tuple, Optional
from .utils import normalize_text, text_similarity


class SpanProjector:
    """
    Класс для восстановления символьных координат (spans) токенов
    в исходном сыром тексте. Критичен для систем, теряющих оффсеты.
    """

    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.normalized_raw = normalize_text(raw_text)
        self.cursor = 0

    def project(self, tokens: List[str]) -> List]

        ]:
    """
    Проецирует список токенов (строк) на сырой текст.
    Возвращает список кортежей (start, end) или None, если токен не найден.
    """
    spans =
    self.cursor = 0  # Сброс курсора для нового предложения

    for token in tokens:
        norm_token = normalize_text(token)
    if not norm_token:  # Пропуск пустых токенов
        spans.append(None)
    continue

    # 1. Попытка точного поиска (с учетом нормализации)
    # Ищем в окне +50 символов, чтобы компенсировать пропущенные пробелы/пунктуацию
    search_window = self.normalized_raw[self.cursor: self.cursor + 50 + len(norm_token)]
    found_idx = search_window.find(norm_token)

    if found_idx != -1:
        start = self.cursor + found_idx
        end = start + len(norm_token)
        spans.append((start, end))
        self.cursor = end
    else:
        # 2. Fallback: Нечеткий поиск (для опечаток или галлюцинаций LLM)
        # Если точный поиск не дал результата, ищем "наилучшее" совпадение в окне
        best_match = None
        best_score = 0.0

        # Эвристика: перебираем подстроки разной длины
        # Это медленно, но надежно для "шумных" данных
        window_text = self.normalized_raw[self.cursor: self.cursor + 20]

        for length in range(len(norm_token) - 2, len(norm_token) + 3):
            if length <= 0: continue
            candidate = window_text[:length]
            score = text_similarity(norm_token, candidate)
            if score > 0.8 and score > best_score:
                best_score = score
                best_match = length

        if best_match:
            start = self.cursor
            end = start + best_match
            spans.append((start, end))
            self.cursor = end
        else:
            # Токен не найден (Hallucination or massive skip)
            spans.append(None)


return spans