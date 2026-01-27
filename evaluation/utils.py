import unicodedata


def normalize_text(text: str) -> str:
    """
    Нормализация текста для сравнения:
    1. NFKC нормализация (для единообразия символов Unicode).
    2. Приведение к нижнему регистру.
    3. Замена 'ё' на 'е'.

    Обоснование: Устраняет шум, описанный в пункте 2.1 (вариативность орфографии)[cite: 21, 131].
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    return text.lower().replace("ё", "е")


def compute_iou(span1: tuple, span2: tuple) -> float:
    """
    Вычисление Intersection over Union (IoU) для двух символьных диапазонов.

    Args:
        span1: Кортеж (start, end)
        span2: Кортеж (start, end)

    Returns:
        float: Значение IoU от 0.0 до 1.0.

    Реализация формулы из раздела 6.1 отчета[cite: 133].
    """
    if not span1 or not span2:
        return 0.0

    # start = max(start1, start2)
    start = max(span1[0], span2[0])
    # end = min(end1, end2)
    end = min(span1[1], span2[1])

    if end <= start:
        return 0.0

    intersection = end - start
    # Union = len(A) + len(B) - Intersection
    len1 = span1[1] - span1[0]
    len2 = span2[1] - span2[0]
    union = len1 + len2 - intersection

    if union == 0:
        return 0.0

    return intersection / union