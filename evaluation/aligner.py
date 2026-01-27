import networkx as nx
from rapidfuzz import fuzz
from .utils import compute_iou, normalize_text


class SpanProjector:
    """
    Отвечает за проекцию токенов (из CONLL-U или вывода модели) на сырой текст.
    Реализует алгоритм, описанный в п. 3.2 отчета[cite: 38, 49].
    """

    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        # Для поиска используем нормализованную версию, но возвращаем индексы оригинала
        self.normalized_raw = normalize_text(raw_text)
        self.cursor = 0

    def project(self, tokens: list) -> list:
        """
        Возвращает список кортежей (start, end) для каждого токена.
        Args:
            tokens: Список строк (форм токенов)
        """
        spans = []
        self.cursor = 0

        for token_form in tokens:
            clean_form = normalize_text(token_form)
            if not clean_form:  # Пропуск пустых токенов или Null Nodes
                spans.append(None)
                continue

            # 1. Жадный поиск точного совпадения от текущего курсора
            start = self.normalized_raw.find(clean_form, self.cursor)

            # 2. Эвристика Fuzzy Fallback [cite: 60]
            # Если точное совпадение не найдено или оно слишком далеко (>20 символов),
            # запускаем нечеткий поиск в окне, чтобы компенсировать опечатки/галлюцинации.
            if start == -1 or (start - self.cursor) > 20:
                window_size = 20 + len(clean_form)
                window_text = self.normalized_raw[self.cursor: self.cursor + window_size]

                # Ищем лучшее частичное совпадение
                # Используем ratio, так как нам нужно структурное сходство
                if window_text:
                    # Простая эвристика: если в окне есть что-то похожее
                    ratio = fuzz.partial_ratio(clean_form, window_text)
                    if ratio > 75:  # Порог уверенности
                        # Пытаемся найти начало этого "похожего" куска грубым перебором в окне
                        # (в реальной оптимизации здесь можно использовать alignment из rapidfuzz)
                        # Для MVP берем ближайший символ, с которого начинается похожая строка
                        best_local_start = -1
                        for i in range(len(window_text)):
                            if window_text[i] == clean_form[0]:
                                best_local_start = i
                                break
                        if best_local_start != -1:
                            start = self.cursor + best_local_start

            if start != -1:
                end = start + len(clean_form)
                spans.append((start, end))
                self.cursor = end  # Сдвигаем курсор [cite: 69]
            else:
                # Токен не найден (Hallucination или сильное искажение)
                spans.append(None)  # [cite: 65]

        return spans


class FuzzyAligner:
    """
    Реализует логику разрешения конфликтов через графы (NetworkX).
    Соответствует п. 3.3 и 6.2 отчета[cite: 72, 143].
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold  # Порог IoU [cite: 145]

    def align(self, sys_tokens_obj: list, gold_tokens_obj: list, raw_text: str) -> list:
        """
        Строит карту выравнивания между токенами системы и золотого стандарта.

        Args:
            sys_tokens_obj: Список объектов токенов системы (должны иметь атрибут .form или быть строками)
            gold_tokens_obj: Список объектов токенов золота
            raw_text: Исходный текст предложения

        Returns:
            List[dict]: Список компонент связности (выровненных групп).
        """
        # Извлечение текстовых форм для проектора
        sys_forms = [t.form if hasattr(t, 'form') else str(t) for t in sys_tokens_obj]
        gold_forms = [t.form if hasattr(t, 'form') else str(t) for t in gold_tokens_obj]

        # 1. Проекция на текст [cite: 147]
        projector = SpanProjector(raw_text)
        sys_spans = projector.project(sys_forms)
        gold_spans = projector.project(gold_forms)

        # 2. Построение графа [cite: 149]
        G = nx.Graph()

        # Добавляем узлы с метаданными
        for i, _ in enumerate(sys_tokens_obj):
            G.add_node(f"SYS_{i}", type='sys', index=i)

        for j, _ in enumerate(gold_tokens_obj):
            G.add_node(f"GOLD_{j}", type='gold', index=j)

        # Добавляем ребра на основе IoU [cite: 154]
        for i, s_span in enumerate(sys_spans):
            if s_span is None: continue

            for j, g_span in enumerate(gold_spans):
                if g_span is None: continue

                iou = compute_iou(s_span, g_span)
                if iou > self.threshold:
                    G.add_edge(f"SYS_{i}", f"GOLD_{j}", weight=iou)

        # 3. Разрешение конфликтов (Компоненты связности) [cite: 157]
        alignment_map = []

        # Обрабатываем все связные компоненты
        for component in nx.connected_components(G):
            sys_indices = []
            gold_indices = []

            for node in component:
                node_type = G.nodes[node]['type']
                node_idx = G.nodes[node]['index']
                if node_type == 'sys':
                    sys_indices.append(node_idx)
                else:
                    gold_indices.append(node_idx)

            sys_indices.sort()
            gold_indices.sort()

            # Классификация типа выравнивания [cite: 162]
            match_type = self._classify_match(len(sys_indices), len(gold_indices))

            alignment_map.append({
                "sys_indices": sys_indices,  # Индексы токенов системы
                "gold_indices": gold_indices,  # Индексы токенов золота
                "type": match_type  # 1:1, 1:N, N:1, N:M
            })

        # Добавляем невыровненные токены (Unaligned) для полноты картины
        # (Это важно для метрики Hallucination Rate [cite: 176])
        aligned_sys = set(idx for m in alignment_map for idx in m["sys_indices"])
        aligned_gold = set(idx for m in alignment_map for idx in m["gold_indices"])

        for i in range(len(sys_tokens_obj)):
            if i not in aligned_sys:
                alignment_map.append({"sys_indices": [i], "gold_indices": [], "type": "hallucination"})

        for j in range(len(gold_tokens_obj)):
            if j not in aligned_gold:
                alignment_map.append({"sys_indices": [], "gold_indices": [j], "type": "omission"})

        return alignment_map

    def _classify_match(self, n_sys: int, n_gold: int) -> str:
        """Классификация типа коллизии согласно таблице[cite: 27]."""
        if n_sys == 1 and n_gold == 1:
            return "exact"  # Идеальное совпадение
        elif n_sys == 1 and n_gold > 1:
            return "split"  # Система объединила токены (Merge error системы, Split mapping)
        elif n_sys > 1 and n_gold == 1:
            return "merge"  # Система разбила токен (Split error системы, Merge mapping)
        elif n_sys > 1 and n_gold > 1:
            return "complex"  # Сложная перестановка
        else:
            return "error"