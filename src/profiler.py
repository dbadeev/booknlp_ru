import logging
import networkx as nx
from conllu.models import TokenList

logger = logging.getLogger(__name__)


class SentenceProfiler:
    def __init__(self):
        # [cite_start]Список глаголов речи для детекции диалогов [cite: 220]
        self.speech_verbs = {
            "сказать", "говорить", "ответить", "отвечать", "спросить", "спрашивать",
            "промолвить", "заявить", "подумать", "воскликнуть", "шептать", "пояснить",
            "заметить", "продолжить", "добавить"
        }

    def profile_sentence(self, sentence: TokenList) -> dict:
        """
        Вычисляет набор метрик для одного предложения.
        """
        return {
            "id": sentence.metadata.get("sent_id", "unknown"),
            "text_len": len(sentence),
            # [cite_start]
            "is_dialogue": self._is_dialogue(sentence),  # [cite: 220]
            "tree_depth": self._calculate_tree_depth(sentence),  # [cite: 221]
            "non_projectivity": self._is_non_projective(sentence),  # [cite: 222]
            "ellipsis_count": self._count_ellipsis(sentence)  # [cite: 223]
        }

    def _is_dialogue(self, sentence: TokenList) -> bool:
        """
        Эвристика: наличие тире/кавычек + глаголы речи.
        """
        text = sentence.metadata.get("text", "")
        tokens = [t for t in sentence if isinstance(t['id'], int)]

        # 1. Проверка пунктуации (тире в начале или кавычки)
        has_dialogue_punct = False
        if text.strip().startswith(("-", "–", "—")):
            has_dialogue_punct = True
        elif "«" in text and "»" in text:
            has_dialogue_punct = True

        if not has_dialogue_punct:
            return False

        # 2. Проверка наличия лемматизированных глаголов речи
        # Примечание: предполагаем, что в корпусе есть леммы. Если нет - проверка будет неточной.
        has_speech_verb = False
        for t in tokens:
            lemma = t.get("lemma", "").lower() if t.get("lemma") else t["form"].lower()
            if lemma in self.speech_verbs:
                has_speech_verb = True
                break

        return has_speech_verb

    def _calculate_tree_depth(self, sentence: TokenList) -> int:
        """
        Вычисляет максимальную глубину дерева зависимостей.
        """
        g = nx.DiGraph()
        roots = []

        # Строим граф, исключая мульти-токены (1-2) и пустые узлы (1.1)
        valid_tokens = [t for t in sentence if isinstance(t['id'], int)]

        for t in valid_tokens:
            node_id = t['id']
            head_id = t['head']

            g.add_node(node_id)

            if head_id == 0:
                roots.append(node_id)
            else:
                g.add_edge(head_id, node_id)

        if not roots:
            return 0

        max_depth = 0
        try:
            # Для каждого корня ищем самый длинный путь
            for root in roots:
                # shortest_path в невзвешенном графе дает BFS уровни
                lengths = nx.shortest_path_length(g, source=root)
                current_max = max(lengths.values())
                if current_max > max_depth:
                    max_depth = current_max
        except Exception:
            # Если в дереве есть циклы (ошибка разметки), networkx может упасть
            return -1

        return max_depth

    def _is_non_projective(self, sentence: TokenList) -> bool:
        """
        Проверка на пересечение дуг (start < end < start < end).
        """
        arcs = []
        valid_tokens = [t for t in sentence if isinstance(t['id'], int)]

        for t in valid_tokens:
            dep = t['id']
            head = t['head']
            if head == 0:
                continue
            # Дуга всегда от min к max для проверки пересечений
            start, end = sorted((dep, head))
            arcs.append((start, end))

        # [cite_start]Сравниваем каждую дугу с каждой [cite: 222]
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                s1, e1 = arcs[i]
                s2, e2 = arcs[j]

                # Условие пересечения: одна дуга начинается внутри другой, но заканчивается снаружи
                if s1 < s2 < e1 < e2:
                    return True
                if s2 < s1 < e2 < e1:
                    return True

        return False

    def _count_ellipsis(self, sentence: TokenList) -> int:
        """
        Считает количество пустых узлов (Null nodes), которые в CoNLL-U имеют дробные ID (x.1).
        """
        count = 0
        for token in sentence:
            # В conllu ID пустых узлов представляются кортежами, например (8, '.', 1)
            if isinstance(token['id'], tuple):
                count += 1
        return count