# src/ingestion/validators.py
from conllu import TokenList
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """DTO для результатов валидации."""

    def __init__(self, is_valid: bool, errors: List[str]):
        self.is_valid = is_valid
        self.errors = errors


class DataValidator:
    """
    Валидатор предложений в формате CoNLL-U.
    Реализует проверки структуры, метаданных и целостности графа.
    """

    @staticmethod
    def validate_sentence(token_list: TokenList, strict: bool = True) -> ValidationResult:
        errors = [] # Исправлено: инициализация пустым списком

        # 1. Проверка метаданных
        # Для BookNLP критично наличие sent_id для маппинга результатов
        if 'sent_id' not in token_list.metadata:
            errors.append("ERROR: Отсутствует метаполе 'sent_id'")

        # text необходим для выравнивания токенов (fuzzy alignment)
        if 'text' not in token_list.metadata:
            # В lenient режиме можем допустить отсутствие text, если есть токены
            if strict:
                errors.append("ERROR: Отсутствует метаполе 'text'")

        # 2. Проверка токенов
        roots = 0
        for token in token_list:
            token_id = token['id']

            # Пропуск мульти-словных токенов (диапазонов) при проверке HEAD
            # Пример ID: (1, "-", 2)
            if isinstance(token_id, tuple):
                continue

            # Обязательные поля
            if not token['form']:
                errors.append(f"Token {token_id}: Пустое поле FORM")

            # Проверка HEAD
            head = token['head']
            if head == 0:
                roots += 1
            elif isinstance(head, int):
                # Проверяем, что HEAD ссылается на существующий токен
                # (В оптимизированной версии можно использовать set ID-шников)
                if not any(t['id'] == head for t in token_list):
                    errors.append(f"Token {token_id}: HEAD {head} ссылается на несуществующий ID")

        # 3. Структурная проверка
        # В проективном дереве должен быть ровно один корень
        if roots != 1:
            if strict:
                errors.append(f"ERROR: Найдено {roots} корней (ожидается 1)")
            else:
                # В Taiga или веб-текстах может быть "мусор", логируем как warning
                pass

        return ValidationResult(len(errors) == 0, errors)

    @staticmethod
    def validate_batch(sentences: List, strict: bool = True) -> Dict[str, Any]:
        """Агрегированная статистика валидации набора предложений."""
        stats = {
            "total": len(sentences),
            "valid": 0,
            "invalid": 0,
            "errors": []  # Исправлено: инициализация пустым списком
        }

        # Проверка уникальности ID внутри файла
        seen_ids = set()

        for sent in sentences:
            res = DataValidator.validate_sentence(sent, strict)

            sent_id = sent.metadata.get('sent_id')
            if sent_id and sent_id in seen_ids:
                res.is_valid = False
                res.errors.append(f"Duplicate sent_id: {sent_id}")

            if sent_id:
                seen_ids.add(sent_id)

            if res.is_valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                if sent_id:  # Логируем только если есть ID, иначе непонятно, что это
                    stats["errors"].append({"id": sent_id, "issues": res.errors})

        return stats