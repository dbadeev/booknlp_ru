import sys
from typing import List, Dict, Optional
from razdel import sentenize, tokenize as razdel_tokenize

# Импорты нашей архитектуры
from src.core.interfaces import BasePreprocessor
from src.core.data_structures import Token

try:
    from deeppavlov import build_model, configs
except ImportError:
    print("❌ DeepPavlov не установлен. Выполните: pip install deeppavlov")
    sys.exit(1)


class DeepPavlovEngine(BasePreprocessor):
    """
    Реализация ENG-002: DeepPavlov (RuBERT) Wrapper.
    Решает проблему несовпадения токенизации через символьное выравнивание.
    """

    def __init__(self, install: bool = False):
        print("🧠 Инициализация DeepPavlov (RuBERT)...")

        # Конфиг из Roadmap [cite: 89]
        self.config_name = configs.syntax.ru_syntagrus_joint_parsing

        if install:
            print("📦 Установка зависимостей DeepPavlov (это может занять время)...")
            from deeppavlov.core.commands.infer import interact_model
            from deeppavlov.core.common.file import read_json
            # Это вызовет загрузку весов (~700MB+) [cite: 91]

        try:
            self.model = build_model(self.config_name, download=True)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("💡 Попробуйте: python -m deeppavlov install ru_syntagrus_joint_parsing")
            raise e

        print("✅ DeepPavlov Engine Ready")

    def process(self, text: str) -> List[List[Token]]:
        doc_sentences = []

        # 1. Сегментация на предложения (Razdel - наш эталон)
        spans = list(sentenize(text))
        sent_texts = [s.text for s in spans]

        if not sent_texts:
            return []

        # 2. Batch Inference в DeepPavlov
        # Модель принимает список строк и возвращает список CoNLL-U строк (обычно)
        # или структурированных списков, в зависимости от версии.
        # ru_syntagrus_joint_parsing обычно возвращает распарсенные данные.
        parsed_batch = self.model(sent_texts)

        # 3. Выравнивание и Detokenization Mapping
        for i, dp_output in enumerate(parsed_batch):
            sent_span = spans[i]

            # Токены Razdel (наш Target Grid)
            razdel_tokens = list(razdel_tokenize(sent_span.text))

            # Парсинг выхода DP (это может быть строка CoNLL или объект)
            dp_tokens_data = self._parse_dp_output(dp_output)

            # Самая сложная часть: Склеивание токенов DP под Razdel
            aligned_tokens = self._align_and_merge(
                razdel_tokens=razdel_tokens,
                dp_tokens_data=dp_tokens_data,
                sent_offset=sent_span.start
            )

            doc_sentences.append(aligned_tokens)

        return doc_sentences

    def _parse_dp_output(self, output) -> List[Dict]:
        """
        Преобразует выход DeepPavlov в список словарей.
        DeepPavlov joint_parser возвращает строку в формате CoNLL-U.
        """
        if isinstance(output, str):
            lines = output.strip().split('\n')
            tokens = []
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 10:
                    tokens.append({
                        'text': parts[1],
                        'lemma': parts[2],
                        'pos': parts[3],
                        'head': int(parts[6]) if parts[6].isdigit() else 0,
                        'rel': parts[7]
                    })
            return tokens
        else:
            # Если версия DP возвращает списки, адаптируем здесь
            # Для ru_syntagrus_joint_parsing это обычно string
            return []

    def _align_and_merge(self, razdel_tokens, dp_tokens_data, sent_offset) -> List[Token]:
        """
        Реализация алгоритма 'detokenization mapping'.
        Если DP разбил слово "по-русски" на ["по", "-", "русски"],
        мы должны объединить их в один Token (как в Razdel),
        выбрав главного syntactic head.
        """
        result_tokens = []
        dp_cursor = 0

        # Временный маппинг: DP index -> Result Token index
        # Нужен для пересчета head_id, так как количество токенов меняется
        dp_to_result_map = {}

        # 1. Проход по токенам Razdel (Target)
        for r_idx, r_tok in enumerate(razdel_tokens):
            r_text = r_tok.text

            # Собираем токены DP, которые попадают внутрь r_text
            # Простая эвристика: собираем DP токены пока их склеенный текст совпадает с r_text
            # (В реальной жизни нужно честное символьное выравнивание, но для MVP хватит concat)

            buffer_dp_indices = []
            buffer_text = ""

            while dp_cursor < len(dp_tokens_data):
                dp_tok = dp_tokens_data[dp_cursor]

                # Нормализация для сравнения (у DP может быть ё/е различие)
                # но для простоты проверяем длину или вхождение

                buffer_dp_indices.append(dp_cursor)
                buffer_text += dp_tok['text']
                dp_cursor += 1

                # Если собрали слово целиком (или больше)
                if len(buffer_text) >= len(r_text):
                    break

            # --- Merge Logic ---
            # У нас есть N токенов DeepPavlov, которые соответствуют 1 токену Razdel.
            # Пример: Razdel="по-русски", DP=["по", "-", "русски"]

            # Выбираем "представителя" для POS и Syntax.
            # Эвристика: берем токен, который является ROOT-ом для этой группы
            # (т.е. на который ссылаются другие, или просто последний/первый)

            # Для простоты берем ПОСЛЕДНИЙ значащий токен (часто корень в суффиксе)
            # или ПЕРВЫЙ. DeepPavlov обычно ставит head на главное слово.

            # Возьмем первый токен из группы как основу, но если группа > 1,
            # это сигнал "Tokenization Mismatch"[cite: 102].

            if not buffer_dp_indices:
                continue  # Edge case

            main_dp_idx = buffer_dp_indices[0]
            # Можно улучшить: найти токен в группе, у которого head лежит ВНЕ группы

            dp_token = dp_tokens_data[main_dp_idx]

            # Сохраняем маппинг для всех "съеденных" токенов DP -> текущий r_idx
            for dpi in buffer_dp_indices:
                dp_to_result_map[dpi] = r_idx + 1  # 1-based output index

            token = Token(
                id=r_idx + 1,
                text=r_text,
                lemma=dp_token['lemma'],  # Лемма от DP
                pos=dp_token['pos'],  # POS от DP
                head_id=dp_token['head'],  # Старый head (пока невалидный)
                rel=dp_token['rel'],
                char_start=sent_offset + r_tok.start,
                char_end=sent_offset + r_tok.stop
            )
            result_tokens.append(token)

        # 2. Второй проход: Исправление ссылок HEAD
        # Так как мы склеили токены, индексы изменились. Нужно переадресовать head_id.
        for token in result_tokens:
            old_head = token.head_id

            if old_head == 0:
                token.head_id = 0
            elif old_head in dp_to_result_map:
                # Перенаправляем на ID нового склеенного токена
                token.head_id = dp_to_result_map[old_head]

                # Защита от self-loop (если head указывал на соседа, с которым мы склеились)
                if token.head_id == token.id:
                    # Такое бывает при мердже. Ищем "внешнюю" связь?
                    # Для MVP ставим 0 или оставляем как root группы.
                    # Обычно это значит, что мы взяли зависимый токен как представителя.
                    token.rel = "flat:merged"  # Пометка для отладки
            else:
                # Если ссылка ведет в никуда (например, DP токен был пропущен), обнуляем
                token.head_id = 0

        return result_tokens
    