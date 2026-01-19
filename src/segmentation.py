import logging
from dataclasses import dataclass
from typing import List, Generator
from razdel import tokenize as razdel_tokenize
from razdel import sentenize as razdel_sentenize

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """
    Унифицированное представление токена с привязкой к исходному тексту.
    [cite_start]Требование[cite: 268]: поля text, start_char, end_char.
    """
    text: str
    start_char: int
    end_char: int

    # Дополнительные поля для удобства отладки
    id: int = None  # Порядковый номер в предложении (1-based)

    def __repr__(self):
        return f"Token('{self.text}', {self.start_char}, {self.end_char})"


@dataclass
class Sentence:
    text: str
    start_char: int
    end_char: int
    tokens: List[Token]


class RazdelSegmenter:
    """
    Обертка над библиотекой Razdel для сегментации текста.
    [cite_start]Реализует требования Карточки 2.1[cite: 258, 488].
    """

    def tokenize(self, text: str) -> List[Token]:
        """
        Простая токенизация текста.
        """
        tokens = []
        for item in razdel_tokenize(text):
            tokens.append(Token(
                text=item.text,
                start_char=item.start,
                end_char=item.stop
            ))
        return tokens

    def split_sentences(self, text: str) -> List[Sentence]:
        """
        Разбиение на предложения с токенизацией внутри.
        [cite_start]Гарантирует сохранение глобальных оффсетов[cite: 271].
        """
        sentences = []

        # 1. Разбиваем на предложения
        sent_spans = list(razdel_sentenize(text))

        for sent_span in sent_spans:
            # sent_span.text - это подстрока
            # sent_span.start, sent_span.stop - глобальные границы

            # 2. Токенизируем само предложение
            raw_tokens = list(razdel_tokenize(sent_span.text))

            sentence_tokens = []
            for i, rt in enumerate(raw_tokens, 1):
                # rt.start и rt.stop здесь локальные (относительно начала предложения)
                # Нам нужны глобальные:
                global_start = sent_span.start + rt.start
                global_stop = sent_span.start + rt.stop

                # Санити-чек: проверяем, что оффсеты указывают на тот же текст
                original_slice = text[global_start:global_stop]
                if original_slice != rt.text:
                    logger.warning(
                        f"Offset mismatch! Token: {rt.text}, Slice: {original_slice}, "
                        f"Global: {global_start}-{global_stop}"
                    )

                sentence_tokens.append(Token(
                    text=rt.text,
                    start_char=global_start,
                    end_char=global_stop,
                    id=i
                ))

            sentences.append(Sentence(
                text=sent_span.text,
                start_char=sent_span.start,
                end_char=sent_span.stop,
                tokens=sentence_tokens
            ))

        return sentences
    