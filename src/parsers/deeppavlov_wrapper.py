#!/usr/bin/env python3
"""
Обёртка для DeepPavlov (через Modal).

Поддерживает два режима токенизации:
1. 'razdel' (по умолчанию) - использует Razdel для качественной токенизации
2. 'native' - использует встроенный токенизатор DeepPavlov
"""

import logging
import modal
from typing import List, Dict, Any, Literal

logger = logging.getLogger(__name__)


class DeepPavlovParser:
    """
    Клиент для DeepPavlov, запущенного в Modal.

    Args:
        tokenizer: 'razdel' или 'native'
            - 'razdel' (рекомендуется): использует Razdel для качественной токенизации
            - 'native': встроенный простой токенизатор DeepPavlov
    """

    def __init__(self, tokenizer: Literal['razdel', 'native'] = 'razdel'):  # ДОБАВЛЕН ПАРАМЕТР
        self.logger = logging.getLogger(__name__)
        self.tokenizer_type = tokenizer  # СОХРАНЯЕМ ВЫБОР

        try:
            self.service = modal.Cls.from_name("booknlp-ru-deeppavlov", "DeepPavlovService")()
            self.logger.info(f"Connected to DeepPavlov via Modal (tokenizer: {tokenizer}).")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal app: {e}")
            raise e

    def parse_text(self, text: str) -> List[List[Dict[str, Any]]]:
        """
        Парсит текст с выбранным токенизатором.

        Возвращает: List[List[Dict]] - список предложений с токенами.
        Поля: id, form, lemma, upos, xpos, feats, head, deprel, deps, misc, startchar, endchar
        """
        try:
            # УСЛОВНЫЙ ВЫЗОВ в зависимости от типа токенизатора
            if self.tokenizer_type == 'razdel':
                results = self.service.parse_text.remote(text)
            elif self.tokenizer_type == 'native':
                results = self.service.parse_text_native.remote(text)
            else:
                raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")

            return results if results else []
        except Exception as e:
            self.logger.error(f"Error during DeepPavlov parsing: {e}")
            raise e

    def parse_batch(self, texts: List[str]) -> List[List[List[Dict[str, Any]]]]:
        """
        Парсит батч текстов с выбранным токенизатором.
        """
        try:
            if self.tokenizer_type == 'razdel':
                return self.service.parse_batch.remote(texts)
            elif self.tokenizer_type == 'native':
                # Для native - вызываем parse_text_native для каждого текста
                return [self.service.parse_text_native.remote(text) for text in texts]
            else:
                raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")
        except Exception as e:
            self.logger.error(f"Error during DeepPavlov batch parsing: {e}")
            raise e


if __name__ == "__main__":
    import pandas as pd
    import argparse

    logging.basicConfig(level=logging.INFO)

    # Парсинг аргументов командной строки
    parser_args = argparse.ArgumentParser(
        description='Test DeepPavlov parser with different tokenizers'
    )
    parser_args.add_argument(
        '--tokenizer',
        type=str,
        choices=['razdel', 'native'],
        default='razdel',
        help='Choose tokenizer: razdel (recommended) or native'
    )
    args = parser_args.parse_args()

    test_text = "Мама мыла раму."

    print(f"{'=' * 60}")
    print(f"🚀 Testing DeepPavlov with {args.tokenizer.upper()} tokenizer")
    print(f"{'=' * 60}")

    try:
        # Создаём parser с выбранным токенизатором
        parser = DeepPavlovParser(tokenizer=args.tokenizer)
        sentences = parser.parse_text(test_text)

        print(f"\nReceived {len(sentences)} sentence(s)")

        # Преобразуем в DataFrame
        all_tokens = [token for sent in sentences for token in sent]
        df = pd.DataFrame(all_tokens)

        print(f"\n--- DeepPavlov Joint Parsing ({args.tokenizer}) ---")
        if not df.empty:
            cols = ['id', 'form', 'lemma', 'upos', 'head', 'deprel']
            available_cols = [col for col in cols if col in df.columns]
            print(df[available_cols].to_string(index=False))

            if 'feats' in df.columns:
                print(f"\n--- Morphological Features ---")
                print(df[['form', 'feats']].to_string(index=False))

            # Character offsets только для razdel
            if 'startchar' in df.columns and args.tokenizer == 'razdel':
                print(f"\n--- Character Offsets (Razdel) ---")
                print(df[['form', 'startchar', 'endchar']].to_string(index=False))
        else:
            print("Empty result")

        print(f"\n✅ Test completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
