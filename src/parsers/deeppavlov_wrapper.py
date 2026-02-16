#!/usr/bin/env python3
"""
Обёртка для DeepPavlov (через Modal) с поддержкой ПОЛНОГО выхода.

Поддерживает три режима:
1. output_format='dict' - текущий формат (список словарей)
2. output_format='conllu' - нативный CoNLL-U формат (строка)
3. output_format='full' - ПОЛНЫЙ выход с probas/logits (словарь)

Также поддерживает два токенизатора:
- 'razdel' (рекомендуется) - качественная токенизация с символьными смещениями
- 'native' - встроенный токенизатор DeepPavlov
"""

import logging
import modal
from typing import List, Dict, Any, Literal, Union

logger = logging.getLogger(__name__)


class DeepPavlovParser:
    """
    Клиент для DeepPavlov, запущенного в Modal.

    Args:
        tokenizer: 'razdel' или 'native'
            - 'razdel' (рекомендуется): использует Razdel для качественной токенизации
            - 'native': встроенный простой токенизатор DeepPavlov
    """

    def __init__(self, tokenizer: Literal['razdel', 'native'] = 'razdel'):
        self.logger = logging.getLogger(__name__)
        self.tokenizer_type = tokenizer

        try:
            self.service = modal.Cls.from_name("booknlp-ru-deeppavlov", "DeepPavlovService")()
            self.logger.info(f"Connected to DeepPavlov via Modal (tokenizer: {tokenizer}).")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal app: {e}")
            raise e

    def parse_text(
        self, 
        text: str, 
        output_format: str = 'dict',
        use_cache: bool = False
    ) -> Union[List[List[Dict[str, Any]]], str, Dict[str, Any]]:
        """
        Парсит текст с выбранным токенизатором и возвращает в указанном формате.

        Args:
            text: входной текст для разбора
            output_format: формат выходных данных
                - 'dict' (по умолчанию): текущий формат - список предложений,
                  каждое предложение - список словарей с полями:
                  id, form, lemma, upos, xpos, feats, head, deprel, deps, misc,
                  startchar, endchar (для razdel токенизатора)

                - 'conllu': нативный CoNLL-U формат - текстовая строка
                  с 10 колонками (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD,
                  DEPREL, DEPS, MISC). Предложения разделены пустой строкой.

                - 'full': ПОЛНЫЙ выход с probas/logits - словарь со структурой:
                  {
                      'format': 'full',
                      'conllu': <CoNLL-U строка>,
                      'sentences': [
                          [
                              {
                                  # Стандартные поля CoNLL-U
                                  'id': 1, 'form': 'Мама', 'lemma': 'мама',
                                  'upos': 'NOUN', 'head': 2, 'deprel': 'nsubj',
                                  ...
                                  # ДОПОЛНИТЕЛЬНО: probas/logits
                                  'heads_proba': [0.05, 0.88, 0.03, ...],
                                  'deps_proba': {'nsubj': 0.92, 'obj': 0.05, ...},
                                  'upos_proba': 0.98
                              },
                              ...
                          ]
                      ],
                      'metadata': {
                          'model': 'ru_syntagrus_joint_parsing',
                          'tokenizer': 'razdel',
                          'vocab': {'deprels': [...]}
                      }
                  }

            use_cache: использовать кэширование результатов (ускоряет повторные запросы)

        Returns:
            - Если output_format='dict': List[List[Dict]] - список предложений
            - Если output_format='conllu': str - строка в формате CoNLL-U
            - Если output_format='full': Dict - полная структура с probas

        Examples:
            >>> parser = DeepPavlovParser(tokenizer='razdel')

            >>> # Текущий формат (dict)
            >>> result = parser.parse_text("Мама мыла раму.", output_format='dict')
            >>> print(result[0][0]['form'])  # 'Мама'

            >>> # Нативный формат (CoNLL-U)
            >>> result = parser.parse_text("Мама мыла раму.", output_format='conllu')
            >>> print(result)  # "1\tМама\t..."

            >>> # Полный формат с probas
            >>> result = parser.parse_text("Мама мыла раму.", output_format='full')
            >>> token = result['sentences'][0][0]
            >>> print(token['form'])  # 'Мама'
            >>> print(token['heads_proba'])  # [0.05, 0.88, 0.03, 0.04]
            >>> print(token['deps_proba'])  # {'nsubj': 0.92, 'obj': 0.05, ...}
        """
        try:
            # ====================================================================
            # УСЛОВНЫЙ ВЫЗОВ в зависимости от типа токенизатора и формата выхода
            # ====================================================================
            if self.tokenizer_type == 'razdel':
                results = self.service.parse_text.remote(
                    text, 
                    output_format=output_format,
                    use_cache=use_cache
                )
            elif self.tokenizer_type == 'native':
                if output_format == 'full':
                    self.logger.warning(
                        "Full format not supported with native tokenizer. "
                        "Falling back to dict format."
                    )
                    output_format = 'dict'

                results = self.service.parse_text_native.remote(
                    text, 
                    output_format=output_format
                )
            else:
                raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")
            # ====================================================================

            # Обработка пустых результатов
            if output_format == 'dict':
                return results if results else []
            elif output_format == 'conllu':
                return results if results else ''
            else:  # 'full'
                return results if results else {
                    'format': 'full',
                    'conllu': '',
                    'sentences': [],
                    'metadata': {}
                }

        except Exception as e:
            self.logger.error(f"Error during DeepPavlov parsing: {e}")
            raise e

    def parse_batch(
        self, 
        texts: List[str], 
        output_format: str = 'dict',
        use_cache: bool = False
    ) -> Union[List[List[List[Dict[str, Any]]]], List[str], List[Dict[str, Any]]]:
        """
        Парсит батч текстов с выбранным токенизатором.

        Args:
            texts: список текстов
            output_format: 'dict', 'conllu' или 'full'
            use_cache: использовать кэширование

        Returns:
            Список результатов в указанном формате
        """
        try:
            if self.tokenizer_type == 'razdel':
                return self.service.parse_batch.remote(
                    texts, 
                    output_format=output_format,
                    use_cache=use_cache
                )
            elif self.tokenizer_type == 'native':
                if output_format == 'full':
                    self.logger.warning(
                        "Full format not supported with native tokenizer. "
                        "Falling back to dict format."
                    )
                    output_format = 'dict'

                # Для native - вызываем parse_text_native для каждого текста
                return [
                    self.service.parse_text_native.remote(text, output_format=output_format) 
                    for text in texts
                ]
            else:
                raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")
        except Exception as e:
            self.logger.error(f"Error during DeepPavlov batch parsing: {e}")
            raise e


if __name__ == "__main__":
    import pandas as pd
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    # =========================================================================
    # ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
    # =========================================================================
    parser_args = argparse.ArgumentParser(
        description='Test DeepPavlov parser with different tokenizers and formats'
    )
    parser_args.add_argument(
        '--tokenizer',
        type=str,
        choices=['razdel', 'native'],
        default='razdel',
        help='Choose tokenizer: razdel (recommended) or native'
    )
    parser_args.add_argument(
        '--output-format',
        type=str,
        choices=['dict', 'conllu', 'full'],
        default='dict',
        help='Choose output format: dict (default), conllu, or full (with probas)'
    )
    parser_args.add_argument(
        '--use-cache',
        action='store_true',
        help='Enable caching for faster repeated queries'
    )
    args = parser_args.parse_args()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    print(f"{'=' * 70}")
    print(f"🚀 Testing DeepPavlov with {args.tokenizer.upper()} tokenizer")
    print(f"   Output format: {args.output_format.upper()}")
    print(f"   Caching: {'ENABLED' if args.use_cache else 'DISABLED'}")
    print(f"{'=' * 70}")

    try:
        # Создаём parser с выбранным токенизатором
        parser = DeepPavlovParser(tokenizer=args.tokenizer)

        # ====================================================================
        # ВЫЗОВ С УКАЗАНИЕМ ФОРМАТА ВЫХОДА
        # ====================================================================
        result = parser.parse_text(
            test_text, 
            output_format=args.output_format,
            use_cache=args.use_cache
        )
        # ====================================================================

        # ====================================================================
        # ОБРАБОТКА РЕЗУЛЬТАТА В ЗАВИСИМОСТИ ОТ ФОРМАТА
        # ====================================================================
        if args.output_format == 'conllu':
            # ================================================================
            # CoNLL-U ФОРМАТ - просто выводим строку
            # ================================================================
            print(f"\n--- DeepPavlov CoNLL-U Output ({args.tokenizer}) ---\n")
            print(result)

        elif args.output_format == 'full':
            # ================================================================
            # FULL ФОРМАТ - показываем структуру и пример токена с probas
            # ================================================================
            print(f"\n--- DeepPavlov FULL Output ({args.tokenizer}) ---\n")

            print(f"📊 Structure:")
            print(f"  format: {result['format']}")
            print(f"  conllu: <{len(result['conllu'])} chars>")
            print(f"  sentences: {len(result['sentences'])} sentence(s)")
            print(f"  metadata: {list(result['metadata'].keys())}")

            # Показываем CoNLL-U часть
            print(f"\n📄 CoNLL-U representation:")
            print(result['conllu'])

            # Показываем пример токена с probas
            if result['sentences']:
                print(f"\n📋 Example token with probas/logits:")
                first_token = result['sentences'][0][0]

                print(f"\n  Basic fields:")
                print(f"    form: {first_token['form']}")
                print(f"    lemma: {first_token['lemma']}")
                print(f"    upos: {first_token['upos']}")
                print(f"    head: {first_token['head']}")
                print(f"    deprel: {first_token['deprel']}")

                print(f"\n  Probabilities:")
                print(f"    upos_proba: {first_token.get('upos_proba', 'N/A')}")

                if 'heads_proba' in first_token:
                    heads_p = first_token['heads_proba']
                    print(f"    heads_proba (length={len(heads_p)}): {heads_p[:5]}... (showing first 5)")
                    print(f"      → probability for chosen head ({first_token['head']}): "
                          f"{heads_p[first_token['head']]:.3f}")

                if 'deps_proba' in first_token:
                    deps_p = first_token['deps_proba']
                    print(f"    deps_proba (top 5):")
                    for deprel, prob in sorted(deps_p.items(), key=lambda x: -x[1])[:5]:
                        marker = " ← CHOSEN" if deprel == first_token['deprel'] else ""
                        print(f"      - {deprel}: {prob:.4f}{marker}")

                # Показываем весь токен в JSON для полноты
                print(f"\n  Full token as JSON:")
                print(json.dumps(first_token, indent=4, ensure_ascii=False))

        else:  # 'dict'
            # ================================================================
            # DICT ФОРМАТ - преобразуем в DataFrame для наглядности
            # ================================================================
            sentences = result
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
        # ====================================================================

        print(f"\n✅ Test completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
