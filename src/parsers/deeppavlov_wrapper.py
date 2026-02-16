#!/usr/bin/env python3
"""
Обёртка для DeepPavlov (через Modal) с поддержкой ПОЛНОГО выхода.
"""

import logging
import modal
from typing import List, Dict, Any, Literal, Union
import json

logger = logging.getLogger(__name__)


class DeepPavlovParser:
    """Клиент для DeepPavlov, запущенного в Modal."""

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
        """Парсит текст с выбранным токенизатором."""
        try:
            if self.tokenizer_type == 'razdel':
                results = self.service.parse_text.remote(
                    text, 
                    output_format=output_format,
                    use_cache=use_cache
                )
            elif self.tokenizer_type == 'native':
                if output_format == 'full':
                    self.logger.warning("Full format not supported with native tokenizer.")
                    output_format = 'dict'
                results = self.service.parse_text_native.remote(text, output_format=output_format)
            else:
                raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")

            return results if results else ([] if output_format == 'dict' else '')
        except Exception as e:
            self.logger.error(f"Error during DeepPavlov parsing: {e}")
            raise e

    def parse_batch(
        self, 
        texts: List[str], 
        output_format: str = 'dict',
        use_cache: bool = False
    ) -> Union[List, List[str], List[Dict]]:
        """Парсит батч текстов."""
        try:
            if self.tokenizer_type == 'razdel':
                return self.service.parse_batch.remote(texts, output_format=output_format, use_cache=use_cache)
            elif self.tokenizer_type == 'native':
                if output_format == 'full':
                    self.logger.warning("Full format not supported with native tokenizer.")
                    output_format = 'dict'
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

    logging.basicConfig(level=logging.INFO)

    parser_args = argparse.ArgumentParser(description='Test DeepPavlov parser')
    parser_args.add_argument('--tokenizer', type=str, choices=['razdel', 'native'], default='razdel')
    parser_args.add_argument('--output-format', type=str, choices=['dict', 'conllu', 'full', 'both'], default='both')
    parser_args.add_argument('--use-cache', action='store_true')
    args = parser_args.parse_args()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    print(f"{'=' * 70}")
    print(f"🚀 Testing DeepPavlov with {args.tokenizer.upper()} tokenizer")
    print(f"   Output format: {args.output_format.upper()}")
    print(f"   Caching: {'ENABLED' if args.use_cache else 'DISABLED'}")
    print(f"{'=' * 70}")

    try:
        parser = DeepPavlovParser(tokenizer=args.tokenizer)

        # ====================================================================
        # BOTH: Тестируем оба формата (standard dict + full)
        # ====================================================================
        if args.output_format == 'both':
            # ================================================================
            # ВАРИАНТ 1: STANDARD (dict)
            # ================================================================
            print(f"\n{'═'*70}")
            print(f"📊 ВАРИАНТ 1: STANDARD (dict)")
            print(f"{'═'*70}")

            result_dict = parser.parse_text(test_text, output_format='dict', use_cache=args.use_cache)
            sentences = result_dict
            print(f"\nReceived {len(sentences)} sentence(s)")

            all_tokens = [token for sent in sentences for token in sent]
            df = pd.DataFrame(all_tokens)

            print(f"\n{'─'*70}")
            print(f"📄 DeepPavlov Joint Parsing ({args.tokenizer})")
            print(f"{'─'*70}")
            if not df.empty:
                cols = ['id', 'form', 'lemma', 'upos', 'head', 'deprel']
                available_cols = [col for col in cols if col in df.columns]
                print(df[available_cols].to_string(index=False))

                print(f"\n{'─'*70}")
                print(f"📋 Morphological Features")
                print(f"{'─'*70}")
                if 'feats' in df.columns:
                    print(df[['form', 'feats']].to_string(index=False))

                if 'startchar' in df.columns and args.tokenizer == 'razdel':
                    print(f"\n{'─'*70}")
                    print(f"📍 Character Offsets (Razdel)")
                    print(f"{'─'*70}")
                    print(df[['form', 'startchar', 'endchar']].to_string(index=False))

            # ================================================================
            # ВАРИАНТ 2: FULL (с probas)
            # ================================================================
            print(f"\n{'═'*70}")
            print(f"📊 ВАРИАНТ 2: FULL (с probas)")
            print(f"{'═'*70}")

            result_full = parser.parse_text(test_text, output_format='full', use_cache=args.use_cache)

            print(f"\n📋 Structure:")
            print(f"  format: {result_full['format']}")
            print(f"  conllu: <{len(result_full['conllu'])} chars>")
            print(f"  sentences: {len(result_full['sentences'])} sentence(s)")

            # Выводим ПЕРВЫЕ 3 ТОКЕНА детально
            print(f"\n{'─'*70}")
            print(f"📊 First 3 tokens with probas:")
            print(f"{'─'*70}")

            if result_full['sentences']:
                first_sent = result_full['sentences'][0]
                for tok_idx, token in enumerate(first_sent[:3], 1):
                    print(f"\n  [{tok_idx}] {token['form']}")
                    print(f"      {'─'*62}")
                    print(f"      ID: {token['id']}")
                    print(f"      Lemma: {token['lemma']}")
                    print(f"      UPOS: {token['upos']}")

                    # UPOS proba с визуализацией
                    upos_proba = token.get('upos_proba', 0)
                    bar = '█' * int(upos_proba * 20)
                    print(f"      UPOS confidence: {upos_proba:.4f} {bar}")

                    print(f"\n      Head: {token['head']}")
                    print(f"      Deprel: {token['deprel']}")

                    # Heads probabilities (TOP-5)
                    heads_p = token.get('heads_proba', [])
                    if heads_p:
                        print(f"\n      Heads probabilities (TOP-5 from K+1={len(heads_p)}):")
                        heads_enum = [(i, p) for i, p in enumerate(heads_p)]
                        heads_enum.sort(key=lambda x: -x[1])

                        for head_idx, prob in heads_enum[:5]:
                            if head_idx == 0:
                                head_label = "ROOT"
                            else:
                                if head_idx <= len(first_sent):
                                    head_form = first_sent[head_idx-1]['form']
                                    head_label = f"→ {head_form} (id={head_idx})"
                                else:
                                    head_label = f"id={head_idx}"

                            marker = " ✓" if head_idx == token['head'] else ""
                            bar = '█' * int(prob * 20)
                            print(f"        [{head_idx:2d}] {head_label:20s} {prob:.4f} {bar}{marker}")

                    # Dependency relation probabilities (TOP-5)
                    deps_p = token.get('deps_proba', {})
                    if deps_p:
                        print(f"\n      Dependency relation probabilities (TOP-5):")
                        top_deps = sorted(deps_p.items(), key=lambda x: -x[1])[:5]

                        for deprel, prob in top_deps:
                            marker = " ✓" if deprel == token['deprel'] else ""
                            bar = '█' * int(prob * 20)
                            print(f"        {deprel:12s} {prob:.4f} {bar}{marker}")

                # Сокращённый вывод для остальных токенов
                if len(first_sent) > 3:
                    print(f"\n  ... и ещё {len(first_sent) - 3} токен(ов) с probas")

            # Статистика
            print(f"\n{'─'*70}")
            print(f"📈 Confidence Statistics:")
            print(f"{'─'*70}")

            all_upos = []
            all_heads = []
            all_deps = []

            for sent in result_full['sentences']:
                for token in sent:
                    all_upos.append(token.get('upos_proba', 0))

                    heads_p = token.get('heads_proba', [])
                    if heads_p and token['head'] < len(heads_p):
                        all_heads.append(heads_p[token['head']])

                    deps_p = token.get('deps_proba', {})
                    if token['deprel'] in deps_p:
                        all_deps.append(deps_p[token['deprel']])

            if all_upos:
                print(f"\nUPOS confidence:")
                print(f"  Average: {sum(all_upos)/len(all_upos):.4f}")
                print(f"  Min: {min(all_upos):.4f}")
                print(f"  Max: {max(all_upos):.4f}")

            if all_heads:
                print(f"\nHead attachment confidence:")
                print(f"  Average: {sum(all_heads)/len(all_heads):.4f}")
                print(f"  Min: {min(all_heads):.4f}")
                print(f"  Max: {max(all_heads):.4f}")

            if all_deps:
                print(f"\nDependency relation confidence:")
                print(f"  Average: {sum(all_deps)/len(all_deps):.4f}")
                print(f"  Min: {min(all_deps):.4f}")
                print(f"  Max: {max(all_deps):.4f}")

        # ====================================================================
        # Одиночные форматы
        # ====================================================================
        elif args.output_format == 'conllu':
            result = parser.parse_text(test_text, output_format='conllu', use_cache=args.use_cache)
            print(f"\n{'─'*70}")
            print(f"📄 CoNLL-U Output ({args.tokenizer})")
            print(f"{'─'*70}\n")
            print(result)

        elif args.output_format == 'full':
            result_full = parser.parse_text(test_text, output_format='full', use_cache=args.use_cache)

            print(f"\n{'─'*70}")
            print(f"📊 FULL Output with probas")
            print(f"{'─'*70}")

            # Полный вывод ВСЕХ токенов
            for sent_idx, sent in enumerate(result_full['sentences'], 1):
                print(f"\n{'═'*70}")
                print(f"Sentence {sent_idx}: {len(sent)} tokens")
                print(f"{'═'*70}")

                for tok_idx, token in enumerate(sent, 1):
                    print(f"\n  [{tok_idx}] {token['form']}")
                    print(f"      {'─'*62}")
                    print(f"      ID: {token['id']}")
                    print(f"      Lemma: {token['lemma']}")
                    print(f"      UPOS: {token['upos']}")

                    upos_proba = token.get('upos_proba', 0)
                    bar = '█' * int(upos_proba * 20)
                    print(f"      UPOS confidence: {upos_proba:.4f} {bar}")

                    print(f"\n      Head: {token['head']}")
                    print(f"      Deprel: {token['deprel']}")

                    heads_p = token.get('heads_proba', [])
                    if heads_p:
                        print(f"\n      Heads probabilities (TOP-5):")
                        heads_enum = [(i, p) for i, p in enumerate(heads_p)]
                        heads_enum.sort(key=lambda x: -x[1])

                        for head_idx, prob in heads_enum[:5]:
                            if head_idx == 0:
                                head_label = "ROOT"
                            else:
                                if head_idx <= len(sent):
                                    head_form = sent[head_idx-1]['form']
                                    head_label = f"→ {head_form} (id={head_idx})"
                                else:
                                    head_label = f"id={head_idx}"

                            marker = " ✓" if head_idx == token['head'] else ""
                            bar = '█' * int(prob * 20)
                            print(f"        [{head_idx:2d}] {head_label:20s} {prob:.4f} {bar}{marker}")

                    deps_p = token.get('deps_proba', {})
                    if deps_p:
                        print(f"\n      Dependency relation probabilities (TOP-5):")
                        top_deps = sorted(deps_p.items(), key=lambda x: -x[1])[:5]

                        for deprel, prob in top_deps:
                            marker = " ✓" if deprel == token['deprel'] else ""
                            bar = '█' * int(prob * 20)
                            print(f"        {deprel:12s} {prob:.4f} {bar}{marker}")

        else:  # 'dict'
            result = parser.parse_text(test_text, output_format='dict', use_cache=args.use_cache)
            sentences = result
            print(f"\nReceived {len(sentences)} sentence(s)")

            all_tokens = [token for sent in sentences for token in sent]
            df = pd.DataFrame(all_tokens)

            print(f"\n{'─'*70}")
            print(f"📄 DeepPavlov Joint Parsing ({args.tokenizer})")
            print(f"{'─'*70}")
            if not df.empty:
                cols = ['id', 'form', 'lemma', 'upos', 'head', 'deprel']
                available_cols = [col for col in cols if col in df.columns]
                print(df[available_cols].to_string(index=False))

                print(f"\n{'─'*70}")
                print(f"📋 Morphological Features")
                print(f"{'─'*70}")
                if 'feats' in df.columns:
                    print(df[['form', 'feats']].to_string(index=False))

                if 'startchar' in df.columns and args.tokenizer == 'razdel':
                    print(f"\n{'─'*70}")
                    print(f"📍 Character Offsets (Razdel)")
                    print(f"{'─'*70}")
                    print(df[['form', 'startchar', 'endchar']].to_string(index=False))

        print(f"\n{'='*70}")
        print(f"✅ Test completed!")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
