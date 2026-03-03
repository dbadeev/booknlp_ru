#!/usr/bin/env python3
"""
Обёртка для DeepPavlov (через Modal) с поддержкой ПОЛНОГО выхода.
Chunking по 32 предложения выполняется на стороне wrapper,
Modal распределяет чанки по контейнерам через .map().
"""

import logging
import modal
from razdel import sentenize                        # ← новый импорт
from typing import List, Dict, Any, Literal, Union, Tuple

logger = logging.getLogger(__name__)


class DeepPavlovParser:
    """Клиент для DeepPavlov, запущенного в Modal."""

    def __init__(self, tokenizer: Literal['razdel', 'native'] = 'razdel'):
        self.logger = logging.getLogger(__name__)
        self.tokenizer_type = tokenizer
        try:
            self.service = modal.Cls.from_name(
                "booknlp-ru-deeppavlov", "DeepPavlovService"
            )()
            self.logger.info(
                f"Connected to DeepPavlov via Modal (tokenizer: {tokenizer})."
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal app: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_to_chunks(
        text: str,
        chunk_size: int,
        base_offset: int = 0,
    ) -> List[List[Tuple[str, int]]]:
        """
        Разбивает текст на чанки по chunk_size предложений.
        Возвращает List[List[(sentence_text, start_char_in_original_text)]].
        base_offset — смещение text внутри более крупного документа (для parse_batch).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        sentences = list(sentenize(text))
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = [
                (s.text, base_offset + s.start)
                for s in sentences[i:i + chunk_size]
            ]
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _format_conllu_local(sentences: List[List[Dict]]) -> str:
        """Локальное форматирование CoNLL-U (зеркало modal._format_conllu_output)."""
        blocks = []
        for sent in sentences:
            lines = []
            for t in sent:
                if '-' in str(t.get('id', '')):
                    continue
                lines.append('\t'.join([
                    str(t.get('id', 0)),
                    t.get('form', '_'),
                    t.get('lemma', '_'),
                    t.get('upos', '_'),
                    t.get('xpos', '_'),
                    t.get('feats', '_'),
                    str(t.get('head', 0)),
                    t.get('deprel', '_'),
                    t.get('deps', '_'),
                    t.get('misc', '_'),
                ]))
            blocks.append('\n'.join(lines))
        return '\n\n'.join(blocks) + '\n'

    @staticmethod
    def _merge_chunks(
        chunk_results: List[Union[List[List[Dict]], Dict]],
        output_format: str,
    ) -> Union[List[List[Dict]], str, Dict]:
        """
        Склеивает результаты чанков в единый результат.
        chunk_results при output_format="conllu" содержат dict,
        потому что CoNLL-U форматируется локально после сборки.
        """
        if output_format == "full":
            all_sentences: List[List[Dict]] = []
            all_conllu_parts: List[str] = []
            metadata: Dict = {}
            for cr in chunk_results:
                all_sentences.extend(cr['sentences'])
                all_conllu_parts.append(cr['conllu'].rstrip('\n'))
                metadata = cr.get('metadata', metadata)
            return {
                'format': 'full',
                'conllu': '\n\n'.join(all_conllu_parts) + '\n',
                'sentences': all_sentences,
                'metadata': metadata,
            }

        # dict или conllu: чанки пришли как List[List[Dict]]
        all_sents: List[List[Dict]] = [
            sent for cr in chunk_results for sent in cr
        ]
        if output_format == "conllu":
            return DeepPavlovParser._format_conllu_local(all_sents)
        return all_sents  # "dict"

    @staticmethod
    def _split_to_sentence_chunks(
            text: str,
            chunk_size: int,
    ) -> List[List[str]]:
        """
        Разбивает текст на чанки предложений (только тексты, без офсетов).
        Используется для native-токенизатора.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        sentences = list(sentenize(text))
        return [
            [s.text for s in sentences[i:i + chunk_size]]
            for i in range(0, len(sentences), chunk_size)
        ]

    # ------------------------------------------------------------------ #
    #  Публичный API                                                      #
    # ------------------------------------------------------------------ #
    # razdel:  sentenize → _split_to_chunks         → .map(parse_sentence_chunk)
    # native:  sentenize → _split_to_sentence_chunks → .map(parse_sentence_chunk_native)
    # Обе ветки используют .map() с единственным различием — razdel несёт символьные офсеты, native — нет.

    def parse_text(
        self,
        text: str,
        output_format: str = "dict",
        chunk_size: int = 32,
    ) -> Union[List[List[Dict[str, Any]]], str, Dict[str, Any]]:
        try:
            if self.tokenizer_type == "native":
                if output_format == "full":
                    self.logger.warning("Full format not supported with native tokenizer.")
                    output_format = "dict"
                internal_format = "dict" if output_format == "conllu" else output_format

                chunks = self._split_to_sentence_chunks(text, chunk_size)
                if not chunks:
                    return []

                if len(chunks) == 1:
                    res = self.service.parse_sentence_chunk_native.remote(
                        chunks[0], output_format=internal_format
                    )
                    return self._merge_chunks([res], output_format)

                # .map() — Modal распределяет чанки по контейнерам
                chunk_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        chunks,
                        kwargs={"output_format": internal_format},
                    )
                )
                return self._merge_chunks(chunk_results, output_format)

            # razdel: wrapper делает chunking
            chunks = self._split_to_chunks(text, chunk_size)
            if not chunks:
                return (
                    {'format': 'full', 'conllu': '', 'sentences': [], 'metadata': {}}
                    if output_format == "full" else []
                )

            # Для conllu запрашиваем dict и форматируем локально
            internal_format = "dict" if output_format == "conllu" else output_format

            if len(chunks) == 1:
                result = self.service.parse_sentence_chunk.remote(
                    chunks[0], output_format=internal_format
                )
                # Оборачиваем в список для _merge_chunks
                return self._merge_chunks([result], output_format)

            # Несколько чанков → .map() → Modal распределяет по контейнерам
            chunk_results = list(
                self.service.parse_sentence_chunk.map(
                    chunks,
                    kwargs={"output_format": internal_format},
                )
            )
            return self._merge_chunks(chunk_results, output_format)

        except Exception as e:
            self.logger.error(f"Error during DeepPavlov parsing: {e}")
            raise

    def parse_batch(
        self,
        texts: List[str],
        output_format: str = "dict",
        chunk_size: int = 32,
    ) -> List[Union[List[List[Dict[str, Any]]], str, Dict[str, Any]]]:
        """
        Разбивает ВСЕ тексты на чанки по chunk_size предложений,
        отправляет все чанки сразу через .map() — Modal распределяет
        их по доступным контейнерам (max_containers=4).
        """
        try:
            if self.tokenizer_type == "native":
                if output_format == "full":
                    raise ValueError(
                        "output_format='full' is not supported with native tokenizer."
                    )

                internal_format = "dict" if output_format == "conllu" else output_format

                native_chunks: List[List[str]] = []
                native_chunks_per_text: List[int] = []

                for text in texts:
                    text_chunks = self._split_to_sentence_chunks(text, chunk_size)
                    native_chunks_per_text.append(len(text_chunks))
                    native_chunks.extend(text_chunks)

                all_chunk_results = list(
                    self.service.parse_sentence_chunk_native.map(
                        native_chunks,
                        kwargs={"output_format": internal_format},
                    )
                )

                # Reassemble — идентично razdel-ветке
                results = []
                offset = 0
                for n_chunks in native_chunks_per_text:
                    results.append(self._merge_chunks(
                        all_chunk_results[offset:offset + n_chunks], output_format
                    ))
                    offset += n_chunks
                return results

            # razdel: wrapper делает chunking по всем текстам
            internal_format = "dict" if output_format == "conllu" else output_format

            all_chunks: List[List[Tuple[str, int]]] = []
            chunks_per_text: List[int] = []      # сколько чанков у каждого текста

            for text in texts:
                text_chunks = self._split_to_chunks(text, chunk_size)
                chunks_per_text.append(len(text_chunks))
                all_chunks.extend(text_chunks)

            if not all_chunks:
                return [[] for _ in texts]

            # Один .map() — все чанки всех текстов одновременно
            all_chunk_results = list(
                self.service.parse_sentence_chunk.map(
                    all_chunks,
                    kwargs={"output_format": internal_format},
                )
            )

            # Reassemble: собираем результаты обратно по текстам
            results = []
            offset = 0
            for n_chunks in chunks_per_text:
                text_chunk_results = all_chunk_results[offset:offset + n_chunks]
                results.append(self._merge_chunks(text_chunk_results, output_format))
                offset += n_chunks

            return results

        except Exception as e:
            self.logger.error(f"Error during DeepPavlov batch parsing: {e}")
            raise


if __name__ == "__main__":
    import pandas as pd
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser_args = argparse.ArgumentParser(description='Test DeepPavlov parser')
    parser_args.add_argument('--tokenizer', type=str, choices=['razdel', 'native'], default='razdel')
    parser_args.add_argument('--output-format', type=str, choices=['dict', 'conllu', 'full', 'both'], default='both')
    args = parser_args.parse_args()

    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    # test_text = "Мама без мыла мыла раму."

    print(f"{'=' * 70}")
    print(f"🚀 Testing DeepPavlov with {args.tokenizer.upper()} tokenizer")
    print(f"   Output format: {args.output_format.upper()}")
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

            result_dict = parser.parse_text(test_text, output_format='dict')
            sentences = result_dict
            print(f"\nReceived {len(sentences)} sentence(s)")

            all_tokens = [token for sent in sentences for token in sent]
            df = pd.DataFrame(all_tokens)

            print(f"\n{'─'*70}")
            print(f"📄 DeepPavlov Joint Parsing ({args.tokenizer})")
            print(f"{'─'*70}")
            if not df.empty:
                cols = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
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

            result_full = parser.parse_text(test_text, output_format='full')

            print(f"\n📋 Structure:")
            print(f"  format: {result_full['format']}")
            print(f"  conllu: <{len(result_full['conllu'])} chars>")
            print(f"  sentences: {len(result_full['sentences'])} sentence(s)")

            # Выводим ПЕРВЫЕ 3 ТОКЕНА детально
            print(f"\n{'─'*70}")
            print(f"📊 Tokens with probas:")
            print(f"{'─'*70}")

            if result_full['sentences']:
                first_sent = result_full['sentences'][0]
                for tok_idx, token in enumerate(first_sent, 1):
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
            result = parser.parse_text(test_text, output_format='conllu')
            print(f"\n{'─'*70}")
            print(f"📄 CoNLL-U Output ({args.tokenizer})")
            print(f"{'─'*70}\n")
            print(result)

        elif args.output_format == 'full':
            result_full = parser.parse_text(test_text, output_format='full')

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
            result = parser.parse_text(test_text, output_format='dict')
            sentences = result
            print(f"\nReceived {len(sentences)} sentence(s)")

            all_tokens = [token for sent in sentences for token in sent]
            df = pd.DataFrame(all_tokens)

            print(f"\n{'─'*70}")
            print(f"📄 DeepPavlov Joint Parsing ({args.tokenizer})")
            print(f"{'─'*70}")
            if not df.empty:
                cols = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
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
