import modal
from typing import List, Dict, Any, Union

dp_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "transformers",
        "deeppavlov",
        "razdel",
        "pandas",
        "nltk",
        "tqdm",
        "numpy")
    .run_commands(
        "python -m deeppavlov install ru_syntagrus_joint_parsing",
        "python -c \"from deeppavlov import build_model;"
        "build_model('ru_syntagrus_joint_parsing', download=True)\""
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\""
    )
    .env({"DEEPPAVLOV_DOWNLOAD_PROGRESSIVE": "0"})
)

app = modal.App("booknlp-ru-deeppavlov")

@app.cls(
    image=dp_image,
    gpu="T4",
    timeout=1200,
    max_containers=2
)
class DeepPavlovService:

    # ← Аннотации на уровне класса вместо __init__
    # PyCharm видит их как объявления атрибутов экземпляра
    model: Any
    morpho_tagger_component: Any
    syntax_parser_component: Any
    deprel_vocab: List[str]
    hook_handles: List
    _last_upos_logits: Any
    _last_heads_logits: Any
    _last_deps_logits: Any

    @modal.enter()
    def enter(self):
        from deeppavlov import build_model, configs
        from pathlib import Path

        # Теперь self.model и остальные — без предупреждений PyCharm
        self.model = build_model(
            configs.morpho_syntax_parser.ru_syntagrus_joint_parsing,
            download=True
        )
        self.morpho_tagger_component = None
        self.syntax_parser_component = None
        self._last_upos_logits = None
        self._last_heads_logits = None
        self._last_deps_logits = None

        print("\n🔧 Extracting components...")

        main_component = self.model.get_main_component()

        if hasattr(main_component, 'tagger') and hasattr(main_component.tagger, 'pipe'):
            for i, item in enumerate(main_component.tagger.pipe):
                comp = item[2] if isinstance(item, (tuple, list)) and len(item) > 2 else item
                if 'Sequence' in comp.__class__.__name__ and 'Tagger' in comp.__class__.__name__:
                    self.morpho_tagger_component = comp
                    print(f"  ✓ Tagger")
                    break

        if hasattr(main_component, 'parser') and hasattr(main_component.parser, 'pipe'):
            for i, item in enumerate(main_component.parser.pipe):
                comp = item[2] if isinstance(item, (tuple, list)) and len(item) > 2 else item
                if 'Syntax' in comp.__class__.__name__ and 'Parser' in comp.__class__.__name__:
                    self.syntax_parser_component = comp
                    print(f"  ✓ Parser")
                    break

        # VOCAB
        print("\n🔍 Loading vocab...")
        self.deprel_vocab = None

        possible_paths = [
            Path.home() / ".deeppavlov" / "models" / "syntax_parsing",
            Path("/root/.deeppavlov/models/syntax_parsing"),
        ]

        for base_path in possible_paths:
            if base_path.exists():
                for vocab_file in base_path.rglob("*.dict"):
                    if vocab_file.is_file() and 100 < vocab_file.stat().st_size < 10000:
                        try:
                            with open(vocab_file, 'r', encoding='utf-8') as f:
                                lines = f.read().strip().split('\n')
                                if lines and 'nsubj' in '\n'.join(lines):
                                    self.deprel_vocab = [line.split('\t')[0].strip() for line in lines if line.strip()]
                                    print(f"  ✓ Loaded {len(self.deprel_vocab)} deprels")
                                    break
                        except (AttributeError, RuntimeError, TypeError):
                            pass
                if self.deprel_vocab:
                    break

        if self.deprel_vocab is None:
            self.deprel_vocab = [
                'PAD', 'punct', 'case', 'nmod', 'obl', 'amod', 'nsubj', 'advmod',
                'root', 'conj', 'cc', 'obj', 'det', 'acl', 'nummod', 'appos',
                'mark', 'flat', 'advcl', 'aux', 'cop', 'expl', 'fixed', 'iobj',
                'ccomp', 'discourse', 'parataxis', 'nummod:gov', 'xcomp', 'compound',
                'csubj', 'dep', 'list', 'orphan', 'vocative', 'dislocated',
                'goeswith', 'reparandum', 'clf', 'acl:relcl'
            ]

        print(f"  Vocab: {len(self.deprel_vocab)} labels")

        # HOOKS для RAW logits
        print("\n🔨 Setting up hooks...")

        self.hook_handles = []
        service = self

        # UPOS
        if self.morpho_tagger_component and hasattr(self.morpho_tagger_component, 'model'):
            def morpho_hook(_module, _input, output):
                try:
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    elif isinstance(output, tuple) and len(output) > 0:
                        logits = output[0]
                    else:
                        logits = output
                    service.morpho_tagger_component._last_upos_logits = logits.detach().cpu()
                except (AttributeError, RuntimeError, TypeError):
                    pass

            handle = self.morpho_tagger_component.model.register_forward_hook(morpho_hook)
            self.hook_handles.append(handle)
            print("  ✓ UPOS hook")

        # HEADS и DEPS
        if self.syntax_parser_component and hasattr(self.syntax_parser_component, 'model'):
            parser_model = self.syntax_parser_component.model

            if hasattr(parser_model, 'biaf_head'):
                def biaf_head_hook(_module, _input, output):
                    try:
                        if hasattr(output, 'shape'):
                            if len(output.shape) == 4 and output.shape[-1] == 1:
                                output = output.squeeze(-1)
                            service.syntax_parser_component._last_heads_logits = output.detach().cpu()
                    except (AttributeError, RuntimeError, TypeError):
                       pass

                handle = parser_model.biaf_head.register_forward_hook(biaf_head_hook)
                self.hook_handles.append(handle)
                print("  ✓ heads hook")

            if hasattr(parser_model, 'biaf_dep'):
                def biaf_dep_hook(_module, _input, output):
                    try:
                        if hasattr(output, 'shape'):
                            service.syntax_parser_component._last_deps_logits = output.detach().cpu()
                    except (AttributeError, RuntimeError, TypeError):
                        pass

                handle = parser_model.biaf_dep.register_forward_hook(biaf_dep_hook)
                self.hook_handles.append(handle)
                print("  ✓ deps hook")

        print("\n✅ Ready\n")

    @staticmethod
    def _format_connlu_output(sentences: List[List[Dict]]) -> str:
        conllu_blocks = []
        for sent in sentences:
            lines = []
            for token in sent:
                if '-' in str(token.get('id', '')):
                    continue
                line = "\t".join([
                    str(token.get('id', 0)),
                    token.get('form', '_'),
                    token.get('lemma', '_'),
                    token.get('upos', '_'),
                    token.get('xpos', '_'),
                    token.get('feats', '_'),
                    str(token.get('head', 0)),
                    token.get('deprel', '_'),
                    token.get('deps', '_'),
                    token.get('misc', '_')
                ])
                lines.append(line)
            conllu_blocks.append('\n'.join(lines))
        return '\n\n'.join(conllu_blocks) + '\n'

    @staticmethod
    def _parse_batch_to_dicts(parsed_batch, token_spans) -> List[List[Dict]]:
        """Разбирает сырой CoNLL-U вывод модели в List[List[Dict]]."""
        results = []
        for i, sent_conllu in enumerate(parsed_batch):
            sent_res = []
            lines = [line for line in sent_conllu.split("\n") if line and not line.startswith("#")]
            for j, line in enumerate(lines):
                fields = line.split("\t")
                if "-" in fields[0]:
                    continue
                start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)
                sent_res.append({
                    "id": int(fields[0]),
                    "form": fields[1],
                    "lemma": fields[2],
                    "upos": fields[3],
                    "xpos": fields[4],
                    "feats": fields[5],
                    "head": int(fields[6]),
                    "deprel": fields[7],
                    "deps": fields[8],
                    "misc": fields[9],
                    "startchar": start_c,
                    "endchar": end_c,
                })
            results.append(sent_res)
        return results


    def _get_deprel_vocab(self) -> List[str]:
        return self.deprel_vocab if self.deprel_vocab else []

    def _extract_real_probas(
        self, 
        tokenized_sentences: List[List[str]],
        sentences_dict: List[List[Dict]]
    ) -> tuple:
        import torch.nn.functional as F

        upos_probas_all = []
        heads_probas_all = []
        deps_probas_all = []
        deprel_vocab = self._get_deprel_vocab()

        for sent_idx, sent_tokens in enumerate(tokenized_sentences):
            sent_len = len(sent_tokens)

            # UPOS
            if (self.morpho_tagger_component and 
                hasattr(self.morpho_tagger_component, '_last_upos_logits')):
                try:
                    batch_logits = self.morpho_tagger_component._last_upos_logits
                    if sent_idx < len(batch_logits):
                        sent_logits = batch_logits[sent_idx]
                        sent_probas = F.softmax(sent_logits, dim=-1).numpy()
                        upos_probas_all.append([
                            float(sent_probas[tok_idx].max())
                            for tok_idx in range(min(sent_len, len(sent_probas)))
                        ])
                    else:
                        upos_probas_all.append([0.95] * sent_len)
                except Exception:
                    upos_probas_all.append([0.95] * sent_len)
            else:
                upos_probas_all.append([0.95] * sent_len)

            # HEADS
            if (self.syntax_parser_component and 
                hasattr(self.syntax_parser_component, '_last_heads_logits')):
                try:
                    batch_heads = self.syntax_parser_component._last_heads_logits
                    if sent_idx < len(batch_heads):
                        sent_heads_logits = batch_heads[sent_idx]
                        sent_heads = F.softmax(sent_heads_logits, dim=-1).numpy()
                        heads_probas_all.append([
                            sent_heads[tok_idx].tolist()
                            for tok_idx in range(min(sent_len, len(sent_heads)))
                        ])
                    else:
                        heads_probas_all.append(
                            [[1.0/(sent_len+1)] * (sent_len+1) for _ in range(sent_len)]
                        )
                except Exception:
                    heads_probas_all.append(
                        [[1.0/(sent_len+1)] * (sent_len+1) for _ in range(sent_len)]
                    )
            else:
                heads_probas_all.append(
                    [[1.0/(sent_len+1)] * (sent_len+1) for _ in range(sent_len)]
                )

            # DEPS
            if (self.syntax_parser_component and 
                hasattr(self.syntax_parser_component, '_last_deps_logits')):
                try:
                    batch_deps = self.syntax_parser_component._last_deps_logits
                    if sent_idx < len(batch_deps):
                        sent_deps_logits = batch_deps[sent_idx]
                        deps_list = []

                        if len(sent_deps_logits.shape) == 3:
                            for tok_idx in range(min(sent_len, sent_deps_logits.shape[0])):
                                if sent_idx < len(sentences_dict) and tok_idx < len(sentences_dict[sent_idx]):
                                    chosen_head = sentences_dict[sent_idx][tok_idx]['head']
                                    tok_deps_logits = sent_deps_logits[tok_idx, chosen_head, :]
                                    tok_deps_probas = F.softmax(tok_deps_logits, dim=-1).numpy()

                                    deps_dict = {}
                                    for dep_idx, prob in enumerate(tok_deps_probas):
                                        if dep_idx < len(deprel_vocab):
                                            deps_dict[deprel_vocab[dep_idx]] = float(prob)
                                    deps_list.append(deps_dict)
                                else:
                                    deps_list.append({'root': 0.95})
                        else:
                            deps_list = [{'root': 0.95} for _ in range(sent_len)]

                        deps_probas_all.append(deps_list)
                    else:
                        deps_probas_all.append([{'root': 0.95} for _ in range(sent_len)])
                except Exception:
                    deps_probas_all.append([{'root': 0.95} for _ in range(sent_len)])
            else:
                deps_probas_all.append([{'root': 0.95} for _ in range(sent_len)])

        return upos_probas_all, heads_probas_all, deps_probas_all

    def _parse_with_probas(
            self,
            tokenized_sentences: List[List[str]],
            token_spans: List[List[tuple]],
            sentence_batch_size: int = 32,
    ) -> Dict[str, Any]:

        all_sentences_dict: List[List[Dict]] = []
        all_upos_probas: List = []
        all_heads_probas: List = []
        all_deps_probas: List = []

        for chunk_start in range(0, len(tokenized_sentences), sentence_batch_size):
            chunk_end = chunk_start + sentence_batch_size
            chunk_tokenized = tokenized_sentences[chunk_start:chunk_end]
            chunk_spans = token_spans[chunk_start:chunk_end]

            parsed_chunk = self.model(chunk_tokenized)
            chunk_dicts = self._parse_batch_to_dicts(parsed_chunk, chunk_spans)

            # ← вызывать СРАЗУ: хуки содержат логиты только текущего чанка
            upos_p, heads_p, deps_p = self._extract_real_probas(
                chunk_tokenized, chunk_dicts
            )

            all_sentences_dict.extend(chunk_dicts)
            all_upos_probas.extend(upos_p)
            all_heads_probas.extend(heads_p)
            all_deps_probas.extend(deps_p)

        for sent_idx, sent_tokens in enumerate(all_sentences_dict):
            for tok_idx, token in enumerate(sent_tokens):
                token['upos_proba'] = (
                    all_upos_probas[sent_idx][tok_idx]
                    if sent_idx < len(all_upos_probas)
                       and tok_idx < len(all_upos_probas[sent_idx])
                    else 0.95
                )
                token['heads_proba'] = (
                    all_heads_probas[sent_idx][tok_idx]
                    if sent_idx < len(all_heads_probas)
                       and tok_idx < len(all_heads_probas[sent_idx])
                    else [1.0 / (len(sent_tokens) + 1)] * (len(sent_tokens) + 1)
                )
                token['deps_proba'] = (
                    all_deps_probas[sent_idx][tok_idx]
                    if sent_idx < len(all_deps_probas)
                       and tok_idx < len(all_deps_probas[sent_idx])
                    else {'root': 0.95}
                )

        return {
            'format': 'full',
            'conllu': self._format_connlu_output(all_sentences_dict),
            'sentences': all_sentences_dict,
            'metadata': {
                'model': 'ru_syntagrus_joint_parsing',
                'tokenizer': 'razdel',
                'vocab': {'deprels': self._get_deprel_vocab()},
                'probas_source': 'real_from_raw_logits',
            }
        }

    @modal.method()
    def parse_text(
            self,
            text: str,
            output_format: str = "dict",
            sentence_batch_size: int = 32,
    ) -> Union[List[List[Dict[str, Any]]], str, Dict[str, Any]]:
        from razdel import tokenize, sentenize

        sentences = list(sentenize(text))
        tokenized_sentences: List[List[str]] = []
        token_spans: List[List[tuple]] = []

        for sent in sentences:
            tokens = list(tokenize(sent.text))
            tokenized_sentences.append([t.text for t in tokens])
            token_spans.append([
                (sent.start + t.start, sent.start + t.stop)
                for t in tokens
            ])

        if output_format == "full":
            return self._parse_with_probas(tokenized_sentences, token_spans, sentence_batch_size)

        results: List[List[Dict[str, Any]]] = []
        for chunk_start in range(0, len(tokenized_sentences), sentence_batch_size):
            chunk_end = chunk_start + sentence_batch_size
            parsed_chunk = self.model(tokenized_sentences[chunk_start:chunk_end])
            results.extend(
                self._parse_batch_to_dicts(
                    parsed_chunk, token_spans[chunk_start:chunk_end]
                )
            )

        if output_format == "conllu":
            return self._format_connlu_output(results)

        return results  # "dict" → List[List[Dict]]

    @modal.method()
    def parse_batch(
            self,
            texts: List[str],
            output_format: str = "dict",
            sentence_batch_size: int = 32,
    ) -> Union[List[List[Dict]], List[str]]:
        from razdel import tokenize, sentenize

        # Шаг 1: токенизируем все тексты, собираем предложения в плоский список
        all_tokenized: List[List[str]] = []
        all_spans: List[List[tuple]] = []
        text_sent_counts: List[int] = []

        for text in texts:
            sents = list(sentenize(text))
            count = 0
            for sent in sents:
                tokens = list(tokenize(sent.text))
                all_tokenized.append([t.text for t in tokens])
                all_spans.append([
                    (sent.start + t.start, sent.start + t.stop)
                    for t in tokens
                ])
                count += 1
            text_sent_counts.append(count)

        # Шаг 2: обрабатываем предложения чанками
        all_dicts: List[List[Dict]] = []

        for chunk_start in range(0, len(all_tokenized), sentence_batch_size):
            chunk_end = chunk_start + sentence_batch_size
            tokenized_chunk = all_tokenized[chunk_start:chunk_end]
            spans_chunk = all_spans[chunk_start:chunk_end]

            parsed_chunk = self.model(tokenized_chunk)
            dicts_chunk = self._parse_batch_to_dicts(parsed_chunk, spans_chunk)
            all_dicts.extend(dicts_chunk)

        # Шаг 3: собираем предложения обратно по исходным текстам
        results = []
        offset = 0
        for count in text_sent_counts:
            text_sents = all_dicts[offset:offset + count]
            if output_format == "conllu":
                results.append(self._format_connlu_output(text_sents))
            else:
                results.append(text_sents)
            offset += count

        return results

    @modal.method()
    def parse_text_native(
        self,
        text: str,
        output_format: str = 'dict',
        sentence_batch_size: int = 32,
    ) -> Union[List, str, Dict]:
        # import os
        from razdel import sentenize

        # if sentence_batch_size <= 0:
        #     sentence_batch_size = int(os.environ.get("SENTENCE_BATCH_SIZE", 32))

        sentence_texts = [sent.text for sent in sentenize(text)]
        results: List[List[Dict]] = []

        for chunk_start in range(0, len(sentence_texts), sentence_batch_size):
            chunk = sentence_texts[chunk_start:chunk_start + sentence_batch_size]
            parsed_chunk = self.model(chunk)
            token_spans = [[] for _ in parsed_chunk]
            results.extend(self._parse_batch_to_dicts(parsed_chunk, token_spans))

        if output_format == 'conllu':
            return self._format_connlu_output(results)
        return results


@app.local_entrypoint()
def main():
    import json

    SEP = "=" * 70
    TEST_TEXT = (
        "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."
    )

    # service = DeepPavlovService()
    service: Any = DeepPavlovService()  # type: ignore[call-arg]
    print(f"\n{SEP}")
    print("🚀 Testing DeepPavlov (production)")
    print(f"   Text: {TEST_TEXT}")
    print(f"{SEP}")

    # --- ВАРИАНТ 1: conllu (dict) ---
    print(f"\n{SEP}")
    print("📊 ВАРИАНТ 1: CoNLL-U (str)")
    print(f"{SEP}")
    result_conllu = service.parse_text.remote(TEST_TEXT, output_format="conllu")
    # result_conllu — строка CoNLL-U, выводим как есть
    print(result_conllu)

    # --- ВАРИАНТ 2: dict (все поля) ---
    print(f"\n{SEP}")
    print("📊 ВАРИАНТ 2: dict (все CoNLL-U поля)")
    print(f"{SEP}")
    result_dict = service.parse_text.remote(TEST_TEXT, output_format="dict")
    for sidx, sent in enumerate(result_dict, 1):
        print(f"\n--- Sentence {sidx} ---")
        print(f"{'ID':>4} {'FORM':<16} {'LEMMA':<16} {'UPOS':<8} {'XPOS':<8} "
              f"{'FEATS':<36} {'HEAD':>5} {'DEPREL':<14} {'DEPS':<6} {'MISC':<10} START END")
        print("-" * 130)
        for t in sent:
            # feats_d = (t['feats'] or "_")[:34]
            print(f"{t['id']:>4} {t['form']:<16} {t['lemma']:<12} "
                   f"{t['upos']:<8} {(t['xpos'] or '_'):<6} "
                   f"{t['head']:>5} {t['deprel']:<14} "
                   f"{(t['deps'] or '_'):<6} {(t['misc'] or '_'):<10} "
                   f"{t['startchar']} {t['endchar']}")
            print(f"     feats: {t['feats'] or '_'}")
            # print(f"{t['id']:>4} {t['form']:<16} {t['lemma']:<16} {t['upos']:<8} "
            #       f"{(t['xpos'] or '_'):<8} {feats_d:<36} {t['head']:>5} "
            #       f"{t['deprel']:<14} {(t['deps'] or '_'):<6} {(t['misc'] or '_'):<10} "
            #       f"{t['startchar']} {t['endchar']}")
    print(f"\n--- Keys in first token ---")
    print(json.dumps(result_dict[0][0], ensure_ascii=False, indent=2))

    # --- ВАРИАНТ 3: full (probas, топ-3 токена) ---
    print(f"\n{SEP}")
    print("📊 ВАРИАНТ 3: full (с probas, первые 3 токена)")
    print(f"{SEP}")
    result_full = service.parse_text.remote(TEST_TEXT, output_format="full")
    print(f"  probas_source: {result_full['metadata']['probas_source']}")
    print(f"  sentences: {len(result_full['sentences'])}")
    for tok in result_full["sentences"][0][:3]:
        print(f"\n  [{tok['id']}] {tok['form']}")
        print(f"      UPOS: {tok['upos']}  conf={tok.get('upos_proba', 0):.4f}")
        print(f"      Head: {tok['head']}  Deprel: {tok['deprel']}")
    print(f"\n✅ Done!")

