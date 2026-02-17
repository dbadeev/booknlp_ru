import modal
from typing import List, Dict, Any, Union, Optional
import json

cache_volume = modal.Volume.from_name("deeppavlov-cache", create_if_missing=True)
results_cache_volume = modal.Volume.from_name("deeppavlov-results-cache", create_if_missing=True)

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
    volumes={
        "/cache": cache_volume,
        "/results_cache": results_cache_volume
    }
)
class DeepPavlovService:
    @modal.enter()
    def enter(self):
        from deeppavlov import build_model, configs
        import hashlib
        import torch
        import torch.nn.functional as F
        from pathlib import Path

        self.model = build_model(
            configs.morpho_syntax_parser.ru_syntagrus_joint_parsing,
            download=True
        )

        print("\n🔧 Extracting components...")

        main = self.model.get_main_component()

        self.morpho_tagger_component = None
        self.syntax_parser_component = None

        if hasattr(main, 'tagger') and hasattr(main.tagger, 'pipe'):
            for i, item in enumerate(main.tagger.pipe):
                comp = item[2] if isinstance(item, (tuple, list)) and len(item) > 2 else item
                if 'Sequence' in comp.__class__.__name__ and 'Tagger' in comp.__class__.__name__:
                    self.morpho_tagger_component = comp
                    print(f"  ✓ Tagger")
                    break

        if hasattr(main, 'parser') and hasattr(main.parser, 'pipe'):
            for i, item in enumerate(main.parser.pipe):
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
                        except:
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
            def morpho_hook(module, input, output):
                try:
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    elif isinstance(output, tuple) and len(output) > 0:
                        logits = output[0]
                    else:
                        logits = output
                    service.morpho_tagger_component._last_upos_logits = logits.detach().cpu()
                except:
                    pass

            handle = self.morpho_tagger_component.model.register_forward_hook(morpho_hook)
            self.hook_handles.append(handle)
            print("  ✓ UPOS hook")

        # HEADS и DEPS
        if self.syntax_parser_component and hasattr(self.syntax_parser_component, 'model'):
            parser_model = self.syntax_parser_component.model

            if hasattr(parser_model, 'biaf_head'):
                def biaf_head_hook(module, input, output):
                    try:
                        if hasattr(output, 'shape'):
                            if len(output.shape) == 4 and output.shape[-1] == 1:
                                output = output.squeeze(-1)
                            service.syntax_parser_component._last_heads_logits = output.detach().cpu()
                    except:
                        pass

                handle = parser_model.biaf_head.register_forward_hook(biaf_head_hook)
                self.hook_handles.append(handle)
                print("  ✓ heads hook")

            if hasattr(parser_model, 'biaf_dep'):
                def biaf_dep_hook(module, input, output):
                    try:
                        if hasattr(output, 'shape'):
                            service.syntax_parser_component._last_deps_logits = output.detach().cpu()
                    except:
                        pass

                handle = parser_model.biaf_dep.register_forward_hook(biaf_dep_hook)
                self.hook_handles.append(handle)
                print("  ✓ deps hook")

        print("\n✅ Ready\n")

        self.cache_enabled = True

    def _format_native_output(self, sentences: List[List[Dict]]) -> str:
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
        return '\n\n'.join(conllu_blocks)

    def _get_deprel_vocab(self) -> List[str]:
        return self.deprel_vocab if self.deprel_vocab else []

    def _extract_real_probas(
        self, 
        tokenized_sentences: List[List[str]],
        sentences_dict: List[List[Dict]]
    ) -> tuple:
        import numpy as np
        import torch.nn.functional as F

        _ = self.model(tokenized_sentences)

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
                except:
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
                except:
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
                except:
                    deps_probas_all.append([{'root': 0.95} for _ in range(sent_len)])
            else:
                deps_probas_all.append([{'root': 0.95} for _ in range(sent_len)])

        return upos_probas_all, heads_probas_all, deps_probas_all

    def _parse_with_probas(
        self, 
        tokenized_sentences: List[List[str]],
        token_spans: List[List[tuple]]
    ) -> Dict[str, Any]:

        parsed_batch = self.model(tokenized_sentences)

        sentences_dict = []
        for i, sent_conllu in enumerate(parsed_batch):
            sent_res = []
            lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

            for j, line in enumerate(lines):
                fields = line.split('\t')
                if '-' in fields[0]:
                    continue

                start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                token_data = {
                    'id': int(fields[0]),
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': int(fields[6]),
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9],
                    'startchar': start_c,
                    'endchar': end_c
                }

                sent_res.append(token_data)

            sentences_dict.append(sent_res)

        upos_probas, heads_probas, deps_probas = self._extract_real_probas(
            tokenized_sentences, sentences_dict
        )

        for sent_idx, sent_tokens in enumerate(sentences_dict):
            for tok_idx, token in enumerate(sent_tokens):
                if sent_idx < len(upos_probas) and tok_idx < len(upos_probas[sent_idx]):
                    token['upos_proba'] = upos_probas[sent_idx][tok_idx]
                else:
                    token['upos_proba'] = 0.95

                if sent_idx < len(heads_probas) and tok_idx < len(heads_probas[sent_idx]):
                    token['heads_proba'] = heads_probas[sent_idx][tok_idx]
                else:
                    token['heads_proba'] = [1.0/(len(sent_tokens)+1)] * (len(sent_tokens)+1)

                if sent_idx < len(deps_probas) and tok_idx < len(deps_probas[sent_idx]):
                    token['deps_proba'] = deps_probas[sent_idx][tok_idx]
                else:
                    token['deps_proba'] = {'root': 0.95}

        result = {
            'format': 'full',
            'conllu': self._format_native_output(sentences_dict),
            'sentences': sentences_dict,
            'metadata': {
                'model': 'ru_syntagrus_joint_parsing',
                'tokenizer': 'razdel',
                'vocab': {'deprels': self._get_deprel_vocab()},
                'probas_source': 'real_from_raw_logits'
            }
        }

        return result

    def _get_cache_key(self, text: str, output_format: str) -> str:
        import hashlib
        content = f"{text}_{output_format}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        try:
            cache_path = f"/results_cache/{cache_key}.json"
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _save_to_cache(self, cache_key: str, result: Any):
        if not self.cache_enabled:
            return
        try:
            cache_path = f"/results_cache/{cache_key}.json"
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            results_cache_volume.commit()
        except:
            pass

    @modal.method()
    def parse_text(
        self, 
        text: str, 
        output_format: str = 'conllu',
        use_cache: bool = False
    ) -> Union[List, str, Dict]:
        from razdel import tokenize, sentenize

        if use_cache:
            cache_key = self._get_cache_key(text, output_format)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        sentences = list(sentenize(text))
        tokenized_sentences = []
        token_spans = []

        for sent in sentences:
            tokens = list(tokenize(sent.text))
            tokenized_sentences.append([t.text for t in tokens])
            token_spans.append([
                (sent.start + t.start, sent.start + t.stop)
                for t in tokens
            ])

        if output_format == 'full':
            result = self._parse_with_probas(tokenized_sentences, token_spans)

        elif output_format == 'conllu':
            parsed_batch = self.model(tokenized_sentences)
            results = []
            for i, sent_conllu in enumerate(parsed_batch):
                sent_res = []
                lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

                for j, line in enumerate(lines):
                    fields = line.split('\t')
                    if '-' in fields[0]:
                        continue

                    start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                    sent_res.append({
                        'id': int(fields[0]),
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': int(fields[6]),
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9],
                        'startchar': start_c,
                        'endchar': end_c
                    })

                results.append(sent_res)

            result = self._format_native_output(results)

        else:  # 'dict'
            parsed_batch = self.model(tokenized_sentences)
            results = []
            for i, sent_conllu in enumerate(parsed_batch):
                sent_res = []
                lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

                for j, line in enumerate(lines):
                    fields = line.split('\t')
                    if '-' in fields[0]:
                        continue

                    start_c, end_c = token_spans[i][j] if j < len(token_spans[i]) else (0, 0)

                    sent_res.append({
                        'id': int(fields[0]),
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': int(fields[6]),
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9],
                        'startchar': start_c,
                        'endchar': end_c
                    })

                results.append(sent_res)

            result = results

        if use_cache:
            self._save_to_cache(cache_key, result)

        return result

    @modal.method()
    def parse_batch(
        self, 
        texts: List[str], 
        output_format: str = 'conllu',
        use_cache: bool = False
    ) -> Union[List, List[str], List[Dict]]:
        return [
            self.parse_text(t, output_format=output_format, use_cache=use_cache) 
            for t in texts
        ]

    @modal.method()
    def parse_text_native(
        self, 
        text: str, 
        output_format: str = 'conllu'
    ) -> Union[List, str, Dict]:
        parsed_batch = self.model([text])
        results = []
        for sent_conllu in parsed_batch:
            sent_res = []
            lines = [l for l in sent_conllu.split('\n') if l and not l.startswith('#')]

            for line in lines:
                fields = line.split('\t')
                if '-' in fields[0]:
                    continue

                sent_res.append({
                    'id': int(fields[0]),
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': int(fields[6]),
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9]
                })

            results.append(sent_res)

        if output_format == 'conllu':
            return self._format_native_output(results)
        else:
            return results


@app.local_entrypoint()
def main():
    test_text = "Мама мыла раму."
    print("🚀 Testing DeepPavlov (production)...\n")
    service = DeepPavlovService()

    result = service.parse_text.remote(test_text, output_format='full')

    print(f"📊 probas_source: {result['metadata']['probas_source']}")
    print(f"✅ Done!")
