import torch
from torch import nn
from torch import Tensor, LongTensor

from transformers import AutoTokenizer, AutoModel


class WordTransformerEncoder(nn.Module):
    """
    Encodes sentences into word-level embeddings using a pretrained MLM transformer.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Model like BERT, RoBERTa, etc.
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, words: list[list[str]]) -> Tensor:
        """
        Build words embeddings.

        - Tokenizes input sentences into subtokens.
        - Passes the subtokens through the pre-trained transformer model.
        - Aggregates subtoken embeddings into word embeddings using mean pooling.
        """
        batch_size = len(words)

        # BPE tokenization: split words into subtokens, e.g. ['kidding'] -> ['▁ki', 'dding'].
        subtokens = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt'
        )
        subtokens = subtokens.to(self.model.device)
        # Index words from 1 and reserve 0 for special subtokens (e.g. <s>, </s>, padding, etc.).
        # Such numeration makes a following aggregation easier.
        words_ids = torch.stack([
            torch.tensor(
                [word_id + 1 if word_id is not None else 0 for word_id in subtokens.word_ids(batch_idx)],
                dtype=torch.long,
                device=self.model.device
            )
            for batch_idx in range(batch_size)
        ])

        # Run model and extract subtokens embeddings from the last layer.
        subtokens_embeddings = self.model(**subtokens).last_hidden_state

        # Aggreate subtokens embeddings into words embeddings.
        # [batch_size, n_words, embedding_size]
        words_emeddings = self._aggregate_subtokens_embeddings(subtokens_embeddings, words_ids)
        return words_emeddings

    def _aggregate_subtokens_embeddings(
        self,
        subtokens_embeddings: Tensor, # [batch_size, n_subtokens, embedding_size]
        words_ids: LongTensor          # [batch_size, n_subtokens]
    ) -> Tensor:
        """
        Aggregate subtoken embeddings into word embeddings by averaging.

        This method ensures that multiple subtokens corresponding to a single word are combined
        into a single embedding.
        """
        batch_size, n_subtokens, embedding_size = subtokens_embeddings.shape
        # The number of words in a sentence plus an "auxiliary" word in the beginnig.
        n_words = torch.max(words_ids) + 1

        words_embeddings = torch.zeros(
            size=(batch_size, n_words, embedding_size),
            dtype=subtokens_embeddings.dtype,
            device=self.model.device
        )
        words_ids_expanded = words_ids.unsqueeze(-1).expand(batch_size, n_subtokens, embedding_size)

        # Use scatter_reduce_ to average embeddings of subtokens corresponding to the same word.
        # All the padding and special subtokens will be aggregated into an "auxiliary" first embedding,
        # namely into words_embeddings[:, 0, :].
        words_embeddings.scatter_reduce_(
            dim=1,
            index=words_ids_expanded,
            src=subtokens_embeddings,
            reduce="mean",
            include_self=False
        )
        # Now remove the auxiliary word in the beginning.
        words_embeddings = words_embeddings[:, 1:, :]
        return words_embeddings

    def get_embedding_size(self) -> int:
        """Returns the embedding size of the transformer model, e.g. 768 for BERT."""
        return self.model.config.hidden_size

    def get_embeddings_layer(self):
        """Returns the embeddings model."""
        return self.model.embeddings
        
    def get_transformer_layers(self) -> list[nn.Module]:
        """
        Return a flat list of all transformer-*block* layers, excluding embeddings/poolers, etc.
        """
        layers = []
        for sub in self.model.modules():
            # find all ModuleLists (these always hold the actual block layers)
            if isinstance(sub, nn.ModuleList):
                layers.extend(list(sub))
        return layers