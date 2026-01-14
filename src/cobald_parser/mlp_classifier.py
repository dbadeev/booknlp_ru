import torch
from torch import nn
from torch import Tensor, LongTensor

from transformers.activations import ACT2FN


class MlpClassifier(nn.Module):
    """ Simple feed-forward multilayer perceptron classifier. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_classes: int,
        activation: str,
        dropout: float,
        class_weights: list[float] = None,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            ACT2FN[activation],
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.long)
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, embeddings: Tensor, labels: LongTensor = None) -> dict:
        logits = self.classifier(embeddings)
        # Calculate loss.
        loss = 0.0
        if labels is not None:
            # Reshape tensors to match expected dimensions
            loss = self.cross_entropy(
                logits.view(-1, self.n_classes),
                labels.view(-1)
            )
        # Predictions.
        preds = logits.argmax(dim=-1)
        return {'preds': preds, 'loss': loss}
