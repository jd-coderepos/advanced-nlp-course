"""Very small encoders for retrieval."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WeightedBoWEncoder(nn.Module):
    """Encode text as a trainable weighted bag-of-words vector.

    This model is intentionally small and interpretable. It starts as a lexical
    retriever and learns which tokens should receive more or less weight for the
    retrieval task.
    """

    def __init__(self, vocab_size: int, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # Padding token gets masked out; other token weights are learned.
        self.token_weight_logits = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        bag = (one_hot * attention_mask.unsqueeze(-1)).sum(dim=1)
        weights = F.softplus(self.token_weight_logits).unsqueeze(0)
        weighted_bag = bag * weights
        return F.normalize(weighted_bag, p=2, dim=-1)


class MeanEmbeddingEncoder(nn.Module):
    """Encode text as a dense vector using mean-pooled trainable word embeddings.

    This optional model is closer to a neural sentence embedding model, but it is
    also more unstable on the tiny dataset. Use it for bonus experiments.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_embedding = masked_embeddings.sum(dim=1) / lengths
        projected = torch.tanh(self.projection(mean_embedding))
        return F.normalize(projected, p=2, dim=-1)
