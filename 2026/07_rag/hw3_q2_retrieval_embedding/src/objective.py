"""Training objectives for retrieval-oriented embeddings.

The key function in this assignment is `retrieval_hinge_loss`.
It implements a small contrastive ranking objective:

    loss = max(0, margin + score(query, negative_doc) - score(query, positive_doc))

If the positive document already scores at least `margin` higher than the negative
one, the loss is zero. Otherwise, the model is penalized. This pushes relevant
query-document pairs closer together and irrelevant pairs farther apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_scores(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity scores.

    If both tensors have shape (batch_size, dim), this returns row-wise scores
    with shape (batch_size,). If `b` has shape (batch_size, num_negatives, dim),
    this returns one score per negative with shape (batch_size, num_negatives).
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    if b.dim() == 2:
        return torch.sum(a * b, dim=-1)
    if b.dim() == 3:
        return torch.einsum("bd,bnd->bn", a, b)
    raise ValueError(f"Expected b to have 2 or 3 dimensions, got shape {tuple(b.shape)}")


def retrieval_hinge_loss(
    query_embeddings: torch.Tensor,
    positive_doc_embeddings: torch.Tensor,
    negative_doc_embeddings: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute the contrastive hinge loss for retrieval training.

    The model receives a query, a positive document, and a negative document.
    It should assign a higher score to the positive document than to the negative
    document by at least `margin`.

    Formula per training example:

        max(0, margin + s(q, d_neg) - s(q, d_pos))

    where s is cosine similarity.

    Args:
        query_embeddings: Encoded queries, shape (batch_size, dim).
        positive_doc_embeddings: Encoded relevant documents, shape (batch_size, dim).
        negative_doc_embeddings: Encoded irrelevant documents. Shape can be
            (batch_size, dim) for one negative per query, or
            (batch_size, num_negatives, dim) for several negatives per query.
        margin: Required gap between positive and negative scores.

    Returns:
        A scalar tensor: mean loss over the batch.
    """
    positive_scores = cosine_scores(query_embeddings, positive_doc_embeddings)
    negative_scores = cosine_scores(query_embeddings, negative_doc_embeddings)
    if negative_scores.dim() == 2:
        positive_scores = positive_scores.unsqueeze(1)
    losses = torch.relu(margin + negative_scores - positive_scores)
    return losses.mean()
