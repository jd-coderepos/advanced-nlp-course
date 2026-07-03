"""Retrieval evaluation metrics."""

from __future__ import annotations


def rank_of_positive(ranked_doc_ids: list[str], positive_doc_id: str) -> int:
    """Return the 1-based rank of the positive document, or a large value."""
    try:
        return ranked_doc_ids.index(positive_doc_id) + 1
    except ValueError:
        return 10**9


def summarize_ranks(ranks: list[int]) -> dict[str, float]:
    """Compute simple retrieval metrics from positive-document ranks."""
    total = len(ranks)
    if total == 0:
        return {"recall@1": 0.0, "recall@3": 0.0, "mrr": 0.0}
    return {
        "recall@1": sum(rank <= 1 for rank in ranks) / total,
        "recall@3": sum(rank <= 3 for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
    }
