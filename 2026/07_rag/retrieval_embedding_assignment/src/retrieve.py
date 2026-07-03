"""Use a trained tiny retriever to rank documents for a query.

Example:

    python -m src.retrieve "Which Luna has rabbit food?" --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data_utils import Vocabulary, read_documents
from src.model import MeanEmbeddingEncoder, WeightedBoWEncoder
from src.train import retrieve_all


def load_vocab(tokens: list[str]) -> Vocabulary:
    """Rebuild a Vocabulary object from a saved token list."""
    vocab = Vocabulary([])
    vocab.itos = tokens
    vocab.stoi = {token: idx for idx, token in enumerate(tokens)}
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--documents", default="data/documents.tsv")
    parser.add_argument("--model-path", default="outputs/tiny_retriever.pt")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train a model first with: python -m src.train"
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = load_vocab(checkpoint["vocab_itos"])
    documents = read_documents(args.documents)
    model_name = checkpoint["args"].get("model", "bow")
    embedding_dim = checkpoint["args"].get("embedding_dim", 64)

    if model_name == "bow":
        model = WeightedBoWEncoder(len(vocab))
    else:
        model = MeanEmbeddingEncoder(len(vocab), embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ranking = retrieve_all(model, args.query, documents, vocab, torch.device("cpu"))
    print(f"Query: {args.query}\n")
    for rank, (doc_id, score) in enumerate(ranking[: args.top_k], start=1):
        print(f"{rank}. {doc_id} score={score:.3f}")
        print(f"   {documents[doc_id]}")


if __name__ == "__main__":
    main()
