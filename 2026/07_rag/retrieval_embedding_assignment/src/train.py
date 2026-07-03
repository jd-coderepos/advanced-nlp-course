"""Train a tiny retrieval-oriented embedding model.

Run from the repository root:

    python -m src.train --epochs 120

The code first evaluates the untrained model, then trains with a contrastive
hinge objective, and finally evaluates again.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from src.data_utils import (
    QueryExample,
    TripletExample,
    build_vocabulary,
    read_documents,
    read_queries,
    read_triplets,
)
from src.metrics import rank_of_positive, summarize_ranks
from src.model import MeanEmbeddingEncoder, WeightedBoWEncoder
from src.objective import retrieval_hinge_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_batch(texts: list[str], vocab, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of texts into padded token IDs and an attention mask."""
    encoded = [torch.tensor(vocab.encode(text), dtype=torch.long) for text in texts]
    if not encoded:
        raise ValueError("Cannot make a batch from an empty list of texts.")
    padded = pad_sequence(encoded, batch_first=True, padding_value=0).to(device)
    attention_mask = (padded != 0).float().to(device)
    return padded, attention_mask


def encode_texts(
    model: torch.nn.Module,
    texts: list[str],
    vocab,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode a list of texts without gradient tracking."""
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            input_ids, attention_mask = make_batch(batch_texts, vocab, device)
            vectors.append(model(input_ids, attention_mask).cpu())
    return torch.cat(vectors, dim=0)


def retrieve_all(
    model: torch.nn.Module,
    query: str,
    documents: dict[str, str],
    vocab,
    device: torch.device,
) -> list[tuple[str, float]]:
    """Rank all documents for a query using cosine similarity."""
    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id] for doc_id in doc_ids]
    query_vector = encode_texts(model, [query], vocab, device)
    doc_vectors = encode_texts(model, doc_texts, vocab, device)
    scores = (query_vector @ doc_vectors.T).squeeze(0)
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    return [(doc_ids[index], float(scores[index])) for index in ranked_indices]


def evaluate(
    model: torch.nn.Module,
    examples: list[QueryExample],
    documents: dict[str, str],
    vocab,
    device: torch.device,
    label: str,
) -> dict[str, float]:
    """Evaluate retrieval quality and print ranks for the evaluation queries."""
    ranks: list[int] = []
    print(f"\n{label}")
    print("-" * len(label))
    for example in examples:
        ranking = retrieve_all(model, example.query, documents, vocab, device)
        ranked_doc_ids = [doc_id for doc_id, _score in ranking]
        rank = rank_of_positive(ranked_doc_ids, example.positive_doc_id)
        ranks.append(rank)
        top3 = ", ".join(f"{doc_id}:{score:.3f}" for doc_id, score in ranking[:3])
        print(f"{example.query_id}: rank={rank:>2} gold={example.positive_doc_id} top3=[{top3}]")
    metrics = summarize_ranks(ranks)
    print("metrics:", json.dumps(metrics, indent=2))
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    examples: list[TripletExample],
    documents: dict[str, str],
    vocab,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int,
    margin: float,
) -> float:
    """Run one training epoch over query-positive-negative triplets."""
    model.train()
    random.shuffle(examples)
    losses: list[float] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        queries = [example.query for example in batch]
        positives = [documents[example.positive_doc_id] for example in batch]

        # Use all non-positive documents as negatives. This is still tiny here
        # because the corpus has only 24 documents, but it demonstrates the
        # slide-18 objective: sum over positive/negative document pairs.
        negative_lists = [
            [text for doc_id, text in documents.items() if doc_id != example.positive_doc_id]
            for example in batch
        ]
        flat_negatives = [text for negatives in negative_lists for text in negatives]
        num_negatives = len(negative_lists[0])

        query_ids, query_mask = make_batch(queries, vocab, device)
        pos_ids, pos_mask = make_batch(positives, vocab, device)
        neg_ids, neg_mask = make_batch(flat_negatives, vocab, device)

        query_vecs = model(query_ids, query_mask)
        pos_vecs = model(pos_ids, pos_mask)
        flat_neg_vecs = model(neg_ids, neg_mask)
        neg_vecs = flat_neg_vecs.view(len(batch), num_negatives, -1)

        loss = retrieval_hinge_loss(query_vecs, pos_vecs, neg_vecs, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / len(losses)


def main() -> None:
    # Small models are often faster and more predictable with one CPU thread.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default="data/documents.tsv")
    parser.add_argument("--train", default="data/train_triplets.tsv")
    parser.add_argument("--dev", default="data/dev_queries.tsv")
    parser.add_argument("--test", default="data/test_queries.tsv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", choices=["bow", "mean"], default="mean")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    documents = read_documents(args.documents)
    train_examples = read_triplets(args.train)
    dev_examples = read_queries(args.dev)
    test_examples = read_queries(args.test)
    vocab = build_vocabulary(documents, train_examples, dev_examples + test_examples)
    print(f"Loaded {len(documents)} documents, {len(train_examples)} training triplets")
    print(f"Vocabulary size: {len(vocab)}")

    if args.model == "bow":
        model = WeightedBoWEncoder(len(vocab)).to(device)
    else:
        model = MeanEmbeddingEncoder(len(vocab), embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    evaluate(model, dev_examples, documents, vocab, device, label="Before training: dev retrieval")
    evaluate(model, test_examples, documents, vocab, device, label="Before training: test retrieval")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            examples=train_examples,
            documents=documents,
            vocab=vocab,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            margin=args.margin,
        )
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={loss:.4f}")

    evaluate(model, dev_examples, documents, vocab, device, label="After training: dev retrieval")
    evaluate(model, test_examples, documents, vocab, device, label="After training: test retrieval")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_itos": vocab.itos,
            "args": vars(args),
        },
        output_dir / "tiny_retriever.pt",
    )
    print("\nSaved model to outputs/tiny_retriever.pt")


if __name__ == "__main__":
    main()
