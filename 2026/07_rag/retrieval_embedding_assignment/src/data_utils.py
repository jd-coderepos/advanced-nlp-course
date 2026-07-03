"""Data loading and simple text preprocessing utilities."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class TripletExample:
    query_id: str
    query: str
    positive_doc_id: str
    negative_doc_id: str


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    query: str
    positive_doc_id: str


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text using a small regex tokenizer."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def read_documents(path: str | Path) -> dict[str, str]:
    """Read a TSV file with columns: doc_id, text."""
    documents: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            documents[row["doc_id"]] = row["text"]
    return documents


def read_triplets(path: str | Path) -> list[TripletExample]:
    """Read training triplets from a TSV file."""
    examples: list[TripletExample] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            examples.append(
                TripletExample(
                    query_id=row["query_id"],
                    query=row["query"],
                    positive_doc_id=row["positive_doc_id"],
                    negative_doc_id=row["negative_doc_id"],
                )
            )
    return examples


def read_queries(path: str | Path) -> list[QueryExample]:
    """Read evaluation queries from a TSV file."""
    examples: list[QueryExample] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            examples.append(
                QueryExample(
                    query_id=row["query_id"],
                    query=row["query"],
                    positive_doc_id=row["positive_doc_id"],
                )
            )
    return examples


class Vocabulary:
    """A tiny word-level vocabulary for laptop-friendly experiments."""

    def __init__(self, texts: list[str], min_freq: int = 1) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        self.itos = ["<pad>", "<unk>"]
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenize(text)]


def build_vocabulary(
    documents: dict[str, str],
    train_examples: list[TripletExample],
    eval_examples: list[QueryExample],
) -> Vocabulary:
    """Build a vocabulary from documents and all visible assignment queries."""
    texts = list(documents.values())
    texts.extend(example.query for example in train_examples)
    texts.extend(example.query for example in eval_examples)
    return Vocabulary(texts)
