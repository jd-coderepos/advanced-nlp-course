"""Run the DPO objective on the toy preference dataset."""

from __future__ import annotations

import argparse

from src.dpo_objective import dpo_logits, dpo_loss, preference_accuracy
from src.utils import extract_logprob_arrays, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_preferences.jsonl", help="Path to JSONL preference data")
    parser.add_argument("--beta", type=float, default=0.5, help="DPO beta parameter")
    args = parser.parse_args()

    records = load_jsonl(args.data)
    arrays = extract_logprob_arrays(records)

    logits = dpo_logits(*arrays, beta=args.beta)
    try:
        losses = dpo_loss(*arrays, beta=args.beta, reduction="none")
        mean_loss = dpo_loss(*arrays, beta=args.beta, reduction="mean")
    except NotImplementedError as exc:
        print(exc)
        print("Open src/dpo_objective.py and implement dpo_loss, then rerun this script.")
        return

    acc = preference_accuracy(*arrays, beta=args.beta)

    print(f"Loaded {len(records)} preference pairs")
    print(f"beta: {args.beta}")
    print(f"mean DPO loss: {mean_loss:.4f}")
    print(f"preference accuracy from DPO logits: {acc:.3f}")
    print()
    print("Per-example diagnostics:")
    for record, logit, loss in zip(records, logits, losses):
        print(f"{record['id']}: logit={logit: .3f}, loss={loss: .3f}, prompt={record['prompt']}")


if __name__ == "__main__":
    main()
