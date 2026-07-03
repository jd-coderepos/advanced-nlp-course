# Assignment: Implementing the DPO Objective

## Context

Direct Preference Optimization (DPO) trains a language model directly from preference pairs. Each training example contains:

- a prompt,
- a preferred response,
- a rejected response,
- log probabilities assigned by the current policy,
- log probabilities assigned by a fixed reference policy.

The goal is to increase the relative likelihood of the preferred response compared with the rejected response, while measuring this change relative to the reference model.

## Task

Implement the DPO loss in `src/dpo_objective.py`.

The formula is:

```text
L_DPO = - log sigmoid(
    beta * [
        (policy_chosen_logp - policy_rejected_logp)
        -
        (reference_chosen_logp - reference_rejected_logp)
    ]
)
```

Use a numerically stable implementation. A good identity is:

```python
-log(sigmoid(z)) = log(1 + exp(-z)) = np.logaddexp(0, -z)
```

## Required behavior

Your function must:

1. Accept scalar values or NumPy arrays.
2. Support `reduction="none"`, `reduction="mean"`, and `reduction="sum"`.
3. Validate that all log-probability arrays have the same shape.
4. Raise a `ValueError` for invalid beta values or invalid reduction names.
5. Pass all tests in `tests/test_dpo_objective.py`.

## Dataset

The file `data/toy_preferences.jsonl` contains small preference examples. Example fields:

```json
{
  "prompt": "Explain why RLHF can be unstable.",
  "chosen": "RLHF can be unstable because reward optimization uses sampled outputs and high-variance updates.",
  "rejected": "RLHF is always stable and never needs regularization.",
  "policy_chosen_logp": -2.3,
  "policy_rejected_logp": -4.0,
  "ref_chosen_logp": -2.7,
  "ref_rejected_logp": -3.7
}
```

The text fields are included so that the examples look like preference data. The objective only uses the log-probability fields.

