"""
DPO objective starter code.

Students: implement `dpo_loss` only. You may use the helper functions below.
"""

from __future__ import annotations

import numpy as np


def _as_array(x):
    """Convert scalars or array-like values to a NumPy array of floats."""
    return np.asarray(x, dtype=float)


def _validate_same_shape(*arrays: np.ndarray) -> None:
    """Raise ValueError if arrays do not all have the same shape."""
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"All log-probability inputs must have the same shape, got {shapes}")


def dpo_logits(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
):
    """
    Compute the DPO logits.

    z = beta * [
        (log pi_theta(chosen) - log pi_theta(rejected))
        - (log pi_ref(chosen) - log pi_ref(rejected))
    ]
    """
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_chosen_logps = _as_array(policy_chosen_logps)
    policy_rejected_logps = _as_array(policy_rejected_logps)
    ref_chosen_logps = _as_array(ref_chosen_logps)
    ref_rejected_logps = _as_array(ref_rejected_logps)

    _validate_same_shape(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    return beta * (policy_logratios - ref_logratios)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
    reduction: str = "mean",
):
    """
    Compute the DPO loss.

    Parameters
    ----------
    policy_chosen_logps, policy_rejected_logps:
        Log probabilities assigned by the current policy to the chosen and rejected outputs.
    ref_chosen_logps, ref_rejected_logps:
        Log probabilities assigned by the reference policy to the chosen and rejected outputs.
    beta:
        Positive temperature/scaling parameter.
    reduction:
        "none", "mean", or "sum".

    Returns
    -------
    np.ndarray or float
        Per-example loss if reduction="none", otherwise a scalar.

    TODO
    ----
    Implement:
        logits = dpo_logits(...)
        losses = -log(sigmoid(logits))

    Use a numerically stable identity:
        -log(sigmoid(z)) = log(1 + exp(-z)) = np.logaddexp(0, -z)
    """
    # TODO: replace this with your implementation.
    raise NotImplementedError("Implement dpo_loss in src/dpo_objective.py")


def preference_accuracy(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
) -> float:
    """
    Return the fraction of examples where the DPO logit is positive.

    A positive logit means the current policy favors the chosen response over the rejected
    response more strongly than the reference policy does.
    """
    logits = dpo_logits(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    return float(np.mean(logits > 0))
