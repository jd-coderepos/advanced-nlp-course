import numpy as np
import pytest

from src.dpo_objective import dpo_logits, dpo_loss, preference_accuracy


def test_dpo_logits_vectorized_values():
    policy_chosen = np.array([-2.0, -3.0, -1.5])
    policy_rejected = np.array([-4.0, -2.0, -2.0])
    ref_chosen = np.array([-2.5, -2.5, -1.5])
    ref_rejected = np.array([-3.5, -2.0, -2.5])

    # policy log ratios: [2.0, -1.0, 0.5]
    # ref log ratios:    [1.0, -0.5, 1.0]
    # difference:        [1.0, -0.5, -0.5]
    expected = np.array([0.5, -0.25, -0.25])
    actual = dpo_logits(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.5)

    np.testing.assert_allclose(actual, expected)


def test_dpo_loss_none_reduction_known_values():
    logits = np.array([0.5, -0.25, -0.25])
    expected_losses = np.logaddexp(0.0, -logits)

    actual_losses = dpo_loss(
        np.array([-2.0, -3.0, -1.5]),
        np.array([-4.0, -2.0, -2.0]),
        np.array([-2.5, -2.5, -1.5]),
        np.array([-3.5, -2.0, -2.5]),
        beta=0.5,
        reduction="none",
    )

    np.testing.assert_allclose(actual_losses, expected_losses)


def test_dpo_loss_mean_and_sum_reductions():
    losses = dpo_loss(
        [-2.0, -3.0, -1.5],
        [-4.0, -2.0, -2.0],
        [-2.5, -2.5, -1.5],
        [-3.5, -2.0, -2.5],
        beta=0.5,
        reduction="none",
    )

    mean_loss = dpo_loss(
        [-2.0, -3.0, -1.5],
        [-4.0, -2.0, -2.0],
        [-2.5, -2.5, -1.5],
        [-3.5, -2.0, -2.5],
        beta=0.5,
        reduction="mean",
    )

    sum_loss = dpo_loss(
        [-2.0, -3.0, -1.5],
        [-4.0, -2.0, -2.0],
        [-2.5, -2.5, -1.5],
        [-3.5, -2.0, -2.5],
        beta=0.5,
        reduction="sum",
    )

    assert np.isclose(mean_loss, np.mean(losses))
    assert np.isclose(sum_loss, np.sum(losses))


def test_scalar_inputs_are_supported():
    actual = dpo_loss(-2.0, -4.0, -2.5, -3.5, beta=0.5, reduction="mean")
    expected = np.logaddexp(0.0, -0.5)
    assert np.isclose(actual, expected)


def test_preference_accuracy_counts_positive_logits():
    acc = preference_accuracy(
        [-2.0, -3.0, -1.5],
        [-4.0, -2.0, -2.0],
        [-2.5, -2.5, -1.5],
        [-3.5, -2.0, -2.5],
        beta=0.5,
    )
    assert np.isclose(acc, 1 / 3)


def test_invalid_beta_raises_value_error():
    with pytest.raises(ValueError):
        dpo_loss([-1.0], [-2.0], [-1.0], [-2.0], beta=0.0)


def test_invalid_reduction_raises_value_error():
    with pytest.raises(ValueError):
        dpo_loss([-1.0], [-2.0], [-1.0], [-2.0], beta=0.5, reduction="median")


def test_mismatched_shapes_raise_value_error():
    with pytest.raises(ValueError):
        dpo_loss([-1.0, -2.0], [-2.0], [-1.0], [-2.0], beta=0.5)
