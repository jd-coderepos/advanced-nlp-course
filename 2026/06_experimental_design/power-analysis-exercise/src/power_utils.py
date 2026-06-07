"""
Utilities for classroom power-analysis exercises in NLP.

The functions use simulation-based power analysis:
1. Choose assumptions about the true effect and variability.
2. Simulate many possible evaluation datasets.
3. Run the planned significance test on each simulated dataset.
4. Estimate power as the fraction of simulated datasets where p < alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.stats import binomtest, ttest_rel


def ask_float(prompt: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Prompt for a floating-point value in a notebook, with an editable default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if raw == "" else float(raw)
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def ask_int(prompt: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Prompt for an integer value in a notebook, with an editable default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if raw == "" else int(raw)
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def exact_mcnemar_p_value(only_a_correct: int, only_b_correct: int) -> float:
    """
    Exact two-sided McNemar p-value for paired binary outcomes.

    only_a_correct: items where model A is correct and B is wrong
    only_b_correct: items where model B is correct and A is wrong
    """
    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 1.0
    return binomtest(min(only_a_correct, only_b_correct), n=discordant, p=0.5, alternative="two-sided").pvalue


def accuracy_category_probabilities(
    baseline_accuracy: float,
    expected_delta: float,
    agreement_rate: float,
) -> np.ndarray:
    """
    Return probabilities for:
    [both_correct, only_a_correct, only_b_correct, both_wrong]

    expected_delta = accuracy_B - accuracy_A.
    agreement_rate = P(models have the same correctness outcome).
    """
    if not 0 <= baseline_accuracy <= 1:
        raise ValueError("baseline_accuracy must be between 0 and 1.")
    if not 0 <= agreement_rate <= 1:
        raise ValueError("agreement_rate must be between 0 and 1.")
    improved_accuracy = baseline_accuracy + expected_delta
    if not 0 <= improved_accuracy <= 1:
        raise ValueError("baseline_accuracy + expected_delta must be between 0 and 1.")

    discord = 1.0 - agreement_rate
    only_b = (discord + expected_delta) / 2.0
    only_a = (discord - expected_delta) / 2.0
    both_correct = baseline_accuracy - only_a
    both_wrong = agreement_rate - both_correct

    probs = np.array([both_correct, only_a, only_b, both_wrong], dtype=float)
    if np.any(probs < -1e-9):
        raise ValueError(
            "The inputs imply impossible outcome probabilities. "
            "Try a smaller expected_delta, a lower agreement_rate, or a different baseline_accuracy."
        )
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    return probs


def simulate_accuracy_counts(
    n_instances: int,
    baseline_accuracy: float,
    expected_delta: float,
    agreement_rate: float,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Simulate paired correct/incorrect outcomes for two classifiers."""
    probs = accuracy_category_probabilities(baseline_accuracy, expected_delta, agreement_rate)
    both_correct, only_a, only_b, both_wrong = rng.multinomial(n_instances, probs)
    return {
        "both_correct": int(both_correct),
        "only_a_correct": int(only_a),
        "only_b_correct": int(only_b),
        "both_wrong": int(both_wrong),
    }


def estimate_mcnemar_power(
    n_instances: int,
    baseline_accuracy: float,
    expected_delta: float,
    agreement_rate: float,
    alpha: float = 0.05,
    n_trials: int = 2000,
    seed: Optional[int] = 42,
) -> float:
    """Estimate power for an accuracy comparison using exact McNemar tests."""
    rng = np.random.default_rng(seed)
    detections = 0
    target_sign = np.sign(expected_delta)

    for _ in range(n_trials):
        counts = simulate_accuracy_counts(
            n_instances=n_instances,
            baseline_accuracy=baseline_accuracy,
            expected_delta=expected_delta,
            agreement_rate=agreement_rate,
            rng=rng,
        )
        p_value = exact_mcnemar_p_value(counts["only_a_correct"], counts["only_b_correct"])
        observed_delta = (counts["only_b_correct"] - counts["only_a_correct"]) / n_instances
        correct_direction = np.sign(observed_delta) == target_sign if target_sign != 0 else True
        if p_value < alpha and correct_direction:
            detections += 1
    return detections / n_trials


def find_required_n_mcnemar(
    baseline_accuracy: float,
    expected_delta: float,
    agreement_rate: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_trials: int = 2000,
    n_min: int = 20,
    n_max: int = 100_000,
    seed: Optional[int] = 42,
) -> Tuple[Optional[int], Optional[float]]:
    """Find the smallest n that reaches target power for an accuracy comparison."""
    if expected_delta == 0:
        return None, None

    n = n_min
    last_power = None
    while n <= n_max:
        power = estimate_mcnemar_power(
            n, baseline_accuracy, expected_delta, agreement_rate, alpha, n_trials, seed
        )
        last_power = power
        if power >= target_power:
            # Refine by binary search between previous half and current n.
            low = max(n_min, n // 2)
            high = n
            while low < high:
                mid = (low + high) // 2
                mid_power = estimate_mcnemar_power(
                    mid, baseline_accuracy, expected_delta, agreement_rate, alpha, n_trials, seed
                )
                if mid_power >= target_power:
                    high = mid
                    last_power = mid_power
                else:
                    low = mid + 1
            final_power = estimate_mcnemar_power(
                high, baseline_accuracy, expected_delta, agreement_rate, alpha, n_trials, seed
            )
            return high, final_power
        n *= 2
    return None, last_power


def find_mde_mcnemar(
    n_instances: int,
    baseline_accuracy: float,
    agreement_rate: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_trials: int = 2000,
    delta_step: float = 0.001,
    max_delta: float = 0.50,
    seed: Optional[int] = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Find the minimum detectable accuracy improvement for a fixed test size."""
    delta = delta_step
    while delta <= max_delta and baseline_accuracy + delta <= 1.0:
        try:
            power = estimate_mcnemar_power(
                n_instances, baseline_accuracy, delta, agreement_rate, alpha, n_trials, seed
            )
        except ValueError:
            delta += delta_step
            continue
        if power >= target_power:
            return delta, power
        delta += delta_step
    return None, None


def simulate_paired_metric_scores(
    n_instances: int,
    mean_a: float,
    mean_b: float,
    sd_a: float,
    sd_b: float,
    correlation: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate paired per-instance metric scores for two methods."""
    cov = [[sd_a ** 2, correlation * sd_a * sd_b], [correlation * sd_a * sd_b, sd_b ** 2]]
    scores = rng.multivariate_normal([mean_a, mean_b], cov, size=n_instances)
    scores = np.clip(scores, lower, upper)
    return scores[:, 0], scores[:, 1]


def estimate_paired_ttest_power(
    n_instances: int,
    mean_a: float,
    mean_b: float,
    sd_a: float,
    sd_b: float,
    correlation: float,
    lower: float = 0.0,
    upper: float = 1.0,
    alpha: float = 0.05,
    n_trials: int = 2000,
    seed: Optional[int] = 42,
) -> float:
    """Estimate power using a paired t-test on per-instance metric scores."""
    rng = np.random.default_rng(seed)
    detections = 0
    target_sign = np.sign(mean_b - mean_a)
    for _ in range(n_trials):
        a, b = simulate_paired_metric_scores(n_instances, mean_a, mean_b, sd_a, sd_b, correlation, lower, upper, rng)
        result = ttest_rel(b, a)
        p_value = result.pvalue if np.isfinite(result.pvalue) else 1.0
        observed_delta = float(np.mean(b - a))
        correct_direction = np.sign(observed_delta) == target_sign if target_sign != 0 else True
        if p_value < alpha and correct_direction:
            detections += 1
    return detections / n_trials


def find_required_n_paired_metric(
    mean_a: float,
    mean_b: float,
    sd_a: float,
    sd_b: float,
    correlation: float,
    lower: float = 0.0,
    upper: float = 1.0,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_trials: int = 2000,
    n_min: int = 20,
    n_max: int = 100_000,
    seed: Optional[int] = 42,
) -> Tuple[Optional[int], Optional[float]]:
    """Find the smallest n that reaches target power for paired metric scores."""
    n = n_min
    last_power = None
    while n <= n_max:
        power = estimate_paired_ttest_power(
            n, mean_a, mean_b, sd_a, sd_b, correlation, lower, upper, alpha, n_trials, seed
        )
        last_power = power
        if power >= target_power:
            low = max(n_min, n // 2)
            high = n
            while low < high:
                mid = (low + high) // 2
                mid_power = estimate_paired_ttest_power(
                    mid, mean_a, mean_b, sd_a, sd_b, correlation, lower, upper, alpha, n_trials, seed
                )
                if mid_power >= target_power:
                    high = mid
                else:
                    low = mid + 1
            final_power = estimate_paired_ttest_power(
                high, mean_a, mean_b, sd_a, sd_b, correlation, lower, upper, alpha, n_trials, seed
            )
            return high, final_power
        n *= 2
    return None, last_power


def find_mde_paired_metric(
    n_instances: int,
    mean_a: float,
    sd_a: float,
    sd_b: float,
    correlation: float,
    lower: float = 0.0,
    upper: float = 1.0,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_trials: int = 2000,
    delta_step: float = 0.001,
    max_delta: float = 0.50,
    seed: Optional[int] = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Find the minimum detectable improvement for a fixed n and paired metric setup."""
    delta = delta_step
    while delta <= max_delta and mean_a + delta <= upper:
        power = estimate_paired_ttest_power(
            n_instances, mean_a, mean_a + delta, sd_a, sd_b, correlation, lower, upper, alpha, n_trials, seed
        )
        if power >= target_power:
            return delta, power
        delta += delta_step
    return None, None


def paired_bootstrap_p_value(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 5000, seed: Optional[int] = 42) -> float:
    """
    Simple paired bootstrap-style p-value for whether mean(b-a) differs from zero.

    This is intended for teaching intuition, not as a replacement for a careful task-specific test.
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(b) - np.asarray(a)
    observed = np.mean(diffs)
    centered = diffs - observed
    boot_stats = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = rng.choice(centered, size=n, replace=True)
        boot_stats.append(np.mean(sample))
    boot_stats = np.asarray(boot_stats)
    return float(np.mean(np.abs(boot_stats) >= abs(observed)))


def estimate_inputs_from_pilot(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Estimate means, standard deviations, correlation, and observed delta from paired pilot scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return {
        "n": float(len(a)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta": float(np.mean(b) - np.mean(a)),
        "sd_a": float(np.std(a, ddof=1)),
        "sd_b": float(np.std(b, ddof=1)),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "sd_difference": float(np.std(b - a, ddof=1)),
    }


def simulate_likert_experiment(
    n_items: int,
    raters_per_item: int,
    baseline_mean: float,
    expected_effect: float,
    item_sd: float,
    rater_sd: float,
    residual_sd: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a simplified paired human-evaluation experiment.

    Each item receives ratings for model A and model B. The returned values are item-level
    mean ratings after averaging over raters.
    """
    item_quality = rng.normal(0, item_sd, size=n_items)
    a_item_means = []
    b_item_means = []
    for i in range(n_items):
        rater_bias = rng.normal(0, rater_sd, size=raters_per_item)
        a = baseline_mean + item_quality[i] + rater_bias + rng.normal(0, residual_sd, size=raters_per_item)
        b = baseline_mean + expected_effect + item_quality[i] + rater_bias + rng.normal(0, residual_sd, size=raters_per_item)
        a_item_means.append(np.mean(np.clip(a, lower, upper)))
        b_item_means.append(np.mean(np.clip(b, lower, upper)))
    return np.asarray(a_item_means), np.asarray(b_item_means)


def estimate_likert_power(
    n_items: int,
    raters_per_item: int,
    baseline_mean: float,
    expected_effect: float,
    item_sd: float,
    rater_sd: float,
    residual_sd: float,
    lower: float = 0.0,
    upper: float = 1.0,
    alpha: float = 0.05,
    n_trials: int = 2000,
    seed: Optional[int] = 42,
) -> float:
    """Estimate power for a simplified paired Likert-rating comparison."""
    rng = np.random.default_rng(seed)
    detections = 0
    target_sign = np.sign(expected_effect)
    for _ in range(n_trials):
        a, b = simulate_likert_experiment(
            n_items, raters_per_item, baseline_mean, expected_effect,
            item_sd, rater_sd, residual_sd, lower, upper, rng
        )
        result = ttest_rel(b, a)
        p_value = result.pvalue if np.isfinite(result.pvalue) else 1.0
        observed_delta = float(np.mean(b - a))
        correct_direction = np.sign(observed_delta) == target_sign if target_sign != 0 else True
        if p_value < alpha and correct_direction:
            detections += 1
    return detections / n_trials
