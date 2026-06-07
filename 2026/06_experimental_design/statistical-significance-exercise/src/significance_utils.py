
"""
Utility functions for Lecture 6: Statistical Significance Testing exercises.

These functions are intentionally lightweight and classroom-friendly.
They avoid hidden assumptions and expose the main quantities students should inspect.
"""
from __future__ import annotations

from collections import Counter
from math import exp, factorial, log
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel


def mcnemar_exact_test(y_true, pred_a, pred_b):
    """Exact McNemar test for paired classification outputs.

    Returns a dictionary containing:
    - both_correct
    - only_a_correct
    - only_b_correct
    - both_wrong
    - p_value

    The exact test uses the discordant pairs only. Under the null hypothesis,
    Model A and Model B are equally likely to be correct on cases where they disagree.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    both_correct = int(np.sum(a_correct & b_correct))
    only_a_correct = int(np.sum(a_correct & ~b_correct))
    only_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    n_discordant = only_a_correct + only_b_correct
    if n_discordant == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial test with p=0.5 on discordant pairs.
        p_value = binomtest(
            k=min(only_a_correct, only_b_correct),
            n=n_discordant,
            p=0.5,
            alternative='two-sided',
        ).pvalue

    return {
        'both_correct': both_correct,
        'only_a_correct': only_a_correct,
        'only_b_correct': only_b_correct,
        'both_wrong': both_wrong,
        'n_discordant': n_discordant,
        'accuracy_a': float(np.mean(a_correct)),
        'accuracy_b': float(np.mean(b_correct)),
        'difference_b_minus_a': float(np.mean(b_correct) - np.mean(a_correct)),
        'p_value': float(p_value),
    }


def paired_t_test(scores_a, scores_b):
    """Paired t-test for paired numeric scores.

    Use this when every row has a score for Model A and a score for Model B,
    e.g. per-instance ratings, per-document metric scores, or matched random seeds.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    statistic, p_value = ttest_rel(scores_b, scores_a)
    diff = scores_b - scores_a
    return {
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'mean_difference_b_minus_a': float(np.mean(diff)),
        'std_difference': float(np.std(diff, ddof=1)),
        't_statistic': float(statistic),
        'p_value': float(p_value),
    }


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, alpha=0.05, seed=13):
    """Paired bootstrap test for a difference in mean scores.

    This function reports two useful things:
    1. A bootstrap confidence interval for the observed paired difference.
    2. A simple two-sided p-value using a centered bootstrap null distribution.

    It is suitable for classroom demonstration with per-instance metric scores.
    """
    rng = np.random.default_rng(seed)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) != len(scores_b):
        raise ValueError('scores_a and scores_b must have the same length for paired bootstrap.')

    n = len(scores_a)
    diffs = scores_b - scores_a
    observed = float(np.mean(diffs))

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])

    ci_low, ci_high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    # Null distribution by centering the paired differences at zero.
    centered = diffs - np.mean(diffs)
    null_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        null_means[i] = np.mean(centered[idx])

    p_value = float(np.mean(np.abs(null_means) >= abs(observed)))

    return {
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'observed_difference_b_minus_a': observed,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': p_value,
    }


def _ngram_counts(tokens: Sequence[str], n: int):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def simple_corpus_bleu(hypotheses: Sequence[str], references: Sequence[str], max_order: int = 4, smooth: float = 1.0):
    """Small, dependency-free corpus BLEU implementation for teaching.

    This is not a replacement for sacreBLEU. It is included so the randomization
    test notebook can run with dummy data without additional metric libraries.
    """
    matches_by_order = np.zeros(max_order)
    possible_by_order = np.zeros(max_order)
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)
        for order in range(1, max_order + 1):
            hyp_counts = _ngram_counts(hyp_tokens, order)
            ref_counts = _ngram_counts(ref_tokens, order)
            overlap = hyp_counts & ref_counts
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(hyp_tokens) - order + 1, 0)

    precisions = (matches_by_order + smooth) / (possible_by_order + smooth)
    geo_mean = exp(np.mean(np.log(precisions)))
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else exp(1 - ref_len / hyp_len)
    return 100.0 * bp * geo_mean


def paired_randomization_test(outputs_a, outputs_b, references, metric_fn=None, n_permutations=10000, seed=7):
    """Paired randomization/permutation test for system outputs.

    Under the null hypothesis, Model A and Model B are exchangeable for each item.
    For each permutation, randomly swap the two outputs for some items and recompute
    the metric difference.
    """
    rng = np.random.default_rng(seed)
    outputs_a = np.asarray(outputs_a, dtype=object)
    outputs_b = np.asarray(outputs_b, dtype=object)
    references = np.asarray(references, dtype=object)
    if metric_fn is None:
        metric_fn = simple_corpus_bleu

    observed_a = metric_fn(outputs_a.tolist(), references.tolist())
    observed_b = metric_fn(outputs_b.tolist(), references.tolist())
    observed_diff = observed_b - observed_a

    null_diffs = np.empty(n_permutations)
    n = len(outputs_a)
    for i in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_a = outputs_a.copy()
        perm_b = outputs_b.copy()
        perm_a[swap] = outputs_b[swap]
        perm_b[swap] = outputs_a[swap]
        null_diffs[i] = metric_fn(perm_b.tolist(), references.tolist()) - metric_fn(perm_a.tolist(), references.tolist())

    p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))

    return {
        'metric_a': float(observed_a),
        'metric_b': float(observed_b),
        'observed_difference_b_minus_a': float(observed_diff),
        'p_value': p_value,
    }


def interpret_p_value(p_value, alpha=0.05):
    """Classroom-friendly p-value interpretation."""
    if p_value < alpha:
        return f'p = {p_value:.4f} < alpha = {alpha}: reject the null hypothesis.'
    return f'p = {p_value:.4f} >= alpha = {alpha}: do not reject the null hypothesis.'
