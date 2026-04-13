# Kappa Agreement Exercise

This exercise demonstrates how to calculate **Cohen's Kappa** to assess inter-annotator agreement. Refer Lecture 6, Slide 32.

## Dataset

The dataset `annotation_data.csv` contains 10 text items annotated by two annotators with sentiment labels:
- Positive
- Neutral
- Negative

## Instructions

1. Load the dataset using `pandas`.
2. Create a confusion matrix.
3. Calculate observed agreement (`p_o`).
4. Calculate expected agreement (`p_e`).
5. Compute Cohen’s Kappa:

```
kappa = (p_o - p_e) / (1 - p_e)
```

## Reference

Agreement scale (Landis & Koch, 1977):
- < 0.00 — Poor
- 0.00–0.20 — Slight
- 0.21–0.40 — Fair
- 0.41–0.60 — Moderate
- 0.61–0.80 — Substantial
- 0.81–1.00 — Almost perfect
