# Statistical Significance Testing Exercise

This folder accompanies **Lecture 6: Experimental Design**.

A statistical significance test asks:

> Given the evaluation data I already have, is the observed difference between two methods large enough to reject the null hypothesis of no real difference?

The notebooks demonstrate four common tests used in NLP-style evaluations:

1. **McNemar's test** for paired classification accuracy.
2. **Paired bootstrap test** for per-instance metric scores, such as ROUGE-like, F1-like, or other document-level scores.
3. **Paired randomization/permutation test** for paired system outputs, demonstrated with a simple MT-style corpus metric.
4. **Paired Student's t-test** for paired numeric scores, such as human ratings or repeated-run scores.

The exercises use small dummy datasets in `data/dummy/` so that students can inspect the data directly and modify the assumptions.

## Folder structure

```text
statistical-significance-exercise/
  README.md
  requirements.txt
  data/
    dummy/
      classification_predictions.csv
      per_instance_scores.csv
      mt_outputs.csv
      human_ratings.csv
      repeated_runs.csv
  notebooks/
    00_start_here_significance_tests.ipynb
    01_mcnemar_paired_accuracy.ipynb
    02_bootstrap_paired_metric_scores.ipynb
    03_randomization_test_outputs.ipynb
    04_paired_t_test_scores.ipynb
  src/
    significance_utils.py
```

## Setup

From this folder, install the requirements:

```bash
pip install -r requirements.txt
```

Then open the notebooks:

```bash
jupyter notebook notebooks/
```

## What students should learn

By the end of the exercise, students should be able to explain:

- why paired evaluations require paired tests;
- why comparing only average scores can be misleading;
- what the null hypothesis means in model comparison;
- what a p-value does and does not mean;
- which test is appropriate for which kind of NLP result.

## Important distinction from power analysis

- **Significance testing** asks whether an observed result on an existing dataset is statistically reliable.
- **Power analysis** asks whether a planned evaluation setup is likely to detect an expected difference.

This folder is for the first question only.
