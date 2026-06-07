# Power Analysis Exercise

This exercise accompanies **Lecture 6: Experimental Design**, **slide 32**, in particular, on the question:

> How can I estimate whether my test/dev dataset is large enough?

The exercise introduces **power analysis** as a way to reason about the reliability of model comparisons. In NLP, we often compare two systems and ask whether one method is better than another. However, a small observed score difference may be difficult to interpret if the evaluation set is small or noisy.

Power analysis helps make this question more precise by connecting the following concepts:

* **Effect size**: the expected difference between two methods, for example a 1–2 percentage-point accuracy improvement.
* **Sample size**: the number of test instances, dev instances, human ratings, or annotated examples used in the evaluation.
* **Significance threshold**: the criterion for statistical significance, commonly `alpha = 0.05`.
* **Statistical power**: the probability that an experiment will detect a real difference between methods, often targeted at `0.80`.
* **Variance / agreement assumptions**: assumptions about how noisy the metric is, or how similarly two systems behave on the same examples.
* **Minimum detectable effect**: the smallest performance difference that can be detected reliably for a fixed dataset size.

The notebooks use simulation-based power analysis. Students specify assumptions about the expected model scores, dataset size, significance threshold, and target power. The simulations then estimate either:

1. how many evaluation instances are needed to detect a given expected difference, or
2. what size of performance difference can realistically be detected with an already fixed dataset.

The goal is to offer a practical understanding of how dataset size, effect size, and statistical testing interact in experimental NLP.

## Structure

```text
nlp_power_analysis_teaching_exercises/
├── README.md
├── requirements.txt
├── ORIGINAL_REPO_LICENSE
├── data/
│   ├── README.md
│   ├── demo/
│   │   └── paired_metric_scores.csv
│   ├── glue/
│   │   ├── glue_task_test_set_sizes.csv
│   │   └── glue_reported_results.csv
│   ├── squad2/
│   │   ├── models.tsv
│   │   └── pairs_dev.tsv
│   └── human_eval/
│       ├── num_items_vs_effect_size.csv
│       └── likert_ratings.csv
├── src/
│   └── power_utils.py
└── notebooks/
    ├── 00_start_here_power_analysis_concepts.ipynb
    ├── 01_accuracy_mcnemar_power.ipynb
    ├── 02_fixed_dataset_minimum_detectable_effect.ipynb
    ├── 03_generic_paired_metric_power.ipynb
    └── 04_human_likert_power.ipynb
```

## Installation

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order. Each notebook has a clear **Student input** cell. Students either enter values through prompts or replace the parameters with values from their own experiment.

## Source credits

This exercise adapts code and ideas from the paper With Little Power Comes Great Responsibility: Statistical Significance and Power in NLP:

https://aclanthology.org/2020.emnlp-main.745/

The original code repository associated with the paper is available at:

https://github.com/dallascard/NLP-power-analysis


