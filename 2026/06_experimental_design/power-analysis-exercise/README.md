# NLP Power Analysis — Teaching Exercises

This is a cleaned classroom exercise package inspired by Card et al. (2020), *With Little Power Comes Great Responsibility*.

The purpose is not to reproduce every analysis or figure from the paper. Instead, the package gives students practical notebooks for the slide question:

> How can I estimate whether my test/dev dataset is large enough?

Students will learn to:

1. start with an expected effect size, such as a 1–2 percentage-point improvement;
2. choose a significance threshold, usually `alpha = 0.05`;
3. choose desired power, often `0.80`;
4. simulate evaluation data under explicit assumptions;
5. estimate either:
   - how many instances are needed to detect the effect, or
   - what minimum detectable effect is realistic for a fixed dataset size.

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


