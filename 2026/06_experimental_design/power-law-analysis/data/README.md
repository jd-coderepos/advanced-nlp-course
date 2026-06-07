# Curated data for the teaching exercises

This folder intentionally contains only the data files needed for the class notebooks.
It does **not** include the original repository's figure-generation code, data-import notebooks, or large MT data archive.

Included files:

- `glue/glue_task_test_set_sizes.csv` — small table of GLUE/SQuAD test set sizes.
- `glue/glue_reported_results.csv` — reported GLUE model comparisons used for classroom discussion.
- `squad2/models.tsv` and `squad2/pairs_dev.tsv` — SQuAD2 leaderboard/dev-pair information useful for explaining paired model comparisons.
- `human_eval/num_items_vs_effect_size.csv` and `human_eval/likert_ratings.csv` — summarized human-evaluation data for class demos.
- `demo/paired_metric_scores.csv` — synthetic per-instance metric scores for the generic paired-metric notebook.

The original code repository attached by the instructor contained additional folders such as `code_for_figures/`, `analyses/`, and `data_import/`. Those are intentionally omitted here because this package is designed as a clean teaching exercise, not as a copy of the research repository.
