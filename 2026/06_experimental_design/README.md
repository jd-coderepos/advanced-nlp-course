### Lecture 6: Experimental Design

This repository contains two exercises for analysing experimental reliability in NLP: score significance on benchmark datasets and annotation agreement between annotators.

1. `power-analysis-exercise/`
   Introduces power analysis for estimating whether a development or test set is large enough to reliably detect differences between methods. The notebooks guide students through simulation-based power analysis using expected effect size, significance threshold, desired statistical power, and assumptions about variance or model agreement. The exercise also shows the inverse question: given a fixed dataset size, what is the smallest performance difference that can be detected with high confidence?

2. `kappa-agreement-exercise/`
   Demonstrates the application of Cohen’s kappa agreement score on annotations from two annotators. The exercise helps students understand why raw agreement alone can be misleading and how chance-corrected agreement provides a more informative measure of annotation reliability.

