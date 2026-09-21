# Negation Sensitivity Across NLU Tasks

This repository contains the experimental materials for the term paper:

**Important or Incidental? Evaluating Negation Sensitivity Across NLU Tasks**

## Project Overview

This study investigates whether a modern language model performs differently when negation is important for determining the correct answer compared with cases where negation is incidental.

The evaluation covers three natural language understanding tasks:

- QNLI — Natural Language Inference
- CommonsenseQA — Question Answering
- SST-2 — Sentiment Analysis

The study builds on the distinction between important and unimportant negation introduced by Hossain et al. (2022).

## Main Experiment

A balanced evaluation set of 396 examples was constructed from the released negation annotations of Hossain et al. (2022):

- QNLI: 40 examples
- CommonsenseQA: 162 examples
- SST-2: 194 examples

The final set contains:

- 198 important-negation examples
- 198 unimportant-negation examples

GPT-5.6 Sol was evaluated in a zero-shot setting.

## Evaluation

The analysis includes:

- Accuracy
- Macro-F1
- Bootstrap 95% confidence intervals
- Fisher's exact tests
- Holm correction
- Qualitative error analysis

## Main Results

Overall accuracy:

- Important negation: 90.4%
- Unimportant negation: 91.4%

No statistically significant performance difference was observed across the three evaluated tasks.

## Repository Contents

- `negation_balanced_evaluation_396.csv` — balanced evaluation dataset
- `gpt56sol_predictions_396.csv` — model predictions
- `gpt56sol_error_analysis_36.csv` — qualitative error annotations
- `negation_analysis_reproduce.py` — analysis script
- `requirements.txt` — required Python packages

## Counterfactual Intervention

A secondary counterfactual experiment was prepared using 30 important-negation examples, with 10 examples from each task.

The blinded intervention file is included for a fresh evaluation run.

## Source

The negation annotations are based on:

Hossain et al. (2022),  
**An Analysis of Negation in Natural Language Understanding Corpora**

ACL Anthology:  
https://aclanthology.org/2022.acl-short.81/

The 396-example balanced evaluation set used in this repository was constructed specifically for the present study from the released annotations.

## Reproducibility

Install the required packages with:

```bash
pip install -r requirements.txt
