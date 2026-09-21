from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from scipy.stats import fisher_exact


# ============================================================
# SETTINGS
# ============================================================

SEED = 20260920

DATA_FILE = Path("negation_balanced_evaluation_396.csv")
PREDICTION_FILE = Path("gpt56sol_predictions_396.csv")


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Could not find negation_balanced_evaluation_396.csv"
    )

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        "Could not find gpt56sol_predictions_396.csv"
    )

data = pd.read_csv(DATA_FILE)
predictions = pd.read_csv(PREDICTION_FILE)

print("Evaluation examples:", len(data))
print("Prediction rows:", len(predictions))


# ============================================================
# 2. CHECK BALANCED DATASET
# ============================================================

dataset_summary = (
    data.groupby(["dataset", "importance"])
        .size()
        .rename("n")
        .reset_index()
)

print("\nBalanced evaluation set:")
print(dataset_summary.to_string(index=False))

assert len(data) == 396
assert (data["importance"] == "important").sum() == 198
assert (data["importance"] == "unimportant").sum() == 198


# ============================================================
# 3. CREATE BLINDED MODEL INPUT
# ============================================================

blinded_columns = [
    "sample_id",
    "dataset",
    "task_type",
    "input_1",
    "input_2",
    "choices"
]

blinded = data[blinded_columns].copy()

blinded.to_csv(
    "negation_blinded_model_input_396_RECREATED.csv",
    index=False
)

print(
    "\nCreated blinded input file:"
    "\nnegation_blinded_model_input_396_RECREATED.csv"
)


# ============================================================
# 4. MERGE MODEL PREDICTIONS
# ============================================================

required_prediction_columns = {
    "sample_id",
    "model_prediction"
}

if not required_prediction_columns.issubset(predictions.columns):
    raise ValueError(
        "Prediction file must contain sample_id "
        "and model_prediction columns."
    )

prediction_subset = predictions[
    ["sample_id", "model_prediction"]
].copy()

scored = data.merge(
    prediction_subset,
    on="sample_id",
    how="left",
    validate="one_to_one"
)

if scored["model_prediction"].isna().any():
    missing = scored.loc[
        scored["model_prediction"].isna(),
        "sample_id"
    ].tolist()

    raise ValueError(
        f"Missing model predictions for {len(missing)} examples."
    )


# ============================================================
# 5. NORMALIZE LABELS
# ============================================================

def normalize_label(dataset, value):

    text = str(value).strip().lower()

    # QNLI
    if dataset == "QNLI":

        text = (
            text.replace("-", "_")
                .replace(" ", "_")
        )

        aliases = {
            "entailment": "entailment",
            "entails": "entailment",
            "entailed": "entailment",
            "not_entailment": "not_entailment",
            "notentailment": "not_entailment",
            "non_entailment": "not_entailment"
        }

        return aliases.get(text, text)

    # CommonsenseQA
    elif dataset == "CommonsenseQA":

        match = re.search(
            r"\b([a-e])\b",
            text
        )

        if match:
            return match.group(1).upper()

        return str(value).strip().upper()

    # SST-2
    elif dataset == "SST-2":

        if text.startswith("pos"):
            return "positive"

        if text.startswith("neg"):
            return "negative"

        return text

    return text


scored["gold_norm"] = [
    normalize_label(dataset, label)
    for dataset, label
    in zip(
        scored["dataset"],
        scored["gold_label"]
    )
]

scored["pred_norm"] = [
    normalize_label(dataset, prediction)
    for dataset, prediction
    in zip(
        scored["dataset"],
        scored["model_prediction"]
    )
]

scored["correct"] = (
    scored["gold_norm"]
    ==
    scored["pred_norm"]
).astype(int)


scored.to_csv(
    "gpt56sol_scored_396_RECREATED.csv",
    index=False
)


# ============================================================
# 6. ACCURACY AND MACRO-F1
# ============================================================

metric_rows = []

datasets = [
    "QNLI",
    "CommonsenseQA",
    "SST-2"
]

importance_groups = [
    "important",
    "unimportant"
]

for dataset in datasets:

    for importance in importance_groups:

        subset = scored[
            (scored["dataset"] == dataset)
            &
            (scored["importance"] == importance)
        ]

        accuracy = subset["correct"].mean()

        macro_f1 = f1_score(
            subset["gold_norm"],
            subset["pred_norm"],
            average="macro"
        )

        metric_rows.append(
            {
                "dataset": dataset,
                "importance": importance,
                "n": len(subset),
                "correct": int(
                    subset["correct"].sum()
                ),
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }
        )


metrics = pd.DataFrame(metric_rows)

metrics.to_csv(
    "metrics_by_dataset_RECREATED.csv",
    index=False
)

print("\nPerformance by dataset:")
print(metrics.to_string(index=False))


# ============================================================
# 7. OVERALL ACCURACY
# ============================================================

overall = (
    scored.groupby("importance")
          .agg(
              n=("correct", "size"),
              correct=("correct", "sum"),
              accuracy=("correct", "mean")
          )
          .reset_index()
)

overall.to_csv(
    "overall_balanced_summary_RECREATED.csv",
    index=False
)

print("\nOverall performance:")
print(overall.to_string(index=False))


# ============================================================
# 8. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_accuracy_gap(
    important_correct,
    unimportant_correct,
    n_bootstrap=10000,
    seed=SEED
):

    rng = np.random.default_rng(seed)

    important_correct = np.asarray(
        important_correct,
        dtype=float
    )

    unimportant_correct = np.asarray(
        unimportant_correct,
        dtype=float
    )

    differences = []

    for _ in range(n_bootstrap):

        important_sample = rng.choice(
            important_correct,
            size=len(important_correct),
            replace=True
        )

        unimportant_sample = rng.choice(
            unimportant_correct,
            size=len(unimportant_correct),
            replace=True
        )

        difference = (
            unimportant_sample.mean()
            -
            important_sample.mean()
        )

        differences.append(difference)

    lower = np.quantile(
        differences,
        0.025
    )

    upper = np.quantile(
        differences,
        0.975
    )

    return lower, upper


# ============================================================
# 9. FISHER EXACT TESTS
# ============================================================

test_rows = []

for dataset in datasets:

    important = scored[
        (scored["dataset"] == dataset)
        &
        (scored["importance"] == "important")
    ]

    unimportant = scored[
        (scored["dataset"] == dataset)
        &
        (scored["importance"] == "unimportant")
    ]

    important_accuracy = (
        important["correct"].mean()
    )

    unimportant_accuracy = (
        unimportant["correct"].mean()
    )

    accuracy_gap = (
        unimportant_accuracy
        -
        important_accuracy
    )

    ci_low, ci_high = (
        bootstrap_accuracy_gap(
            important["correct"],
            unimportant["correct"]
        )
    )

    contingency_table = [
        [
            int(
                important["correct"].sum()
            ),
            int(
                len(important)
                -
                important["correct"].sum()
            )
        ],
        [
            int(
                unimportant["correct"].sum()
            ),
            int(
                len(unimportant)
                -
                unimportant["correct"].sum()
            )
        ]
    ]

    odds_ratio, p_value = fisher_exact(
        contingency_table,
        alternative="two-sided"
    )

    test_rows.append(
        {
            "dataset": dataset,
            "accuracy_gap_unimportant_minus_important":
                accuracy_gap,
            "bootstrap_95ci_low":
                ci_low,
            "bootstrap_95ci_high":
                ci_high,
            "fisher_odds_ratio":
                odds_ratio,
            "fisher_p_raw":
                p_value
        }
    )


tests = pd.DataFrame(test_rows)


# ============================================================
# 10. HOLM CORRECTION
# ============================================================

number_of_tests = len(tests)

order = np.argsort(
    tests["fisher_p_raw"].to_numpy()
)

adjusted_p_values = np.empty(
    number_of_tests
)

previous_adjusted = 0.0

for rank, index in enumerate(order):

    adjusted = min(
        1.0,
        (
            number_of_tests
            -
            rank
        )
        *
        tests.loc[
            index,
            "fisher_p_raw"
        ]
    )

    adjusted = max(
        previous_adjusted,
        adjusted
    )

    adjusted_p_values[
        index
    ] = adjusted

    previous_adjusted = adjusted


tests["fisher_p_holm"] = (
    adjusted_p_values
)

tests["significant_holm_0.05"] = (
    tests["fisher_p_holm"]
    <
    0.05
)

tests.to_csv(
    "statistical_tests_RECREATED.csv",
    index=False
)

print("\nStatistical tests:")
print(tests.to_string(index=False))


# ============================================================
# 11. EXTRACT MODEL ERRORS
# ============================================================

errors = scored[
    scored["correct"] == 0
].copy()

errors.to_csv(
    "errors_for_manual_analysis.csv",
    index=False
)

print(
    "\nTotal incorrect predictions:",
    len(errors)
)


# ============================================================
# 12. FIGURE:
# ACCURACY BY DATASET AND IMPORTANCE
# ============================================================

accuracy_table = (
    metrics.pivot(
        index="dataset",
        columns="importance",
        values="accuracy"
    )
)

accuracy_table = accuracy_table.loc[
    [
        "QNLI",
        "CommonsenseQA",
        "SST-2"
    ]
]

ax = accuracy_table.plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_ylabel("Accuracy")
ax.set_xlabel("")
ax.set_ylim(0, 1)

ax.set_title(
    "Accuracy by Dataset and Negation Importance"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "accuracy_by_dataset_and_importance_RECREATED.png",
    dpi=200
)

plt.close()


# ============================================================
# 13. FIGURE:
# ACCURACY GAP + CONFIDENCE INTERVAL
# ============================================================

plot_data = tests.copy()

plot_data["error_low"] = (
    plot_data[
        "accuracy_gap_unimportant_minus_important"
    ]
    -
    plot_data[
        "bootstrap_95ci_low"
    ]
)

plot_data["error_high"] = (
    plot_data[
        "bootstrap_95ci_high"
    ]
    -
    plot_data[
        "accuracy_gap_unimportant_minus_important"
    ]
)

x_positions = np.arange(
    len(plot_data)
)

y_values = plot_data[
    "accuracy_gap_unimportant_minus_important"
].to_numpy()

error_values = np.vstack(
    [
        plot_data[
            "error_low"
        ].to_numpy(),
        plot_data[
            "error_high"
        ].to_numpy()
    ]
)

plt.figure(
    figsize=(8, 5)
)

plt.errorbar(
    x_positions,
    y_values,
    yerr=error_values,
    fmt="o",
    capsize=5
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    x_positions,
    plot_data["dataset"]
)

plt.ylabel(
    "Accuracy gap "
    "(unimportant - important)"
)

plt.title(
    "Accuracy Gap with "
    "Bootstrap 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "accuracy_gap_with_ci_RECREATED.png",
    dpi=200
)

plt.close()


# ============================================================
# 14. FIGURE:
# OVERALL ACCURACY
# ============================================================

overall_plot = (
    overall.set_index("importance")
           .loc[
               [
                   "important",
                   "unimportant"
               ]
           ]
)

plt.figure(
    figsize=(6, 5)
)

plt.bar(
    overall_plot.index,
    overall_plot["accuracy"]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Overall Accuracy on "
    "the Balanced Evaluation Set"
)

plt.tight_layout()

plt.savefig(
    "overall_accuracy_RECREATED.png",
    dpi=200
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\nAnalysis complete.")

print(
    "\nExpected approximate results:"
)

print(
    "Important overall accuracy: 0.9040"
)

print(
    "Unimportant overall accuracy: 0.9141"
)

print(
    "\nGenerated CSV files and figures "
    "are saved in the current folder."
)
