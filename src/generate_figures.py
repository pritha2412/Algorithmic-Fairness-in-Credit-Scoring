import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

# Location of this Python file:
#
# project/
# └── src/
#     └── generate_figures.py
#
# Therefore, one level above src = project root.

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# CSV files are inside outputs/tables/
TABLE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "tables"
)


# Figures will be saved inside outputs/figures/
FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "figures"
)


# Create figures folder if it does not already exist
os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# 2. FIND SUMMARY CSV FILES
# ============================================================

def find_summary_csv(prefix):
    """
    Find the summary CSV corresponding to an experiment.

    We specifically look for filenames containing both:
        prefix
        summary

    This prevents accidentally selecting the *_results.csv
    files when both results and summary files exist.
    """

    matching_files = []

    for filename in os.listdir(TABLE_DIR):

        if (
            filename.startswith(prefix)
            and "summary" in filename
            and filename.endswith(".csv")
        ):
            matching_files.append(filename)


    # No file found
    if len(matching_files) == 0:

        raise FileNotFoundError(
            "\nCould not find the required summary CSV.\n\n"
            f"Expected a file starting with:\n"
            f"    {prefix}\n\n"
            f"inside:\n"
            f"    {TABLE_DIR}\n\n"
            "Please check the filename and folder."
        )


    # Sort so selection is deterministic
    matching_files.sort()


    # If more than one matching summary exists,
    # show them so we know what is happening.
    if len(matching_files) > 1:

        print(
            f"\nWARNING: Multiple summary files found "
            f"for '{prefix}':"
        )

        for filename in matching_files:
            print(
                "   -",
                filename
            )

        print(
            "\nUsing:",
            matching_files[0]
        )


    return os.path.join(
        TABLE_DIR,
        matching_files[0]
    )


# ============================================================
# 3. LOCATE THE FOUR FINAL SUMMARY CSVs
# ============================================================

BASELINE_FILE = find_summary_csv(
    "phase2_baseline_summary"
)

REWEIGHING_FILE = find_summary_csv(
    "phase3_reweighing_summary"
)

ADVERSARIAL_FILE = find_summary_csv(
    "phase3_adversarial_summary"
)

CEODDS_FILE = find_summary_csv(
    "phase3_calibrated_equalized_odds_summary"
)


# ============================================================
# 4. LOAD CSV FILES
# ============================================================

print("\n" + "=" * 60)
print("LOADING CSV FILES")
print("=" * 60)


print("\nBaseline:")
print(
    BASELINE_FILE
)


print("\nReweighing:")
print(
    REWEIGHING_FILE
)


print("\nAdversarial:")
print(
    ADVERSARIAL_FILE
)


print("\nCE-Odds:")
print(
    CEODDS_FILE
)


# Read CSV files
baseline = pd.read_csv(
    BASELINE_FILE
)

reweighing = pd.read_csv(
    REWEIGHING_FILE
)

adversarial = pd.read_csv(
    ADVERSARIAL_FILE
)

ceodds = pd.read_csv(
    CEODDS_FILE
)


print("\nCSV files loaded successfully.")


print(
    "\nBaseline shape:",
    baseline.shape
)

print(
    "Reweighing shape:",
    reweighing.shape
)

print(
    "Adversarial shape:",
    adversarial.shape
)

print(
    "CE-Odds shape:",
    ceodds.shape
)


# ============================================================
# 5. STANDARDIZE DATASET AND MODEL NAMES
# ============================================================

def standardize_dataset_name(name):
    """
    Convert different possible dataset names into
    the standard names used in the figures.
    """

    name = str(name).strip().lower()

    if "adult" in name:
        return "Adult"

    if "german" in name:
        return "German"

    return str(name).strip()


def standardize_model_name(name):
    """
    Convert different possible model names into
    standard names used in the figures.
    """

    name = str(name).strip().lower()

    if "logistic" in name:
        return "Logistic Regression"

    if "random" in name and "forest" in name:
        return "Random Forest"

    if "xgb" in name or "xgboost" in name:
        return "XGBoost"

    return str(name).strip()


# Apply dataset-name standardization to every CSV

baseline["dataset"] = baseline["dataset"].apply(
    standardize_dataset_name
)

reweighing["dataset"] = reweighing["dataset"].apply(
    standardize_dataset_name
)

adversarial["dataset"] = adversarial["dataset"].apply(
    standardize_dataset_name
)

ceodds["dataset"] = ceodds["dataset"].apply(
    standardize_dataset_name
)


# Apply model-name standardization wherever a model column exists

if "model" in baseline.columns:
    baseline["model"] = baseline["model"].apply(
        standardize_model_name
    )

if "model" in reweighing.columns:
    reweighing["model"] = reweighing["model"].apply(
        standardize_model_name
    )

if "model" in adversarial.columns:
    adversarial["model"] = adversarial["model"].apply(
        standardize_model_name
    )

if "model" in ceodds.columns:
    ceodds["model"] = ceodds["model"].apply(
        standardize_model_name
    )


# ============================================================
# 6. BASIC SETTINGS
# ============================================================

DATASETS = [
    "Adult",
    "German"
]

MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

MODEL_SHORT = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "XGBoost": "XGB"
}


# Print standardized values so we can verify them

print("\nStandardized dataset names:")

print(
    "Baseline:",
    baseline["dataset"].unique()
)

print(
    "Reweighing:",
    reweighing["dataset"].unique()
)

print(
    "Adversarial:",
    adversarial["dataset"].unique()
)

print(
    "CE-Odds:",
    ceodds["dataset"].unique()
)

# ============================================================
# 6. HELPER FUNCTION
# ============================================================

def save_figure(filename):
    """
    Save the current figure inside outputs/figures/
    at 300 DPI for research-paper quality.
    """

    file_path = os.path.join(
        FIGURE_DIR,
        filename
    )


    plt.tight_layout()


    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nSaved:",
        file_path
    )


# ============================================================
# FIGURE 1
# BASELINE DISPARATE IMPACT
# ============================================================

print("\n" + "=" * 60)
print("GENERATING FIGURE 1")
print("=" * 60)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# Three protected-attribute dimensions
protected_attributes = [
    ("age_di_mean", "Age"),
    ("sex_di_mean", "Sex"),
    ("intersection_di_mean", "Age × Sex")
]


# Positions of the three models
x = np.arange(
    len(MODELS)
)


width = 0.35


for ax, (column, title) in zip(
    axes,
    protected_attributes
):

    # --------------------------------------------------------
    # Adult
    # --------------------------------------------------------

    adult_values = baseline[
        baseline["dataset"] == "Adult"
    ][column].values


    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    german_values = baseline[
        baseline["dataset"] == "German"
    ][column].values


    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    ax.bar(
        x - width / 2,
        adult_values,
        width,
        label="Adult"
    )


    ax.bar(
        x + width / 2,
        german_values,
        width,
        label="German"
    )


    # --------------------------------------------------------
    # Ideal DI = 1
    # --------------------------------------------------------

    ax.axhline(
        y=1,
        linestyle="--",
        linewidth=1.5
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(
        title + " Disparate Impact",
        fontsize=13,
        fontweight="bold"
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        [
            MODEL_SHORT[model]
            for model in MODELS
        ]
    )


    ax.set_ylabel(
        "Disparate Impact"
    )


    ax.set_ylim(
        0,
        1.1
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


axes[0].legend()


fig.suptitle(
    "Baseline Disparate Impact Across Protected Attributes",
    fontsize=16,
    fontweight="bold"
)


save_figure(
    "Figure_1_Baseline_DI.png"
)


# ============================================================
# FIGURE 2
# FAIRNESS VS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("GENERATING FIGURE 2")
print("=" * 60)


def calculate_fairness_distance(row):
    """
    Calculate a descriptive distance from the fairness ideal.

    For:
        DPD = 0 is ideal
        EOD = 0 is ideal
        EOPD = 0 is ideal
        DI = 1 is ideal

    This is ONLY a descriptive summary measure.
    It is NOT an optimization score.
    """

    distances = []


    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    distances.append(
        abs(row["age_dpd_mean"])
    )

    distances.append(
        abs(row["age_eod_mean"])
    )

    distances.append(
        abs(row["age_eopd_mean"])
    )

    distances.append(
        abs(row["age_di_mean"] - 1)
    )


    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    distances.append(
        abs(row["sex_dpd_mean"])
    )

    distances.append(
        abs(row["sex_eod_mean"])
    )

    distances.append(
        abs(row["sex_eopd_mean"])
    )

    distances.append(
        abs(row["sex_di_mean"] - 1)
    )


    # --------------------------------------------------------
    # Age × Sex intersection
    # --------------------------------------------------------

    distances.append(
        abs(row["intersection_dpd_mean"])
    )

    distances.append(
        abs(row["intersection_eod_mean"])
    )

    distances.append(
        abs(row["intersection_eopd_mean"])
    )

    distances.append(
        abs(row["intersection_di_mean"] - 1)
    )


    return np.mean(
        distances
    )


# ------------------------------------------------------------
# Add method labels
# ------------------------------------------------------------

baseline_plot = baseline.copy()

baseline_plot["Method"] = "Baseline"

baseline_plot["Fairness_Distance"] = baseline_plot.apply(
    calculate_fairness_distance,
    axis=1
)


reweighing_plot = reweighing.copy()

reweighing_plot["Method"] = "Reweighing"

reweighing_plot["Fairness_Distance"] = reweighing_plot.apply(
    calculate_fairness_distance,
    axis=1
)


adversarial_plot = adversarial.copy()

adversarial_plot["Method"] = "Adversarial"

adversarial_plot["Fairness_Distance"] = adversarial_plot.apply(
    calculate_fairness_distance,
    axis=1
)


ceodds_plot = ceodds.copy()

ceodds_plot["Method"] = "CE-Odds"

ceodds_plot["Fairness_Distance"] = ceodds_plot.apply(
    calculate_fairness_distance,
    axis=1
)


# ------------------------------------------------------------
# Combine all methods
# ------------------------------------------------------------

all_methods = pd.concat(
    [
        baseline_plot,
        reweighing_plot,
        adversarial_plot,
        ceodds_plot
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# Average configurations
# ------------------------------------------------------------

summary = (
    all_methods
    .groupby(
        [
            "dataset",
            "Method"
        ]
    )
    .agg(
        Accuracy=(
            "accuracy_mean",
            "mean"
        ),
        Fairness_Distance=(
            "Fairness_Distance",
            "mean"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for dataset in DATASETS:

    data = summary[
        summary["dataset"] == dataset
    ]


    for _, row in data.iterrows():

        ax.scatter(
            row["Fairness_Distance"],
            row["Accuracy"],
            s=100
        )


        ax.annotate(
            f"{dataset} - {row['Method']}",
            (
                row["Fairness_Distance"],
                row["Accuracy"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9
        )


ax.set_xlabel(
    "Mean Distance from Fairness Ideal",
    fontsize=12
)


ax.set_ylabel(
    "Mean Accuracy",
    fontsize=12
)


ax.set_title(
    "Fairness–Accuracy Positioning Across Methods",
    fontsize=15,
    fontweight="bold"
)


ax.grid(
    alpha=0.3
)


fig.text(
    0.5,
    0.01,
    "Lower fairness distance indicates closer proximity to the fairness ideals. "
    "This is a descriptive summary, not an optimized trade-off score.",
    ha="center",
    fontsize=9
)


plt.subplots_adjust(
    bottom=0.15
)


save_figure(
    "Figure_2_Fairness_Accuracy_Tradeoff.png"
)


# ============================================================
# FIGURE 3
# INTERSECTIONAL DI BEFORE AND AFTER MITIGATION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING FIGURE 3")
print("=" * 60)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


methods = [
    "Baseline",
    "Reweighing",
    "Adversarial",
    "CE-Odds"
]


for ax, dataset in zip(
    axes,
    DATASETS
):

    values = []


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_value = baseline[
        baseline["dataset"] == dataset
    ]["intersection_di_mean"].mean()


    values.append(
        baseline_value
    )


    # --------------------------------------------------------
    # Reweighing
    # --------------------------------------------------------

    reweighing_value = reweighing[
        reweighing["dataset"] == dataset
    ]["intersection_di_mean"].mean()


    values.append(
        reweighing_value
    )


    # --------------------------------------------------------
    # Adversarial
    # --------------------------------------------------------

    adversarial_value = adversarial[
        adversarial["dataset"] == dataset
    ]["intersection_di_mean"].mean()


    values.append(
        adversarial_value
    )


    # --------------------------------------------------------
    # CE-Odds
    # --------------------------------------------------------

    ceodds_value = ceodds[
        ceodds["dataset"] == dataset
    ]["intersection_di_mean"].mean()


    values.append(
        ceodds_value
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    x = np.arange(
        len(methods)
    )


    ax.bar(
        x,
        values
    )


    # Ideal DI = 1
    ax.axhline(
        y=1,
        linestyle="--",
        linewidth=1.5
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        methods,
        rotation=20
    )


    ax.set_ylabel(
        "Intersectional Disparate Impact"
    )


    ax.set_title(
        dataset,
        fontsize=14,
        fontweight="bold"
    )


    ax.set_ylim(
        0,
        1.1
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


fig.suptitle(
    "Intersectional Disparate Impact Before and After Mitigation",
    fontsize=16,
    fontweight="bold"
)


fig.text(
    0.5,
    0.01,
    "Values are means across the available model, target-attribute, "
    "and adversarial-strength configurations.",
    ha="center",
    fontsize=9
)


plt.subplots_adjust(
    bottom=0.16
)


save_figure(
    "Figure_3_Intersectional_DI_Mitigation.png"
)


# ============================================================
# FIGURE 4
# EFFECT OF ADVERSARIAL DEBIASING STRENGTH
# ============================================================

print("\n" + "=" * 60)
print("GENERATING FIGURE 4")
print("=" * 60)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


target_attributes = [
    "Age",
    "Sex"
]


strength_order = [
    0.1,
    1,
    10
]


for row_index, dataset in enumerate(
    DATASETS
):

    for col_index, protected_attribute in enumerate(
        target_attributes
    ):

        ax = axes[
            row_index,
            col_index
        ]


        # ----------------------------------------------------
        # Select relevant adversarial rows
        # ----------------------------------------------------

        data = adversarial[
            (adversarial["dataset"] == dataset)
            &
            (
                adversarial["protected_attribute"]
                == protected_attribute
            )
        ].copy()


        # Sort by alpha
        data = data.sort_values(
            "alpha"
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        ax.plot(
            data["alpha"],
            data["accuracy_mean"],
            marker="o",
            linewidth=2,
            label="Accuracy"
        )


        # Logarithmic alpha scale
        ax.set_xscale(
            "log"
        )


        ax.set_xticks(
            strength_order
        )


        ax.set_xticklabels(
            [
                "0.1",
                "1",
                "10"
            ]
        )


        ax.set_xlabel(
            "Adversarial Strength (α)"
        )


        ax.set_ylabel(
            "Accuracy"
        )


        # ----------------------------------------------------
        # Intersectional DI
        # ----------------------------------------------------

        ax2 = ax.twinx()


        ax2.plot(
            data["alpha"],
            data["intersection_di_mean"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Intersectional DI"
        )


        # Ideal DI = 1
        ax2.axhline(
            y=1,
            linestyle=":",
            linewidth=1.2
        )


        ax2.set_ylabel(
            "Intersectional Disparate Impact"
        )


        ax.set_title(
            f"{dataset} — {protected_attribute}-targeted",
            fontsize=13,
            fontweight="bold"
        )


        ax.grid(
            alpha=0.3
        )


# ------------------------------------------------------------
# Overall title
# ------------------------------------------------------------

fig.suptitle(
    "Effect of Adversarial Debiasing Strength",
    fontsize=16,
    fontweight="bold"
)


# ------------------------------------------------------------
# Get legend handles from first panel
# ------------------------------------------------------------

first_primary_axis = axes[
    0,
    0
]


handles_1, labels_1 = (
    first_primary_axis
    .get_legend_handles_labels()
)


# Find the twin axis belonging to first panel.
#
# Matplotlib stores it after the primary axis.
first_twin_axis = fig.axes[1]


handles_2, labels_2 = (
    first_twin_axis
    .get_legend_handles_labels()
)


fig.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    loc="lower center",
    ncol=2
)


plt.subplots_adjust(
    bottom=0.12
)


save_figure(
    "Figure_4_Adversarial_Strength.png"
)


# ============================================================
# FIGURE 5
# PREDICTIVE PERFORMANCE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("GENERATING FIGURE 5")
print("=" * 60)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


# ------------------------------------------------------------
# Add method labels
# ------------------------------------------------------------

baseline_temp = baseline.copy()

baseline_temp["Method"] = "Baseline"


reweighing_temp = reweighing.copy()

reweighing_temp["Method"] = "Reweighing"


adversarial_temp = adversarial.copy()

adversarial_temp["Method"] = "Adversarial"


ceodds_temp = ceodds.copy()

ceodds_temp["Method"] = "CE-Odds"


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

performance = pd.concat(
    [
        baseline_temp,
        reweighing_temp,
        adversarial_temp,
        ceodds_temp
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# Average available configurations
# ------------------------------------------------------------

performance_summary = (
    performance
    .groupby(
        [
            "dataset",
            "Method"
        ]
    )
    .agg(
        Accuracy=(
            "accuracy_mean",
            "mean"
        ),
        F1=(
            "f1_mean",
            "mean"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = np.arange(
    len(methods)
)


width = 0.35


for ax, dataset in zip(
    axes,
    DATASETS
):

    data = performance_summary[
        performance_summary["dataset"] == dataset
    ]


    accuracy_values = []

    f1_values = []


    for method in methods:

        row = data[
            data["Method"] == method
        ]


        accuracy_values.append(
            row["Accuracy"].iloc[0]
        )


        f1_values.append(
            row["F1"].iloc[0]
        )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    ax.bar(
        x - width / 2,
        accuracy_values,
        width,
        label="Accuracy"
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    ax.bar(
        x + width / 2,
        f1_values,
        width,
        label="F1-score"
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        methods,
        rotation=20
    )


    ax.set_ylabel(
        "Score"
    )


    ax.set_ylim(
        0,
        1
    )


    ax.set_title(
        dataset,
        fontsize=14,
        fontweight="bold"
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


axes[0].legend()


fig.suptitle(
    "Predictive Performance Across Mitigation Methods",
    fontsize=16,
    fontweight="bold"
)


fig.text(
    0.5,
    0.01,
    "Values represent means across the available model and mitigation configurations.",
    ha="center",
    fontsize=9
)


plt.subplots_adjust(
    bottom=0.16
)


save_figure(
    "Figure_5_Performance_Comparison.png"
)


# ============================================================
# 7. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)


print(
    "\nFigures saved in:"
)


print(
    FIGURE_DIR
)


print(
    "\nGenerated PNG files:"
)


for filename in sorted(
    os.listdir(FIGURE_DIR)
):

    if filename.endswith(".png"):

        print(
            " -",
            filename
        )


print(
    "\nDone!"
)