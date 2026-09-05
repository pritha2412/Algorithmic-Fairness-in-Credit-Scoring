import pandas as pd
from pathlib import Path
import numpy as np

from phase2_models import (
    load_german,
    load_adult,
    create_split,
    create_preprocessor,
    create_model,
    evaluate_fairness,
    create_intersectional_groups,
    RANDOM_SEEDS
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# REWEIGHING WEIGHTS
# ============================================================

def calculate_reweighing_weights(protected_attribute, y):

    data = pd.DataFrame({
        "protected": protected_attribute,
        "target": y
    })

    total = len(data)

    group_counts = data["protected"].value_counts()
    target_counts = data["target"].value_counts()

    group_target_counts = (
        data.groupby(["protected", "target"])
        .size()
    )

    weights = []

    for _, row in data.iterrows():

        group = row["protected"]
        target = row["target"]

        weight = (
            group_counts[group] * target_counts[target]
        ) / (
            total * group_target_counts[group, target]
        )

        weights.append(weight)

    return np.array(weights)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    datasets = {
        "German Credit": load_german,
        "Adult Income": load_adult
    }

    results = []

    protected_attributes_names = [
        "Age",
        "Sex"
    ]

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    for dataset_name, load_function in datasets.items():

        print(f"\n========== {dataset_name} ==========")

        X, y, age_group, sex = load_function()

        protected_attributes = {
            "Age": age_group,
            "Sex": sex
        }

        for model_name in model_names:

            for protected_name in protected_attributes_names:

                for seed in RANDOM_SEEDS:

                    print(
                        f"Running Reweighing | "
                        f"{dataset_name} | "
                        f"{model_name} | "
                        f"{protected_name} | "
                        f"Seed {seed}"
                    )

                    # ----------------------------------------
                    # Same split as Phase 2
                    # ----------------------------------------

                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        age_train,
                        age_test,
                        sex_train,
                        sex_test
                    ) = create_split(
                        X,
                        y,
                        age_group,
                        sex,
                        seed
                    )

                    # ----------------------------------------
                    # Select protected attribute
                    # ----------------------------------------

                    if protected_name == "Age":
                        protected_train = age_train
                    else:
                        protected_train = sex_train

                    # ----------------------------------------
                    # Calculate training weights
                    # ----------------------------------------

                    weights = calculate_reweighing_weights(
                        protected_train,
                        y_train
                    )

                    # ----------------------------------------
                    # Preprocessing
                    # ----------------------------------------

                    preprocessor = create_preprocessor(
                        X_train
                    )

                    # ----------------------------------------
                    # Model
                    # ----------------------------------------

                    model = create_model(
                        model_name,
                        preprocessor
                    )

                    # ----------------------------------------
                    # Weighted training
                    # ----------------------------------------

                    model.fit(
                        X_train,
                        y_train,
                        classifier__sample_weight=weights
                    )

                    # ----------------------------------------
                    # Predictions
                    # ----------------------------------------

                    y_pred = model.predict(X_test)

                    y_prob = model.predict_proba(
                        X_test
                    )[:, 1]

                    # ----------------------------------------
                    # Performance
                    # ----------------------------------------

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred
                    )

                    auc = roc_auc_score(
                        y_test,
                        y_prob
                    )

                    # ----------------------------------------
                    # Age fairness
                    # ----------------------------------------

                    (
                        age_dpd,
                        age_eod,
                        age_eopd,
                        age_di
                    ) = evaluate_fairness(
                        y_test,
                        y_pred,
                        age_test
                    )

                    # ----------------------------------------
                    # Sex fairness
                    # ----------------------------------------

                    (
                        sex_dpd,
                        sex_eod,
                        sex_eopd,
                        sex_di
                    ) = evaluate_fairness(
                        y_test,
                        y_pred,
                        sex_test
                    )

                    # ----------------------------------------
                    # Intersectional fairness
                    # ----------------------------------------

                    intersection_test = (
                        create_intersectional_groups(
                            age_test,
                            sex_test
                        )
                    )

                    (
                        intersection_dpd,
                        intersection_eod,
                        intersection_eopd,
                        intersection_di
                    ) = evaluate_fairness(
                        y_test,
                        y_pred,
                        intersection_test
                    )

                    # ----------------------------------------
                    # Store result
                    # ----------------------------------------

                    results.append({

                        "dataset": dataset_name,
                        "mitigation": "Reweighing",
                        "protected_attribute": protected_name,
                        "model": model_name,
                        "seed": seed,

                        "accuracy": accuracy,
                        "f1": f1,
                        "roc_auc": auc,

                        "age_dpd": age_dpd,
                        "age_eod": age_eod,
                        "age_eopd": age_eopd,
                        "age_di_ratio": age_di,

                        "sex_dpd": sex_dpd,
                        "sex_eod": sex_eod,
                        "sex_eopd": sex_eopd,
                        "sex_di_ratio": sex_di,

                        "intersection_dpd": intersection_dpd,
                        "intersection_eod": intersection_eod,
                        "intersection_eopd": intersection_eopd,
                        "intersection_di_ratio": intersection_di
                    })


    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\nReweighing Results:")
    print(results_df)

    # ========================================================
    # MEAN ± STANDARD DEVIATION
    # ========================================================

    summary = results_df.groupby(
        [
            "dataset",
            "mitigation",
            "protected_attribute",
            "model"
        ]
    ).agg(

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),

        auc_mean=("roc_auc", "mean"),
        auc_std=("roc_auc", "std"),

        age_dpd_mean=("age_dpd", "mean"),
        age_dpd_std=("age_dpd", "std"),

        age_eod_mean=("age_eod", "mean"),
        age_eod_std=("age_eod", "std"),

        age_eopd_mean=("age_eopd", "mean"),
        age_eopd_std=("age_eopd", "std"),

        age_di_mean=("age_di_ratio", "mean"),
        age_di_std=("age_di_ratio", "std"),

        sex_dpd_mean=("sex_dpd", "mean"),
        sex_dpd_std=("sex_dpd", "std"),

        sex_eod_mean=("sex_eod", "mean"),
        sex_eod_std=("sex_eod", "std"),

        sex_eopd_mean=("sex_eopd", "mean"),
        sex_eopd_std=("sex_eopd", "std"),

        sex_di_mean=("sex_di_ratio", "mean"),
        sex_di_std=("sex_di_ratio", "std"),

        intersection_dpd_mean=("intersection_dpd", "mean"),
        intersection_dpd_std=("intersection_dpd", "std"),

        intersection_eod_mean=("intersection_eod", "mean"),
        intersection_eod_std=("intersection_eod", "std"),

        intersection_eopd_mean=("intersection_eopd", "mean"),
        intersection_eopd_std=("intersection_eopd", "std"),

        intersection_di_mean=("intersection_di_ratio", "mean"),
        intersection_di_std=("intersection_di_ratio", "std")
    )

    print("\n\nReweighing Mean ± Standard Deviation:")
    print(summary)

    # ========================================================
    # SAVE
    # ========================================================

    output_folder = Path("outputs") / "tables"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_folder / "phase3_reweighing_results.csv",
        index=False
    )

    summary.to_csv(
        output_folder / "phase3_reweighing_summary.csv"
    )

    print("\nReweighing results saved to:")
    print(
        output_folder /
        "phase3_reweighing_results.csv"
    )

    print("\nReweighing summary saved to:")
    print(
        output_folder /
        "phase3_reweighing_summary.csv"
    )