import pandas as pd
from pathlib import Path
import numpy as np

from phase2_models import (
    load_german,
    load_adult,
    create_split,
    create_preprocessor,
    evaluate_fairness,
    create_intersectional_groups,
    RANDOM_SEEDS
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from fairlearn.adversarial import AdversarialFairnessClassifier


# ============================================================
# ADVERSARIAL DEBIASING
# ============================================================

def create_adversarial_model(alpha, seed):
    """
    Create an adversarial fairness classifier.

    alpha controls how strongly fairness is emphasized.
    Larger alpha means stronger adversarial pressure.
    """

    model = AdversarialFairnessClassifier(
        predictor_model=[32, 16, 1],
        adversary_model=[16, 8, 1],
        alpha=alpha,
        epochs=30,
        batch_size=64,
        random_state=seed,
        progress_updates=0
    )

    return model


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    datasets = {
        "German Credit": load_german,
        "Adult Income": load_adult
    }

    # --------------------------------------------------------
    # Adversarial fairness strengths
    # --------------------------------------------------------

    # Weak, medium and strong fairness pressure
    ALPHA_VALUES = {
        "weak": 0.1,
        "medium": 1.0,
        "strong": 10.0
    }

    # --------------------------------------------------------
    # Store experiment results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    for dataset_name, load_function in datasets.items():

        print(f"\n========== {dataset_name} ==========")

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        X, y, age_group, sex = load_function()

        # ----------------------------------------------------
        # Protected attributes
        # ----------------------------------------------------

        protected_attributes = {
            "Age": age_group,
            "Sex": sex
        }

        # ----------------------------------------------------
        # Run each protected attribute
        # ----------------------------------------------------

        for protected_name, protected_data in protected_attributes.items():

            # ------------------------------------------------
            # Run each fairness strength
            # ------------------------------------------------

            for strength_name, alpha in ALPHA_VALUES.items():

                # ------------------------------------------------
                # Run each random seed
                # ------------------------------------------------

                for seed in RANDOM_SEEDS:

                    print(
                        f"Running Adversarial | "
                        f"{dataset_name} | "
                        f"{protected_name} | "
                        f"{strength_name} | "
                        f"Seed {seed}"
                    )

                    # ==========================================
                    # CREATE SAME SPLIT AS PHASE 2
                    # ==========================================

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

                    # ==========================================
                    # PREPROCESS
                    # ==========================================

                    # Fit preprocessing ONLY on training data
                    preprocessor = create_preprocessor(X_train)

                    X_train_processed = preprocessor.fit_transform(
                        X_train
                    )

                    X_test_processed = preprocessor.transform(
                        X_test
                    )

                    # ==========================================
                    # CONVERT SPARSE TO DENSE
                    # ==========================================

                    # Fairlearn's adversarial classifier
                    # requires dense input.

                    if hasattr(X_train_processed, "toarray"):
                        X_train_processed = X_train_processed.toarray()

                    if hasattr(X_test_processed, "toarray"):
                        X_test_processed = X_test_processed.toarray()

                    # ==========================================
                    # SELECT PROTECTED ATTRIBUTE
                    # ==========================================

                    if protected_name == "Age":
                        protected_train = age_train
                    else:
                        protected_train = sex_train

                    # ==========================================
                    # CONVERT PROTECTED ATTRIBUTE TO BINARY
                    # ==========================================

                    protected_train_binary = (
                        protected_train
                        .astype("category")
                        .cat.codes
                        .to_numpy()
                    )

                    # ==========================================
                    # CREATE MODEL
                    # ==========================================

                    model = create_adversarial_model(
                        alpha,
                        seed
                    )

                    # ==========================================
                    # TRAIN MODEL
                    # ==========================================

                    model.fit(
                        X_train_processed,
                        y_train.to_numpy(),
                        sensitive_features=protected_train_binary
                    )

                    # ==========================================
                    # TEST PREDICTIONS
                    # ==========================================

                    y_pred = model.predict(
                        X_test_processed
                    )

                    # ==========================================
                    # PREDICTIVE PERFORMANCE
                    # ==========================================

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred
                    )

                    # Fairlearn 0.14.0 exposes hard predictions
                    # but does not provide predict_proba().
                    #
                    # Therefore ROC-AUC cannot be calculated
                    # consistently for this estimator.

                    auc = np.nan

                    # ==========================================
                    # AGE FAIRNESS
                    # ==========================================

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

                    # ==========================================
                    # SEX FAIRNESS
                    # ==========================================

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

                    # ==========================================
                    # INTERSECTIONAL FAIRNESS
                    # ==========================================

                    intersection_test = create_intersectional_groups(
                        age_test,
                        sex_test
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

                    # ==========================================
                    # STORE RESULT
                    # ==========================================

                    results.append({

                        "dataset": dataset_name,

                        "mitigation":
                            "Adversarial Debiasing",

                        "protected_attribute":
                            protected_name,

                        "strength":
                            strength_name,

                        "alpha":
                            alpha,

                        "seed":
                            seed,

                        # Performance
                        "accuracy":
                            accuracy,

                        "f1":
                            f1,

                        "roc_auc":
                            auc,

                        # Age fairness
                        "age_dpd":
                            age_dpd,

                        "age_eod":
                            age_eod,

                        "age_eopd":
                            age_eopd,

                        "age_di_ratio":
                            age_di,

                        # Sex fairness
                        "sex_dpd":
                            sex_dpd,

                        "sex_eod":
                            sex_eod,

                        "sex_eopd":
                            sex_eopd,

                        "sex_di_ratio":
                            sex_di,

                        # Intersectional fairness
                        "intersection_dpd":
                            intersection_dpd,

                        "intersection_eod":
                            intersection_eod,

                        "intersection_eopd":
                            intersection_eopd,

                        "intersection_di_ratio":
                            intersection_di
                    })


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\nAdversarial Debiasing Results:")
    print(results_df)


    # ========================================================
    # MEAN ± STANDARD DEVIATION
    # ========================================================

    summary = results_df.groupby(
        [
            "dataset",
            "mitigation",
            "protected_attribute",
            "strength",
            "alpha"
        ]
    ).agg(

        # -------------------------
        # Performance
        # -------------------------

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),

        auc_mean=("roc_auc", "mean"),
        auc_std=("roc_auc", "std"),

        # -------------------------
        # Age fairness
        # -------------------------

        age_dpd_mean=("age_dpd", "mean"),
        age_dpd_std=("age_dpd", "std"),

        age_eod_mean=("age_eod", "mean"),
        age_eod_std=("age_eod", "std"),

        age_eopd_mean=("age_eopd", "mean"),
        age_eopd_std=("age_eopd", "std"),

        age_di_mean=("age_di_ratio", "mean"),
        age_di_std=("age_di_ratio", "std"),

        # -------------------------
        # Sex fairness
        # -------------------------

        sex_dpd_mean=("sex_dpd", "mean"),
        sex_dpd_std=("sex_dpd", "std"),

        sex_eod_mean=("sex_eod", "mean"),
        sex_eod_std=("sex_eod", "std"),

        sex_eopd_mean=("sex_eopd", "mean"),
        sex_eopd_std=("sex_eopd", "std"),

        sex_di_mean=("sex_di_ratio", "mean"),
        sex_di_std=("sex_di_ratio", "std"),

        # -------------------------
        # Intersectional fairness
        # -------------------------

        intersection_dpd_mean=("intersection_dpd", "mean"),
        intersection_dpd_std=("intersection_dpd", "std"),

        intersection_eod_mean=("intersection_eod", "mean"),
        intersection_eod_std=("intersection_eod", "std"),

        intersection_eopd_mean=("intersection_eopd", "mean"),
        intersection_eopd_std=("intersection_eopd", "std"),

        intersection_di_mean=("intersection_di_ratio", "mean"),
        intersection_di_std=("intersection_di_ratio", "std")
    )


    print("\n\nAdversarial Mean ± Standard Deviation:")
    print(summary)


    # ========================================================
    # SAVE ADVERSARIAL RESULTS
    # ========================================================

    output_folder = Path("outputs") / "tables"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save individual experiment results
    # --------------------------------------------------------

    results_df.to_csv(
        output_folder /
        "phase3_adversarial_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.to_csv(
        output_folder /
        "phase3_adversarial_summary.csv"
    )

    print("\nAdversarial results saved to:")

    print(
        output_folder /
        "phase3_adversarial_results.csv"
    )

    print("\nAdversarial summary saved to:")

    print(
        output_folder /
        "phase3_adversarial_summary.csv"
    )