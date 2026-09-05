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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import (
    CalibratedEqOddsPostprocessing
)


# ============================================================
# CONVERT PROTECTED ATTRIBUTE TO BINARY
# ============================================================

def convert_to_binary(protected):
    """
    Convert a binary protected attribute into 0/1 values.
    """

    values = pd.Series(protected).astype(str)

    unique_values = sorted(values.unique())

    if len(unique_values) != 2:
        raise ValueError(
            "Protected attribute must contain exactly two groups."
        )

    mapping = {
        unique_values[0]: 0,
        unique_values[1]: 1
    }

    return values.map(mapping).to_numpy()


# ============================================================
# CREATE AIF360 DATASET
# ============================================================

def create_aif360_dataset(
    y,
    scores,
    protected
):
    """
    Create an AIF360 BinaryLabelDataset containing
    labels, protected attribute and prediction scores.
    """

    df = pd.DataFrame({
        "label": np.asarray(y).astype(float),
        "protected": np.asarray(protected).astype(float)
    })

    dataset = BinaryLabelDataset(
        favorable_label=1.0,
        unfavorable_label=0.0,
        df=df,
        label_names=["label"],
        protected_attribute_names=["protected"]
    )

    dataset.scores = np.asarray(scores).reshape(-1, 1)

    return dataset


# ============================================================
# APPLY CALIBRATED EQUALIZED ODDS
# ============================================================

def apply_calibrated_equalized_odds(
    y_cal,
    cal_prob,
    protected_cal,
    test_prob,
    protected_test,
    seed
):
    """
    Fit Calibrated Equalized Odds using the calibration set
    and apply it to the untouched test predictions.
    """

    # --------------------------------------------------------
    # BASE MODEL PREDICTIONS
    # --------------------------------------------------------

    cal_pred_labels = (
        cal_prob >= 0.5
    ).astype(float)

    test_pred_labels = (
        test_prob >= 0.5
    ).astype(float)


    # --------------------------------------------------------
    # TRUE CALIBRATION DATASET
    # --------------------------------------------------------

    calibration_true = create_aif360_dataset(
        y_cal,
        cal_prob,
        protected_cal
    )


    # --------------------------------------------------------
    # PREDICTED CALIBRATION DATASET
    #
    # Contains:
    # - base model predicted labels
    # - calibrated probability scores
    # --------------------------------------------------------

    calibration_pred = create_aif360_dataset(
        cal_pred_labels,
        cal_prob,
        protected_cal
    )


    # --------------------------------------------------------
    # CREATE CE-ODDS POSTPROCESSOR
    #
    # FNR cost constraint is used for the final experiment.
    # --------------------------------------------------------

    postprocessor = CalibratedEqOddsPostprocessing(

        unprivileged_groups=[
            {"protected": 0.0}
        ],

        privileged_groups=[
            {"protected": 1.0}
        ],

        cost_constraint="fnr",

        seed=seed
    )


    # --------------------------------------------------------
    # FIT CE-ODDS
    #
    # ONLY calibration data is used here.
    # Test data is NOT used.
    # --------------------------------------------------------

    postprocessor.fit(
        calibration_true,
        calibration_pred
    )


    # --------------------------------------------------------
    # TEST PREDICTION DATASET
    #
    # Contains base model predictions and calibrated scores.
    # True y_test is NOT supplied.
    # --------------------------------------------------------

    test_pred_dataset = create_aif360_dataset(
        test_pred_labels,
        test_prob,
        protected_test
    )


    # --------------------------------------------------------
    # APPLY CE-ODDS
    # --------------------------------------------------------

    mitigated_dataset = postprocessor.predict(
        test_pred_dataset
    )


    # --------------------------------------------------------
    # EXTRACT FINAL MITIGATED PREDICTIONS
    # --------------------------------------------------------

    y_pred = (
        mitigated_dataset.labels
        .ravel()
        .astype(int)
    )

    return y_pred


# ============================================================
# CREATE BASE CLASSIFIER
# ============================================================

def create_classifier(model_name, seed):
    """
    Create the base classifier.

    Probability calibration is handled separately.
    """

    if model_name == "Logistic Regression":

        classifier = LogisticRegression(
            max_iter=1000,
            random_state=seed
        )

    elif model_name == "Random Forest":

        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        )

    elif model_name == "XGBoost":

        classifier = XGBClassifier(
            n_estimators=100,
            random_state=seed,
            eval_metric="logloss"
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return classifier


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    datasets = {
        "German Credit": load_german,
        "Adult Income": load_adult
    }


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results = []


    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    for dataset_name, load_function in datasets.items():

        print(
            f"\n========== {dataset_name} =========="
        )


        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        X, y, age_group, sex = load_function()


        # ----------------------------------------------------
        # PROTECTED ATTRIBUTES
        # ----------------------------------------------------

        protected_attributes = {
            "Age": age_group,
            "Sex": sex
        }


        # ====================================================
        # PROTECTED ATTRIBUTE
        # ====================================================

        for protected_name, protected_data in protected_attributes.items():


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            for model_name in models:


                # ------------------------------------------------
                # RANDOM SEED
                # ------------------------------------------------

                for seed in RANDOM_SEEDS:

                    print(
                        f"Running Calibrated Equalized Odds | "
                        f"{dataset_name} | "
                        f"{protected_name} | "
                        f"{model_name} | Seed {seed}"
                    )


                    # ==========================================
                    # SAME 70/30 SPLIT AS PHASE 2
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
                    # SELECT PROTECTED ATTRIBUTE
                    # ==========================================

                    if protected_name == "Age":

                        protected_train = age_train
                        protected_test = age_test

                    else:

                        protected_train = sex_train
                        protected_test = sex_test


                    # ==========================================
                    # SPLIT TRAINING DATA
                    #
                    # 75% → model training
                    # 25% → CE-Odds calibration
                    #
                    # Original 30% test remains untouched.
                    # ==========================================

                    (
                        X_model_train,
                        X_cal,
                        y_model_train,
                        y_cal,
                        protected_model_train,
                        protected_cal
                    ) = train_test_split(

                        X_train,
                        y_train,
                        protected_train,

                        test_size=0.25,

                        random_state=seed,

                        stratify=y_train
                    )


                    # ==========================================
                    # PREPROCESSING
                    # ==========================================

                    # Create preprocessor using ONLY
                    # model-training data.

                    preprocessor = create_preprocessor(
                        X_model_train
                    )


                    # Fit preprocessing ONLY on
                    # model-training data.

                    X_model_train_processed = (
                        preprocessor.fit_transform(
                            X_model_train
                        )
                    )


                    # Transform calibration data.

                    X_cal_processed = (
                        preprocessor.transform(
                            X_cal
                        )
                    )


                    # Transform untouched test data.

                    X_test_processed = (
                        preprocessor.transform(
                            X_test
                        )
                    )


                    # ==========================================
                    # CREATE BASE CLASSIFIER
                    # ==========================================

                    base_model = create_classifier(
                        model_name,
                        seed
                    )


                    # ==========================================
                    # CALIBRATE BASE MODEL
                    #
                    # 5-fold calibration is performed ONLY
                    # within model-training data.
                    # ==========================================

                    calibrated_model = CalibratedClassifierCV(

                        estimator=base_model,

                        method="sigmoid",

                        cv=5
                    )


                    # ==========================================
                    # TRAIN + CALIBRATE
                    # ==========================================

                    calibrated_model.fit(
                        X_model_train_processed,
                        y_model_train
                    )


                    # ==========================================
                    # GET CALIBRATED PROBABILITY SCORES
                    # ==========================================

                    cal_prob = (
                        calibrated_model.predict_proba(
                            X_cal_processed
                        )[:, 1]
                    )

                    test_prob = (
                        calibrated_model.predict_proba(
                            X_test_processed
                        )[:, 1]
                    )


                    # ==========================================
                    # CONVERT PROTECTED ATTRIBUTES
                    # ==========================================

                    protected_cal_binary = convert_to_binary(
                        protected_cal
                    )

                    protected_test_binary = convert_to_binary(
                        protected_test
                    )


                    # ==========================================
                    # APPLY CE-ODDS
                    # ==========================================

                    y_pred = apply_calibrated_equalized_odds(

                        y_cal,

                        cal_prob,

                        protected_cal_binary,

                        test_prob,

                        protected_test_binary,

                        seed
                    )


                    # ==========================================
                    # PERFORMANCE
                    # ==========================================

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )


                    f1 = f1_score(
                        y_test,
                        y_pred
                    )


                    # ------------------------------------------------
                    # ROC-AUC
                    #
                    # CE-Odds produces final binary predictions.
                    #
                    # We therefore do NOT report the base model's
                    # AUC as the mitigated model's AUC.
                    # ------------------------------------------------

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


                    # ==========================================
                    # SAVE RESULT
                    # ==========================================

                    results.append({

                        "dataset":
                            dataset_name,

                        "mitigation":
                            "Calibrated Equalized Odds",

                        "protected_attribute":
                            protected_name,

                        "model":
                            model_name,

                        "seed":
                            seed,


                        # Predictive performance

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


    print(
        "\n\nCalibrated Equalized Odds Results:"
    )

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

        # ---------------------------------------------
        # PERFORMANCE
        # ---------------------------------------------

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        auc_mean=(
            "roc_auc",
            "mean"
        ),

        auc_std=(
            "roc_auc",
            "std"
        ),


        # ---------------------------------------------
        # AGE FAIRNESS
        # ---------------------------------------------

        age_dpd_mean=(
            "age_dpd",
            "mean"
        ),

        age_dpd_std=(
            "age_dpd",
            "std"
        ),

        age_eod_mean=(
            "age_eod",
            "mean"
        ),

        age_eod_std=(
            "age_eod",
            "std"
        ),

        age_eopd_mean=(
            "age_eopd",
            "mean"
        ),

        age_eopd_std=(
            "age_eopd",
            "std"
        ),

        age_di_mean=(
            "age_di_ratio",
            "mean"
        ),

        age_di_std=(
            "age_di_ratio",
            "std"
        ),


        # ---------------------------------------------
        # SEX FAIRNESS
        # ---------------------------------------------

        sex_dpd_mean=(
            "sex_dpd",
            "mean"
        ),

        sex_dpd_std=(
            "sex_dpd",
            "std"
        ),

        sex_eod_mean=(
            "sex_eod",
            "mean"
        ),

        sex_eod_std=(
            "sex_eod",
            "std"
        ),

        sex_eopd_mean=(
            "sex_eopd",
            "mean"
        ),

        sex_eopd_std=(
            "sex_eopd",
            "std"
        ),

        sex_di_mean=(
            "sex_di_ratio",
            "mean"
        ),

        sex_di_std=(
            "sex_di_ratio",
            "std"
        ),


        # ---------------------------------------------
        # INTERSECTIONAL FAIRNESS
        # ---------------------------------------------

        intersection_dpd_mean=(
            "intersection_dpd",
            "mean"
        ),

        intersection_dpd_std=(
            "intersection_dpd",
            "std"
        ),

        intersection_eod_mean=(
            "intersection_eod",
            "mean"
        ),

        intersection_eod_std=(
            "intersection_eod",
            "std"
        ),

        intersection_eopd_mean=(
            "intersection_eopd",
            "mean"
        ),

        intersection_eopd_std=(
            "intersection_eopd",
            "std"
        ),

        intersection_di_mean=(
            "intersection_di_ratio",
            "mean"
        ),

        intersection_di_std=(
            "intersection_di_ratio",
            "std"
        )
    )


    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print(
        "\n\nCalibrated Equalized Odds "
        "Mean ± Standard Deviation:"
    )

    print(summary)


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_folder = (
        Path("outputs") / "tables"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Individual results
    # --------------------------------------------------------

    results_df.to_csv(

        output_folder /
        "phase3_calibrated_equalized_odds_results.csv",

        index=False
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary.to_csv(

        output_folder /
        "phase3_calibrated_equalized_odds_summary.csv"
    )


    print(
        "\nCalibrated Equalized Odds results saved to:"
    )

    print(
        output_folder /
        "phase3_calibrated_equalized_odds_results.csv"
    )


    print(
        "\nCalibrated Equalized Odds summary saved to:"
    )

    print(
        output_folder /
        "phase3_calibrated_equalized_odds_summary.csv"
    )