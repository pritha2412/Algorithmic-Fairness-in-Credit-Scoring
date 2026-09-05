import pandas as pd
from pathlib import Path
import numpy as np

# Scikit-learn tools for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Used to replace missing values during preprocessing
from sklearn.impute import SimpleImputer

# Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Metrics for evaluating predictions
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# XGBoost model
from xgboost import XGBClassifier

# Fairness metrics
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
    demographic_parity_ratio
)

# -----------------------------
# Dataset paths
# -----------------------------

GERMAN_PATH = Path("data") / "german_credit_data_risk.csv"
ADULT_PATH = Path("data") / "adult.data"


# -----------------------------
# Load German Credit
# -----------------------------

def load_german():
    df = pd.read_csv(GERMAN_PATH)

    # Protected attributes
    age_group = df["Age"].apply(
        lambda age: "under_25" if age < 25 else "25_and_above"
    )

    sex = df["Sex"]

    # Target
    y = (df["Risk"] == "good").astype(int)

    # Remove protected attributes and target from model features
    X = df.drop(columns=["Risk", "Age", "Sex", "Unnamed: 0"])
    return X, y, age_group, sex


# -----------------------------
# Load Adult Income
# -----------------------------

def load_adult():
    columns = [
        "age", "workclass", "fnlwgt", "education",
        "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain",
        "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    df = pd.read_csv(
        ADULT_PATH,
        names=columns,
        skipinitialspace=True
    )
    # Convert '?' into proper missing values (NaN)
    df = df.replace("?", np.nan)

    # Protected attributes
    age_group = df["age"].apply(
        lambda age: "under_25" if age < 25 else "25_and_above"
    )

    sex = df["sex"]

    # Target
    y = (df["income"] == ">50K").astype(int)

    # Remove protected attributes and target
    X = df.drop(columns=["income", "age", "sex"])

    return X, y, age_group, sex


# ---------------------------------
# Create preprocessing pipeline
# ---------------------------------

def create_preprocessor(X):

    # Find all numerical columns
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Find all categorical/text columns
    # Anything that is not numerical is treated as categorical
    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Scale numerical features
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Handle missing categorical values, then convert categories to numbers
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    # Apply the correct preprocessing to each type of feature
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# ---------------------------------
# Random seeds for repeated testing
# ---------------------------------

RANDOM_SEEDS = [42, 123, 456, 789, 1000]


# ---------------------------------
# Create one train/test split
# ---------------------------------

def create_split(X, y, age_group, sex, seed):

    # Split X, y, and protected attributes together
    # This keeps every person's information aligned
    (
        X_train,
        X_test,
        y_train,
        y_test,
        age_train,
        age_test,
        sex_train,
        sex_test
    ) = train_test_split(
        X,
        y,
        age_group,
        sex,
        test_size=0.30,
        random_state=seed,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        age_train,
        age_test,
        sex_train,
        sex_test
    )

# ---------------------------------
# Create Logistic Regression model
# ---------------------------------

def create_logistic_model(preprocessor):

    # Pipeline:
    # 1. Preprocess the data
    # 2. Train Logistic Regression
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])

    return model


# ---------------------------------
# Create Random Forest model
# ---------------------------------

def create_random_forest_model(preprocessor):

    # Pipeline:
    # 1. Preprocess the data
    # 2. Train Random Forest
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    return model


# ---------------------------------
# Create XGBoost model
# ---------------------------------

def create_xgboost_model(preprocessor):

    # Pipeline:
    # 1. Preprocess the data
    # 2. Train XGBoost
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss"
        ))
    ])

    return model

# ---------------------------------
# Evaluate one model on one split
# ---------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):

    # Train the model using only the training data
    model.fit(X_train, y_train)

    # Predict the class (0 or 1)
    y_pred = model.predict(X_test)

    # Get probability of the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate predictive metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return accuracy, f1, auc, y_pred

# ---------------------------------
# Calculate fairness metrics
# ---------------------------------

def evaluate_fairness(y_test, y_pred, protected_attribute):

    # Demographic Parity Difference
    # Measures difference in positive prediction rates
    dpd = demographic_parity_difference(
        y_test,
        y_pred,
        sensitive_features=protected_attribute
    )

    # Equalized Odds Difference
    # Considers both TPR and FPR differences
    eod = equalized_odds_difference(
        y_test,
        y_pred,
        sensitive_features=protected_attribute
    )

    # Equal Opportunity Difference
    # Considers only the TPR difference
    eopd = equal_opportunity_difference(
        y_test,
        y_pred,
        sensitive_features=protected_attribute
    )

    # Disparate Impact Ratio
    # Compares the lowest positive prediction rate
    # with the highest positive prediction rate
    di_ratio = demographic_parity_ratio(
        y_test,
        y_pred,
        sensitive_features=protected_attribute
    )

    return dpd, eod, eopd, di_ratio

# ---------------------------------
# Create intersectional Age × Sex groups
# ---------------------------------

def create_intersectional_groups(age_group, sex):

    # Combine age group and sex into one group label
    intersection = age_group.astype(str) + "_" + sex.astype(str)

    return intersection

# ---------------------------------
# Create a model based on its name
# ---------------------------------

def create_model(model_name, preprocessor):

    # Logistic Regression
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=1000
        )

    # Random Forest
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # XGBoost
    elif model_name == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss"
        )

    # Prevent accidentally using an unknown model name
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Put preprocessing + classifier into one pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return model

# Models used in the baseline experiment
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]



# ---------------------------------
# Run baseline experiments
# ---------------------------------

if __name__ == "__main__":

    # Load both datasets
    datasets = {
        "German Credit": load_german,
        "Adult Income": load_adult
    }

    # Store every experiment result
    results = []

    # Loop through each dataset
    for dataset_name, load_function in datasets.items():

        print(f"\n========== {dataset_name} ==========")

        # Load the current dataset
        X, y, age_group, sex = load_function()

        # Try every model
        for model_name in MODEL_NAMES:

            # Try every random seed
            for seed in RANDOM_SEEDS:

                print(f"Running {model_name} | Seed {seed}")

                # Create a train/test split
                # Protected attributes are split at the same time
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

                # Create preprocessing using only training data
                preprocessor = create_preprocessor(X_train)

                # Create the selected model
                model = create_model(
                    model_name,
                    preprocessor
                )

                # Train model and calculate predictive metrics
                accuracy, f1, auc, y_pred = evaluate_model(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )

                # Calculate fairness for Age
                age_dpd, age_eod, age_eopd, age_di = evaluate_fairness(
                    y_test,
                    y_pred,
                    age_test
                )

                # Calculate fairness for Sex
                sex_dpd, sex_eod, sex_eopd, sex_di = evaluate_fairness(
                    y_test,
                    y_pred,
                    sex_test
                )
                # Create Age × Sex intersectional groups
                intersection_test = create_intersectional_groups(
                    age_test,
                    sex_test
                )

                # Calculate fairness across the four intersectional groups
                intersection_dpd, intersection_eod, intersection_eopd, intersection_di = evaluate_fairness(
                    y_test,
                    y_pred,
                    intersection_test
                )

                # Store all results from this experiment
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,

                    # Predictive performance
                    "accuracy": accuracy,
                    "f1": f1,
                    "roc_auc": auc,

                    # Age fairness
                    "age_dpd": age_dpd,
                    "age_eod": age_eod,
                    "age_eopd": age_eopd,
                    "age_di_ratio": age_di,

                    # Sex fairness
                    "sex_dpd": sex_dpd,
                    "sex_eod": sex_eod,
                    "sex_eopd": sex_eopd,
                    "sex_di_ratio": sex_di,

                    # Intersectional Age × Sex fairness
                    "intersection_dpd": intersection_dpd,
                    "intersection_eod": intersection_eod,
                    "intersection_eopd": intersection_eopd,
                    "intersection_di_ratio": intersection_di
                })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    # Display the complete results
    print("\nRaw Results:")
    print(results_df)

    # Calculate mean and standard deviation
    summary = results_df.groupby(
        ["dataset", "model"]
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

    print("\nMean ± Standard Deviation:")
    print(summary)

    # ---------------------------------
    # Save baseline results
    # ---------------------------------

    # Create the output folder if it doesn't already exist
    output_folder = Path("outputs") / "tables"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save all 30 individual experiment results
    results_df.to_csv(
        output_folder / "phase2_baseline_results.csv",
        index=False
    )

    print("\nBaseline results saved to:")
    print(output_folder / "phase2_baseline_results.csv")

    # Save mean ± standard deviation summary
    summary.to_csv(
        output_folder / "phase2_baseline_summary.csv"
    )

    print("Baseline summary saved to:")
    print(output_folder / "phase2_baseline_summary.csv")



