import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data") / "adult.data"


def load_adult_data():
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
    ]

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=column_names,
        skipinitialspace=True
    )

    return df


def basic_eda(df):
    print("\n" + "=" * 50)
    print("DATASET SHAPE")
    print("=" * 50)
    print(df.shape)

    print("\n" + "=" * 50)
    print("DATA TYPES")
    print("=" * 50)
    print(df.dtypes)

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(df.describe())

    print("\n" + "=" * 50)
    print("INCOME DISTRIBUTION")
    print("=" * 50)
    print(df["income"].value_counts())

    print("\n" + "=" * 50)
    print("SEX DISTRIBUTION")
    print("=" * 50)
    print(df["sex"].value_counts())

    print("\n" + "=" * 50)
    print("AGE STATISTICS")
    print("=" * 50)
    print(df["age"].describe())

    print("\n" + "=" * 50)
    print("MISSING VALUES REPRESENTED BY '?'")
    print("=" * 50)

    for column in df.columns:
        missing = (df[column] == "?").sum()

        if missing > 0:
            print(column, ":", missing)

    print("\n" + "=" * 50)
    print("SEX VALUES")
    print("=" * 50)
    print(df["sex"].value_counts())

    print("\n" + "=" * 50)
    print("RACE VALUES")
    print("=" * 50)
    print(df["race"].value_counts())

def group_income_analysis(df):
    df["age_group"] = df["age"].apply(
        lambda age: "under_25" if age < 25 else "25_and_above"
    )

    print("\n" + "=" * 50)
    print("AGE GROUP DISTRIBUTION")
    print("=" * 50)
    print(df["age_group"].value_counts())

    print("\n" + "=" * 50)
    print("INCOME BY AGE GROUP")
    print("=" * 50)
    print(
        pd.crosstab(
            df["age_group"],
            df["income"],
            normalize="index"
        ) * 100
    )

    print("\n" + "=" * 50)
    print("INCOME BY SEX")
    print("=" * 50)
    print(
        pd.crosstab(
            df["sex"],
            df["income"],
            normalize="index"
        ) * 100
    )

    print("\n" + "=" * 50)
    print("INCOME BY AGE × SEX")
    print("=" * 50)
    print(
        pd.crosstab(
            [df["sex"], df["age_group"]],
            df["income"],
            normalize="index"
        ) * 100
    )

def plot_group_income(df):

    # Income by age group
    age_income = pd.crosstab(
        df["age_group"],
        df["income"],
        normalize="index"
    ) * 100

    age_income.plot(kind="bar", figsize=(8, 5))

    plt.title("Income Distribution by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Income")
    plt.tight_layout()

    plt.savefig("outputs/figures/income_by_age_group_adult.png")
    plt.show()

    # Income by sex
    sex_income = pd.crosstab(
        df["sex"],
        df["income"],
        normalize="index"
    ) * 100

    sex_income.plot(kind="bar", figsize=(8, 5))

    plt.title("Income Distribution by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Income")
    plt.tight_layout()

    plt.savefig("outputs/figures/income_by_sex_adult.png")
    plt.show()

def plot_correlation_heatmap(df):

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Adult Dataset Correlation Heatmap")
    plt.tight_layout()

    plt.savefig("outputs/figures/adult_correlation_heatmap.png")
    plt.show()



if __name__ == "__main__":
    df = load_adult_data()

    basic_eda(df)
    group_income_analysis(df)
    plot_group_income(df)
    plot_correlation_heatmap(df)