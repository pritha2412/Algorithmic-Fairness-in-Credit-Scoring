import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

def basic_info(df):
    """
    Display basic information about the dataset.
    """

    print("=" * 50)
    print("FIRST FIVE ROWS")
    print("=" * 50)
    print(df.head())

    print("\n")

    print("=" * 50)
    print("DATASET SHAPE")
    print("=" * 50)
    print(df.shape)

    print("\n")

    print("=" * 50)
    print("COLUMN NAMES")
    print("=" * 50)
    print(df.columns)

    print("\n")

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    df.info()

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(df.describe())

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("TARGET DISTRIBUTION")
    print("=" * 50)
    print(df["Risk"].value_counts())

    print("\nSex distribution:")
    print(df["Sex"].value_counts())

    print("\nAge statistics:")
    print(df["Age"].describe())




def group_risk_analysis(df):
    # Create age groups
    df["Age_group"] = df["Age"].apply(
        lambda age: "under_25" if age < 25 else "25_and_above"
    )

    print("\nAge group distribution:")
    print(df["Age_group"].value_counts())

    print("\nRisk by age group:")
    print(pd.crosstab(
        df["Age_group"],
        df["Risk"],
        normalize="index"
    ) * 100)

    print("\nRisk by sex:")
    print(pd.crosstab(
        df["Sex"],
        df["Risk"],
        normalize="index"
    ) * 100)

    print("\n" + "=" * 50)
    print("RISK BY AGE × SEX")
    print("=" * 50)

    print(
        pd.crosstab(
            [df["Sex"], df["Age_group"]],
            df["Risk"],
            normalize="index"
        ) * 100
    )

def plot_group_risk(df):

    # Risk by age group
    age_risk = pd.crosstab(
        df["Age_group"],
        df["Risk"],
        normalize="index"
    ) * 100

    age_risk.plot(kind="bar", figsize=(8, 5))

    plt.title("Risk Distribution by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Risk")
    plt.tight_layout()

    plt.savefig("outputs/figures/risk_by_age_group.png")
    plt.show()

    # Risk by sex
    sex_risk = pd.crosstab(
        df["Sex"],
        df["Risk"],
        normalize="index"
    ) * 100

    sex_risk.plot(kind="bar", figsize=(8, 5))

    plt.title("Risk Distribution by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Risk")
    plt.tight_layout()

    plt.savefig("outputs/figures/risk_by_sex.png")
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

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig("outputs/figures/german_correlation_heatmap.png")
    plt.show()