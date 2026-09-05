from src.load_data import load_data
from src.phase1a_eda_german import (
    basic_info,
    group_risk_analysis,
    plot_group_risk,
    plot_correlation_heatmap
)


def main():
    df = load_data()

    basic_info(df)
    group_risk_analysis(df)
    plot_group_risk(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()
