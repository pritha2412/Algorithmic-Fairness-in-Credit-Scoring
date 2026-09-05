import pandas as pd
from pathlib import Path

DATA_PATH = Path("data") / "german_credit_data_risk.csv"

def load_data():
    """
    Load the German Credit dataset.
    """
    df = pd.read_csv(DATA_PATH)
    return df
