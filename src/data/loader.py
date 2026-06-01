"""
loader.py

Handles loading Home Credit dataset files.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def load_train_data():
    """
    Load application_train.csv
    """
    file_path = DATA_DIR / "application_train.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded training data: {df.shape}")

    return df


def load_test_data():
    """
    Load application_test.csv
    """
    file_path = DATA_DIR / "application_test.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded test data: {df.shape}")

    return df


def get_target(df):
    """
    Extract target column.
    """
    return df["TARGET"]


if __name__ == "__main__":

    train_df = load_train_data()

    print(train_df.head())