import os
import pandas as pd
from typing import Union


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet based on file extension.

    :param path: Path to the CSV or Parquet file.
    :return: pandas DataFrame.
    :raises ValueError: if file extension is unsupported.
    """
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.lower().endswith(('.csv', '.txt')):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format for loading: {path}")


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame as a Parquet file, ensuring the directory exists.

    :param df: DataFrame to save.
    :param path: Destination path for the Parquet file.
    """
    # Ensure parent directory exists
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)

    # Write to Parquet
    df.to_parquet(path, index=False)
