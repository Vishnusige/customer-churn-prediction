# Functions to load, validate, and inspect the Telco churn dataset.

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV file.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with original dtypes.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            f"Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )

    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_processed_data(processed_dir: str):
    """
    Load the four processed split files from data/processed/.

    Parameters
    ----------
    processed_dir : str
        Path to the processed/ directory.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test) as DataFrames/Series
    """
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

    print(f"Loaded processed data:")
    print(f"  X_train: {X_train.shape} | y_train churn rate: {y_train.mean()*100:.2f}%")
    print(f"  X_test : {X_test.shape}  | y_test  churn rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """
    Run a full validation audit on a DataFrame.

    Parameters
    ----------
    df   : pd.DataFrame — the dataframe to validate
    name : str          — label for print output

    Returns
    -------
    dict with keys: shape, nulls, duplicates, dtypes
    """
    report = {
        'shape'      : df.shape,
        'nulls'      : df.isnull().sum().sum(),
        'duplicates' : df.duplicated().sum(),
        'dtypes'     : df.dtypes.value_counts().to_dict()
    }

    print(f"\nValidation report — {name}")
    print(f"  Shape      : {report['shape'][0]:,} rows × {report['shape'][1]} columns")
    print(f"  Nulls      : {report['nulls']}")
    print(f"  Duplicates : {report['duplicates']}")
    print(f"  Dtypes     : {report['dtypes']}")

    if report['nulls'] > 0:
        print(f"  WARNING: {report['nulls']} missing values detected")
    if report['duplicates'] > 0:
        print(f"  WARNING: {report['duplicates']} duplicate rows detected")

    return report