# Functions for data cleaning, encoding, and feature scaling.

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning operations to the raw Telco churn dataset.

    Steps:
    1. Fix TotalCharges dtype (object → float)
    2. Drop rows with missing TotalCharges (tenure=0, new customers)
    3. Drop customerID column
    4. Encode binary columns (Yes/No → 1/0, gender → 1/0)

    Parameters
    ----------
    df : pd.DataFrame — raw dataset

    Returns
    -------
    pd.DataFrame — cleaned dataset
    """
    df = df.copy()

    # Fix TotalCharges dtype
    df['TotalCharges'] = df['TotalCharges'].str.strip().replace('', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing rows
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

    # Drop ID column
    df = df.drop(columns=['customerID'], errors='ignore')

    # Encode gender
    df['gender'] = (df['gender'].str.strip().str.lower()
                    .map({'male': 1, 'female': 0}))

    # Encode all Yes/No binary columns
    yes_no_cols = [c for c in df.select_dtypes('object').columns
                   if set(df[c].dropna().unique()).issubset({'Yes', 'No'})]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode target
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"Cleaning complete: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


def encode_features(df: pd.DataFrame, save_path: str = None):
    """
    One-hot encode multi-class categorical columns.

    Parameters
    ----------
    df        : pd.DataFrame — cleaned dataset
    save_path : str or None  — if provided, saves encoder column list as .pkl

    Returns
    -------
    pd.DataFrame — fully encoded dataset (all numeric)
    """
    text_cols  = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=text_cols,
                                drop_first=True, dtype=int)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(text_cols, save_path)
        print(f"Encoder column list saved to: {save_path}")

    print(f"Encoding complete: {len(text_cols)} columns encoded")
    print(f"Final shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")
    return df_encoded


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   cols: list, save_path: str = None):
    """
    Fit StandardScaler on X_train and transform both X_train and X_test.
    Scaler is NEVER fit on X_test to prevent data leakage.

    Parameters
    ----------
    X_train   : pd.DataFrame — training features
    X_test    : pd.DataFrame — test features
    cols      : list         — column names to scale
    save_path : str or None  — if provided, saves scaler as .pkl

    Returns
    -------
    tuple : (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()

    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()

    X_train_s[cols] = scaler.fit_transform(X_train[cols])
    X_test_s[cols]  = scaler.transform(X_test[cols])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to: {save_path}")

    return X_train_s, X_test_s, scaler


def split_data(df_encoded: pd.DataFrame, target: str = 'Churn',
               test_size: float = 0.2, random_state: int = 42):
    """
    Separate features and target, then perform stratified train/test split.

    Parameters
    ----------
    df_encoded   : pd.DataFrame — fully encoded dataset
    target       : str          — name of target column
    test_size    : float        — fraction for test set (default 0.2)
    random_state : int          — reproducibility seed

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    print(f"Split complete:")
    print(f"  Train: {X_train.shape[0]:,} rows | churn: {y_train.mean()*100:.2f}%")
    print(f"  Test : {X_test.shape[0]:,} rows  | churn: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test