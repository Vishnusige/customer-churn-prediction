# Shared helper functions used across notebooks and src modules.

import os
import time
import joblib
import numpy as np
import random


def set_random_seed(seed: int = 42):
    """
    Set random seed for full reproducibility across numpy and random.

    Parameters
    ----------
    seed : int — seed value (default 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def save_model(model, filepath: str):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model    : fitted sklearn/xgboost model
    filepath : str — full path including filename and .pkl extension
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"Model saved: {filepath}  ({size_kb:.0f} KB)")


def load_model(filepath: str):
    """
    Load a saved model from disk.

    Parameters
    ----------
    filepath : str — path to the .pkl file

    Returns
    -------
    Fitted model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded: {filepath}")
    return model


def timer(func):
    """
    Decorator that prints the execution time of a function.

    Usage
    -----
    @timer
    def my_function():
        ...
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        end    = time.time()
        print(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result
    return wrapper


def assign_risk_tier(prob: float) -> str:
    """
    Assign a customer to a churn risk tier based on predicted probability.

    Parameters
    ----------
    prob : float — predicted churn probability (0.0 to 1.0)

    Returns
    -------
    str — 'High', 'Medium', or 'Low'
    """
    if prob >= 0.70:
        return 'High'
    elif prob >= 0.40:
        return 'Medium'
    else:
        return 'Low'


def format_currency(value: float) -> str:
    """Return a formatted currency string: $1,234.56"""
    return f"${value:,.2f}"