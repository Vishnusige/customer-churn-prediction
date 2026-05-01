# Unit tests for preprocessing functions in src/preprocessing.py
# Run with: python -m pytest tests/ -v

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_data, encode_features, split_data


@pytest.fixture
def sample_raw_df():
    """
    Create a small sample dataframe that mimics the raw Telco dataset.
    This lets us test cleaning logic without needing the full 7,032 row file.
    """
    return pd.DataFrame({
        'customerID'     : ['A001', 'A002', 'A003', 'A004', 'A005'],
        'gender'         : ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen'  : [0, 0, 1, 0, 0],
        'Partner'        : ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents'     : ['No', 'No', 'Yes', 'No', 'No'],
        'tenure'         : [1, 34, 2, 45, 0],
        'PhoneService'   : ['No', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines'  : ['No phone service', 'No', 'No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'DSL', 'Fiber optic', 'DSL', 'No'],
        'OnlineSecurity' : ['No', 'Yes', 'No', 'Yes', 'No internet service'],
        'OnlineBackup'   : ['Yes', 'No', 'No', 'No', 'No internet service'],
        'DeviceProtection':['No', 'Yes', 'No', 'Yes', 'No internet service'],
        'TechSupport'    : ['No', 'No', 'No', 'Yes', 'No internet service'],
        'StreamingTV'    : ['No', 'No', 'No', 'No', 'No internet service'],
        'StreamingMovies': ['No', 'No', 'No', 'No', 'No internet service'],
        'Contract'       : ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling':['Yes', 'No', 'Yes', 'No', 'No'],
        'PaymentMethod'  : ['Electronic check', 'Mailed check',
                            'Electronic check', 'Bank transfer (automatic)',
                            'Credit card (automatic)'],
        'MonthlyCharges' : [29.85, 56.95, 53.85, 42.30, 70.70],
        'TotalCharges'   : ['29.85', '1889.50', ' ', '1840.75', '0.00'],
        'Churn'          : ['No', 'No', 'Yes', 'No', 'No']
    })


class TestCleanData:
    """Tests for the clean_data() function."""

    def test_total_charges_converted_to_float(self, sample_raw_df):
        """TotalCharges must be float64 after cleaning."""
        cleaned = clean_data(sample_raw_df)
        assert cleaned['TotalCharges'].dtype == np.float64

    def test_blank_total_charges_rows_dropped(self, sample_raw_df):
        """Rows with blank TotalCharges (new customers) must be removed."""
        cleaned = clean_data(sample_raw_df)
        # Row A003 has ' ' in TotalCharges — should be dropped
        assert cleaned.isnull().sum().sum() == 0

    def test_customer_id_dropped(self, sample_raw_df):
        """customerID column must not exist after cleaning."""
        cleaned = clean_data(sample_raw_df)
        assert 'customerID' not in cleaned.columns

    def test_gender_encoded_as_binary(self, sample_raw_df):
        """gender column must contain only 0 and 1 after cleaning."""
        cleaned = clean_data(sample_raw_df)
        assert set(cleaned['gender'].unique()).issubset({0, 1})

    def test_churn_encoded_as_binary(self, sample_raw_df):
        """Churn column must contain only 0 and 1 after cleaning."""
        cleaned = clean_data(sample_raw_df)
        assert set(cleaned['Churn'].unique()).issubset({0, 1})

    def test_no_missing_values_after_cleaning(self, sample_raw_df):
        """Cleaned dataframe must have zero missing values."""
        cleaned = clean_data(sample_raw_df)
        assert cleaned.isnull().sum().sum() == 0


class TestEncodeFeatures:
    """Tests for the encode_features() function."""

    def test_no_text_columns_after_encoding(self, sample_raw_df):
        """No object dtype columns should remain after one-hot encoding."""
        cleaned = clean_data(sample_raw_df)
        encoded = encode_features(cleaned)
        text_cols = encoded.select_dtypes(include='object').columns.tolist()
        assert len(text_cols) == 0

    def test_output_is_dataframe(self, sample_raw_df):
        """encode_features must return a DataFrame."""
        cleaned = clean_data(sample_raw_df)
        encoded = encode_features(cleaned)
        assert isinstance(encoded, pd.DataFrame)

    def test_row_count_unchanged(self, sample_raw_df):
        """Encoding must not add or remove rows."""
        cleaned = clean_data(sample_raw_df)
        encoded = encode_features(cleaned)
        assert len(encoded) == len(cleaned)


class TestSplitData:
    """Tests for the split_data() function."""

    def test_correct_split_proportions(self, sample_raw_df):
        """Test set should be 20% of total rows."""
        cleaned = clean_data(sample_raw_df)
        encoded = encode_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(encoded)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.15  # tolerance for small sample

    def test_no_overlap_between_splits(self, sample_raw_df):
        """Training and test sets must have no overlapping indices."""
        cleaned = clean_data(sample_raw_df)
        encoded = encode_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(encoded)
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        assert len(train_idx.intersection(test_idx)) == 0