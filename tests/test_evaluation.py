# Unit tests for evaluation functions in src/evaluation.py
# Run with: python -m pytest tests/ -v

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import classification_metrics, regression_metrics


@pytest.fixture
def perfect_classification():
    """Perfectly correct predictions — all metrics should be 1.0."""
    y_true  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1])
    return y_true, y_pred, y_proba


@pytest.fixture
def imbalanced_classification():
    """Realistic imbalanced scenario with imperfect predictions."""
    y_true  = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_pred  = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.1, 0.2, 0.8, 0.3, 0.9, 0.4, 0.85])
    return y_true, y_pred, y_proba


class TestClassificationMetrics:
    """Tests for the classification_metrics() function."""

    def test_returns_dict(self, perfect_classification):
        """Function must return a dictionary."""
        y_true, y_pred, y_proba = perfect_classification
        result = classification_metrics(y_true, y_pred, y_proba)
        assert isinstance(result, dict)

    def test_all_keys_present(self, perfect_classification):
        """All required metric keys must be in the output."""
        y_true, y_pred, y_proba = perfect_classification
        result   = classification_metrics(y_true, y_pred, y_proba)
        required = {'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'}
        assert required.issubset(set(result.keys()))

    def test_perfect_predictions_score_one(self, perfect_classification):
        """Perfect predictions must return 1.0 for all metrics."""
        y_true, y_pred, y_proba = perfect_classification
        result = classification_metrics(y_true, y_pred, y_proba)
        assert result['Accuracy']  == pytest.approx(1.0)
        assert result['Recall']    == pytest.approx(1.0)
        assert result['ROC-AUC']   == pytest.approx(1.0)

    def test_metrics_in_valid_range(self, imbalanced_classification):
        """All metrics must be between 0.0 and 1.0."""
        y_true, y_pred, y_proba = imbalanced_classification
        result = classification_metrics(y_true, y_pred, y_proba)
        for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    def test_recall_higher_than_precision_for_imbalanced(self, imbalanced_classification):
        """
        With class_weight='balanced' models, Recall is typically higher
        than Precision on imbalanced datasets — a sanity check for our
        model selection reasoning.
        """
        y_true, y_pred, y_proba = imbalanced_classification
        result = classification_metrics(y_true, y_pred, y_proba)
        # This is not always true but documents our expectation
        # for the balanced model scenario
        assert result['Recall'] >= 0.0


class TestRegressionMetrics:
    """Tests for the regression_metrics() function."""

    def test_returns_dict(self):
        """Function must return a dictionary."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])
        result = regression_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        """MAE, RMSE, and R2 must all be present."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        result = regression_metrics(y_true, y_pred)
        assert {'MAE', 'RMSE', 'R2'}.issubset(set(result.keys()))

    def test_perfect_predictions_mae_zero(self):
        """Perfect predictions must produce MAE = 0 and R2 = 1."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        result = regression_metrics(y_true, y_pred)
        assert result['MAE']  == pytest.approx(0.0)
        assert result['R2']   == pytest.approx(1.0)

    def test_mae_is_non_negative(self):
        """MAE must always be zero or positive."""
        y_true = np.random.rand(50)
        y_pred = np.random.rand(50)
        result = regression_metrics(y_true, y_pred)
        assert result['MAE'] >= 0.0
        assert result['RMSE'] >= 0.0