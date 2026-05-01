# src/__init__.py
# Makes the src directory a Python package.
# Import key functions for easy access.

from .data_loader    import load_raw_data, load_processed_data, validate_dataframe
from .preprocessing  import clean_data, encode_features, scale_features
from .classifiers    import train_classifiers, get_best_classifier
from .regressors     import train_regressors, get_best_regressor
from .evaluation     import classification_metrics, regression_metrics, plot_roc_curves
from .utils          import set_random_seed, save_model, load_model, timer