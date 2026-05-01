# Classification model training and cross-validation.

import numpy as np
import joblib
import os
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost                 import XGBClassifier


def get_classifiers(scale_pos_weight: float = 2.7) -> dict:
    """
    Return a dictionary of configured classification models.

    Parameters
    ----------
    scale_pos_weight : float — neg/pos ratio for XGBoost imbalance handling

    Returns
    -------
    dict : {model_name: model_object}
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc', random_state=42, verbosity=0
        )
    }


def train_classifiers(X_train, y_train,
                       cv_folds: int = 5,
                       scoring: str = 'roc_auc') -> dict:
    """
    Train all classifiers with cross-validation and fit on full training set.

    Parameters
    ----------
    X_train  : array-like — training features
    y_train  : array-like — training labels
    cv_folds : int        — number of CV folds (default 5)
    scoring  : str        — sklearn scoring metric

    Returns
    -------
    dict : {model_name: {'model': fitted_model, 'cv_scores': array}}
    """
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    spw   = neg / pos

    models   = get_classifiers(scale_pos_weight=spw)
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results  = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring=scoring)
        model.fit(X_train, y_train)

        results[name] = {'model': model, 'cv_scores': cv_scores}
        print(f"{name:<25} CV {scoring}: "
              f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


def get_best_classifier(results: dict,
                         metric: str = 'recall',
                         X_test=None,
                         y_test=None):
    """
    Select the best classifier.

    If metric='cv_auc'  : select by cross-validation AUC mean.
    If metric='recall'  : select by test set recall (requires X_test, y_test).

    Parameters
    ----------
    results : dict   — output from train_classifiers()
    metric  : str    — 'cv_auc' or 'recall'
    X_test  : array  — test features (required if metric='recall')
    y_test  : array  — test labels   (required if metric='recall')

    Returns
    -------
    tuple : (best_name, best_model)
    """
    from sklearn.metrics import recall_score

    if metric == 'cv_auc':
        best_name = max(results, key=lambda n: results[n]['cv_scores'].mean())
    elif metric == 'recall':
        best_name = max(
            results,
            key=lambda n: recall_score(
                y_test, results[n]['model'].predict(X_test)
            )
        )
    else:
        raise ValueError(f"metric must be 'cv_auc' or 'recall', got '{metric}'")

    best_model = results[best_name]['model']
    print(f"\nBest classifier (by {metric}): {best_name}")
    return best_name, best_model