# Revenue forecasting modelled as a regression problem.
# Target: MonthlyCharges weighted by churn probability (revenue at risk).

import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import mean_absolute_error, r2_score, mean_squared_error


def get_regressors() -> dict:
    """
    Return a dictionary of configured regression models.

    Returns
    -------
    dict : {model_name: model_object}
    """
    return {
        'Linear Regression'        : LinearRegression(),
        'Ridge Regression'         : Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor'  : RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
    }


def build_regression_target(df_test: pd.DataFrame,
                              churn_proba: np.ndarray) -> pd.Series:
    """
    Build the regression target: revenue at risk per customer.

    Formula: revenue_at_risk = MonthlyCharges × churn_probability

    Parameters
    ----------
    df_test     : pd.DataFrame — original test set with MonthlyCharges
    churn_proba : np.ndarray   — predicted churn probabilities

    Returns
    -------
    pd.Series — revenue_at_risk per customer
    """
    return pd.Series(
        df_test['MonthlyCharges'].values * churn_proba,
        name='revenue_at_risk'
    )


def train_regressors(X_train, y_reg_train,
                     cv_folds: int = 5) -> dict:
    """
    Train all regressors with cross-validation on revenue-at-risk target.

    Parameters
    ----------
    X_train      : array-like — training features
    y_reg_train  : array-like — revenue at risk values (regression target)
    cv_folds     : int        — number of CV folds

    Returns
    -------
    dict : {model_name: {'model': fitted_model, 'cv_r2': array}}
    """
    models  = get_regressors()
    cv      = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        cv_r2 = cross_val_score(model, X_train, y_reg_train,
                                cv=cv, scoring='r2')
        model.fit(X_train, y_reg_train)

        results[name] = {'model': model, 'cv_r2': cv_r2}
        print(f"{name:<30} CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    return results


def get_best_regressor(results: dict) -> tuple:
    """
    Select the best regressor by highest mean cross-validation R² score.

    Parameters
    ----------
    results : dict — output from train_regressors()

    Returns
    -------
    tuple : (best_name, best_model)
    """
    best_name  = max(results, key=lambda n: results[n]['cv_r2'].mean())
    best_model = results[best_name]['model']
    print(f"\nBest regressor (by CV R²): {best_name}")
    return best_name, best_model


def evaluate_regressor(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted regressor on the test set.

    Parameters
    ----------
    model  : fitted regressor
    X_test : array-like — test features
    y_test : array-like — true revenue at risk values

    Returns
    -------
    dict : {mae, rmse, r2}
    """
    y_pred = model.predict(X_test)
    return {
        'MAE'  : mean_absolute_error(y_test, y_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2'   : r2_score(y_test, y_pred)
    }