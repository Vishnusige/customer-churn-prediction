# Reusable evaluation functions for classification and regression.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)


def classification_metrics(y_true, y_pred, y_proba,
                             model_name: str = '') -> dict:
    """
    Compute all classification metrics for a single model.

    Parameters
    ----------
    y_true     : array-like — true labels
    y_pred     : array-like — hard predictions (0/1)
    y_proba    : array-like — churn probabilities
    model_name : str        — label for display

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    metrics = {
        'Model'     : model_name,
        'Accuracy'  : accuracy_score(y_true, y_pred),
        'Precision' : precision_score(y_true, y_pred, zero_division=0),
        'Recall'    : recall_score(y_true, y_pred),
        'F1 Score'  : f1_score(y_true, y_pred),
        'ROC-AUC'   : roc_auc_score(y_true, y_proba)
    }
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics.
    """
    from sklearn.metrics import (mean_absolute_error,
                                  mean_squared_error, r2_score)
    return {
        'MAE'  : mean_absolute_error(y_true, y_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2'   : r2_score(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           ax=None, save_path: str = None):
    """
    Plot a labelled confusion matrix heatmap.
    """
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = ['Retained (0)', 'Churned (1)']
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f'{model_name}\nTP={tp} FP={fp} TN={tn} FN={fn}',
                 fontsize=11, fontweight='bold')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def plot_roc_curves(models_proba: dict, y_true,
                    save_path: str = None):
    """
    Plot ROC curves for multiple models on one chart.

    Parameters
    ----------
    models_proba : dict — {model_name: y_pred_proba}
    y_true       : array-like — true labels
    save_path    : str or None — path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = ['#185FA5', '#1D9E75', '#D85A30']

    for (name, proba), color in zip(models_proba.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name}  (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1.5,
            label='Random baseline (AUC = 0.500)')
    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — test set', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()