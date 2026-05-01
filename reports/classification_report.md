# Classification Report

## Model: Logistic Regression (Selected)
**Selection reason:** Highest Recall — minimises missed churners

### Test Set Results

| Metric     | Score  |
|------------|-------:|
| ROC-AUC    | 0.8353 |
| Recall     | 0.7941 |
| Precision  | 0.4901 |
| F1 Score   | 0.6061 |
| Accuracy   | 0.7257 |

### All Models Comparison

| Model                | ROC-AUC | Recall | Precision | F1     |
|---------------------|--------:|-------:|----------:|-------:|
| Logistic Regression | 0.8353  | **0.7941** | 0.4901 | 0.6061 |
| Random Forest       | **0.8376**  | 0.7701 | **0.5363** | **0.6323** |
| XGBoost             | 0.8206  | 0.7246 | 0.5192 | 0.6049 |

> Fill all values from notebooks/06_evaluation.ipynb output.

### Confusion Matrix Interpretation
- True Positives (churners caught): [value]
- False Negatives (churners missed): [value]
- False Positives (false alarms): [value]
- True Negatives (correctly ignored): [value]

### Key Finding
Logistic Regression selected over higher AUC models because Recall
is the correct primary metric for this asymmetric cost problem.
A missed churner (FN) = permanent revenue loss.
A false alarm (FP) = small recoverable retention cost.