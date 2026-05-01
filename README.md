# Customer Churn Prediction & Revenue Forecasting

**Author:** Vishnu Rao Sige | IEC2022097 | IIIT Allahabad 

---

## Project Overview

A complete end-to-end supervised machine learning project that predicts
customer churn for a telecom company and translates model predictions
into quantified revenue risk — enabling targeted retention campaigns.

**Business Question:**  
*Which customers are about to leave, how much revenue do they represent,
and what should we do about it?*

**Result:**  
Logistic Regression model selected (highest Recall) identifying customers
at risk. Revenue forecasting reveals monthly revenue at risk with a
full retention ROI model for campaign planning.

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Place in:** `data/raw/`
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` column (Yes/No → 1/0)

---

## Project Structure

customer-churn-prediction/
│
├── data/
│   ├── raw/                    ← Place downloaded dataset here
│   └── processed/              ← Cleaned and split datasets (auto-generated)
│       ├── churn_cleaned.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/                  ← Run in order 01 → 07
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_revenue_forecasting.ipynb
│
├── models/                     ← Saved trained models (auto-generated)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── reports/
│   └── figures/                ← All charts (auto-generated)
│
├── requirements.txt
├── .gitignore
└── README.md
---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Vishnusige/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install all dependencies
pip install -r requirements.txt

### 4. Download the dataset
Download from Kaggle and place the CSV file in `data/raw/`

### 5. Run notebooks in order
jupyter notebook
Run notebooks 01 through 07 in sequence.  
Each notebook is self-contained and saves its outputs for the next stage.

---

## Pipeline Stages

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_data_exploration.ipynb` | Initial dataset inspection |
| 2 | `02_data_cleaning.ipynb` | Fix dtypes, handle nulls, encode binaries |
| 3 | `03_eda.ipynb` | Distributions, churn rates, correlation heatmap |
| 4 | `04_feature_engineering.ipynb` | One-hot encoding, scaling, train/test split |
| 5 | `05_model_training.ipynb` | Train LR, RF, XGBoost with 5-fold CV |
| 6 | `06_evaluation.ipynb` | Confusion matrix, ROC curves, full metrics |
| 7 | `07_revenue_forecasting.ipynb` | Revenue at risk, risk tiers, ROI forecast |

---

## Key Results

### Model Performance (Test Set)

| Model                | ROC-AUC | Recall | F1 Score |
|---------------------|--------:|-------:|---------:|
| Logistic Regression | 0.8353  | **0.7941** | 0.6061 |
| Random Forest       | **0.8376**  | 0.7701 | **0.6323** |
| XGBoost             | 0.8206  | 0.7246 | 0.6049 |


**Selected model:** Logistic Regression  
**Reason:** Highest Recall — in churn prediction, missing a churner (False Negative)
is far more costly than a false alarm (False Positive). Recall is the
correct optimisation objective for this asymmetric cost problem.

### Revenue Forecast
- Monthly revenue at risk quantified per customer
- Three-tier risk segmentation (High / Medium / Low)
- Top 20 priority customers identified for immediate retention action
- Retention ROI model shows break-even rate and expected returns

---

## Key Findings from EDA

1. **Churn rate is 26.5%** — a meaningful class imbalance requiring special handling
2. **Tenure is the strongest predictor** — new customers (0–12 months) churn at ~47%
3. **Contract type is critical** — month-to-month: ~43% churn vs two-year: ~3%
4. **Fiber optic customers churn at ~42%** — double the overall average
5. **Higher monthly charges correlate with higher churn** — churned customers pay ~$13/month more

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost |
| Model persistence | joblib |
| Environment | Jupyter Notebook |
| Version control | Git, GitHub |

---

## Git History

All work is tracked with meaningful commits across feature branches,
merged into `main` via the standard branch → commit → merge → push workflow.