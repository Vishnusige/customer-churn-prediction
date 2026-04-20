# Customer Churn Prediction & Revenue Forecasting

A supervised machine learning project to predict customer churn and forecast revenue impact using the Telco Customer Churn dataset.

## Project Structure
- `data/` — raw and processed datasets
- `notebooks/` — Jupyter notebooks for each pipeline stage
- `src/` — reusable Python utility scripts
- `models/` — saved trained model files
- `reports/` — final report, presentation, and figures

## Setup Instructions

1. Clone this repository:
git clone: https://github.com/Vishnusige/customer-churn-prediction.git

2. Create and activate virtual environment:
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Download the dataset from Kaggle:
   [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   Place it in `data/raw/`

## Dataset
- Source: IBM Telco Customer Churn Dataset (via Kaggle)
- Records: 7,043 customers
- Target: `Churn` column (Yes/No)

## Tech Stack
Python 3.11 | Pandas | Scikit-learn | XGBoost | Matplotlib | Seaborn | Jupyter