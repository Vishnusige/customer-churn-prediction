# Customer Churn Prediction & Revenue Forecasting
## Using Supervised Machine Learning

**Author:** Vishnu Rao Sige | IEC2022097 | IIIT Allahabad  
**Date:** May 2026

---

## 1. Executive Summary

This project builds a complete machine learning pipeline to predict customer
churn for a telecom company and quantify the resulting revenue risk.

Using the IBM Telco Customer Churn dataset (7,032 customers, 21 features),
three supervised learning models were trained and evaluated. Logistic
Regression was selected as the final model based on its superior Recall
score — the correct metric for this asymmetric cost problem where missing
a churning customer is far more costly than a false alarm.

The revenue forecasting stage translates model predictions into dollar
values, producing a prioritised list of at-risk customers and a retention
campaign ROI model.

**Key outcomes:**
- Churn rate identified at 26.5% with clear high-risk customer segments
- Model identifies the majority of churners before they leave
- Monthly revenue at risk quantified with annual projection
- Top 20 priority customers listed for immediate retention action
- Break-even retention rate calculated for campaign planning

---

## 2. Problem Statement

### Business Context
Customer churn — when a customer stops using a service — is one of the
most expensive problems in subscription-based businesses. Acquiring a new
customer costs 5–7× more than retaining an existing one. For a telecom
company, each churned customer represents lost monthly recurring revenue
that is difficult and expensive to replace.

### Objective
Build a model that predicts which customers are likely to churn, then
quantify the revenue impact of those predictions to enable targeted,
cost-effective retention campaigns.

### Success Criteria
A model that catches as many actual churners as possible (high Recall)
while providing accurate revenue risk estimates to prioritise retention spend.

---

## 3. Dataset Description

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn (Kaggle) |
| Original size | 7,043 rows × 21 columns |
| After cleaning | 7,032 rows × 20 columns |
| Target variable | Churn (Yes/No → 1/0) |
| Churn rate | 26.5% (1,869 churned, 5,163 retained) |
| Class imbalance | 2.7:1 (retained:churned) |

**Feature categories:**
- Customer demographics: gender, SeniorCitizen, Partner, Dependents
- Account details: tenure, Contract, PaperlessBilling, PaymentMethod
- Financial: MonthlyCharges, TotalCharges
- Services: PhoneService, InternetService, OnlineSecurity, and 6 more

---

## 4. Methodology — Data Science Pipeline

### Stage 1: Data Exploration
Initial inspection revealed the dataset structure, data types, and the
26.5% churn rate. The class imbalance was documented and flagged for
handling in model training.

### Stage 2: Data Cleaning
Three cleaning operations were performed:
- `TotalCharges` column fixed from object to float64 dtype (hidden blank spaces)
- 11 rows with missing TotalCharges dropped (new customers, tenure=0)
- Binary categorical columns encoded to 0/1

### Stage 3: Exploratory Data Analysis
Key findings (see figures 01–08 in reports/figures/):
- New customers (0–12 months) churn at ~47% vs 6% for long-tenure customers
- Month-to-month contracts have 43% churn vs 3% for two-year contracts
- Fiber optic internet customers churn at ~42% — double the average
- Churned customers pay ~$13/month more than retained customers
- TotalCharges and tenure are highly correlated (0.83) — multicollinearity noted

### Stage 4: Feature Engineering
- One-hot encoding applied to 7 multi-class categorical columns
- `drop_first=True` prevents the dummy variable trap
- 80/20 stratified train/test split (random_state=42)
- StandardScaler fit on training data only — no data leakage
- Final feature matrix: ~30 features

### Stage 5: Model Training
Three models trained with 5-fold stratified cross-validation:

| Model                | CV Mean AUC | CV Std AUC |
|---------------------|------------:|-----------:|
| Logistic Regression | 0.8459      | 0.0040     |
| Random Forest       | 0.8479      | 0.0045     |
| XGBoost             | 0.8381      | 0.0050     |


All models used `class_weight='balanced'` or equivalent to handle the
2.7:1 class imbalance.

### Stage 6: Model Evaluation
Test set evaluation results (unseen data):

| Model                | ROC-AUC | Recall | Precision | F1     |
|---------------------|--------:|-------:|----------:|-------:|
| Logistic Regression | 0.8353  | 0.7941 | 0.4901    | 0.6061 |
| Random Forest       | 0.8376  | 0.7701 | 0.5363    | 0.6323 |
| XGBoost             | 0.8206  | 0.7246 | 0.5192    | 0.6049 |


**Model selection justification:**
Logistic Regression was selected despite not having the highest ROC-AUC.
In churn prediction, a False Negative (missed churner) represents permanent
revenue loss, while a False Positive (false alarm) results only in a small,
recoverable retention cost. This asymmetric cost structure means Recall is
the correct primary metric, and Logistic Regression achieves the highest
Recall of the three models evaluated.

---

## 5. Revenue Forecasting Results

### Core Formula
Revenue at Risk (per customer) = MonthlyCharges × Churn Probability

### Summary
[Fill in from your Stage 7 Cell 5 output:]
- Total monthly revenue (test set): $[amount]
- Expected revenue at risk: $[amount]/month
- Annual projection: $[amount]/year
- Percentage of revenue at risk: [X]%

### Risk Tier Segmentation

| Tier   | Threshold | Customers | Revenue at Risk |
|--------|----------:|----------:|----------------:|
| High   | >70%      | 378       | $24,153.86      |
| Medium | 40–70%    | 343       | $12,565.74      |
| Low    | <40%      | 686       | $6,082.97       |

### Retention ROI
Targeting high-risk customers with a $30/customer campaign:
- Break-even at approximately [X]% retention success rate
- At 50% retention success: $[amount]/month net saved
- Annual net saving at 50%: $[amount]/year

---

## 6. Business Recommendations

**Immediate (this week):**
Contact the top 20 highest revenue-at-risk customers with personalised
retention offers. These represent the highest ROI intervention available.

**Short term (this month):**
Launch a structured retention campaign targeting all high-risk customers.
Focus on converting month-to-month customers to annual contracts with
an incentive offer.

**Medium term (this quarter):**
Conduct a service quality audit for Fiber Optic infrastructure.
The disproportionate churn rate among Fiber Optic customers suggests
a product quality issue rather than a pricing problem.

**Ongoing:**
Deploy the model as an automated monthly scoring system.
Flag any customer crossing the 0.70 probability threshold for
proactive outreach before they make the decision to leave.

---

## 7. Conclusion

This project successfully demonstrates a complete, production-grade data
science pipeline from raw data to business-actionable revenue forecasting.

The selected Logistic Regression model provides the highest Recall among
the three models evaluated, correctly prioritising the minimisation of
missed churners over the minimisation of false alarms — the correct
trade-off given the asymmetric costs in this business context.

The revenue forecasting stage provides direct business value by converting
probability scores into dollar figures, enabling cost-benefit analysis of
retention campaigns and clear prioritisation of customer outreach.

The high-risk segment contains only 26.9% of customers but contributes
56.4% of total revenue at risk, indicating strong risk concentration
and making it the highest priority for retention efforts.

---

## 8. References

1. IBM Telco Customer Churn Dataset — Kaggle
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. Scikit-learn Documentation — https://scikit-learn.org

3. XGBoost Documentation — https://xgboost.readthedocs.io

4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
   KDD '16 Proceedings.