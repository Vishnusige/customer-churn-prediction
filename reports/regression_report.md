# Regression Report

## Revenue Forecasting Model

### Problem Framing
Revenue forecasting is treated as a regression problem where the target
variable is revenue_at_risk per customer:

    revenue_at_risk = MonthlyCharges × churn_probability

### Models Evaluated

| Model                     | Test MAE | Test RMSE | Test R² |
|--------------------------|----------|-----------|---------|
| Linear Regression        | 0.0259   | 0.0343    | 0.9988  |
| Ridge Regression         | 0.0259   | 0.0343    | 0.9988  |
| Random Forest Regressor  | 0.0303   | 0.0430    | 0.9981  |
| Gradient Boosting Regressor | (not used) | (not used) | (not used) |


### Selected Model

Best regressor: Linear Regression

Reason:
Linear Regression was selected due to its simplicity and nearly identical performance to Ridge Regression (R² ≈ 0.9988). The relationship between features and MonthlyCharges is largely linear, making a simple model sufficient.

### Revenue Forecast Summary

| Metric | Value |
|--------|-------|
| Total monthly revenue (test set) | $91,737.95 |
| Expected revenue at risk | $38,248.27/month |
| Annual projection | $458,979.24/year |
| % of revenue at risk | 41.7% |

### Risk Tier Segmentation

| Tier | Customers | Revenue at Risk |
|------|-----------|-----------------|
| High (>70%) | 378 | $19,172.82 |
| Medium (40–70%) | 343 | $12,515.03 |
| Low (<40%) | 686 | $6,560.42 |