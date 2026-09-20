# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card transactions.

**Stack:** Python, XGBoost, FastAPI, Streamlit, Docker

---

## Problem Statement

Credit card fraud costs the global economy billions of dollars annually. Banks need to
detect fraudulent transactions in real-time, but face two challenges:

1. **Extreme class imbalance** — only 0.17% of transactions are fraudulent
2. **Cost asymmetry** — missing a fraud costs far more than a false alarm

This project builds a production-ready ML service that classifies transactions as
**fraudulent** or **normal** with high precision and recall.

**Business goal:** catch as many frauds as possible while keeping false alarms low.

---

## Dataset

**ULB Credit Card Fraud Detection** (Kaggle, `mlg-ulb/creditcardfraud`)

- **284,807** transactions, **31** features
- **492** fraudulent (0.17%), **284,315** normal
- Features **V1–V28** are PCA components (anonymized for privacy)
- Only **Time** and **Amount** are in original units

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Key EDA Findings

- **Severe class imbalance** (0.17%) → accuracy is useless. Use Precision, Recall, F1, PR-AUC
- **Amount is right-skewed** → apply log1p transformation
- **Time shows daily periodicity** → extract `hour` as a feature
- **Top correlated features:** V17 (−0.33), V14 (−0.30), V12 (−0.26), V10 (−0.22)
- **Boxplots confirm** these features clearly separate fraud from normal

### Class Distribution

![Class Distribution](images/class_distribution.png)

---

## Approach

1. **Data cleaning** — remove duplicates to avoid train/val leakage
2. **Feature engineering** — create `hour` (from Time) and `log_amount` (from Amount)
3. **Split** — 60/20/20 train/val/test with `stratify=y`
4. **Scaling** — StandardScaler on `log_amount` and `hour` (fit on train only)
5. **Models** — Logistic Regression, Random Forest, XGBoost
6. **Imbalance handling** — compared SMOTE vs `scale_pos_weight`
7. **Threshold tuning** — F1-optimal threshold via `precision_recall_curve`
8. **Explainability** — SHAP for feature importance
9. **Deployment** — FastAPI + Streamlit + Docker

---

## Results

### Model Comparison (Validation Set, F1-optimal threshold)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.844 | 0.768 | 0.804 |
| Random Forest | 0.879 | 0.808 | 0.842 |
| XGBoost + SMOTE | 0.963 | 0.798 | 0.873 |
| **XGBoost + scale_pos_weight** | **0.976** | **0.828** | **0.896** |

### Test Set — Final Model

**XGBoost with `scale_pos_weight`**

| Metric | Value |
|---|---|
| Precision | 0.960 |
| Recall | 0.758 |
| F1 | 0.847 |
| ROC-AUC | 0.956 |
| PR-AUC | 0.789 |

**Winner:** `scale_pos_weight` outperformed SMOTE on all metrics — it avoids
synthetic examples and gives a cleaner model.

---

## SHAP — Model Explainability

Top-5 features by mean |SHAP value|:

V14, V4, V12, V10, **log_amount**

- **`log_amount`** in top-5 → feature engineering validated
- Fraud transactions tend to have **lower V14, V12, V10** — confirms correlations
- Different top features vs SMOTE model — `scale_pos_weight` uses original data

### SHAP Feature Importance

![SHAP Feature Importance](images/shap_bar.png)

---

## Project Structure

```text
credit-card-fraud/
├── api/
│   └── app.py              # FastAPI application & REST endpoints
├── models/                 # Model artifacts
│   ├── feature_names.pkl   # Saved feature order
│   ├── fraud_model.pkl     # Trained XGBoost model
│   ├── scaler.pkl          # Trained StandardScaler
│   └── threshold.pkl       # F1-optimal threshold
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory Data Analysis & experiments
├── src/
│   ├── predict.py          # Prediction & preprocessing logic
│   └── train.py            # Model training & threshold evaluation
├── wheels/
│   └── xgboost-2.1.4-*.whl # Offline wheel for Docker build
├── .dockerignore           # Files excluded from Docker builds
├── Dockerfile              # Dockerfile for FastAPI backend
├── requirements.txt        # Full local development environment
├── requirements-docker.txt # Pinned minimal dependencies for Docker
├── streamlit_app.py        # Interactive Streamlit dashboard
└── README.md