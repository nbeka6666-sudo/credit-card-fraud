"""
Streamlit dashboard for Credit Card Fraud Detection.

Run locally:
    py -3.12 -m streamlit run streamlit_app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODELS_DIR = "models"


@st.cache_resource
def load_artifacts():
    """Load model, scaler, threshold, feature names (cached)."""
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, threshold, feature_names


def preprocess(transaction, scaler, feature_names):
    """Apply feature engineering + scaling to a single transaction."""
    df = pd.DataFrame([transaction])
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time", "Amount"])
    df = df[feature_names]

    cols_to_scale = ["log_amount", "hour"]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


# ---------- Page ----------
st.set_page_config(page_title="Fraud Detection", page_icon="🚨", layout="wide")

st.title("🚨 Credit Card Fraud Detection")
st.write("Enter transaction parameters and check if it's fraudulent.")

model, scaler, threshold, feature_names = load_artifacts()

# ---------- Sidebar inputs ----------
st.sidebar.header("Transaction Parameters")

amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=149.62, step=1.0)
time = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=406.0, step=1.0)

st.sidebar.subheader("PCA Components (V1–V28)")
st.sidebar.caption("Adjust the most important features:")

v1 = st.sidebar.slider("V1", -10.0, 10.0, -1.36, 0.01)
v4 = st.sidebar.slider("V4", -10.0, 10.0, 1.38, 0.01)
v10 = st.sidebar.slider("V10", -10.0, 10.0, 0.09, 0.01)
v12 = st.sidebar.slider("V12", -10.0, 10.0, -0.62, 0.01)
v14 = st.sidebar.slider("V14", -10.0, 10.0, -0.31, 0.01)
v17 = st.sidebar.slider("V17", -10.0, 10.0, 0.21, 0.01)

# Build transaction dict — other V's = 0
transaction = {"Time": time, "Amount": amount}
for i in range(1, 29):
    transaction[f"V{i}"] = 0.0

transaction["V1"] = v1
transaction["V4"] = v4
transaction["V10"] = v10
transaction["V12"] = v12
transaction["V14"] = v14
transaction["V17"] = v17

# ---------- Predict button ----------
if st.button("🔍 Check Transaction", type="primary"):
    X = preprocess(transaction, scaler, feature_names)
    proba = float(model.predict_proba(X)[0, 1])
    label = "FRAUD" if proba > threshold else "NORMAL"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fraud Probability", f"{proba:.4f}")
    with col2:
        st.metric("Decision Threshold", f"{threshold:.4f}")

    if label == "FRAUD":
        st.error(f"🚨 **{label}** — this transaction looks fraudulent")
    else:
        st.success(f"✅ **{label}** — this transaction looks normal")

st.markdown("---")
st.caption("Model: XGBoost with scale_pos_weight | Threshold: F1-optimal on validation")