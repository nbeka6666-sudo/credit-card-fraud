"""
FastAPI service for Credit Card Fraud Detection.

Endpoints:
    GET  /         — health check
    POST /predict  — classify a single transaction

Usage:
    uvicorn api.app:app --reload --port 8000
"""

import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


MODELS_DIR = "models"


# ---------- Load artifacts at startup ----------
model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


# ---------- Schema ----------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ---------- App ----------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost model with scale_pos_weight",
    version="1.0",
)


def preprocess(transaction: dict) -> pd.DataFrame:
    """Apply feature engineering and scaling to a single transaction."""
    df = pd.DataFrame([transaction])

    # Feature engineering
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time", "Amount"])

    # Reorder columns to match training
    df = df[feature_names]

    # Scale log_amount and hour
    cols_to_scale = ["log_amount", "hour"]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


@app.get("/")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model": "XGBoost + scale_pos_weight",
        "threshold": threshold,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    """Classify a single transaction."""
    X = preprocess(transaction.model_dump())
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba > threshold)
    return {
        "fraud_probability": round(proba, 4),
        "threshold": threshold,
        "prediction": "FRAUD" if label == 1 else "NORMAL",
    }