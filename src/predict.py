"""
Prediction script for Credit Card Fraud Detection.

Loads the trained model + scaler + threshold from models/,
then classifies a single transaction.

Usage:
    py -3.12 src/predict.py
"""

import os
import joblib
import numpy as np
import pandas as pd


MODELS_DIR = "models"


def load_artifacts():
    """Load model, scaler, threshold, and feature names from disk."""
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, threshold, feature_names


def preprocess(transaction, scaler, feature_names):
    """
    Preprocess a single transaction dict:
    - feature engineering (hour, log_amount)
    - drop Time / Amount
    - reorder columns to match training
    - scale log_amount and hour
    """
    df = pd.DataFrame([transaction])

    # Feature engineering
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time", "Amount"])

    # Ensure same column order as in training
    df = df[feature_names]

    # Scale the same columns as during training
    cols_to_scale = ["log_amount", "hour"]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def predict(transaction, model, scaler, threshold, feature_names):
    """Return (probability, label) for a transaction."""
    X = preprocess(transaction, scaler, feature_names)
    proba = model.predict_proba(X)[0, 1]
    label = int(proba > threshold)
    return float(proba), label


def main():
    print("Loading artifacts...")
    model, scaler, threshold, feature_names = load_artifacts()
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Features:  {len(feature_names)}")

    # Example transaction (V1-V28 are PCA components; Time and Amount in raw units)
    example = {
        "Time": 406.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311170, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62,
    }

    proba, label = predict(example, model, scaler, threshold, feature_names)

    print("\nPrediction for example transaction:")
    print(f"  Fraud probability: {proba:.4f}")
    print(f"  Threshold:         {threshold:.4f}")
    print(f"  Prediction:        {'FRAUD' if label == 1 else 'NORMAL'}")


if __name__ == "__main__":
    main()