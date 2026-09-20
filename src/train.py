"""
Training script for Credit Card Fraud Detection.

Trains XGBoost with scale_pos_weight (which outperformed SMOTE),
saves model artifacts to models/.

Usage:
    python src/train.py
"""

import os
import joblib
import kagglehub
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


def load_data():
    """Download and load the ULB Credit Card Fraud dataset."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(os.path.join(path, "creditcard.csv"))
    return df


def clean_data(df):
    """Remove duplicates to prevent train/val leakage."""
    print(f"  Before dedup: {df.shape}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After dedup: {df.shape}")
    return df


def feature_engineering(df):
    """Create hour and log_amount features, drop original Time and Amount."""
    df = df.copy()
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time", "Amount"])
    return df


def split_data(df, random_state=1):
    """Split into 60/20/20 train/val/test with stratification."""
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df["Class"]
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=random_state,
        stratify=df_full_train["Class"]
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["Class"].values
    y_val = df_val["Class"].values
    y_test = df_test["Class"].values

    del df_train["Class"]
    del df_val["Class"]
    del df_test["Class"]

    return df_train, df_val, df_test, y_train, y_val, y_test


def scale_features(df_train, df_val, df_test):
    """Scale log_amount and hour, fit on train only."""
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    scaler = StandardScaler()
    cols = ["log_amount", "hour"]

    df_train[cols] = scaler.fit_transform(df_train[cols])
    df_val[cols] = scaler.transform(df_val[cols])
    df_test[cols] = scaler.transform(df_test[cols])

    return df_train, df_val, df_test, scaler


def find_best_threshold(y_val, y_proba):
    """Find F1-optimal threshold via precision_recall_curve."""
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def train_model(df_train, y_train):
    """Train XGBoost with scale_pos_weight."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=1,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(df_train, y_train)
    return model


def save_artifacts(model, scaler, threshold, feature_names, output_dir="models"):
    """Save model, scaler, threshold, and feature names to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(threshold, os.path.join(output_dir, "threshold.pkl"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))
    print(f"  Saved artifacts to {output_dir}/")


def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)

    print("Feature engineering...")
    df = feature_engineering(df)

    print("Splitting data...")
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    print(f"  Train: {df_train.shape} | fraud: {y_train.sum()}")
    print(f"  Val:   {df_val.shape} | fraud: {y_val.sum()}")
    print(f"  Test:  {df_test.shape} | fraud: {y_test.sum()}")

    print("Scaling features...")
    df_train, df_val, df_test, scaler = scale_features(df_train, df_val, df_test)

    print("Training XGBoost with scale_pos_weight...")
    model = train_model(df_train, y_train)

    print("Finding F1-optimal threshold...")
    y_pred_proba = model.predict_proba(df_val)[:, 1]
    threshold = find_best_threshold(y_val, y_pred_proba)
    print(f"  Threshold: {threshold:.4f}")

    print("Saving artifacts...")
    save_artifacts(model, scaler, threshold, list(df_train.columns))

    print("\nDone.")


if __name__ == "__main__":
    main()