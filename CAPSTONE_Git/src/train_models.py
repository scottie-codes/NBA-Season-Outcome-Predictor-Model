"""
train_models.py
Trains Random Forest models to predict NBA team performance.
Regressor model predicts exact win %
Classifier model predicts top half vs bottom half finish (for playoff qualification estimation)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
import joblib

# File paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

MASTER_FEATURES = PROC / "master_features.csv"
METRICS_CLS = PROC / "metrics_classification.csv"
METRICS_REG = PROC / "metrics_regression.csv"
CLF_PKL = PROC / "classifier.pkl"
REG_PKL = PROC / "regressor.pkl"
FEATURES_FILE = PROC / "clf_features.json"


def load_features():
    """Load the master features file."""
    if not MASTER_FEATURES.exists():
        raise FileNotFoundError(f"Missing file: {MASTER_FEATURES}")
    return pd.read_csv(MASTER_FEATURES)


def select_feature_columns(df):
    """
    Select numeric columns to use as features.
    Excludes ID columns and target/outcome columns to prevent data leakage.
    """
    exclude = {
        "TEAM_ID", "Season",
        "TARGET_WIN_PCT", "WINPCT_TOP",
        "ADV_ROUND1", "ADV_ROUND2", "ADV_ROUND3", "ADV_FINALS",
        "MADE_PLAYOFFS",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col not in exclude]

    return features


def train_models():
    """Main training function."""
    print("Loading data...")
    df = load_features().dropna(subset=["TARGET_WIN_PCT"])

    # Create binary label: 1 if team is in top half of league that season
    if "WINPCT_TOP" not in df.columns:
        season_median = df.groupby("Season")["TARGET_WIN_PCT"].transform("median")
        df["WINPCT_TOP"] = (df["TARGET_WIN_PCT"] >= season_median).astype(int)

    # prepare features and targets
    features = select_feature_columns(df)
    X = df[features].fillna(0.0).to_numpy(dtype=np.float32)
    y_regression = df["TARGET_WIN_PCT"].to_numpy(dtype=np.float32)
    y_classification = df["WINPCT_TOP"].to_numpy(dtype=int)

    # Split data (same split for both models to keep rows aligned)
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_regression, y_classification,
        test_size=0.2,
        random_state=42,
        stratify=y_classification
    )

    # Train Random forests regression model
    print("Training regression model...")
    regressor = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(X_train, y_train_reg)

    # evaluate regression
    pred_reg = regressor.predict(X_test)
    mae = mean_absolute_error(y_test_reg, pred_reg)
    r2 = r2_score(y_test_reg, pred_reg)
    print(f"Regression - MAE: {mae:.4f}, R2: {r2:.4f}")

    # train Random forests classification model
    print("Training classification model...")
    classifier = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train_cls)

    # Evaluate classification
    pred_cls = classifier.predict(X_test)
    acc = accuracy_score(y_test_cls, pred_cls)

    # Calculate AUC if possible
    try:
        proba = classifier.predict_proba(X_test)
        auc = roc_auc_score(y_test_cls, proba[:, 1])
    except Exception:
        auc = float("nan")

    print(f"Classification - Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Save everything
    PROC.mkdir(parents=True, exist_ok=True)

    # Save the validation metrics to CSV
    pd.DataFrame([{"ACC": acc, "AUC": auc}]).to_csv(METRICS_CLS, index=False)
    pd.DataFrame([{"MAE": mae, "R2": r2}]).to_csv(METRICS_REG, index=False)

    # Save trained models
    joblib.dump(classifier, CLF_PKL)
    joblib.dump(regressor, REG_PKL)

    # save feature list
    with open(FEATURES_FILE, "w") as f:
        json.dump(features, f)

    print(f"Training finished")


if __name__ == "__main__":
    train_models()