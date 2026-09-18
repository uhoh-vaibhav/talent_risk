"""
Baseline Modeling Module
Trains and evaluates Logistic Regression baseline models for Attrition and Promotion.
"""

import logging
import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.features.store import FeatureStore, DATA_DIR

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "models" / "saved_models"


def evaluate_classification(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive classification metrics suited for imbalanced targets."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4)
    }


def train_baseline_attrition(df_attrition: pd.DataFrame) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Trains Logistic Regression baseline for Attrition prediction."""
    logger.info("Training Logistic Regression baseline for Attrition...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fs = FeatureStore()
    X_trans = fs.get_serving_features(df_attrition)
    y = df_attrition["target_attrition"].values

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, y_proba)
    logger.info("Attrition Baseline Metrics: %s", metrics)

    # Save artifact
    joblib.dump(model, MODEL_DIR / "baseline_attrition_lr.joblib")
    return model, metrics


def train_baseline_promotion(df_promotion: pd.DataFrame) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Trains Logistic Regression baseline for Promotion prediction."""
    logger.info("Training Logistic Regression baseline for Promotion...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fs = FeatureStore()
    X_trans = fs.get_serving_features(df_promotion)
    y = df_promotion["target_promotion"].values

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, y_proba)
    logger.info("Promotion Baseline Metrics: %s", metrics)

    # Save artifact
    joblib.dump(model, MODEL_DIR / "baseline_promotion_lr.joblib")
    return model, metrics


if __name__ == "__main__":
    df_attrition = pd.read_parquet(DATA_DIR / "unified_attrition.parquet")
    df_promotion = pd.read_parquet(DATA_DIR / "unified_promotion.parquet")
    
    train_baseline_attrition(df_attrition)
    train_baseline_promotion(df_promotion)
