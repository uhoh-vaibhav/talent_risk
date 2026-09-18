"""
Attrition Model Training & Optimization Module
Trains XGBoost model for Attrition Prediction with MLflow tracking, SHAP explainer, and Fairness Audit.
"""

import os
import joblib
import logging
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.features.store import FeatureStore, DATA_DIR
from src.models.baseline import evaluate_classification
from src.explainability.fairness import FairnessAuditor
from src.explainability.shap_explainer import ModelExplainer

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
EXPERIMENT_NAME = "Talent_Attrition_Risk_Prediction"


def train_attrition_model(
    df_attrition: pd.DataFrame,
    use_mlflow: bool = True
) -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, Any]]:
    """Trains and optimizes XGBoost model for employee attrition risk prediction."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing Attrition Model Training...")

    # Load feature pipeline & extract features
    fs = FeatureStore()
    X_trans = fs.get_serving_features(df_attrition)
    feature_names = fs.feature_names
    y = df_attrition["target_attrition"].values

    # Compute class ratio for scale_pos_weight
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_trans, y, df_attrition, test_size=0.2, random_state=42, stratify=y
    )

    # Configure MLflow
    if use_mlflow:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name="XGBoost_Attrition_Main")
        except Exception as e:
            logger.warning("MLflow initialization notice: %s", str(e))
            use_mlflow = False

    # Define hyperparams
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate test metrics
    y_test_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, y_test_proba)

    logger.info("XGBoost Attrition Metrics: %s", metrics)

    # Fairness Audit
    df_test_eval = df_test.copy()
    df_test_eval["prediction"] = (y_test_proba >= 0.5).astype(int)
    fairness_report = FairnessAuditor.run_full_fairness_audit(df_test_eval, pred_column="prediction")

    # SHAP Explainer initialization
    explainer = ModelExplainer(model, feature_names)

    # Save artifacts
    model_path = SAVED_MODELS_DIR / "attrition_xgb_model.joblib"
    explainer_path = SAVED_MODELS_DIR / "attrition_shap_explainer.joblib"

    joblib.dump({"model": model, "feature_names": feature_names, "metrics": metrics}, model_path)
    joblib.dump(explainer, explainer_path)
    logger.info("Saved trained attrition model to %s", model_path)

    # MLflow logging
    if use_mlflow:
        try:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(fairness_report, "fairness_audit_report.json")
            mlflow.end_run()
        except Exception as e:
            logger.warning("Could not log to MLflow: %s", str(e))

    return model, metrics, fairness_report


if __name__ == "__main__":
    df_attrition = pd.read_parquet(DATA_DIR / "unified_attrition.parquet")
    train_attrition_model(df_attrition)
