"""
Train Promotion Model Module
Trains and evaluates models for predicting employee promotion readiness.
"""

import logging
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import train_test_split

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.features.store import FeatureStore, DATA_DIR
from src.models.comparison import ModelComparer, evaluate_classification
from src.models.threshold import ThresholdOptimizer
from src.explainability.fairness import FairnessAuditor
from src.explainability.shap_explainer import ModelExplainer

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"


def train_promotion_model(df_promotion: pd.DataFrame, use_mlflow: bool = True) -> Dict[str, Any]:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train/test split
    X = df_promotion.drop(columns=["target_promotion"])
    y = df_promotion["target_promotion"]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit FeatureStore pipeline on TRAINING data only
    fs = FeatureStore(model_type="promotion")
    df_train = pd.concat([X_train_raw, y_train], axis=1)
    df_test = pd.concat([X_test_raw, y_test], axis=1)
    
    X_train_trans, feature_names = fs.fit_and_save(df_train)
    X_test_trans = fs.transform(df_test)
    
    # Compute class distribution + scale_pos_weight
    pos_count = int(sum(y_train == 1))
    neg_count = int(sum(y_train == 0))
    imbalance_ratio = float(neg_count / max(1, pos_count))
    class_distribution = {
        "positive": pos_count,
        "negative": neg_count,
        "imbalance_ratio": round(imbalance_ratio, 2)
    }
    
    # Model comparison with CV
    comparer = ModelComparer(class_weight_ratio=imbalance_ratio)
    comparison_results = comparer.compare(X_train_trans, y_train, X_test_trans, y_test)
    
    # Get best model
    best_model_name, best_model = comparer.get_best_model()
    y_test_proba = best_model.predict_proba(X_test_trans)[:, 1]
    
    # Threshold optimization
    threshold_info = ThresholdOptimizer.find_optimal_threshold(y_test, y_test_proba)
    optimal_thresh = float(threshold_info.get("optimal_threshold", 0.5))
    y_test_pred = (y_test_proba >= optimal_thresh).astype(int)
    
    # Fairness audit on test set
    df_eval = X_test_raw.copy()
    df_eval["prediction"] = y_test_pred
    fairness_results = FairnessAuditor.run_full_fairness_audit(df_eval, pred_column="prediction")
    
    # SHAP Explainer + Global Importance
    explainer = ModelExplainer(best_model, feature_names=feature_names)
    shap_sample = X_test_trans[:min(200, len(X_test_trans))]
    global_importance = explainer.compute_global_importance(shap_sample, top_k=15)
    
    # Save artifacts
    model_artifact = {
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": comparison_results["per_model_results"][best_model_name]["test_metrics"],
        "threshold_info": threshold_info,
        "comparison_summary": comparison_results["comparison_table"],
        "class_distribution": class_distribution,
        "hyperparameters": best_model.get_params() if hasattr(best_model, "get_params") else {}
    }
    
    joblib.dump(model_artifact, SAVED_MODELS_DIR / "promotion_xgb_model.joblib")
    joblib.dump(model_artifact, SAVED_MODELS_DIR / "promotion_model.joblib")
    joblib.dump(explainer, SAVED_MODELS_DIR / "promotion_shap_explainer.joblib")
    joblib.dump(explainer, SAVED_MODELS_DIR / "promotion_explainer.joblib")
    
    with open(SAVED_MODELS_DIR / "promotion_fairness.json", "w") as f:
        json.dump(fairness_results, f, indent=2)
        
    with open(SAVED_MODELS_DIR / "promotion_global_shap.json", "w") as f:
        json.dump(global_importance, f, indent=2)
        
    # MLflow logging
    if use_mlflow and HAS_MLFLOW:
        try:
            mlflow.set_experiment("Talent_Promotion_Readiness_Prediction")
            with mlflow.start_run():
                mlflow.log_params({k: str(v) for k, v in model_artifact["hyperparameters"].items() if isinstance(v, (int, float, str, bool))})
                mlflow.log_metrics({
                    "test_roc_auc": model_artifact["metrics"]["roc_auc"],
                    "test_pr_auc": model_artifact["metrics"]["pr_auc"],
                    "test_f1": model_artifact["metrics"]["f1_score"],
                    "optimal_threshold": optimal_thresh
                })
        except Exception as e:
            logger.warning(f"MLflow logging skipped/failed: {e}")
            
    return {
        "best_model_name": best_model_name,
        "metrics": model_artifact["metrics"],
        "threshold_info": threshold_info,
        "comparison_table": comparison_results["comparison_table"],
        "fairness": fairness_results,
        "global_shap": global_importance
    }


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "processed_promotion.parquet")
    train_promotion_model(df, use_mlflow=False)
