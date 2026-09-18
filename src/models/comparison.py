"""
Model Comparison Module
Trains and compares multiple classifiers (Logistic Regression, Random Forest, XGBoost)
for systematic model selection with cross-validation.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

def evaluate_classification(y_true, y_pred_proba, threshold=0.5) -> Dict[str, Any]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold_used": float(threshold)
    }

class ModelComparer:
    """Compares multiple classifiers using cross-validation and holdout evaluation."""

    def __init__(self, class_weight_ratio: float = 1.0):
        self.class_weight_ratio = class_weight_ratio
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def _build_candidates(self) -> Dict[str, Any]:
        """Creates candidate model instances."""
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced",
                random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42, eval_metric="logloss", n_jobs=-1
            )
        }

    def compare(self, X_train, y_train, X_test, y_test, cv_folds=5) -> Dict[str, Any]:
        """
        Trains all candidates, evaluates with CV on train and holdout on test.
        Returns comparison results dict.
        """
        candidates = self._build_candidates()
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        per_model_results = {}
        comparison_table = []
        
        best_pr_auc = -1.0
        
        for name, model in candidates.items():
            logger.info(f"Training and evaluating {name}...")
            
            # Cross-validation
            cv_roc_aucs = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train[train_idx], y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                X_va, y_va = X_train[val_idx], y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                y_va_proba = model.predict_proba(X_va)[:, 1]
                cv_roc_aucs.append(roc_auc_score(y_va, y_va_proba))
                
            mean_cv_roc_auc = np.mean(cv_roc_aucs)
            std_cv_roc_auc = np.std(cv_roc_aucs)
            
            # Fit on full train
            model.fit(X_train, y_train)
            
            # Holdout evaluation
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = evaluate_classification(y_test, y_test_proba)
            
            per_model_results[name] = {
                "train_cv_roc_auc_mean": float(mean_cv_roc_auc),
                "train_cv_roc_auc_std": float(std_cv_roc_auc),
                "test_metrics": test_metrics,
                "model_object": model
            }
            
            comparison_table.append({
                "Model": name,
                "CV ROC-AUC": f"{mean_cv_roc_auc:.4f} ± {std_cv_roc_auc:.4f}",
                "Test ROC-AUC": test_metrics['roc_auc'],
                "Test PR-AUC": test_metrics['pr_auc'],
                "Test F1": test_metrics['f1_score']
            })
            
            if test_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = test_metrics['pr_auc']
                self.best_model_name = name
                self.best_model = model

        self.models = candidates
        self.results = {
            "per_model_results": per_model_results,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "comparison_table": comparison_table
        }
        return self.results

    def get_best_model(self) -> Tuple[str, Any]:
        """Returns (name, model) of the best performing model."""
        return self.best_model_name, self.best_model
