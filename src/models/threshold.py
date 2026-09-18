"""
Threshold Optimization Module
Finds optimal classification thresholds by analyzing precision-recall tradeoffs.
"""

import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Finds optimal decision threshold by maximizing F1 score."""

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1"
    ) -> Dict[str, Any]:
        """
        Evaluates thresholds from 0.1 to 0.9 in steps of 0.05.
        Returns optimal threshold and metrics at all evaluated thresholds.
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        
        best_threshold = 0.5
        best_metric_val = -1.0
        best_metrics = {}
        
        analysis = []
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            p = float(precision_score(y_true, y_pred, zero_division=0))
            r = float(recall_score(y_true, y_pred, zero_division=0))
            f = float(f1_score(y_true, y_pred, zero_division=0))
            
            analysis.append({
                "threshold": float(t),
                "precision": p,
                "recall": r,
                "f1": f
            })
            
            val = f if metric == "f1" else (p if metric == "precision" else r)
            
            if val > best_metric_val:
                best_metric_val = val
                best_threshold = t
                best_metrics = {"precision": p, "recall": r, "f1": f}
                
        return {
            "optimal_threshold": float(best_threshold),
            "optimal_f1": best_metrics["f1"],
            "optimal_precision": best_metrics["precision"],
            "optimal_recall": best_metrics["recall"],
            "threshold_analysis": analysis
        }
