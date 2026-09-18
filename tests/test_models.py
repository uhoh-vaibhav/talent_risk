"""
Unit tests for model comparison, evaluation metrics, threshold optimization, and 2x2 matrix assignment.
"""

import pytest
import numpy as np
from src.models.comparison import evaluate_classification, ModelComparer
from src.models.threshold import ThresholdOptimizer
from src.models.matrix import TalentMatrixEngine


def test_evaluate_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.1])

    metrics = evaluate_classification(y_true, y_proba, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["f1_score"] == 1.0


def test_threshold_optimizer():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    res = ThresholdOptimizer.find_optimal_threshold(y_true, y_proba)

    assert "optimal_threshold" in res
    assert "optimal_f1" in res
    assert 0.0 < res["optimal_threshold"] < 1.0
    assert len(res["threshold_analysis"]) > 0


def test_model_comparer_candidates():
    comparer = ModelComparer(class_weight_ratio=2.0)
    candidates = comparer._build_candidates()

    assert "Logistic Regression" in candidates
    assert "Random Forest" in candidates
    assert "XGBoost" in candidates


def test_talent_matrix_quadrants():
    # Quadrant 1: High Attrition, High Promotion
    q1 = TalentMatrixEngine.evaluate_quadrant(0.75, 0.85, 0.5, 0.5)
    assert q1["quadrant"] == TalentMatrixEngine.QUADRANT_RETAIN_URGENTLY
    assert q1["priority_level"] == "CRITICAL"
    assert "disclaimer" in q1

    # Quadrant 2: Low Attrition, High Promotion
    q2 = TalentMatrixEngine.evaluate_quadrant(0.20, 0.80, 0.5, 0.5)
    assert q2["quadrant"] == TalentMatrixEngine.QUADRANT_INVEST_FAST_TRACK
    assert q2["priority_level"] == "HIGH"

    # Quadrant 3: High Attrition, Low Promotion
    q3 = TalentMatrixEngine.evaluate_quadrant(0.80, 0.30, 0.5, 0.5)
    assert q3["quadrant"] == TalentMatrixEngine.QUADRANT_MONITOR_ENGAGE
    assert q3["priority_level"] == "MEDIUM"

    # Quadrant 4: Low Attrition, Low Promotion
    q4 = TalentMatrixEngine.evaluate_quadrant(0.15, 0.25, 0.5, 0.5)
    assert q4["quadrant"] == TalentMatrixEngine.QUADRANT_CORE_PERFORMER
    assert q4["priority_level"] == "LOW"


def test_talent_matrix_custom_thresholds():
    # With 0.7 threshold, 0.65 is LOW
    q_custom = TalentMatrixEngine.evaluate_quadrant(0.65, 0.65, attrition_threshold=0.7, promotion_threshold=0.7)
    assert q_custom["quadrant"] == TalentMatrixEngine.QUADRANT_CORE_PERFORMER
