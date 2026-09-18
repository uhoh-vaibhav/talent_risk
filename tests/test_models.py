"""
Unit tests for model training, metrics evaluation, and 2x2 talent matrix assignment.
"""

import pytest
import numpy as np
from src.models.baseline import evaluate_classification
from src.models.matrix import TalentMatrixEngine


def test_evaluate_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.1])

    metrics = evaluate_classification(y_true, y_proba, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["f1_score"] == 1.0


def test_talent_matrix_quadrants():
    # Quadrant 1: High Attrition, High Promotion
    q1 = TalentMatrixEngine.evaluate_quadrant(0.75, 0.85)
    assert q1["quadrant"] == TalentMatrixEngine.QUADRANT_RETAIN_URGENTLY
    assert q1["priority_level"] == "CRITICAL"

    # Quadrant 2: Low Attrition, High Promotion
    q2 = TalentMatrixEngine.evaluate_quadrant(0.20, 0.80)
    assert q2["quadrant"] == TalentMatrixEngine.QUADRANT_INVEST_FAST_TRACK
    assert q2["priority_level"] == "HIGH"

    # Quadrant 3: High Attrition, Low Promotion
    q3 = TalentMatrixEngine.evaluate_quadrant(0.80, 0.30)
    assert q3["quadrant"] == TalentMatrixEngine.QUADRANT_MONITOR_ENGAGE

    # Quadrant 4: Low Attrition, Low Promotion
    q4 = TalentMatrixEngine.evaluate_quadrant(0.15, 0.25)
    assert q4["quadrant"] == TalentMatrixEngine.QUADRANT_CORE_PERFORMER
