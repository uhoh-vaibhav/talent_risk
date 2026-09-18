"""
Unit tests for bias and disparate impact fairness auditor.
"""

import pytest
import pandas as pd
from src.explainability.fairness import FairnessAuditor, DISPARATE_IMPACT_THRESHOLD


def test_disparate_impact_audit_pass():
    df_eval = pd.DataFrame({
        "gender": ["Male"] * 100 + ["Female"] * 100,
        "prediction": [1] * 50 + [0] * 50 + [1] * 45 + [0] * 55
    })

    audit = FairnessAuditor.audit_disparate_impact(
        df_eval, protected_column="gender", unprivileged_value="Female", privileged_value="Male"
    )

    # 45% vs 50% = 0.90 ratio >= 0.80 threshold
    assert audit["disparate_impact_ratio"] == 0.90
    assert audit["passed_fairness_audit"] is True


def test_disparate_impact_audit_fail():
    df_eval = pd.DataFrame({
        "gender": ["Male"] * 100 + ["Female"] * 100,
        "prediction": [1] * 60 + [0] * 40 + [1] * 20 + [0] * 80
    })

    audit = FairnessAuditor.audit_disparate_impact(
        df_eval, protected_column="gender", unprivileged_value="Female", privileged_value="Male"
    )

    # 20% vs 60% = 0.3333 ratio < 0.80 threshold
    assert audit["disparate_impact_ratio"] < DISPARATE_IMPACT_THRESHOLD
    assert audit["passed_fairness_audit"] is False
