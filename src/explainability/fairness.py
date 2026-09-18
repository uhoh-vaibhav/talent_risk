"""
Bias & Fairness Audit Module
Calculates Disparate Impact Ratio and Demographic Parity metrics across protected groups.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DISPARATE_IMPACT_THRESHOLD = 0.80


class FairnessAuditor:
    """Audits machine learning predictions for algorithmic bias and disparate impact."""

    @staticmethod
    def audit_disparate_impact(
        df_eval: pd.DataFrame,
        protected_column: str,
        unprivileged_value: str,
        privileged_value: str,
        pred_column: str = "prediction"
    ) -> Dict[str, Any]:
        """Calculates Disparate Impact Ratio for a protected feature."""
        df_unpriv = df_eval[df_eval[protected_column] == unprivileged_value]
        df_priv = df_eval[df_eval[protected_column] == privileged_value]

        if len(df_unpriv) == 0 or len(df_priv) == 0:
            logger.warning("Insufficient samples to evaluate fairness for %s.", protected_column)
            return {"disparate_impact": 1.0, "status": "SKIPPED"}

        rate_unpriv = df_unpriv[pred_column].mean()
        rate_priv = df_priv[pred_column].mean()

        if rate_priv == 0:
            disparate_impact = 1.0
        else:
            disparate_impact = rate_unpriv / rate_priv

        passed = disparate_impact >= DISPARATE_IMPACT_THRESHOLD

        report = {
            "protected_feature": protected_column,
            "unprivileged_group": unprivileged_value,
            "privileged_group": privileged_value,
            "unprivileged_selection_rate": round(float(rate_unpriv), 4),
            "privileged_selection_rate": round(float(rate_priv), 4),
            "disparate_impact_ratio": round(float(disparate_impact), 4),
            "threshold": DISPARATE_IMPACT_THRESHOLD,
            "passed_fairness_audit": bool(passed)
        }

        if not passed:
            logger.warning("FAIRNESS VIOLATION DETECTED for %s! Disparate Impact = %.4f (Threshold: %.2f)",
                           protected_column, disparate_impact, DISPARATE_IMPACT_THRESHOLD)
        else:
            logger.info("Fairness audit passed for %s. Disparate Impact = %.4f", protected_column, disparate_impact)

        return report

    @classmethod
    def run_full_fairness_audit(
        cls,
        df_eval: pd.DataFrame,
        pred_column: str = "prediction"
    ) -> Dict[str, Any]:
        """Runs audit across Gender and Age attributes."""
        audits = []

        # Gender audit
        if "gender" in df_eval.columns:
            gender_audit = cls.audit_disparate_impact(
                df_eval, protected_column="gender", unprivileged_value="Female", privileged_value="Male", pred_column=pred_column
            )
            audits.append(gender_audit)

        # Age audit (<30 vs >=30)
        if "age" in df_eval.columns:
            df_eval_copy = df_eval.copy()
            df_eval_copy["age_group"] = np.where(df_eval_copy["age"] < 30, "<30", ">=30")
            age_audit = cls.audit_disparate_impact(
                df_eval_copy, protected_column="age_group", unprivileged_value="<30", privileged_value=">=30", pred_column=pred_column
            )
            audits.append(age_audit)

        all_passed = all(a.get("passed_fairness_audit", True) for a in audits)
        
        return {
            "overall_fairness_passed": all_passed,
            "subgroup_audits": audits
        }
