"""
Data & Prediction Drift Detection Module
Calculates Kolmogorov-Smirnov (KS) statistic and Population Stability Index (PSI) to detect distribution shifts.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current distribution."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(reference, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    ref_counts, _ = np.histogram(reference, bins=buckets)
    curr_counts, _ = np.histogram(current, bins=buckets)

    ref_pct = np.maximum(ref_counts / len(reference), 1e-4)
    curr_pct = np.maximum(curr_counts / len(current), 1e-4)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


class DriftDetector:
    """Monitors feature and prediction distribution drift against baseline distributions."""

    @staticmethod
    def detect_feature_drift(
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        numerical_features: List[str],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Runs Kolmogorov-Smirnov test and PSI across numerical features."""
        drift_results = []
        drift_count = 0

        for col in numerical_features:
            if col not in df_reference.columns or col not in df_current.columns:
                continue

            ref_vals = df_reference[col].dropna().values
            curr_vals = df_current[col].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            # KS-test
            ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
            psi_score = calculate_psi(ref_vals, curr_vals)

            is_drifted = bool(p_value < alpha or psi_score > 0.25)
            if is_drifted:
                drift_count += 1

            drift_results.append({
                "feature": col,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "psi_score": round(float(psi_score), 4),
                "drift_detected": is_drifted
            })

        drift_percentage = (drift_count / max(len(numerical_features), 1)) * 100
        should_retrain = drift_percentage > 20.0 or any(r["psi_score"] > 0.25 for r in drift_results)

        logger.info("Drift Audit Completed: %d/%d features drifted (Retrain Triggered: %s)",
                    drift_count, len(numerical_features), should_retrain)

        return {
            "total_features_monitored": len(numerical_features),
            "drifted_feature_count": drift_count,
            "drift_percentage": round(drift_percentage, 2),
            "retraining_alert_triggered": should_retrain,
            "feature_details": drift_results
        }
