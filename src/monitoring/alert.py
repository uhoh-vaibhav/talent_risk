"""
Alerting Module
Sends alerts and triggers automated pipeline retraining when drift or fairness thresholds are violated.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RetrainingAlertManager:
    """Manages drift alerts and triggers model retraining routines."""

    @staticmethod
    def evaluate_and_alert(drift_report: Dict[str, Any], fairness_report: Dict[str, Any]) -> bool:
        """Evaluates drift and fairness reports to trigger alerts."""
        retrain_needed = False

        if drift_report.get("retraining_alert_triggered", False):
            logger.warning("[ALERT] Data drift detected across %.2f%% of features! Retraining recommended.",
                           drift_report.get("drift_percentage", 0.0))
            retrain_needed = True

        if not fairness_report.get("overall_fairness_passed", True):
            logger.error("[CRITICAL ALERT] Algorithmic fairness audit failed! Disparate impact threshold violated.")
            retrain_needed = True

        if not retrain_needed:
            logger.info("Monitoring status clean. No retraining required.")

        return retrain_needed
