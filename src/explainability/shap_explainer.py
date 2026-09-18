"""
SHAP Explainability Module
Generates SHAP values for global feature importance and top individual prediction drivers.
"""

import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Computes SHAP feature attributions for tree models and linear baselines."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer."""
        try:
            # Tree explainer for XGBoost/LightGBM
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to Kernel or Linear explainer
            logger.info("Using Kernel / Linear SHAP explainer fallback.")
            self.explainer = shap.Explainer(self.model)

    def explain_sample(self, X_sample: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Computes top SHAP feature drivers for a single employee prediction sample."""
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-output or single array
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif shap_values.ndim == 3:
                shap_vals = shap_values[0, :, 1]
            elif shap_values.ndim == 2:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            # Sort by absolute impact
            abs_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

            drivers = []
            for idx in abs_indices:
                feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                impact_val = float(shap_vals[idx])
                feature_val = float(X_sample[0, idx])
                drivers.append({
                    "feature_name": feat_name,
                    "shap_impact": round(impact_val, 4),
                    "feature_value": round(feature_val, 4),
                    "direction": "increases_risk" if impact_val > 0 else "decreases_risk"
                })
            return drivers
        except Exception as e:
            logger.error("Error computing SHAP values: %s", str(e))
            return []
