"""
SHAP Explainability Module
Generates SHAP values for global feature importance and top individual prediction drivers.
"""

import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Computes SHAP feature attributions for tree models, linear models, and ensemble baselines."""

    def __init__(self, model, background_or_features=None, feature_names: Optional[List[str]] = None):
        self.model = model
        if isinstance(background_or_features, (list, tuple)) and (len(background_or_features) == 0 or isinstance(background_or_features[0], str)):
            self.feature_names = list(background_or_features)
            self.background_data = None
        else:
            self.background_data = background_or_features
            self.feature_names = feature_names or []
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer."""
        try:
            # Tree explainer for XGBoost/Tree models
            self.explainer = shap.TreeExplainer(self.model)
            return
        except Exception:
            pass

        # Try LinearExplainer for models with coef_
        if hasattr(self.model, "coef_"):
            try:
                n_features = self.model.coef_.shape[1] if self.model.coef_.ndim > 1 else len(self.model.coef_)
                bg = self.background_data if self.background_data is not None else np.zeros((1, n_features))
                self.explainer = shap.LinearExplainer(self.model, bg)
                return
            except Exception as e:
                logger.warning("LinearExplainer init failed: %s", e)

        # Fallback to general Explainer
        try:
            if hasattr(self.model, "predict_proba") and self.background_data is not None:
                self.explainer = shap.Explainer(self.model.predict_proba, self.background_data)
            else:
                self.explainer = shap.Explainer(self.model)
        except Exception as e:
            logger.warning("Fallback SHAP explainer failed: %s", e)
            self.explainer = None

    def explain_sample(self, X_sample: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Computes top SHAP feature drivers for a single employee prediction sample."""
        if self.explainer is None:
            return []

        if hasattr(X_sample, "values"):
            X_sample = X_sample.values
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-output or single array
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif hasattr(shap_values, "values"):
                vals = shap_values.values
                if vals.ndim == 3:
                    shap_vals = vals[0, :, 1]
                elif vals.ndim == 2:
                    shap_vals = vals[0]
                else:
                    shap_vals = vals
            elif hasattr(shap_values, "ndim"):
                if shap_values.ndim == 3:
                    shap_vals = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
            else:
                shap_vals = np.array(shap_values)

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

    def compute_global_importance(self, X_sample: np.ndarray, top_k: int = 15) -> List[Dict[str, Any]]:
        """Computes global SHAP feature importance from a sample of records."""
        if self.explainer is None:
            return []

        if hasattr(X_sample, "values"):
            X_sample = X_sample.values

        try:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif hasattr(shap_values, "values"):
                vals = shap_values.values
                if vals.ndim == 3:
                    shap_vals = vals[:, :, 1]
                else:
                    shap_vals = vals
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            order = np.argsort(mean_abs_shap)[::-1][:top_k]

            importance = []
            for rank, idx in enumerate(order):
                feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                importance.append({
                    "feature": feat_name,
                    "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                    "rank": rank + 1
                })
            return importance
        except Exception as e:
            logger.error("Error computing global SHAP importance: %s", str(e))
            return []
