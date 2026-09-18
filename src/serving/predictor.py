"""
Predictor Engine Module
Handles inference, feature transformation, dual-model scoring, SHAP driver computation, and matrix mapping.
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.features.store import FeatureStore
from src.models.matrix import TalentMatrixEngine
from src.serving.schemas import EmployeeProfileRequest, TalentScoreResponse, ShapDriver

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved_models"


class TalentPredictorEngine:
    """Production predictor engine executing end-to-end model inference."""

    def __init__(self):
        self.feature_store = FeatureStore()
        self.attrition_model = None
        self.promotion_model = None
        self.attrition_explainer = None
        self.promotion_explainer = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads saved models, pipelines, and SHAP explainers."""
        try:
            attr_artifact = joblib.load(SAVED_MODELS_DIR / "attrition_xgb_model.joblib")
            promo_artifact = joblib.load(SAVED_MODELS_DIR / "promotion_xgb_model.joblib")

            self.attrition_model = attr_artifact["model"]
            self.promotion_model = promo_artifact["model"]
            self.feature_names = attr_artifact["feature_names"]

            # Load explainers if available
            attr_exp_path = SAVED_MODELS_DIR / "attrition_shap_explainer.joblib"
            promo_exp_path = SAVED_MODELS_DIR / "promotion_shap_explainer.joblib"

            if attr_exp_path.exists():
                self.attrition_explainer = joblib.load(attr_exp_path)
            if promo_exp_path.exists():
                self.promotion_explainer = joblib.load(promo_exp_path)

            logger.info("Successfully loaded dual prediction models and explainers.")
        except Exception as e:
            logger.error("Error loading model artifacts: %s", str(e))

    def predict_single(self, profile: EmployeeProfileRequest) -> TalentScoreResponse:
        """Runs dual prediction for a single employee profile."""
        df_input = pd.DataFrame([profile.model_dump()])
        X_trans = self.feature_store.get_serving_features(df_input)

        # Dual probabilities
        p_attrition = float(self.attrition_model.predict_proba(X_trans)[0, 1])
        p_promotion = float(self.promotion_model.predict_proba(X_trans)[0, 1])

        # SHAP drivers
        attr_drivers_raw = self.attrition_explainer.explain_sample(X_trans, top_k=3) if self.attrition_explainer else []
        promo_drivers_raw = self.promotion_explainer.explain_sample(X_trans, top_k=3) if self.promotion_explainer else []

        attr_drivers = [ShapDriver(**d) for d in attr_drivers_raw]
        promo_drivers = [ShapDriver(**d) for d in promo_drivers_raw]

        # 2x2 Matrix mapping
        matrix_res = TalentMatrixEngine.evaluate_quadrant(p_attrition, p_promotion)

        return TalentScoreResponse(
            employee_id=profile.employee_id,
            attrition_risk_score=matrix_res["attrition_risk_score"],
            promotion_readiness_score=matrix_res["promotion_readiness_score"],
            quadrant=matrix_res["quadrant"],
            action_code=matrix_res["action_code"],
            priority_level=matrix_res["priority_level"],
            hr_action_recommendation=matrix_res["hr_action_recommendation"],
            top_attrition_drivers=attr_drivers,
            top_promotion_drivers=promo_drivers
        )

    def predict_batch(self, profiles: List[EmployeeProfileRequest]) -> List[TalentScoreResponse]:
        """Runs batch scoring for multiple employee profiles."""
        return [self.predict_single(p) for p in profiles]
