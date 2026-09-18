"""
Predictor Engine Module
Handles dual-model inference, feature transformation, SHAP driver computation, and matrix mapping.
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.features.store import FeatureStore
from src.models.matrix import TalentMatrixEngine
from src.serving.schemas import EmployeeProfileRequest, TalentScoreResponse, ShapDriver

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved_models"


class TalentPredictorEngine:
    """Production predictor engine executing end-to-end model inference across dual models."""

    def __init__(self):
        self.attrition_store = FeatureStore(model_type="attrition")
        self.promotion_store = FeatureStore(model_type="promotion")
        self.attrition_model = None
        self.promotion_model = None
        self.attrition_explainer = None
        self.promotion_explainer = None
        self.attrition_threshold = 0.50
        self.promotion_threshold = 0.50
        self.attrition_feature_names = []
        self.promotion_feature_names = []
        self.model_metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads saved models, pipelines, and SHAP explainers."""
        try:
            # Load feature stores
            self.attrition_store.load()
            self.promotion_store.load()

            # Load models
            attr_artifact = joblib.load(SAVED_MODELS_DIR / "attrition_model.joblib")
            promo_artifact = joblib.load(SAVED_MODELS_DIR / "promotion_model.joblib")

            self.attrition_model = attr_artifact["model"]
            self.promotion_model = promo_artifact["model"]
            self.attrition_feature_names = attr_artifact.get("feature_names", self.attrition_store.feature_names)
            self.promotion_feature_names = promo_artifact.get("feature_names", self.promotion_store.feature_names)

            # Extract calibrated thresholds
            if "threshold_info" in attr_artifact:
                self.attrition_threshold = float(attr_artifact["threshold_info"].get("optimal_threshold", 0.50))
            if "threshold_info" in promo_artifact:
                self.promotion_threshold = float(promo_artifact["threshold_info"].get("optimal_threshold", 0.50))

            # Load explainers
            attr_exp_path = SAVED_MODELS_DIR / "attrition_explainer.joblib"
            promo_exp_path = SAVED_MODELS_DIR / "promotion_explainer.joblib"

            if attr_exp_path.exists():
                self.attrition_explainer = joblib.load(attr_exp_path)
            if promo_exp_path.exists():
                self.promotion_explainer = joblib.load(promo_exp_path)

            self.model_metadata = {
                "attrition": {
                    "best_model": attr_artifact.get("best_model_name", "Unknown"),
                    "threshold": self.attrition_threshold,
                    "metrics": attr_artifact.get("metrics", {})
                },
                "promotion": {
                    "best_model": promo_artifact.get("best_model_name", "Unknown"),
                    "threshold": self.promotion_threshold,
                    "metrics": promo_artifact.get("metrics", {})
                }
            }

            logger.info("Successfully initialized TalentPredictorEngine with dual models and calibrated thresholds.")
        except Exception as e:
            logger.error("Error loading model artifacts in TalentPredictorEngine: %s", str(e))

    def _prepare_attrition_record(self, p: Dict[str, Any]) -> pd.DataFrame:
        """Prepares input DataFrame matching Attrition schema."""
        row = {
            "age": int(p.get("age", 30)),
            "education_level": int(p.get("education_level", 3)),
            "monthly_income": float(p.get("monthly_income", 5000.0)),
            "job_satisfaction": float(p.get("job_satisfaction") or p.get("satisfaction_score", 3.0)),
            "environment_satisfaction": float(p.get("environment_satisfaction", 3.0)),
            "work_life_balance": float(p.get("work_life_balance", 3.0)),
            "performance_rating": float(p.get("performance_rating", 3.0)),
            "stock_option_level": int(p.get("stock_option_level", 1)),
            "tenure_years": float(p.get("tenure_years", 3.0)),
            "years_since_promotion": float(p.get("years_since_promotion", 1.0)),
            "training_times_last_year": int(p.get("training_times_last_year") or p.get("num_trainings_last_year", 2)),
            "num_companies_worked": int(p.get("num_companies_worked", 1)),
            "distance_from_home": float(p.get("distance_from_home", 5.0)),
            "job_involvement": float(p.get("job_involvement", 3.0)),
            "job_level": int(p.get("job_level", 2)),
            "relationship_satisfaction": float(p.get("relationship_satisfaction", 3.0)),
            "percent_salary_hike": float(p.get("percent_salary_hike", 14.0)),
            "overtime": int(p.get("overtime") if p.get("overtime") is not None else p.get("overtime_status", 0)),
            "department": str(p.get("department", "Operations")),
            "gender": str(p.get("gender", "Female")),
            "marital_status": str(p.get("marital_status", "Single"))
        }
        return pd.DataFrame([row])

    def _prepare_promotion_record(self, p: Dict[str, Any]) -> pd.DataFrame:
        """Prepares input DataFrame matching Promotion schema."""
        row = {
            "age": int(p.get("age", 30)),
            "education_level": int(p.get("education_level", 3)),
            "no_of_trainings": int(p.get("no_of_trainings") or p.get("num_trainings_last_year", 2)),
            "previous_year_rating": float(p.get("previous_year_rating") or p.get("performance_rating", 3.0)),
            "length_of_service": float(p.get("length_of_service") or p.get("tenure_years", 3.0)),
            "kpis_met_above_80": int(p.get("kpis_met_above_80") if p.get("kpis_met_above_80") is not None else p.get("kpi_met_above_80", 1)),
            "awards_won": int(p.get("awards_won", 0)),
            "avg_training_score": float(p.get("avg_training_score", 65.0)),
            "department": str(p.get("department", "Operations")),
            "gender": str(p.get("gender", "Female")),
            "recruitment_channel": str(p.get("recruitment_channel", "sourcing"))
        }
        return pd.DataFrame([row])

    def predict_single(
        self,
        profile: EmployeeProfileRequest,
        attr_threshold: Optional[float] = None,
        promo_threshold: Optional[float] = None
    ) -> TalentScoreResponse:
        """Runs dual prediction for a single employee profile."""
        p_dict = profile.model_dump()

        # 1. Attrition scoring
        df_attr = self._prepare_attrition_record(p_dict)
        X_attr_trans = self.attrition_store.transform(df_attr)
        p_attrition = float(self.attrition_model.predict_proba(X_attr_trans)[0, 1])

        # 2. Promotion scoring
        df_promo = self._prepare_promotion_record(p_dict)
        X_promo_trans = self.promotion_store.transform(df_promo)
        p_promotion = float(self.promotion_model.predict_proba(X_promo_trans)[0, 1])

        # 3. SHAP drivers
        attr_drivers_raw = self.attrition_explainer.explain_sample(X_attr_trans, top_k=3) if self.attrition_explainer else []
        promo_drivers_raw = self.promotion_explainer.explain_sample(X_promo_trans, top_k=3) if self.promotion_explainer else []

        attr_drivers = [ShapDriver(**d) for d in attr_drivers_raw]
        promo_drivers = [ShapDriver(**d) for d in promo_drivers_raw]

        # 4. Threshold application & 2x2 Matrix mapping
        th_attr = attr_threshold if attr_threshold is not None else self.attrition_threshold
        th_promo = promo_threshold if promo_threshold is not None else self.promotion_threshold

        matrix_res = TalentMatrixEngine.evaluate_quadrant(
            p_attrition=p_attrition,
            p_promotion=p_promotion,
            attrition_threshold=th_attr,
            promotion_threshold=th_promo
        )

        return TalentScoreResponse(
            employee_id=profile.employee_id,
            attrition_risk_score=matrix_res["attrition_risk_score"],
            promotion_readiness_score=matrix_res["promotion_readiness_score"],
            quadrant=matrix_res["quadrant"],
            action_code=matrix_res["action_code"],
            priority_level=matrix_res["priority_level"],
            hr_action_recommendation=matrix_res["hr_action_recommendation"],
            attrition_threshold_applied=matrix_res["attrition_threshold"],
            promotion_threshold_applied=matrix_res["promotion_threshold"],
            top_attrition_drivers=attr_drivers,
            top_promotion_drivers=promo_drivers,
            disclaimer=matrix_res.get("disclaimer")
        )

    def predict_batch(
        self,
        profiles: List[EmployeeProfileRequest],
        attr_threshold: Optional[float] = None,
        promo_threshold: Optional[float] = None
    ) -> List[TalentScoreResponse]:
        """Runs batch scoring for multiple employee profiles."""
        return [self.predict_single(p, attr_threshold, promo_threshold) for p in profiles]
