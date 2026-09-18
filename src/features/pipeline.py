"""
Feature Engineering Pipeline Module
Defines separate scikit-learn pipelines for Attrition and Promotion models.
Each model has its own feature set — no artificial feature unification.
"""

import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ingestion.schema_mapper import (
    ATTRITION_NUMERICAL_FEATURES, ATTRITION_CATEGORICAL_FEATURES,
    PROMOTION_NUMERICAL_FEATURES, PROMOTION_CATEGORICAL_FEATURES
)

logger = logging.getLogger(__name__)

PIPELINE_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "pipelines"
FEATURE_SET_VERSION = "v1.0.0"

class AttritionFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        tenure_safe = np.where(X_out["tenure_years"] == 0, 1.0, X_out["tenure_years"])
        
        X_out["income_per_tenure"] = X_out["monthly_income"] / tenure_safe
        X_out["promotion_stagnation"] = X_out["years_since_promotion"] / tenure_safe
        X_out["satisfaction_composite"] = (
            X_out["job_satisfaction"] + 
            X_out["environment_satisfaction"] + 
            X_out["work_life_balance"]
        ) / 3.0
        X_out["flight_risk_signal"] = (
            (X_out["performance_rating"] >= 3.5) & 
            (X_out["satisfaction_composite"] <= 2.0)
        ).astype(int)
        
        return X_out

class PromotionFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out["training_effectiveness"] = X_out["avg_training_score"] * X_out["kpis_met_above_80"] / 100.0
        return X_out


ATTRITION_DERIVED = ["income_per_tenure", "promotion_stagnation", "satisfaction_composite", "flight_risk_signal"]
ATTRITION_ENGINEERED_NUMERICAL = ATTRITION_NUMERICAL_FEATURES + ATTRITION_DERIVED

PROMOTION_DERIVED = ["training_effectiveness"]
PROMOTION_ENGINEERED_NUMERICAL = PROMOTION_NUMERICAL_FEATURES + PROMOTION_DERIVED


def _build_column_transformer(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

def create_attrition_pipeline() -> Pipeline:
    preprocessor = _build_column_transformer(ATTRITION_ENGINEERED_NUMERICAL, ATTRITION_CATEGORICAL_FEATURES)
    return Pipeline(steps=[
        ("feature_engineer", AttritionFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

def create_promotion_pipeline() -> Pipeline:
    preprocessor = _build_column_transformer(PROMOTION_ENGINEERED_NUMERICAL, PROMOTION_CATEGORICAL_FEATURES)
    return Pipeline(steps=[
        ("feature_engineer", PromotionFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

def create_pipeline(model_type: str) -> Pipeline:
    if model_type == "attrition":
        return create_attrition_pipeline()
    elif model_type == "promotion":
        return create_promotion_pipeline()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Backward compatibility alias
create_feature_pipeline = create_attrition_pipeline


def fit_and_save_pipeline(
    df: pd.DataFrame,
    model_type: str,
    version: str = FEATURE_SET_VERSION,
    out_dir: Path = PIPELINE_ARTIFACTS_DIR
) -> Tuple[Pipeline, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = create_pipeline(model_type)
    
    pipeline.fit(df)

    cat_features = ATTRITION_CATEGORICAL_FEATURES if model_type == "attrition" else PROMOTION_CATEGORICAL_FEATURES
    num_features = ATTRITION_ENGINEERED_NUMERICAL if model_type == "attrition" else PROMOTION_ENGINEERED_NUMERICAL

    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = num_features + cat_feature_names

    artifact_path = out_dir / f"{model_type}_feature_pipeline_{version}.joblib"
    joblib.dump({"pipeline": pipeline, "feature_names": feature_names, "version": version, "model_type": model_type}, artifact_path)
    logger.info("Saved fitted %s feature pipeline %s to %s", model_type, version, artifact_path)

    return pipeline, feature_names

def load_feature_pipeline(
    model_type: str,
    version: str = FEATURE_SET_VERSION,
    artifacts_dir: Path = PIPELINE_ARTIFACTS_DIR
) -> Tuple[Pipeline, List[str]]:
    artifact_path = artifacts_dir / f"{model_type}_feature_pipeline_{version}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Feature pipeline artifact not found at {artifact_path}")
    
    data = joblib.load(artifact_path)
    logger.info("Loaded %s feature pipeline %s from %s", model_type, data["version"], artifact_path)
    return data["pipeline"], data["feature_names"]
