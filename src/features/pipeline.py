"""
Feature Engineering Pipeline Module
Defines reusable scikit-learn transformers and feature pipeline for training & serving parity.
"""

import os
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

logger = logging.getLogger(__name__)

FEATURE_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MODEL_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "features"

CATEGORICAL_FEATURES = ["department", "gender"]
NUMERICAL_FEATURES = [
    "age",
    "education_level",
    "tenure_years",
    "years_since_promotion",
    "num_trainings_last_year",
    "performance_rating",
    "kpi_met_above_80",
    "awards_won",
    "overtime_status",
    "satisfaction_score",
    "monthly_income",
    "stock_option_level"
]

FEATURE_SET_VERSION = "v1.0.0"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer creating derived interaction features."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Interaction features
        # 1. Income per tenure year
        tenure_safe = np.where(X_out["tenure_years"] == 0, 1.0, X_out["tenure_years"])
        X_out["income_per_tenure"] = X_out["monthly_income"] / tenure_safe

        # 2. Ratio of years since promotion to total tenure
        X_out["promotion_stagnation_index"] = X_out["years_since_promotion"] / tenure_safe

        # 3. High performer low satisfaction flag
        high_perf = (X_out["performance_rating"] >= 3.5) | (X_out["kpi_met_above_80"] == 1)
        low_sat = X_out["satisfaction_score"] <= 2.5
        X_out["flight_risk_signal"] = (high_perf & low_sat).astype(int)

        return X_out


ENGINEERED_NUMERICAL = NUMERICAL_FEATURES + ["income_per_tenure", "promotion_stagnation_index", "flight_risk_signal"]


def create_feature_pipeline() -> Pipeline:
    """Constructs the full feature preprocessing and scaling sklearn Pipeline."""

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, ENGINEERED_NUMERICAL),
            ("cat", cat_transformer, CATEGORICAL_FEATURES)
        ]
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline


def fit_and_save_pipeline(
    df: pd.DataFrame,
    version: str = FEATURE_SET_VERSION,
    out_dir: Path = MODEL_ARTIFACTS_DIR
) -> Tuple[Pipeline, List[str]]:
    """Fits the feature pipeline on DataFrame and saves joblib artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = create_feature_pipeline()
    
    # Fit pipeline
    pipeline.fit(df)

    # Extract feature names
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = ENGINEERED_NUMERICAL + cat_feature_names

    # Save pipeline joblib
    artifact_path = out_dir / f"feature_pipeline_{version}.joblib"
    joblib.dump({"pipeline": pipeline, "feature_names": feature_names, "version": version}, artifact_path)
    logger.info("Saved fitted feature pipeline %s to %s", version, artifact_path)

    return pipeline, feature_names


def load_feature_pipeline(
    version: str = FEATURE_SET_VERSION,
    artifacts_dir: Path = MODEL_ARTIFACTS_DIR
) -> Tuple[Pipeline, List[str]]:
    """Loads serialised feature pipeline from disk."""
    artifact_path = artifacts_dir / f"feature_pipeline_{version}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Feature pipeline artifact not found at {artifact_path}")
    
    data = joblib.load(artifact_path)
    logger.info("Loaded feature pipeline %s from %s", data["version"], artifact_path)
    return data["pipeline"], data["feature_names"]
