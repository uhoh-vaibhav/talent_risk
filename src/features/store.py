"""
Feature Store Logic & Versioning Module
Manages feature store artifacts, versioning, and feature transformations for training & serving.
"""

import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from src.features.pipeline import fit_and_save_pipeline, load_feature_pipeline, FEATURE_SET_VERSION

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class FeatureStore:
    """Lightweight feature store interface for versioned dataset preparation."""

    def __init__(self, version: str = FEATURE_SET_VERSION):
        self.version = version
        self.pipeline = None
        self.feature_names = None

    def fit_transform_store(self, df_master: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """Fits feature pipeline on master dataset and saves version metadata."""
        logger.info("Fitting and transforming master feature set (version: %s)...", self.version)
        self.pipeline, self.feature_names = fit_and_save_pipeline(df_master, version=self.version)
        X_trans = self.pipeline.transform(df_master)
        return X_trans, self.feature_names

    def get_serving_features(self, df_input: pd.DataFrame) -> np.ndarray:
        """Transforms input employee profile DataFrame at serving time using loaded pipeline."""
        if self.pipeline is None:
            self.pipeline, self.feature_names = load_feature_pipeline(version=self.version)
        
        return self.pipeline.transform(df_input)


if __name__ == "__main__":
    df_master = pd.read_parquet(DATA_DIR / "unified_master.parquet")
    fs = FeatureStore()
    X_trans, feature_names = fs.fit_transform_store(df_master)
    print(f"Feature Store Successfully Prepared {X_trans.shape[0]} samples with {X_trans.shape[1]} features.")
