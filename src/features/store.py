"""
Feature Store Logic & Versioning Module
Manages model-specific feature pipelines for training & serving.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from src.features.pipeline import (
    fit_and_save_pipeline, load_feature_pipeline, FEATURE_SET_VERSION
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class FeatureStore:
    """Model-specific feature store for versioned dataset preparation."""

    def __init__(self, model_type: str = "attrition", version: str = FEATURE_SET_VERSION):
        self.model_type = model_type
        self.version = version
        self.pipeline = None
        self.feature_names = None

    def fit_and_save(self, df_train: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """Fits feature pipeline on TRAINING data only and saves version metadata."""
        logger.info("Fitting %s feature pipeline (version: %s)...", self.model_type, self.version)
        self.pipeline, self.feature_names = fit_and_save_pipeline(
            df_train, model_type=self.model_type, version=self.version
        )
        X_trans = self.pipeline.transform(df_train)
        return X_trans, self.feature_names

    def transform(self, df_input: pd.DataFrame) -> np.ndarray:
        """Transforms input data using loaded pipeline."""
        if self.pipeline is None:
            self.load()
        return self.pipeline.transform(df_input)

    def load(self):
        """Loads pre-fitted pipeline from disk."""
        self.pipeline, self.feature_names = load_feature_pipeline(
            model_type=self.model_type, version=self.version
        )

    # Backward compatibility aliases
    fit_transform_store = fit_and_save
    get_serving_features = transform



if __name__ == "__main__":
    df_attrition = pd.read_parquet(DATA_DIR / "processed_attrition.parquet")
    fs_attr = FeatureStore(model_type="attrition")
    X_attr, names_attr = fs_attr.fit_and_save(df_attrition)
    print(f"Attrition Feature Store: Prepared {X_attr.shape[0]} samples with {X_attr.shape[1]} features.")

    df_promo = pd.read_parquet(DATA_DIR / "processed_promotion.parquet")
    fs_promo = FeatureStore(model_type="promotion")
    X_promo, names_promo = fs_promo.fit_and_save(df_promo)
    print(f"Promotion Feature Store: Prepared {X_promo.shape[0]} samples with {X_promo.shape[1]} features.")
