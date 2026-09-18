"""
Unit tests for feature engineering pipeline and feature store.
"""

import pytest
import pandas as pd
import numpy as np
from src.ingestion.download import generate_synthetic_ibm_attrition
from src.ingestion.schema_mapper import map_ibm_to_unified
from src.features.pipeline import create_feature_pipeline, fit_and_save_pipeline
from src.features.store import FeatureStore


def test_feature_pipeline_transformation():
    df_raw = generate_synthetic_ibm_attrition(num_samples=60)
    df_unified = map_ibm_to_unified(df_raw)

    pipeline = create_feature_pipeline()
    pipeline.fit(df_unified)
    X_trans = pipeline.transform(df_unified)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 60
    assert X_trans.shape[1] >= 15


def test_feature_store_serving_transformation():
    df_raw = generate_synthetic_ibm_attrition(num_samples=40)
    df_unified = map_ibm_to_unified(df_raw)

    fs = FeatureStore(version="test_v1")
    fs.fit_transform_store(df_unified)

    # Transform new sample profile
    sample_df = df_unified.iloc[:5].copy()
    X_serving = fs.get_serving_features(sample_df)

    assert X_serving.shape[0] == 5
    assert X_serving.shape[1] == len(fs.feature_names)
