"""
Unit tests for feature engineering pipelines and feature stores.
"""

import pytest
import pandas as pd
import numpy as np
from src.ingestion.download import generate_synthetic_ibm_attrition, generate_synthetic_hr_promotion
from src.ingestion.schema_mapper import map_ibm_to_processed, map_promotion_to_processed
from src.features.pipeline import (
    create_attrition_pipeline,
    create_promotion_pipeline,
    create_pipeline
)
from src.features.store import FeatureStore


def test_attrition_feature_pipeline():
    df_raw = generate_synthetic_ibm_attrition(num_samples=60)
    df_proc = map_ibm_to_processed(df_raw)

    pipeline = create_attrition_pipeline()
    pipeline.fit(df_proc)
    X_trans = pipeline.transform(df_proc)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 60
    assert X_trans.shape[1] >= 20


def test_promotion_feature_pipeline():
    df_raw = generate_synthetic_hr_promotion(num_samples=60)
    df_proc = map_promotion_to_processed(df_raw)

    pipeline = create_promotion_pipeline()
    pipeline.fit(df_proc)
    X_trans = pipeline.transform(df_proc)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 60
    assert X_trans.shape[1] >= 15


def test_create_pipeline_factory():
    pipe_attr = create_pipeline("attrition")
    pipe_promo = create_pipeline("promotion")

    assert pipe_attr is not None
    assert pipe_promo is not None

    with pytest.raises(ValueError):
        create_pipeline("invalid_model")


def test_feature_store_attrition_serving():
    df_raw = generate_synthetic_ibm_attrition(num_samples=50)
    df_proc = map_ibm_to_processed(df_raw)

    fs = FeatureStore(model_type="attrition")
    X_trans, feature_names = fs.fit_and_save(df_proc)

    assert X_trans.shape[0] == 50
    assert len(feature_names) == X_trans.shape[1]

    # Test transforming sample records
    sample_df = df_proc.iloc[:5].copy()
    X_serving = fs.transform(sample_df)
    assert X_serving.shape[0] == 5
    assert X_serving.shape[1] == len(feature_names)


def test_feature_store_promotion_serving():
    df_raw = generate_synthetic_hr_promotion(num_samples=50)
    df_proc = map_promotion_to_processed(df_raw)

    fs = FeatureStore(model_type="promotion")
    X_trans, feature_names = fs.fit_and_save(df_proc)

    assert X_trans.shape[0] == 50
    assert len(feature_names) == X_trans.shape[1]

    sample_df = df_proc.iloc[:5].copy()
    X_serving = fs.transform(sample_df)
    assert X_serving.shape[0] == 5
    assert X_serving.shape[1] == len(feature_names)
