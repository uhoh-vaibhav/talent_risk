"""
Unit tests for data ingestion, validation, and schema mapping.
"""

import pytest
import pandas as pd
import numpy as np
from src.ingestion.download import generate_synthetic_ibm_attrition, generate_synthetic_hr_promotion
from src.ingestion.validator import DataValidator
from src.ingestion.schema_mapper import map_ibm_to_unified, map_promotion_to_unified, pseudonymize_id


def test_synthetic_data_generation():
    df_ibm = generate_synthetic_ibm_attrition(num_samples=100)
    df_promo = generate_synthetic_hr_promotion(num_samples=200)

    assert len(df_ibm) == 100
    assert len(df_promo) == 200
    assert "Attrition" in df_ibm.columns
    assert "is_promoted" in df_promo.columns


def test_pseudonymize_id():
    id1 = pseudonymize_id("IBM", "1001")
    id2 = pseudonymize_id("IBM", "1001")
    id3 = pseudonymize_id("IBM", "1002")

    assert id1.startswith("EMP_")
    assert id1 == id2
    assert id1 != id3


def test_schema_mapping_and_validation():
    df_ibm_raw = generate_synthetic_ibm_attrition(num_samples=50)
    df_ibm_unified = map_ibm_to_unified(df_ibm_raw)

    valid, missing = DataValidator.validate_unified_schema(df_ibm_unified)
    assert valid is True
    assert len(missing) == 0
    assert df_ibm_unified["age"].min() >= 18
    assert "monthly_income" in df_ibm_unified.columns
