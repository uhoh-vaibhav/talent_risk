"""
Unit tests for data ingestion, validation, and schema mapping.
"""

import pytest
import pandas as pd
import numpy as np
from src.ingestion.download import (
    generate_synthetic_ibm_attrition,
    generate_synthetic_hr_promotion,
    get_data_mode
)
from src.ingestion.validator import DataValidator
from src.ingestion.schema_mapper import (
    map_ibm_to_processed,
    map_promotion_to_processed,
    pseudonymize_id,
    map_department_name
)


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


def test_map_department_name():
    assert map_department_name("Research & Development") == "R&D"
    assert map_department_name("sales department") == "Sales"
    assert map_department_name("Technology Dept") == "Technology"
    assert map_department_name("Human Resources") == "Human Resources"
    assert map_department_name("Unknown Dept") == "Operations"


def test_ibm_schema_mapping_and_validation():
    df_ibm_raw = generate_synthetic_ibm_attrition(num_samples=50)
    valid_raw, missing_raw = DataValidator.validate_raw_ibm_schema(df_ibm_raw)
    assert valid_raw is True
    assert len(missing_raw) == 0

    df_ibm_proc = map_ibm_to_processed(df_ibm_raw)
    valid_proc, missing_proc = DataValidator.validate_processed_attrition_schema(df_ibm_proc)
    assert valid_proc is True
    assert len(missing_proc) == 0
    assert "monthly_income" in df_ibm_proc.columns
    assert "overtime" in df_ibm_proc.columns
    assert "target_attrition" in df_ibm_proc.columns
    # Ensure fabricated promotion columns are NOT in attrition schema
    assert "kpis_met_above_80" not in df_ibm_proc.columns


def test_promotion_schema_mapping_and_validation():
    df_promo_raw = generate_synthetic_hr_promotion(num_samples=50)
    valid_raw, missing_raw = DataValidator.validate_raw_promotion_schema(df_promo_raw)
    assert valid_raw is True
    assert len(missing_raw) == 0

    df_promo_proc = map_promotion_to_processed(df_promo_raw)
    valid_proc, missing_proc = DataValidator.validate_processed_promotion_schema(df_promo_proc)
    assert valid_proc is True
    assert len(missing_proc) == 0
    assert "avg_training_score" in df_promo_proc.columns
    assert "target_promotion" in df_promo_proc.columns
    # Ensure fabricated columns are NOT in promotion schema
    assert "monthly_income" not in df_promo_proc.columns
    assert "overtime" not in df_promo_proc.columns


def test_validator_detects_negative_tenure():
    df_invalid = pd.DataFrame({
        "employee_id": ["EMP_01"],
        "age": [30],
        "tenure_years": [-2.0]
    })
    valid, missing = DataValidator.validate_processed_attrition_schema(df_invalid)
    assert valid is False


def test_get_data_mode_callable():
    modes = get_data_mode()
    assert isinstance(modes, dict)
    assert "ibm_attrition" in modes
    assert "hr_promotion" in modes
