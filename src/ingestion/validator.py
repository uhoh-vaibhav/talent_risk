"""
Data Validation Module
Provides schema validation, null boundary checks, data type verifications, and sanity assertions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Expected raw columns
EXPECTED_IBM_COLUMNS = [
    "Age", "Attrition", "Department", "Education", "Gender",
    "JobSatisfaction", "MonthlyIncome", "OverTime", "PerformanceRating",
    "StockOptionLevel", "YearsAtCompany", "YearsSinceLastPromotion"
]

EXPECTED_PROMOTION_COLUMNS = [
    "employee_id", "department", "education", "gender", "no_of_trainings",
    "age", "previous_year_rating", "length_of_service",
    "awards_won", "avg_training_score", "is_promoted"
]

# Unified schema definitions
UNIFIED_REQUIRED_COLUMNS = [
    "employee_id", "department", "gender", "age", "education_level",
    "tenure_years", "years_since_promotion", "num_trainings_last_year",
    "performance_rating", "kpi_met_above_80", "awards_won",
    "overtime_status", "satisfaction_score", "monthly_income",
    "stock_option_level"
]


class DataValidator:
    """Validates raw and unified datasets prior to modeling pipelines."""

    @staticmethod
    def validate_raw_ibm_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates that raw IBM HR Attrition DataFrame contains required columns."""
        missing = [col for col in EXPECTED_IBM_COLUMNS if col not in df.columns]
        if missing:
            logger.error("IBM Attrition raw validation failed! Missing columns: %s", missing)
            return False, missing
        logger.info("IBM Attrition raw schema validation passed.")
        return True, []

    @staticmethod
    def validate_raw_promotion_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates that raw HR Promotion DataFrame contains required columns."""
        missing = [col for col in EXPECTED_PROMOTION_COLUMNS if col not in df.columns]
        if missing:
            logger.error("HR Promotion raw validation failed! Missing columns: %s", missing)
            return False, missing
        logger.info("HR Promotion raw schema validation passed.")
        return True, []

    @staticmethod
    def validate_unified_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates transformed unified DataFrame schema and sanity ranges."""
        missing = [col for col in UNIFIED_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.error("Unified schema validation failed! Missing columns: %s", missing)
            return False, missing

        # Range checks
        if (df["age"] < 16).any() or (df["age"] > 90).any():
            logger.warning("Unusual age values detected outside 16-90 range.")
        
        if (df["tenure_years"] < 0).any():
            logger.error("Negative tenure detected!")
            return False, ["Negative tenure values found"]

        null_summary = df[UNIFIED_REQUIRED_COLUMNS].isnull().mean()
        high_nulls = null_summary[null_summary > 0.3].index.tolist()
        if high_nulls:
            logger.warning("Columns exceeding 30%% null threshold: %s", high_nulls)

        logger.info("Unified schema validation successfully completed for %d records.", len(df))
        return True, []
