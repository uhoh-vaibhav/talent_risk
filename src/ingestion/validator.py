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
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "MonthlyIncome", "OverTime", "PerformanceRating",
    "StockOptionLevel", "YearsAtCompany", "YearsSinceLastPromotion",
    "TrainingTimesLastYear", "NumCompaniesWorked", "DistanceFromHome",
    "JobInvolvement", "JobLevel", "MaritalStatus",
    "RelationshipSatisfaction", "PercentSalaryHike"
]

EXPECTED_PROMOTION_COLUMNS = [
    "employee_id", "department", "education", "gender", "no_of_trainings",
    "age", "previous_year_rating", "length_of_service",
    "awards_won", "avg_training_score", "is_promoted", "recruitment_channel"
]

PROCESSED_ATTRITION_COLUMNS = [
    "employee_id", "department", "gender", "age", "education_level",
    "monthly_income", "overtime", "job_satisfaction", "environment_satisfaction",
    "work_life_balance", "performance_rating", "stock_option_level",
    "tenure_years", "years_since_promotion", "training_times_last_year",
    "num_companies_worked", "distance_from_home", "job_involvement",
    "job_level", "marital_status", "relationship_satisfaction",
    "percent_salary_hike", "target_attrition"
]

PROCESSED_PROMOTION_COLUMNS = [
    "employee_id", "department", "gender", "age", "education_level",
    "no_of_trainings", "previous_year_rating", "length_of_service",
    "kpis_met_above_80", "awards_won", "avg_training_score",
    "recruitment_channel", "target_promotion"
]


class DataValidator:
    """Validates raw and processed datasets prior to modeling pipelines."""

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
    def _validate_processed_schema(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> Tuple[bool, List[str]]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"{dataset_name} schema validation failed! Missing columns: %s", missing)
            return False, missing

        if (df["age"] < 16).any() or (df["age"] > 90).any():
            logger.warning(f"Unusual age values detected outside 16-90 range in {dataset_name}.")
        
        null_summary = df[required_columns].isnull().mean()
        high_nulls = null_summary[null_summary > 0.3].index.tolist()
        if high_nulls:
            logger.warning(f"Columns exceeding 30%% null threshold in {dataset_name}: %s", high_nulls)

        logger.info(f"{dataset_name} schema validation successfully completed for %d records.", len(df))
        return True, []

    @staticmethod
    def validate_processed_attrition_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        if "tenure_years" in df.columns and (df["tenure_years"] < 0).any():
            logger.error("Negative tenure detected!")
            return False, ["Negative tenure values found"]
        return DataValidator._validate_processed_schema(df, PROCESSED_ATTRITION_COLUMNS, "Processed Attrition")

    @staticmethod
    def validate_processed_promotion_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        if "length_of_service" in df.columns and (df["length_of_service"] < 0).any():
            logger.error("Negative length_of_service detected!")
            return False, ["Negative length_of_service values found"]
        return DataValidator._validate_processed_schema(df, PROCESSED_PROMOTION_COLUMNS, "Processed Promotion")

    # Backward compatibility alias
    validate_unified_schema = validate_processed_attrition_schema

