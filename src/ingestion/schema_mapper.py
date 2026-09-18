"""
Schema Mapping & Pseudonymization Module
Maps raw IBM Attrition and HR Promotion datasets into their respective processed schemas.
Each dataset maintains its own honest feature schema — NO artificial feature fabrication.
"""

import hashlib
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from src.ingestion.validator import DataValidator

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# ═══════════════════════════════════════════════════════════
# DATASET-SPECIFIC FEATURE SCHEMAS
# These constants define which features each model uses.
# They are imported by pipeline.py to build per-model pipelines.
# ═══════════════════════════════════════════════════════════

ATTRITION_NUMERICAL_FEATURES = [
    "age", "education_level", "monthly_income", "job_satisfaction",
    "environment_satisfaction", "work_life_balance", "performance_rating",
    "stock_option_level", "tenure_years", "years_since_promotion",
    "training_times_last_year", "num_companies_worked", "distance_from_home",
    "job_involvement", "job_level", "relationship_satisfaction",
    "percent_salary_hike", "overtime"
]
ATTRITION_CATEGORICAL_FEATURES = ["department", "gender", "marital_status"]
ATTRITION_TARGET = "target_attrition"

PROMOTION_NUMERICAL_FEATURES = [
    "age", "education_level", "no_of_trainings", "previous_year_rating",
    "length_of_service", "kpis_met_above_80", "awards_won", "avg_training_score"
]
PROMOTION_CATEGORICAL_FEATURES = ["department", "gender", "recruitment_channel"]
PROMOTION_TARGET = "target_promotion"


def pseudonymize_id(prefix: str, original_id: str) -> str:
    """Creates a deterministic secure SHA-256 hash pseudonym for an employee ID."""
    raw_str = f"{prefix}_{original_id}"
    hash_obj = hashlib.sha256(raw_str.encode("utf-8"))
    return f"EMP_{hash_obj.hexdigest()[:10].upper()}"


def map_department_name(dept: str) -> str:
    """Normalizes department names across heterogeneous sources."""
    if not isinstance(dept, str):
        return "Operations"
    dept_lower = dept.lower()
    if "research" in dept_lower or "r&d" in dept_lower:
        return "R&D"
    elif "sales" in dept_lower or "marketing" in dept_lower:
        return "Sales"
    elif "hr" in dept_lower or "human" in dept_lower:
        return "Human Resources"
    elif "tech" in dept_lower or "analytics" in dept_lower:
        return "Technology"
    elif "op" in dept_lower or "procure" in dept_lower:
        return "Operations"
    elif "fin" in dept_lower:
        return "Finance"
    else:
        return "Operations"


def map_ibm_to_processed(df_ibm: pd.DataFrame) -> pd.DataFrame:
    """Maps raw IBM HR Attrition dataset into attrition schema."""
    logger.info("Mapping IBM Attrition dataset (%d records)...", len(df_ibm))
    
    df = df_ibm.copy()
    mapped = pd.DataFrame()

    mapped["employee_id"] = df["EmployeeNumber"].apply(lambda x: pseudonymize_id("IBM", str(x)))
    mapped["employee_number"] = df["EmployeeNumber"].astype(str)
    mapped["department"] = df["Department"].apply(map_department_name)
    mapped["gender"] = df["Gender"].apply(lambda x: "Male" if x == "Male" else "Female")
    mapped["age"] = df["Age"].astype(int)
    mapped["education_level"] = df["Education"].fillna(3).astype(int)
    mapped["monthly_income"] = df["MonthlyIncome"].astype(float)
    mapped["overtime"] = df["OverTime"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    mapped["job_satisfaction"] = df["JobSatisfaction"].astype(float)
    mapped["environment_satisfaction"] = df["EnvironmentSatisfaction"].astype(float)
    mapped["work_life_balance"] = df["WorkLifeBalance"].astype(float)
    mapped["performance_rating"] = df["PerformanceRating"].astype(float)
    mapped["stock_option_level"] = df["StockOptionLevel"].astype(int)
    mapped["tenure_years"] = df["YearsAtCompany"].astype(float)
    mapped["years_since_promotion"] = df["YearsSinceLastPromotion"].astype(float)
    mapped["training_times_last_year"] = df["TrainingTimesLastYear"].astype(int)
    mapped["num_companies_worked"] = df["NumCompaniesWorked"].astype(int)
    mapped["distance_from_home"] = df["DistanceFromHome"].astype(float)
    mapped["job_involvement"] = df["JobInvolvement"].astype(float)
    mapped["job_level"] = df["JobLevel"].astype(int)
    mapped["marital_status"] = df["MaritalStatus"].astype(str)
    mapped["relationship_satisfaction"] = df["RelationshipSatisfaction"].astype(float)
    mapped["percent_salary_hike"] = df["PercentSalaryHike"].astype(float)
    
    mapped["target_attrition"] = df["Attrition"].apply(lambda x: 1.0 if str(x).lower() == "yes" else 0.0)

    return mapped


def map_promotion_to_processed(df_promo: pd.DataFrame) -> pd.DataFrame:
    """Maps raw HR Promotion dataset into promotion schema."""
    logger.info("Mapping HR Promotion dataset (%d records)...", len(df_promo))
    
    df = df_promo.copy()
    mapped = pd.DataFrame()

    mapped["employee_id"] = df["employee_id"].apply(lambda x: pseudonymize_id("PROMO", str(x)))
    mapped["employee_number"] = df["employee_id"].astype(str)
    mapped["department"] = df["department"].apply(map_department_name)
    mapped["gender"] = df["gender"].apply(lambda x: "Female" if str(x).lower() == "f" else "Male")
    mapped["age"] = df["age"].astype(int)

    def parse_edu(val):
        if pd.isna(val):
            return 3
        v = str(val).lower()
        if "below" in v or "secondary" in v:
            return 1
        elif "bachelor" in v:
            return 3
        elif "master" in v or "above" in v or "phd" in v:
            return 4
        return 3

    mapped["education_level"] = df["education"].apply(parse_edu)
    mapped["no_of_trainings"] = df["no_of_trainings"].astype(int)
    mapped["previous_year_rating"] = df["previous_year_rating"].fillna(3.0).astype(float)
    mapped["length_of_service"] = df["length_of_service"].astype(float)
    
    if "KPIs_met_80pct" in df.columns:
        mapped["kpis_met_above_80"] = df["KPIs_met_80pct"].fillna(0).astype(int)
    elif "KPIs_met >80%" in df.columns:
        mapped["kpis_met_above_80"] = df["KPIs_met >80%"].fillna(0).astype(int)
    else:
        mapped["kpis_met_above_80"] = (mapped["previous_year_rating"] >= 4.0).astype(int)

    mapped["awards_won"] = df["awards_won"].fillna(0).astype(int)
    mapped["avg_training_score"] = df["avg_training_score"].astype(float)
    mapped["recruitment_channel"] = df["recruitment_channel"].astype(str)
    
    mapped["target_promotion"] = df["is_promoted"].astype(float)

    return mapped


def process_and_save_datasets(
    raw_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "raw",
    processed_dir: Path = PROCESSED_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests raw data, applies validation & mapping, and saves separate processed outputs."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    ibm_raw_path = raw_dir / "ibm_attrition_raw.csv"
    promo_raw_path = raw_dir / "hr_promotion_raw.csv"

    df_ibm_raw = pd.read_csv(ibm_raw_path)
    df_promo_raw = pd.read_csv(promo_raw_path)

    # Validate
    DataValidator.validate_raw_ibm_schema(df_ibm_raw)
    DataValidator.validate_raw_promotion_schema(df_promo_raw)

    # Map
    df_ibm_processed = map_ibm_to_processed(df_ibm_raw)
    df_promo_processed = map_promotion_to_processed(df_promo_raw)

    # Validate processed
    DataValidator.validate_processed_attrition_schema(df_ibm_processed)
    DataValidator.validate_processed_promotion_schema(df_promo_processed)

    # Save
    ibm_out = processed_dir / "processed_attrition.parquet"
    promo_out = processed_dir / "processed_promotion.parquet"

    df_ibm_processed.to_parquet(ibm_out, index=False)
    df_promo_processed.to_parquet(promo_out, index=False)

    df_ibm_processed.to_csv(processed_dir / "processed_attrition.csv", index=False)
    df_promo_processed.to_csv(processed_dir / "processed_promotion.csv", index=False)

    logger.info("Successfully saved processed datasets to %s", processed_dir)
    return df_ibm_processed, df_promo_processed


# Backward compatibility aliases
map_ibm_to_unified = map_ibm_to_processed
map_promotion_to_unified = map_promotion_to_processed
process_and_save_unified_data = process_and_save_datasets


if __name__ == "__main__":
    process_and_save_datasets()

