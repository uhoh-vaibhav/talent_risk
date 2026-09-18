"""
Schema Mapping & Pseudonymization Module
Maps raw IBM Attrition and HR Promotion datasets into the Unified Employee Feature Schema.
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


def map_ibm_to_unified(df_ibm: pd.DataFrame) -> pd.DataFrame:
    """Maps raw IBM HR Attrition dataset into Unified Employee Schema."""
    logger.info("Mapping IBM Attrition dataset (%d records)...", len(df_ibm))
    
    df = df_ibm.copy()
    mapped = pd.DataFrame()

    mapped["employee_id"] = df["EmployeeNumber"].apply(lambda x: pseudonymize_id("IBM", str(x)))
    mapped["employee_number"] = df["EmployeeNumber"].astype(str)
    mapped["department"] = df["Department"].apply(map_department_name)
    mapped["gender"] = df["Gender"].apply(lambda x: "Male" if x == "Male" else "Female")
    mapped["age"] = df["Age"].astype(int)
    mapped["education_level"] = df["Education"].fillna(3).astype(int)
    mapped["tenure_years"] = df["YearsAtCompany"].astype(float)
    mapped["years_since_promotion"] = df["YearsSinceLastPromotion"].astype(float)
    mapped["num_trainings_last_year"] = df["TrainingTimesLastYear"].astype(int)
    mapped["performance_rating"] = df["PerformanceRating"].astype(float)
    
    mapped["kpi_met_above_80"] = (df["PerformanceRating"] >= 4).astype(int)
    mapped["awards_won"] = ((df["PerformanceRating"] == 4) & (df["PercentSalaryHike"] >= 20)).astype(int)
    mapped["overtime_status"] = df["OverTime"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    
    satisfaction_avg = (
        df.get("JobSatisfaction", 3) + 
        df.get("EnvironmentSatisfaction", 3) + 
        df.get("WorkLifeBalance", 3)
    ) / 3.0
    mapped["satisfaction_score"] = satisfaction_avg.round(2)
    mapped["monthly_income"] = df["MonthlyIncome"].astype(float)
    mapped["stock_option_level"] = df["StockOptionLevel"].astype(int)
    
    mapped["target_attrition"] = df["Attrition"].apply(lambda x: 1.0 if str(x).lower() == "yes" else 0.0)
    mapped["target_promotion"] = np.nan

    return mapped


def map_promotion_to_unified(df_promo: pd.DataFrame) -> pd.DataFrame:
    """Maps raw HR Promotion dataset into Unified Employee Schema."""
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
    mapped["tenure_years"] = df["length_of_service"].astype(float)
    mapped["years_since_promotion"] = (df["length_of_service"] * 0.4).round(1)
    mapped["num_trainings_last_year"] = df["no_of_trainings"].astype(int)
    
    mapped["performance_rating"] = df["previous_year_rating"].fillna(3.0).astype(float)

    if "KPIs_met_80pct" in df.columns:
        mapped["kpi_met_above_80"] = df["KPIs_met_80pct"].fillna(0).astype(int)
    elif "KPIs_met >80%" in df.columns:
        mapped["kpi_met_above_80"] = df["KPIs_met >80%"].fillna(0).astype(int)
    else:
        mapped["kpi_met_above_80"] = (mapped["performance_rating"] >= 4.0).astype(int)

    mapped["awards_won"] = df["awards_won"].fillna(0).astype(int)
    mapped["overtime_status"] = ((df["no_of_trainings"] > 2) | (df["avg_training_score"] > 80)).astype(int)

    avg_score = df["avg_training_score"].fillna(60)
    satisfaction = 1.0 + ((avg_score - 40) / 60.0) * 4.0
    mapped["satisfaction_score"] = satisfaction.clip(1.0, 5.0).round(2)

    base_salary = 3000.0 + mapped["age"] * 80.0 + mapped["tenure_years"] * 250.0 + mapped["education_level"] * 1000.0
    mapped["monthly_income"] = base_salary.round(2)
    mapped["stock_option_level"] = np.clip((mapped["tenure_years"] // 3).astype(int), 0, 3)

    mapped["target_attrition"] = np.nan
    mapped["target_promotion"] = df["is_promoted"].astype(float)

    return mapped


def process_and_save_unified_data(
    raw_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "raw",
    processed_dir: Path = PROCESSED_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingests raw data, applies validation & mapping, and saves processed unified outputs."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    ibm_raw_path = raw_dir / "ibm_attrition_raw.csv"
    promo_raw_path = raw_dir / "hr_promotion_raw.csv"

    df_ibm_raw = pd.read_csv(ibm_raw_path)
    df_promo_raw = pd.read_csv(promo_raw_path)

    # Validate
    DataValidator.validate_raw_ibm_schema(df_ibm_raw)
    DataValidator.validate_raw_promotion_schema(df_promo_raw)

    # Map
    df_ibm_unified = map_ibm_to_unified(df_ibm_raw)
    df_promo_unified = map_promotion_to_unified(df_promo_raw)

    # Validate unified
    DataValidator.validate_unified_schema(df_ibm_unified)
    DataValidator.validate_unified_schema(df_promo_unified)

    # Combined master dataset
    df_master_unified = pd.concat([df_ibm_unified, df_promo_unified], ignore_index=True)

    # Save
    ibm_out = processed_dir / "unified_attrition.parquet"
    promo_out = processed_dir / "unified_promotion.parquet"
    master_out = processed_dir / "unified_master.parquet"

    df_ibm_unified.to_parquet(ibm_out, index=False)
    df_promo_unified.to_parquet(promo_out, index=False)
    df_master_unified.to_parquet(master_out, index=False)

    df_ibm_unified.to_csv(processed_dir / "unified_attrition.csv", index=False)
    df_promo_unified.to_csv(processed_dir / "unified_promotion.csv", index=False)
    df_master_unified.to_csv(processed_dir / "unified_master.csv", index=False)

    logger.info("Successfully saved unified datasets to %s", processed_dir)
    return df_ibm_unified, df_promo_unified, df_master_unified


if __name__ == "__main__":
    process_and_save_unified_data()
