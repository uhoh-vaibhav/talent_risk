"""
Attrition Dataset EDA
Computes comprehensive exploratory statistics for the IBM HR Attrition dataset.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "eda"


def compute_attrition_eda(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Computes and returns attrition EDA statistics."""
    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed_attrition.parquet")
    
    target = "target_attrition"
    eda = {}
    
    # 1. Class distribution
    class_counts = df[target].value_counts().to_dict()
    total = len(df)
    eda["class_distribution"] = {
        "total_records": total,
        "attrition_yes": int(class_counts.get(1.0, 0)),
        "attrition_no": int(class_counts.get(0.0, 0)),
        "attrition_rate": round(class_counts.get(1.0, 0) / total, 4)
    }
    
    # 2. Attrition by department
    dept_stats = df.groupby("department")[target].agg(["mean", "count"]).reset_index()
    eda["by_department"] = [
        {"department": row["department"], "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in dept_stats.iterrows()
    ]
    
    # 3. Attrition by overtime
    ot_stats = df.groupby("overtime")[target].agg(["mean", "count"]).reset_index()
    eda["by_overtime"] = [
        {"overtime": int(row["overtime"]), "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in ot_stats.iterrows()
    ]
    
    # 4. Attrition by gender
    gender_stats = df.groupby("gender")[target].agg(["mean", "count"]).reset_index()
    eda["by_gender"] = [
        {"gender": row["gender"], "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in gender_stats.iterrows()
    ]
    
    # 5. Attrition by job satisfaction
    sat_stats = df.groupby("job_satisfaction")[target].agg(["mean", "count"]).reset_index()
    eda["by_job_satisfaction"] = [
        {"satisfaction_level": int(row["job_satisfaction"]), "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in sat_stats.iterrows()
    ]
    
    # 6. Attrition by age group
    df_copy = df.copy()
    df_copy["age_group"] = pd.cut(df_copy["age"], bins=[17, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
    age_stats = df_copy.groupby("age_group", observed=True)[target].agg(["mean", "count"]).reset_index()
    eda["by_age_group"] = [
        {"age_group": str(row["age_group"]), "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in age_stats.iterrows()
    ]
    
    # 7. Attrition by marital status
    ms_stats = df.groupby("marital_status")[target].agg(["mean", "count"]).reset_index()
    eda["by_marital_status"] = [
        {"marital_status": row["marital_status"], "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in ms_stats.iterrows()
    ]
    
    # 8. Income distribution stats
    eda["income_stats"] = {
        "mean": round(float(df["monthly_income"].mean()), 2),
        "median": round(float(df["monthly_income"].median()), 2),
        "std": round(float(df["monthly_income"].std()), 2),
        "min": round(float(df["monthly_income"].min()), 2),
        "max": round(float(df["monthly_income"].max()), 2)
    }
    
    # 9. Attrition by income range
    df_copy["income_range"] = pd.cut(df_copy["monthly_income"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    income_stats = df_copy.groupby("income_range", observed=True)[target].agg(["mean", "count"]).reset_index()
    eda["by_income_range"] = [
        {"income_range": str(row["income_range"]), "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in income_stats.iterrows()
    ]
    
    # 10. Attrition by tenure range
    df_copy["tenure_range"] = pd.cut(df_copy["tenure_years"], bins=[0, 2, 5, 10, 20, 40], labels=["0-2", "3-5", "6-10", "11-20", "20+"], right=True)
    tenure_stats = df_copy.groupby("tenure_range", observed=True)[target].agg(["mean", "count"]).reset_index()
    eda["by_tenure_range"] = [
        {"tenure_range": str(row["tenure_range"]), "attrition_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in tenure_stats.iterrows()
    ]
    
    # 11. Numerical feature summary statistics
    num_cols = ["age", "monthly_income", "tenure_years", "years_since_promotion",
                "job_satisfaction", "performance_rating", "distance_from_home"]
    eda["numerical_summary"] = {}
    for col in num_cols:
        if col in df.columns:
            eda["numerical_summary"][col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "std": round(float(df[col].std()), 2)
            }
    
    return eda


def save_attrition_eda(eda_results: Dict[str, Any] = None):
    """Computes and saves attrition EDA to JSON artifact."""
    if eda_results is None:
        eda_results = compute_attrition_eda()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "attrition_eda.json"
    with open(output_path, "w") as f:
        json.dump(eda_results, f, indent=2)
    logger.info("Saved attrition EDA to %s", output_path)


if __name__ == "__main__":
    eda = compute_attrition_eda()
    save_attrition_eda(eda)
    print(f"Attrition EDA: {eda['class_distribution']}")
