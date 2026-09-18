"""
Promotion Dataset EDA
Computes comprehensive exploratory statistics for the HR Promotion dataset.
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

def compute_promotion_eda(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Computes and returns promotion EDA statistics."""
    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed_promotion.parquet")
    
    target = "target_promotion"
    eda = {}
    
    # 1. Class distribution
    class_counts = df[target].value_counts().to_dict()
    total = len(df)
    eda["class_distribution"] = {
        "total_records": total,
        "promotion_yes": int(class_counts.get(1.0, 0)),
        "promotion_no": int(class_counts.get(0.0, 0)),
        "promotion_rate": round(class_counts.get(1.0, 0) / total, 4) if total > 0 else 0
    }
    
    # 2. by department
    dept_stats = df.groupby("department")[target].agg(["mean", "count"]).reset_index()
    eda["by_department"] = [
        {"department": row["department"], "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
        for _, row in dept_stats.iterrows()
    ]
    
    # 3. by education_level (using education)
    edu_col = "education_level" if "education_level" in df.columns else "education"
    if edu_col in df.columns:
        edu_stats = df.groupby(edu_col)[target].agg(["mean", "count"]).reset_index()
        eda["by_education_level"] = [
            {"education_level": str(row[edu_col]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in edu_stats.iterrows()
        ]
        
    # 4. by previous_year_rating
    if "previous_year_rating" in df.columns:
        rating_stats = df.groupby("previous_year_rating")[target].agg(["mean", "count"]).reset_index()
        eda["by_previous_year_rating"] = [
            {"previous_year_rating": float(row["previous_year_rating"]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in rating_stats.iterrows()
        ]
        
    # 5. by avg_training_score bins
    if "avg_training_score" in df.columns:
        df_copy = df.copy()
        df_copy["score_range"] = pd.cut(df_copy["avg_training_score"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
        score_stats = df_copy.groupby("score_range", observed=True)[target].agg(["mean", "count"]).reset_index()
        eda["by_avg_training_score"] = [
            {"score_range": str(row["score_range"]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in score_stats.iterrows()
        ]
        
    # 6. by kpis_met_above_80
    kpi_col = "kpis_met_above_80" if "kpis_met_above_80" in df.columns else "KPIs_met_80pct" if "KPIs_met_80pct" in df.columns else None
    if kpi_col:
        kpi_stats = df.groupby(kpi_col)[target].agg(["mean", "count"]).reset_index()
        eda["by_kpis_met_above_80"] = [
            {"kpis_met_above_80": int(row[kpi_col]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in kpi_stats.iterrows()
        ]
        
    # 7. by awards_won
    award_col = "awards_won" if "awards_won" in df.columns else "awards_won?" if "awards_won?" in df.columns else None
    if award_col:
        award_stats = df.groupby(award_col)[target].agg(["mean", "count"]).reset_index()
        eda["by_awards_won"] = [
            {"awards_won": int(row[award_col]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in award_stats.iterrows()
        ]
        
    # 8. by length_of_service bins
    if "length_of_service" in df.columns:
        df_copy = df.copy() if "df_copy" not in locals() else df_copy
        df_copy["service_range"] = pd.cut(df_copy["length_of_service"], bins=[0, 2, 5, 10, 20, 40], labels=["0-2", "3-5", "6-10", "11-20", "20+"], right=True)
        service_stats = df_copy.groupby("service_range", observed=True)[target].agg(["mean", "count"]).reset_index()
        eda["by_length_of_service"] = [
            {"length_of_service_range": str(row["service_range"]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in service_stats.iterrows()
        ]
        
    # 9. by age group
    if "age" in df.columns:
        df_copy = df.copy() if "df_copy" not in locals() else df_copy
        df_copy["age_group"] = pd.cut(df_copy["age"], bins=[17, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
        age_stats = df_copy.groupby("age_group", observed=True)[target].agg(["mean", "count"]).reset_index()
        eda["by_age_group"] = [
            {"age_group": str(row["age_group"]), "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in age_stats.iterrows()
        ]
        
    # 10. by gender
    if "gender" in df.columns:
        gender_stats = df.groupby("gender")[target].agg(["mean", "count"]).reset_index()
        eda["by_gender"] = [
            {"gender": row["gender"], "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in gender_stats.iterrows()
        ]
        
    # 11. by recruitment_channel
    if "recruitment_channel" in df.columns:
        channel_stats = df.groupby("recruitment_channel")[target].agg(["mean", "count"]).reset_index()
        eda["by_recruitment_channel"] = [
            {"recruitment_channel": row["recruitment_channel"], "promotion_rate": round(row["mean"], 4), "count": int(row["count"])}
            for _, row in channel_stats.iterrows()
        ]
        
    return eda

def save_promotion_eda(eda_results: Dict[str, Any] = None):
    """Computes and saves promotion EDA to JSON artifact."""
    if eda_results is None:
        eda_results = compute_promotion_eda()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "promotion_eda.json"
    with open(output_path, "w") as f:
        json.dump(eda_results, f, indent=2)
    logger.info("Saved promotion EDA to %s", output_path)

if __name__ == "__main__":
    eda = compute_promotion_eda()
    save_promotion_eda(eda)
    print(f"Promotion EDA: {eda['class_distribution']}")
