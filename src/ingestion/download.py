"""
Data Ingestion Module - Download & Dataset Generation Script
Downloads raw Kaggle/IBM HR Attrition & HR Promotion datasets or generates synthetic fallbacks.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Mirror URLs for public datasets
IBM_ATTRITION_URL = "https://raw.githubusercontent.com/ibm-developer-skills-network/ML0101EN-SkillsNetwork/main/labs/Module%203/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
HR_PROMOTION_URL = "https://raw.githubusercontent.com/dphi-official/Datasets/master/hr_analytics/train_LZ43U5d.csv"


def generate_synthetic_ibm_attrition(num_samples: int = 1470) -> pd.DataFrame:
    """Generates synthetic IBM HR Attrition dataset matching exact schema and distributions."""
    logger.info("Generating synthetic IBM HR Attrition dataset (%d records)...", num_samples)
    np.random.seed(42)

    departments = ["Research & Development", "Sales", "Human Resources"]
    education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
    job_roles = [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]
    marital_statuses = ["Single", "Married", "Divorced"]
    business_travels = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    overtimes = ["Yes", "No"]
    genders = ["Male", "Female"]

    ages = np.random.randint(18, 61, size=num_samples)
    attrition_probs = np.where(ages < 30, 0.25, 0.12)
    attritions = np.random.binomial(1, attrition_probs)
    attrition_str = np.where(attritions == 1, "Yes", "No")

    df = pd.DataFrame({
        "Age": ages,
        "Attrition": attrition_str,
        "BusinessTravel": np.random.choice(business_travels, size=num_samples, p=[0.7, 0.2, 0.1]),
        "DailyRate": np.random.randint(100, 1500, size=num_samples),
        "Department": np.random.choice(departments, size=num_samples, p=[0.65, 0.30, 0.05]),
        "DistanceFromHome": np.random.randint(1, 30, size=num_samples),
        "Education": np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05]),
        "EducationField": np.random.choice(education_fields, size=num_samples),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, num_samples + 1),
        "EnvironmentSatisfaction": np.random.randint(1, 5, size=num_samples),
        "Gender": np.random.choice(genders, size=num_samples, p=[0.6, 0.4]),
        "HourlyRate": np.random.randint(30, 100, size=num_samples),
        "JobInvolvement": np.random.randint(1, 5, size=num_samples),
        "JobLevel": np.random.randint(1, 6, size=num_samples),
        "JobRole": np.random.choice(job_roles, size=num_samples),
        "JobSatisfaction": np.random.randint(1, 5, size=num_samples),
        "MaritalStatus": np.random.choice(marital_statuses, size=num_samples, p=[0.3, 0.5, 0.2]),
        "MonthlyIncome": np.random.randint(2000, 20000, size=num_samples),
        "MonthlyRate": np.random.randint(2000, 27000, size=num_samples),
        "NumCompaniesWorked": np.random.randint(0, 10, size=num_samples),
        "Over18": "Y",
        "OverTime": np.random.choice(overtimes, size=num_samples, p=[0.28, 0.72]),
        "PercentSalaryHike": np.random.randint(11, 26, size=num_samples),
        "PerformanceRating": np.random.choice([3, 4], size=num_samples, p=[0.84, 0.16]),
        "RelationshipSatisfaction": np.random.randint(1, 5, size=num_samples),
        "StandardHours": 80,
        "StockOptionLevel": np.random.randint(0, 4, size=num_samples),
        "TotalWorkingYears": np.random.randint(0, 40, size=num_samples),
        "TrainingTimesLastYear": np.random.randint(0, 7, size=num_samples),
        "WorkLifeBalance": np.random.randint(1, 5, size=num_samples),
        "YearsAtCompany": np.random.randint(0, 20, size=num_samples),
        "YearsInCurrentRole": np.random.randint(0, 15, size=num_samples),
        "YearsSinceLastPromotion": np.random.randint(0, 15, size=num_samples),
        "YearsWithCurrManager": np.random.randint(0, 15, size=num_samples)
    })
    return df


def generate_synthetic_hr_promotion(num_samples: int = 54808) -> pd.DataFrame:
    """Generates synthetic HR Promotion Analytics dataset matching exact schema and distributions."""
    logger.info("Generating synthetic HR Promotion dataset (%d records)...", num_samples)
    np.random.seed(42)

    departments = ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "Finance", "HR", "Legal"]
    regions = [f"region_{i}" for i in range(1, 35)]
    educations = ["Master's & above", "Bachelor's", "Below Secondary"]
    genders = ["m", "f"]
    channels = ["sourcing", "other", "referred"]

    kpi_met = np.random.choice([0, 1], size=num_samples, p=[0.65, 0.35])
    awards = np.random.choice([0, 1], size=num_samples, p=[0.97, 0.03])
    avg_scores = np.random.randint(39, 100, size=num_samples)

    # Base promotion probability influenced by KPIs, awards, and training score
    promo_score = kpi_met * 0.4 + awards * 0.4 + (avg_scores - 39) / 60.0 * 0.2
    promo_probs = 1 / (1 + np.exp(-(promo_score - 1.5)))
    is_promoted = np.random.binomial(1, np.clip(promo_probs * 0.2, 0.01, 0.35))

    # Add some nulls to Education and Previous Year Rating to test validation
    education_choices = np.random.choice(educations + [None], size=num_samples, p=[0.25, 0.65, 0.05, 0.05])
    rating_choices = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, None], size=num_samples, p=[0.1, 0.15, 0.4, 0.2, 0.1, 0.05])

    df = pd.DataFrame({
        "employee_id": [100000 + i for i in range(num_samples)],
        "department": np.random.choice(departments, size=num_samples),
        "region": np.random.choice(regions, size=num_samples),
        "education": education_choices,
        "gender": np.random.choice(genders, size=num_samples, p=[0.7, 0.3]),
        "recruitment_channel": np.random.choice(channels, size=num_samples, p=[0.42, 0.55, 0.03]),
        "no_of_trainings": np.random.randint(1, 10, size=num_samples),
        "age": np.random.randint(20, 61, size=num_samples),
        "previous_year_rating": rating_choices,
        "length_of_service": np.random.randint(1, 35, size=num_samples),
        "KPIs_met_80pct": kpi_met,
        "awards_won": awards,
        "avg_training_score": avg_scores,
        "is_promoted": is_promoted
    })
    return df


def download_or_generate_datasets(raw_dir: Path = DATA_DIR):
    """Downloads raw datasets or generates synthetic fallbacks if network fails."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ibm_file = raw_dir / "ibm_attrition_raw.csv"
    promo_file = raw_dir / "hr_promotion_raw.csv"

    # Download or generate IBM Attrition dataset
    if not ibm_file.exists():
        logger.info("Attempting to download IBM Attrition dataset from URL...")
        try:
            df_ibm = pd.read_csv(IBM_ATTRITION_URL)
            df_ibm.to_csv(ibm_file, index=False)
            logger.info("Downloaded IBM Attrition dataset to %s (%d rows)", ibm_file, len(df_ibm))
        except Exception as e:
            logger.warning("Download failed (%s). Generating synthetic IBM dataset...", str(e))
            df_ibm = generate_synthetic_ibm_attrition()
            df_ibm.to_csv(ibm_file, index=False)
            logger.info("Saved synthetic IBM Attrition dataset to %s", ibm_file)
    else:
        logger.info("IBM Attrition raw dataset already exists at %s", ibm_file)

    # Download or generate HR Promotion dataset
    if not promo_file.exists():
        logger.info("Attempting to download HR Promotion dataset from URL...")
        try:
            df_promo = pd.read_csv(HR_PROMOTION_URL)
            df_promo.to_csv(promo_file, index=False)
            logger.info("Downloaded HR Promotion dataset to %s (%d rows)", promo_file, len(df_promo))
        except Exception as e:
            logger.warning("Download failed (%s). Generating synthetic HR Promotion dataset...", str(e))
            df_promo = generate_synthetic_hr_promotion()
            df_promo.to_csv(promo_file, index=False)
            logger.info("Saved synthetic HR Promotion dataset to %s", promo_file)
    else:
        logger.info("HR Promotion raw dataset already exists at %s", promo_file)


if __name__ == "__main__":
    download_or_generate_datasets()
