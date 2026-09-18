"""
Prefect Orchestration Workflow DAG
Automates scheduled data ingestion, feature pipeline transformation, model training, fairness audit, and drift detection.
"""

import logging
import pandas as pd
from pathlib import Path

try:
    from prefect import flow, task
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False
    # Mock decorators if prefect is not installed
    def task(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def flow(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from src.ingestion.download import download_or_generate_datasets
from src.ingestion.schema_mapper import process_and_save_datasets, ATTRITION_NUMERICAL_FEATURES, PROMOTION_NUMERICAL_FEATURES
from src.features.store import FeatureStore, DATA_DIR
from src.models.train_attrition import train_attrition_model
from src.models.train_promotion import train_promotion_model
from src.monitoring.drift import DriftDetector
from src.monitoring.alert import RetrainingAlertManager

logger = logging.getLogger(__name__)


@task(name="Ingest & Schema Mapping")
def ingest_and_map_task():
    logger.info("Executing Ingestion Task...")
    download_or_generate_datasets()
    df_attr, df_promo = process_and_save_datasets()
    return len(df_attr), len(df_promo)


@task(name="Feature Store Transformation")
def feature_transform_task():
    logger.info("Executing Feature Store Task...")
    df_attr = pd.read_parquet(DATA_DIR / "processed_attrition.parquet")
    df_promo = pd.read_parquet(DATA_DIR / "processed_promotion.parquet")

    fs_attr = FeatureStore(model_type="attrition")
    X_attr, names_attr = fs_attr.fit_and_save(df_attr)

    fs_promo = FeatureStore(model_type="promotion")
    X_promo, names_promo = fs_promo.fit_and_save(df_promo)

    return X_attr.shape, X_promo.shape


@task(name="Train Attrition Risk Model")
def train_attrition_task():
    logger.info("Executing Attrition Model Training Task...")
    df_attrition = pd.read_parquet(DATA_DIR / "processed_attrition.parquet")
    results = train_attrition_model(df_attrition, use_mlflow=False)
    return results["metrics"], results.get("fairness", {})


@task(name="Train Promotion Readiness Model")
def train_promotion_task():
    logger.info("Executing Promotion Model Training Task...")
    df_promotion = pd.read_parquet(DATA_DIR / "processed_promotion.parquet")
    results = train_promotion_model(df_promotion, use_mlflow=False)
    return results["metrics"], results.get("fairness", {})


@task(name="Drift Detection & Alert Check")
def drift_monitoring_task(attr_fairness: dict):
    logger.info("Executing Drift Monitoring Task...")
    df_attr = pd.read_parquet(DATA_DIR / "processed_attrition.parquet")
    
    # Split for baseline vs current simulation
    ref_df = df_attr.sample(frac=0.5, random_state=42)
    curr_df = df_attr.drop(ref_df.index)

    drift_report = DriftDetector.detect_feature_drift(ref_df, curr_df, ATTRITION_NUMERICAL_FEATURES[:6])
    retrain_signal = RetrainingAlertManager.evaluate_and_alert(drift_report, attr_fairness)
    return retrain_signal


@flow(name="Talent_Risk_Scoring_Pipeline_Flow")
def talent_risk_pipeline_flow():
    """Main Orchestration Flow executing full ML pipeline."""
    attr_len, promo_len = ingest_and_map_task()
    attr_shape, promo_shape = feature_transform_task()
    attr_metrics, attr_fairness = train_attrition_task()
    promo_metrics, promo_fairness = train_promotion_task()
    retrain_flag = drift_monitoring_task(attr_fairness)

    return {
        "status": "SUCCESS",
        "attrition_records": attr_len,
        "promotion_records": promo_len,
        "attrition_feature_matrix": attr_shape,
        "promotion_feature_matrix": promo_shape,
        "attrition_metrics": attr_metrics,
        "promotion_metrics": promo_metrics,
        "retrain_flag": retrain_flag
    }


if __name__ == "__main__":
    talent_risk_pipeline_flow()
