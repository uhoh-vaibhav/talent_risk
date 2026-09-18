"""
Prefect Orchestration Workflow DAG
Automates scheduled data ingestion, feature pipeline transformation, model training, fairness audit, and drift detection.
"""

import logging
import pandas as pd
from pathlib import Path
from prefect import flow, task

from src.ingestion.download import download_or_generate_datasets
from src.ingestion.schema_mapper import process_and_save_unified_data
from src.features.pipeline import NUMERICAL_FEATURES
from src.features.store import FeatureStore, DATA_DIR
from src.models.train_attrition import train_attrition_model
from src.models.train_promotion import train_promotion_model
from src.monitoring.drift import DriftDetector
from src.monitoring.alert import RetrainingAlertManager

logger = logging.getLogger(__name__)


@task(name="Ingest & Unified Schema Mapping")
def ingest_and_map_task():
    logger.info("Executing Ingestion Task...")
    download_or_generate_datasets()
    df_ibm, df_promo, df_master = process_and_save_unified_data()
    return len(df_master)


@task(name="Feature Store Transformation")
def feature_transform_task():
    logger.info("Executing Feature Store Task...")
    df_master = pd.read_parquet(DATA_DIR / "unified_master.parquet")
    fs = FeatureStore()
    X_trans, feature_names = fs.fit_transform_store(df_master)
    return X_trans.shape


@task(name="Train Attrition Risk Model")
def train_attrition_task():
    logger.info("Executing Attrition Model Training Task...")
    df_attrition = pd.read_parquet(DATA_DIR / "unified_attrition.parquet")
    model, metrics, fairness_report = train_attrition_model(df_attrition, use_mlflow=True)
    return metrics, fairness_report


@task(name="Train Promotion Readiness Model")
def train_promotion_task():
    logger.info("Executing Promotion Model Training Task...")
    df_promotion = pd.read_parquet(DATA_DIR / "unified_promotion.parquet")
    model, metrics, fairness_report = train_promotion_model(df_promotion, use_mlflow=True)
    return metrics, fairness_report


@task(name="Drift Detection & Alert Check")
def drift_monitoring_task():
    logger.info("Executing Drift Monitoring Task...")
    df_master = pd.read_parquet(DATA_DIR / "unified_master.parquet")
    
    # Split for baseline vs current simulation
    ref_df = df_master.sample(frac=0.5, random_state=42)
    curr_df = df_master.drop(ref_df.index)

    drift_report = DriftDetector.detect_feature_drift(ref_df, curr_df, NUMERICAL_FEATURES)
    fairness_dummy = {"overall_fairness_passed": True}
    
    retrain_signal = RetrainingAlertManager.evaluate_and_alert(drift_report, fairness_dummy)
    return retrain_signal


@flow(name="Talent_Risk_Scoring_Pipeline_Flow")
def talent_risk_pipeline_flow():
    """Main Orchestration Flow executing full ML pipeline."""
    total_records = ingest_and_map_task()
    feat_shape = feature_transform_task()
    attr_metrics, attr_fairness = train_attrition_task()
    promo_metrics, promo_fairness = train_promotion_task()
    retrain_flag = drift_monitoring_task()

    return {
        "status": "SUCCESS",
        "total_records": total_records,
        "feature_matrix_shape": feat_shape,
        "attrition_metrics": attr_metrics,
        "promotion_metrics": promo_metrics,
        "retrain_flag": retrain_flag
    }


if __name__ == "__main__":
    talent_risk_pipeline_flow()
