"""
Unit tests for FastAPI endpoints, single/batch inference, EDA, and batch CSV upload.
"""

import io
import pytest
from fastapi.testclient import TestClient
from src.serving.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "") or "application/json" in response.headers.get("content-type", "")


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "data_mode" in data
    assert "calibrated_thresholds" in data


def test_overview_endpoint():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_workforce_records" in data
    assert "benchmark_attrition_rate" in data
    assert "benchmark_promotion_rate" in data


def test_single_prediction_endpoint():
    sample_payload = {
        "employee_id": "TEST_EMP_99",
        "department": "R&D",
        "gender": "Female",
        "age": 30,
        "education_level": 3,
        "tenure_years": 5.0,
        "years_since_promotion": 2.5,
        "monthly_income": 6200.0,
        "overtime_status": 1,
        "satisfaction_score": 2.5,
        "performance_rating": 3.5,
        "kpi_met_above_80": 1,
        "awards_won": 0,
        "num_trainings_last_year": 2,
        "stock_option_level": 1,
        "avg_training_score": 70.0
    }

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["employee_id"] == "TEST_EMP_99"
    assert 0.0 <= data["attrition_risk_score"] <= 1.0
    assert 0.0 <= data["promotion_readiness_score"] <= 1.0
    assert "quadrant" in data
    assert "hr_action_recommendation" in data
    assert "attrition_threshold_applied" in data
    assert "promotion_threshold_applied" in data
    assert "disclaimer" in data
    assert isinstance(data["top_attrition_drivers"], list)
    assert isinstance(data["top_promotion_drivers"], list)


def test_batch_prediction_endpoint():
    batch_payload = {
        "employees": [
            {"employee_id": "EMP_B1", "department": "Sales", "age": 28, "monthly_income": 4000.0},
            {"employee_id": "EMP_B2", "department": "Technology", "age": 35, "monthly_income": 12000.0}
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_batch_upload_endpoint():
    csv_data = (
        "employee_id,department,gender,age,education_level,tenure_years,monthly_income\n"
        "EMP_U1,Operations,Male,40,3,6.0,5500.0\n"
        "EMP_U2,R&D,Female,29,4,3.0,7200.0\n"
    )
    files = {"file": ("test_batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/api/batch-upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["total_scored"] == 2
    assert "quadrant_distribution" in data


def test_fairness_report_endpoint():
    response = client.get("/fairness-report")
    assert response.status_code == 200
    data = response.json()
    assert "disparate_impact_threshold" in data
    assert "models" in data
    assert "attrition_model" in data["models"]
    assert "promotion_model" in data["models"]


def test_model_metrics_endpoint():
    response = client.get("/api/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "attrition_model" in data
    assert "promotion_model" in data
    assert "comparison_table" in data["attrition_model"]
    assert "global_feature_importance" in data["attrition_model"]


def test_eda_endpoints():
    res_attr = client.get("/api/eda/attrition")
    assert res_attr.status_code == 200
    data_attr = res_attr.json()
    assert "class_distribution" in data_attr
    assert "by_department" in data_attr

    res_promo = client.get("/api/eda/promotion")
    assert res_promo.status_code == 200
    data_promo = res_promo.json()
    assert "class_distribution" in data_promo


def test_dataset_explorer_endpoint():
    res_attr = client.get("/api/dataset?dataset_type=attrition&limit=10")
    assert res_attr.status_code == 200
    data = res_attr.json()
    assert "records" in data
    assert len(data["records"]) <= 10

    res_promo = client.get("/api/dataset?dataset_type=promotion&limit=10")
    assert res_promo.status_code == 200
