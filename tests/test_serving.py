"""
Unit tests for FastAPI endpoints.
"""

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


def test_single_prediction_endpoint():
    sample_payload = {
        "employee_id": "TEST_EMP_99",
        "department": "R&D",
        "gender": "Female",
        "age": 30,
        "education_level": 3,
        "tenure_years": 5.0,
        "years_since_promotion": 2.5,
        "num_trainings_last_year": 2,
        "performance_rating": 3.5,
        "kpi_met_above_80": 1,
        "awards_won": 0,
        "overtime_status": 1,
        "satisfaction_score": 2.5,
        "monthly_income": 6200.0,
        "stock_option_level": 1
    }

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["employee_id"] == "TEST_EMP_99"
    assert 0.0 <= data["attrition_risk_score"] <= 1.0
    assert 0.0 <= data["promotion_readiness_score"] <= 1.0
    assert "quadrant" in data
    assert "hr_action_recommendation" in data
    assert isinstance(data["top_attrition_drivers"], list)


def test_fairness_report_endpoint():
    response = client.get("/fairness-report")
    assert response.status_code == 200
    data = response.json()
    assert "disparate_impact_threshold" in data
    assert data["disparate_impact_threshold"] == 0.80
