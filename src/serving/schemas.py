"""
FastAPI Pydantic Schemas
Defines request and response schemas for single and batch Talent Risk Scoring requests.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EmployeeProfileRequest(BaseModel):
    """Employee feature request profile for scoring."""
    employee_id: str = Field(..., json_schema_extra={"example": "EMP_00123"})
    department: str = Field(..., json_schema_extra={"example": "R&D"})
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    age: int = Field(..., ge=18, le=80, json_schema_extra={"example": 32})
    education_level: int = Field(..., ge=1, le=5, json_schema_extra={"example": 3})
    tenure_years: float = Field(..., ge=0, json_schema_extra={"example": 4.5})
    years_since_promotion: float = Field(..., ge=0, json_schema_extra={"example": 2.0})
    num_trainings_last_year: int = Field(..., ge=0, json_schema_extra={"example": 2})
    performance_rating: float = Field(..., ge=1.0, le=5.0, json_schema_extra={"example": 3.5})
    kpi_met_above_80: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    awards_won: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    overtime_status: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    satisfaction_score: float = Field(..., ge=1.0, le=5.0, json_schema_extra={"example": 2.8})
    monthly_income: float = Field(..., ge=1000.0, json_schema_extra={"example": 6500.0})
    stock_option_level: int = Field(..., ge=0, le=3, json_schema_extra={"example": 1})


class BatchEmployeeProfileRequest(BaseModel):
    """Batch scoring request payload."""
    employees: List[EmployeeProfileRequest]


class ShapDriver(BaseModel):
    """SHAP feature attribution driver."""
    feature_name: str
    shap_impact: float
    feature_value: float
    direction: str


class TalentScoreResponse(BaseModel):
    """Prediction output containing dual risk scores, 2x2 matrix, SHAP drivers, and HR action."""
    employee_id: str
    employee_number: Optional[str] = None
    is_new_record: Optional[bool] = False
    attrition_risk_score: float
    promotion_readiness_score: float
    quadrant: str
    action_code: str
    priority_level: str
    hr_action_recommendation: str
    top_attrition_drivers: List[ShapDriver]
    top_promotion_drivers: List[ShapDriver]


class BatchTalentScoreResponse(BaseModel):
    """Batch prediction response container."""
    total_processed: int
    predictions: List[TalentScoreResponse]
