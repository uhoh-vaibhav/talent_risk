"""
FastAPI Pydantic Schemas
Defines request and response schemas for single and batch Talent Risk Scoring requests.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class EmployeeProfileRequest(BaseModel):
    """Employee feature request profile for scoring with support for both model domains."""
    employee_id: str = Field(..., json_schema_extra={"example": "EMP_00123"})
    department: str = Field(default="Operations", json_schema_extra={"example": "R&D"})
    gender: str = Field(default="Female", json_schema_extra={"example": "Female"})
    age: int = Field(default=30, ge=18, le=80, json_schema_extra={"example": 32})
    education_level: int = Field(default=3, ge=1, le=5, json_schema_extra={"example": 3})
    tenure_years: float = Field(default=3.0, ge=0, json_schema_extra={"example": 4.5})
    years_since_promotion: float = Field(default=1.5, ge=0, json_schema_extra={"example": 2.0})
    monthly_income: float = Field(default=6000.0, ge=500.0, json_schema_extra={"example": 6500.0})
    
    # Attrition-specific indicators with sensible defaults & aliases
    overtime_status: Optional[int] = Field(default=0, ge=0, le=1)
    overtime: Optional[int] = Field(default=None, ge=0, le=1)
    satisfaction_score: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    job_satisfaction: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    environment_satisfaction: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    work_life_balance: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    stock_option_level: int = Field(default=1, ge=0, le=3)
    marital_status: Optional[str] = Field(default="Single")
    distance_from_home: Optional[float] = Field(default=5.0, ge=0)
    job_involvement: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    job_level: Optional[int] = Field(default=2, ge=1, le=5)
    relationship_satisfaction: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    percent_salary_hike: Optional[float] = Field(default=14.0, ge=0)
    num_companies_worked: Optional[int] = Field(default=1, ge=0)

    # Promotion-specific indicators
    num_trainings_last_year: Optional[int] = Field(default=2, ge=0)
    no_of_trainings: Optional[int] = Field(default=None, ge=0)
    training_times_last_year: Optional[int] = Field(default=None, ge=0)
    performance_rating: float = Field(default=3.0, ge=1.0, le=5.0, json_schema_extra={"example": 3.5})
    previous_year_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    kpi_met_above_80: Optional[int] = Field(default=1, ge=0, le=1)
    kpis_met_above_80: Optional[int] = Field(default=None, ge=0, le=1)
    awards_won: int = Field(default=0, ge=0, le=1)
    avg_training_score: Optional[float] = Field(default=65.0, ge=0, le=100)
    recruitment_channel: Optional[str] = Field(default="sourcing")

    @model_validator(mode="after")
    def reconcile_aliases(self):
        # Harmonize overtime
        if self.overtime is not None:
            self.overtime_status = self.overtime
        else:
            self.overtime = self.overtime_status

        # Harmonize satisfaction
        if self.job_satisfaction is not None:
            self.satisfaction_score = self.job_satisfaction
        else:
            self.job_satisfaction = self.satisfaction_score

        # Harmonize trainings
        trainings = self.no_of_trainings or self.training_times_last_year or self.num_trainings_last_year or 2
        self.no_of_trainings = trainings
        self.training_times_last_year = trainings
        self.num_trainings_last_year = trainings

        # Harmonize KPI
        if self.kpis_met_above_80 is not None:
            self.kpi_met_above_80 = self.kpis_met_above_80
        else:
            self.kpis_met_above_80 = self.kpi_met_above_80

        # Harmonize rating
        if self.previous_year_rating is None:
            self.previous_year_rating = self.performance_rating

        return self


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
    attrition_threshold_applied: float
    promotion_threshold_applied: float
    top_attrition_drivers: List[ShapDriver]
    top_promotion_drivers: List[ShapDriver]
    disclaimer: Optional[str] = None


class BatchTalentScoreResponse(BaseModel):
    """Batch prediction response container."""
    total_processed: int
    predictions: List[TalentScoreResponse]
