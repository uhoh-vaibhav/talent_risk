# Talent Risk Scoring System

A production-grade machine learning platform built for corporate HR departments. The system dual-predicts **attrition risk** (probability an employee leaves in the next 6-12 months) and **promotion readiness** (probability an employee is ready for advancement), combining both into an actionable **2x2 Talent Risk & Value Matrix**.

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Raw HR Data Ingestion  │
                          │ IBM HR & HR Promotion    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Unified Employee Schema  │
                          │ Data Validation & Hash   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Feature Store & Pipeline │
                          │ Sklearn Transformer v1.0 │
                          └─────────────┬────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │ Attrition XGBoost Model     │               │ Promotion XGBoost Model     │
  │ MLflow & Class Weighting    │               │ MLflow & Class Weighting    │
  └──────────────┬──────────────┘               └──────────────┬──────────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  SHAP & Fairness Audit   │
                          │ Disparate Impact (80%)   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ 2x2 Talent Matrix Engine │
                          │ Combined HR Action Label │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  FastAPI RESTful Service │
                          │ Single & Batch Endpoints │
                          └──────────────────────────┘
```

---

## 2x2 Talent Matrix Business Framing

Traditional HR systems view retention and advancement in isolation. This system unifies both dimensions to help HR leaders prioritize retention interventions and leadership development:

| Quadrant | Attrition Risk | Promotion Readiness | Recommended HR Action Strategy | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Urgent Retention & Key Talent** | **HIGH (>50%)** | **HIGH (>50%)** | **Immediate Intervention**: Conduct stay interview, review market equity compensation, accelerate promotion timeline, and assign executive mentorship. | **CRITICAL** |
| **Invest & Fast-Track** | LOW (≤50%) | **HIGH (>50%)** | **High-Potential Fast-Track**: Enroll in executive leadership programs, assign high-visibility cross-functional initiatives. | **HIGH** |
| **Monitor & Engage** | **HIGH (>50%)** | LOW (≤50%) | **Flight Risk Mitigation**: Conduct 1-on-1 workload & satisfaction reviews; evaluate manager dynamics to improve engagement. | **MEDIUM** |
| **Core Performer / Low Priority** | LOW (≤50%) | LOW (≤50%) | **Steady Engagement**: Maintain standard professional development, routine performance appraisals, and stable project assignments. | **LOW** |

---

## Stage-by-Stage Local Execution Guide

### Prerequisites
- Python 3.11+
- Virtual environment activated

### 1. Ingestion & Schema Unification
Fetch raw datasets (or trigger deterministic statistical synthesizer) and map into unified schema:
```bash
python -m src.ingestion.download
python -m src.ingestion.schema_mapper
```

### 2. Exploratory Data Analysis (EDA)
Launch the interactive Jupyter EDA notebook:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 3. Feature Engineering Pipeline
Fit and store the scikit-learn feature preprocessing pipeline:
```bash
python -m src.features.store
```

### 4. Model Training & MLflow Tracking
Train baseline Logistic Regression models and main XGBoost classifiers:
```bash
# Logistic Regression Baselines
python -m src.models.baseline

# Main XGBoost Models (logs to MLflow & runs Fairness Audit)
python -m src.models.train_attrition
python -m src.models.train_promotion
```

### 5. Serving API (FastAPI)
Run local Uvicorn dev server:
```bash
uvicorn src.serving.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation available at: `http://localhost:8000/docs`

### 6. Orchestration & Monitoring (Prefect DAG)
Execute end-to-end Prefect pipeline:
```bash
python -m pipelines.workflow
```

### 7. Unit Tests
Run full pytest test suite:
```bash
pytest tests/ -v
```

### 8. Containerization (Docker)
Build and run via Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## Bias & Fairness Audit Mechanism

To prevent algorithmic discrimination across demographic groups, the system automatically audits every model post-training using the **Disparate Impact Ratio (80% Rule)**:

$$\text{Disparate Impact} = \frac{P(\hat{Y}=1 \mid \text{Unprivileged Group})}{P(\hat{Y}=1 \mid \text{Privileged Group})}$$

- **Protected Attributes Audited**: `Gender` (Female vs Male) and `Age Group` (<30 vs ≥30).
- **Threshold**: Standard 0.80 threshold. If selection rate ratio falls below 0.80, a fairness violation warning is logged and retraining alerts are triggered.

---

## Non-Technical Business Brief for HR Stakeholders

### Executive Summary
Retaining key talent and identifying future leaders is critical for competitive corporate advantage. The **Talent Risk Scoring System** replaces reactive exit interviews with proactive ML predictions. 

By analyzing work patterns, satisfaction indicators, promotion history, and performance metrics, HR leaders receive a clear 2x2 matrix dashboard identifying exactly **who is at risk**, **who is ready to step up**, and **what specific actions HR should take today**.
"# talent_risk" 
