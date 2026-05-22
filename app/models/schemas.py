"""
Pydantic schemas — request / response models for Consensus Engine API
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusMethod(str, Enum):
    bayesian = "bayesian"
    weighted = "weighted"
    majority = "majority"
    graph    = "graph"


class RiskLevel(str, Enum):
    low      = "low"
    moderate = "moderate"
    high     = "high"
    critical = "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Agent prediction (input)
# ─────────────────────────────────────────────────────────────────────────────

class AgentPrediction(BaseModel):
    agent_id:    str   = Field(..., description="Unique agent identifier, e.g. 'radiology-v2'")
    specialty:   str   = Field(..., description="Medical specialty: radiology, diabetes, ecg, …")
    diagnosis:   str   = Field(..., description="Human-readable diagnosis label")
    confidence:  float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")
    icd10_code:  Optional[str]  = Field(None, description="ICD-10 code if available")
    raw_score:   Optional[float]= Field(None, description="Raw model output before sigmoid")
    metadata:    Optional[dict[str, Any]] = Field(None, description="SHAP values, features, etc.")

    @field_validator("specialty")
    @classmethod
    def normalise_specialty(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("diagnosis")
    @classmethod
    def non_empty_diagnosis(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("diagnosis must not be empty")
        return v.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Consensus request
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusRequest(BaseModel):
    patient_id:   str   = Field(..., description="Patient identifier (anonymised)")
    session_id:   Optional[str] = Field(None, description="Clinical session / visit ID")
    predictions:  list[AgentPrediction] = Field(..., min_length=1, description="Agent outputs")
    method:       ConsensusMethod = Field(ConsensusMethod.bayesian, description="Fusion method")
    quorum:       float = Field(0.6, ge=0.0, le=1.0, description="Vote quorum for majority method")
    graph_iters:  int   = Field(2, ge=1, le=5, description="Propagation iterations for graph method")
    run_all:      bool  = Field(False, description="If true, run all 4 methods and return comparison")

    @model_validator(mode="after")
    def at_least_one_agent(self) -> "ConsensusRequest":
        if len(self.predictions) == 0:
            raise ValueError("At least one agent prediction is required")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "PAT-2024-00042",
                "session_id": "SESS-20240521-001",
                "method": "bayesian",
                "run_all": False,
                "predictions": [
                    {
                        "agent_id": "radiology-v2",
                        "specialty": "radiology",
                        "diagnosis": "Type 2 Diabetes — early retinopathy",
                        "confidence": 0.87,
                        "icd10_code": "E11.3"
                    },
                    {
                        "agent_id": "diabetes-xgb-v3",
                        "specialty": "diabetes",
                        "diagnosis": "Type 2 Diabetes — metabolic syndrome",
                        "confidence": 0.91,
                        "icd10_code": "E11.9"
                    },
                    {
                        "agent_id": "ecg-cnn-v1",
                        "specialty": "ecg",
                        "diagnosis": "Sinus tachycardia — secondary finding",
                        "confidence": 0.79,
                        "icd10_code": "R00.0"
                    },
                    {
                        "agent_id": "lab-gbm-v2",
                        "specialty": "lab",
                        "diagnosis": "HbA1c 7.2% — diabetes confirmed",
                        "confidence": 0.94,
                        "icd10_code": "R73.09"
                    }
                ]
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal weight detail (output)
# ─────────────────────────────────────────────────────────────────────────────

class AgentWeight(BaseModel):
    agent_id:     str
    specialty:    str
    confidence:   float
    weight:       float
    contribution: float


# ─────────────────────────────────────────────────────────────────────────────
# Fusion result
# ─────────────────────────────────────────────────────────────────────────────

class FusionResult(BaseModel):
    method:           ConsensusMethod
    fused_confidence: float = Field(..., description="Final probability 0–1")
    agreement_score:  float = Field(..., description="Inter-agent agreement 0–1")
    dominant_agent:   str
    diagnosis:        str
    recommendation:   str
    agent_weights:    list[AgentWeight]
    num_agents:       int
    metadata:         Optional[dict[str, Any]] = None

    @property
    def risk_level(self) -> RiskLevel:
        if self.fused_confidence >= 0.90:
            return RiskLevel.critical
        elif self.fused_confidence >= 0.75:
            return RiskLevel.high
        elif self.fused_confidence >= 0.55:
            return RiskLevel.moderate
        return RiskLevel.low


# ─────────────────────────────────────────────────────────────────────────────
# API response
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusResponse(BaseModel):
    patient_id:   str
    session_id:   Optional[str]
    timestamp:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result:       FusionResult
    all_results:  Optional[dict[str, FusionResult]] = Field(
        None, description="Populated when run_all=true"
    )
    risk_level:   RiskLevel
    requires_doctor_review: bool


class HealthResponse(BaseModel):
    status:  str
    version: str
    methods: list[str]
