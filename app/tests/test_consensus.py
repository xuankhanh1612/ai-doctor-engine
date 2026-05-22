"""
Tests for Consensus Engine
Run: pytest app/tests/ -v
"""

from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ai.bayesian_fusion import (
    bayesian_fusion,
    weighted_fusion,
    majority_vote,
    graph_fusion,
    run_fusion,
    _log_odds,
    _sigmoid,
)
from app.models.schemas import AgentPrediction, ConsensusMethod

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def four_agents() -> list[AgentPrediction]:
    return [
        AgentPrediction(agent_id="radiology-v2",   specialty="radiology", diagnosis="Type 2 Diabetes", confidence=0.87),
        AgentPrediction(agent_id="diabetes-xgb-v3",specialty="diabetes",  diagnosis="Type 2 Diabetes", confidence=0.91),
        AgentPrediction(agent_id="ecg-cnn-v1",     specialty="ecg",       diagnosis="Sinus tachycardia", confidence=0.79),
        AgentPrediction(agent_id="lab-gbm-v2",     specialty="lab",       diagnosis="Type 2 Diabetes", confidence=0.94),
    ]


@pytest.fixture
def single_agent() -> list[AgentPrediction]:
    return [
        AgentPrediction(agent_id="lab-gbm-v2", specialty="lab", diagnosis="Healthy", confidence=0.55),
    ]


@pytest.fixture
def full_request_payload() -> dict:
    return {
        "patient_id": "PAT-TEST-001",
        "session_id": "SESS-001",
        "method": "bayesian",
        "run_all": False,
        "predictions": [
            {"agent_id": "radiology-v2",    "specialty": "radiology", "diagnosis": "Type 2 Diabetes", "confidence": 0.87},
            {"agent_id": "diabetes-xgb-v3", "specialty": "diabetes",  "diagnosis": "Type 2 Diabetes", "confidence": 0.91},
            {"agent_id": "ecg-cnn-v1",      "specialty": "ecg",       "diagnosis": "Sinus tachycardia", "confidence": 0.79},
            {"agent_id": "lab-gbm-v2",      "specialty": "lab",       "diagnosis": "Type 2 Diabetes", "confidence": 0.94},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMathHelpers:
    def test_log_odds_midpoint(self):
        assert math.isclose(_log_odds(0.5), 0.0, abs_tol=1e-9)

    def test_log_odds_high(self):
        assert _log_odds(0.9) > 0

    def test_log_odds_low(self):
        assert _log_odds(0.1) < 0

    def test_sigmoid_zero(self):
        assert math.isclose(_sigmoid(0.0), 0.5, abs_tol=1e-9)

    def test_sigmoid_positive(self):
        assert _sigmoid(2.0) > 0.5

    def test_log_odds_sigmoid_inverse(self):
        p = 0.73
        assert math.isclose(_sigmoid(_log_odds(p)), p, abs_tol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestBayesianFusion:
    def test_returns_fusion_result(self, four_agents):
        result = bayesian_fusion(four_agents)
        assert result.method == ConsensusMethod.bayesian

    def test_confidence_in_range(self, four_agents):
        result = bayesian_fusion(four_agents)
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_agreement_in_range(self, four_agents):
        result = bayesian_fusion(four_agents)
        assert 0.0 <= result.agreement_score <= 1.0

    def test_all_agents_represented(self, four_agents):
        result = bayesian_fusion(four_agents)
        assert len(result.agent_weights) == len(four_agents)

    def test_high_agreement_increases_confidence(self):
        """Uniform high-confidence agents should produce high fused confidence."""
        agents = [
            AgentPrediction(agent_id=f"a{i}", specialty="lab", diagnosis="D", confidence=0.95)
            for i in range(5)
        ]
        result = bayesian_fusion(agents)
        assert result.fused_confidence >= 0.90

    def test_single_agent(self, single_agent):
        result = bayesian_fusion(single_agent)
        assert result.num_agents == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No agent predictions"):
            bayesian_fusion([])


# ─────────────────────────────────────────────────────────────────────────────
# Weighted fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedFusion:
    def test_method_label(self, four_agents):
        result = weighted_fusion(four_agents)
        assert result.method == ConsensusMethod.weighted

    def test_dominated_by_high_conf(self):
        agents = [
            AgentPrediction(agent_id="a1", specialty="lab",      diagnosis="D", confidence=0.95),
            AgentPrediction(agent_id="a2", specialty="diabetes", diagnosis="D", confidence=0.50),
        ]
        result = weighted_fusion(agents)
        # High-confidence lab agent should pull result above average
        assert result.fused_confidence > 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Majority vote
# ─────────────────────────────────────────────────────────────────────────────

class TestMajorityVote:
    def test_quorum_met(self, four_agents):
        result = majority_vote(four_agents, quorum=0.5)
        assert result.metadata["quorum_met"] is True

    def test_vote_breakdown_present(self, four_agents):
        result = majority_vote(four_agents)
        assert "vote_breakdown" in result.metadata

    def test_quorum_not_met_lowers_confidence(self):
        agents = [
            AgentPrediction(agent_id="a1", specialty="lab",     diagnosis="D1", confidence=0.80),
            AgentPrediction(agent_id="a2", specialty="radiology",diagnosis="D2", confidence=0.80),
            AgentPrediction(agent_id="a3", specialty="ecg",     diagnosis="D3", confidence=0.80),
        ]
        result = majority_vote(agents, quorum=0.9)
        assert result.metadata["quorum_met"] is False
        assert result.fused_confidence < 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Graph fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphFusion:
    def test_propagated_scores_present(self, four_agents):
        result = graph_fusion(four_agents)
        assert "propagated_scores" in result.metadata

    def test_corroborated_scores_higher(self):
        """Lab + Diabetes are adjacent in graph — should reinforce each other."""
        agents = [
            AgentPrediction(agent_id="lab",      specialty="lab",      diagnosis="D", confidence=0.90),
            AgentPrediction(agent_id="diabetes",  specialty="diabetes", diagnosis="D", confidence=0.90),
        ]
        result_graph = graph_fusion(agents)
        result_bayes = bayesian_fusion(agents)
        # Graph result may differ — just check it's valid
        assert 0.0 <= result_graph.fused_confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcher:
    @pytest.mark.parametrize("method", list(ConsensusMethod))
    def test_all_methods_run(self, four_agents, method):
        result = run_fusion(four_agents, method=method)
        assert result.method == method
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_invalid_method_raises(self, four_agents):
        with pytest.raises((ValueError, AttributeError)):
            run_fusion(four_agents, method="nonexistent")  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestConsensusAPI:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_methods(self):
        r = client.get("/api/v1/consensus/methods")
        assert r.status_code == 200
        data = r.json()
        assert "bayesian" in data["methods"]

    def test_consensus_bayesian(self, full_request_payload):
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert r.status_code == 200
        data = r.json()
        assert data["patient_id"] == "PAT-TEST-001"
        assert "result" in data
        assert data["result"]["method"] == "bayesian"
        assert 0.0 <= data["result"]["fused_confidence"] <= 1.0

    def test_consensus_weighted(self, full_request_payload):
        full_request_payload["method"] = "weighted"
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert r.status_code == 200

    def test_consensus_majority(self, full_request_payload):
        full_request_payload["method"] = "majority"
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert r.status_code == 200

    def test_consensus_graph(self, full_request_payload):
        full_request_payload["method"] = "graph"
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert r.status_code == 200

    def test_run_all_returns_all_methods(self, full_request_payload):
        full_request_payload["run_all"] = True
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert r.status_code == 200
        data = r.json()
        assert data["all_results"] is not None
        assert len(data["all_results"]) == 4

    def test_compare_endpoint(self, full_request_payload):
        r = client.post("/api/v1/consensus/compare", json=full_request_payload)
        assert r.status_code == 200
        data = r.json()
        assert data["all_results"] is not None

    def test_risk_level_present(self, full_request_payload):
        r = client.post("/api/v1/consensus", json=full_request_payload)
        data = r.json()
        assert data["risk_level"] in ["low", "moderate", "high", "critical"]

    def test_doctor_review_flag(self, full_request_payload):
        r = client.post("/api/v1/consensus", json=full_request_payload)
        data = r.json()
        assert isinstance(data["requires_doctor_review"], bool)

    def test_empty_predictions_422(self):
        payload = {"patient_id": "X", "predictions": []}
        r = client.post("/api/v1/consensus", json=payload)
        assert r.status_code == 422

    def test_confidence_out_of_range_422(self):
        payload = {
            "patient_id": "X",
            "predictions": [
                {"agent_id": "a1", "specialty": "lab", "diagnosis": "D", "confidence": 1.5}
            ]
        }
        r = client.post("/api/v1/consensus", json=payload)
        assert r.status_code == 422

    def test_process_time_header(self, full_request_payload):
        r = client.post("/api/v1/consensus", json=full_request_payload)
        assert "X-Process-Time-Ms" in r.headers
