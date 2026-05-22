"""
Bayesian Fusion Engine for Multi-Agent Medical Consensus
=========================================================
Combines confidence scores from multiple AI specialist agents
using Bayesian inference to produce a single fused diagnosis.

Methods supported:
  1. bayesian  — log-odds Bayesian fusion (default)
  2. weighted  — confidence-weighted average
  3. majority  — quorum vote with tie-breaking
  4. graph     — graph-based evidence propagation (simplified GNN-style)
"""

from __future__ import annotations

import math
import logging
from enum import Enum
from typing import Optional

from app.models.schemas import (
    AgentPrediction,
    ConsensusMethod,
    FusionResult,
    AgentWeight,
)

logger = logging.getLogger(__name__)

# ── Specialty weights (prior domain reliability from literature) ──────────────
SPECIALTY_PRIORS: dict[str, float] = {
    "radiology":  0.88,
    "diabetes":   0.91,
    "ecg":        0.84,
    "oncology":   0.86,
    "lab":        0.93,
    "pathology":  0.89,
    "neurology":  0.82,
    "cardiology": 0.87,
}
DEFAULT_PRIOR = 0.85

# ── Graph adjacency — which specialties reinforce each other ──────────────────
SPECIALTY_GRAPH: dict[str, list[str]] = {
    "diabetes":   ["lab", "cardiology", "radiology"],
    "lab":        ["diabetes", "oncology", "pathology"],
    "cardiology": ["ecg", "radiology"],
    "ecg":        ["cardiology", "radiology"],
    "radiology":  ["oncology", "pathology", "cardiology"],
    "oncology":   ["pathology", "radiology", "lab"],
    "pathology":  ["oncology", "lab"],
    "neurology":  ["radiology"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, value))


def _log_odds(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _prior_for(specialty: str) -> float:
    return SPECIALTY_PRIORS.get(specialty.lower(), DEFAULT_PRIOR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bayesian log-odds fusion
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_fusion(predictions: list[AgentPrediction]) -> FusionResult:
    """
    Naive-Bayes fusion in log-odds space.

    For each agent i with confidence c_i and specialty prior p_i:
      log_odds_i = log(c_i / (1 - c_i))
      weight_i   = log(p_i / (1 - p_i))   ← how much we trust agent i
      contribution_i = log_odds_i * weight_i / sum(all weights)

    Final probability = sigmoid(sum of contributions)
    """
    if not predictions:
        raise ValueError("No agent predictions provided")

    total_weight = 0.0
    weighted_log_odds = 0.0
    agent_weights: list[AgentWeight] = []

    for pred in predictions:
        prior = _prior_for(pred.specialty)
        specialty_weight = _log_odds(prior)          # trust weight
        agent_log_odds   = _log_odds(pred.confidence)
        contribution     = agent_log_odds * specialty_weight

        weighted_log_odds += contribution
        total_weight      += specialty_weight

        agent_weights.append(AgentWeight(
            agent_id   = pred.agent_id,
            specialty  = pred.specialty,
            confidence = pred.confidence,
            weight     = round(specialty_weight, 4),
            contribution = round(contribution, 4),
        ))

    # Normalise
    if total_weight > 0:
        weighted_log_odds /= total_weight

    fused_confidence = _sigmoid(weighted_log_odds)

    # Agreement: std-dev of confidences (low std = high agreement)
    confs = [p.confidence for p in predictions]
    mean_c = sum(confs) / len(confs)
    variance = sum((c - mean_c) ** 2 for c in confs) / len(confs)
    agreement_score = round(1.0 - math.sqrt(variance), 4)

    # Dominant diagnosis (highest individual confidence)
    dominant = max(predictions, key=lambda p: p.confidence)

    return FusionResult(
        method            = ConsensusMethod.bayesian,
        fused_confidence  = round(fused_confidence, 4),
        agreement_score   = agreement_score,
        dominant_agent    = dominant.agent_id,
        diagnosis         = dominant.diagnosis,
        recommendation    = _build_recommendation(fused_confidence, dominant),
        agent_weights     = agent_weights,
        num_agents        = len(predictions),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Weighted confidence average
# ─────────────────────────────────────────────────────────────────────────────

def weighted_fusion(predictions: list[AgentPrediction]) -> FusionResult:
    """Weighted average — confidence × specialty_prior, normalised."""
    if not predictions:
        raise ValueError("No agent predictions provided")

    total_w = 0.0
    fused   = 0.0
    agent_weights: list[AgentWeight] = []

    for pred in predictions:
        w = pred.confidence * _prior_for(pred.specialty)
        fused   += pred.confidence * w
        total_w += w
        agent_weights.append(AgentWeight(
            agent_id    = pred.agent_id,
            specialty   = pred.specialty,
            confidence  = pred.confidence,
            weight      = round(w, 4),
            contribution= round(pred.confidence * w, 4),
        ))

    fused_confidence = fused / total_w if total_w else 0.0

    confs = [p.confidence for p in predictions]
    mean_c = sum(confs) / len(confs)
    variance = sum((c - mean_c) ** 2 for c in confs) / len(confs)
    agreement_score = round(1.0 - math.sqrt(variance), 4)

    dominant = max(predictions, key=lambda p: p.confidence)

    return FusionResult(
        method           = ConsensusMethod.weighted,
        fused_confidence = round(fused_confidence, 4),
        agreement_score  = agreement_score,
        dominant_agent   = dominant.agent_id,
        diagnosis        = dominant.diagnosis,
        recommendation   = _build_recommendation(fused_confidence, dominant),
        agent_weights    = agent_weights,
        num_agents       = len(predictions),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Majority vote with confidence tie-breaking
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(predictions: list[AgentPrediction], quorum: float = 0.6) -> FusionResult:
    """
    Agents vote for their diagnosis.
    If a diagnosis reaches `quorum` fraction of votes → accepted.
    Ties broken by summed confidence.
    """
    if not predictions:
        raise ValueError("No agent predictions provided")

    vote_counts: dict[str, float] = {}
    vote_confs:  dict[str, float] = {}

    for pred in predictions:
        d = pred.diagnosis.lower().strip()
        vote_counts[d] = vote_counts.get(d, 0) + 1
        vote_confs[d]  = vote_confs.get(d, 0) + pred.confidence

    total_votes = len(predictions)
    winning_diag = max(vote_counts, key=lambda d: (vote_counts[d], vote_confs[d]))
    winning_votes = vote_counts[winning_diag]
    vote_fraction = winning_votes / total_votes
    quorum_met = vote_fraction >= quorum

    # fused_confidence = fraction × mean confidence of winners
    winners = [p for p in predictions if p.diagnosis.lower().strip() == winning_diag]
    mean_winner_conf = sum(p.confidence for p in winners) / len(winners)
    fused_confidence = vote_fraction * mean_winner_conf if quorum_met else mean_winner_conf * 0.6

    agent_weights: list[AgentWeight] = []
    for pred in predictions:
        voted_for_winner = pred.diagnosis.lower().strip() == winning_diag
        agent_weights.append(AgentWeight(
            agent_id    = pred.agent_id,
            specialty   = pred.specialty,
            confidence  = pred.confidence,
            weight      = 1.0,
            contribution= pred.confidence if voted_for_winner else 0.0,
        ))

    dominant = max(winners, key=lambda p: p.confidence)

    return FusionResult(
        method           = ConsensusMethod.majority,
        fused_confidence = round(fused_confidence, 4),
        agreement_score  = round(vote_fraction, 4),
        dominant_agent   = dominant.agent_id,
        diagnosis        = dominant.diagnosis,
        recommendation   = _build_recommendation(fused_confidence, dominant),
        agent_weights    = agent_weights,
        num_agents       = len(predictions),
        metadata         = {
            "quorum_required": quorum,
            "quorum_met":      quorum_met,
            "vote_fraction":   round(vote_fraction, 3),
            "vote_breakdown":  {d: int(v) for d, v in vote_counts.items()},
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph-based evidence propagation
# ─────────────────────────────────────────────────────────────────────────────

def graph_fusion(predictions: list[AgentPrediction], iterations: int = 2) -> FusionResult:
    """
    Simplified message-passing on the specialty graph.
    Each agent propagates its confidence to connected specialties,
    amplifying evidence that is corroborated by adjacent domains.
    """
    if not predictions:
        raise ValueError("No agent predictions provided")

    # Build lookup: specialty → prediction
    spec_map: dict[str, AgentPrediction] = {
        p.specialty.lower(): p for p in predictions
    }

    # Initialise scores
    scores: dict[str, float] = {s: p.confidence for s, p in spec_map.items()}

    # Message-passing iterations
    for _ in range(iterations):
        new_scores: dict[str, float] = {}
        for spec, score in scores.items():
            neighbors = SPECIALTY_GRAPH.get(spec, [])
            present   = [n for n in neighbors if n in scores]
            if present:
                neighbor_mean = sum(scores[n] for n in present) / len(present)
                # Bayesian update: blend own score with neighbor evidence
                new_scores[spec] = _clamp(0.7 * score + 0.3 * neighbor_mean)
            else:
                new_scores[spec] = score
        scores = new_scores

    fused_confidence = sum(scores.values()) / len(scores)

    agent_weights: list[AgentWeight] = []
    for pred in predictions:
        spec   = pred.specialty.lower()
        final  = scores.get(spec, pred.confidence)
        delta  = final - pred.confidence
        agent_weights.append(AgentWeight(
            agent_id    = pred.agent_id,
            specialty   = pred.specialty,
            confidence  = pred.confidence,
            weight      = round(final, 4),
            contribution= round(delta, 4),
        ))

    dominant = max(predictions, key=lambda p: scores.get(p.specialty.lower(), p.confidence))

    confs = list(scores.values())
    mean_c = sum(confs) / len(confs)
    variance = sum((c - mean_c) ** 2 for c in confs) / len(confs)
    agreement_score = round(1.0 - math.sqrt(variance), 4)

    return FusionResult(
        method           = ConsensusMethod.graph,
        fused_confidence = round(fused_confidence, 4),
        agreement_score  = agreement_score,
        dominant_agent   = dominant.agent_id,
        diagnosis        = dominant.diagnosis,
        recommendation   = _build_recommendation(fused_confidence, dominant),
        agent_weights    = agent_weights,
        num_agents       = len(predictions),
        metadata         = {
            "graph_iterations": iterations,
            "propagated_scores": {s: round(v, 4) for s, v in scores.items()},
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion(
    predictions: list[AgentPrediction],
    method: ConsensusMethod = ConsensusMethod.bayesian,
    **kwargs,
) -> FusionResult:
    dispatcher = {
        ConsensusMethod.bayesian: bayesian_fusion,
        ConsensusMethod.weighted: weighted_fusion,
        ConsensusMethod.majority: majority_vote,
        ConsensusMethod.graph:    graph_fusion,
    }
    fn = dispatcher.get(method)
    if fn is None:
        raise ValueError(f"Unknown consensus method: {method}")
    logger.info("Running %s fusion on %d agents", method.value, len(predictions))
    return fn(predictions, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_recommendation(fused_confidence: float, dominant: AgentPrediction) -> str:
    if fused_confidence >= 0.90:
        urgency = "High confidence — recommend immediate clinical action"
    elif fused_confidence >= 0.75:
        urgency = "Moderate confidence — confirm with additional tests"
    elif fused_confidence >= 0.60:
        urgency = "Low-moderate confidence — multi-disciplinary review advised"
    else:
        urgency = "Low confidence — further diagnostic workup required"

    return f"{urgency}. Lead specialist: {dominant.specialty} ({dominant.confidence:.0%} confidence)."
