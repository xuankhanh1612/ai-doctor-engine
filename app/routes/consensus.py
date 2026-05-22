"""
Consensus Router
================
POST /api/v1/consensus          — run fusion on agent predictions
POST /api/v1/consensus/compare  — run all 4 methods, return side-by-side
GET  /api/v1/consensus/methods  — list available methods
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    ConsensusRequest,
    ConsensusResponse,
    ConsensusMethod,
    HealthResponse,
)
from app.services.consensus_service import consensus_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consensus", tags=["Consensus Engine"])


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/consensus
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ConsensusResponse,
    status_code=status.HTTP_200_OK,
    summary="Run consensus fusion on agent predictions",
    description=(
        "Accepts predictions from multiple medical AI agents and fuses them "
        "into a single consensus diagnosis using the specified method. "
        "Set `run_all=true` to compare all four methods in one call."
    ),
)
async def run_consensus(req: ConsensusRequest) -> ConsensusResponse:
    try:
        return consensus_service.process(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Consensus engine error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal fusion error — check server logs",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/consensus/compare
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/compare",
    response_model=ConsensusResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare all fusion methods side-by-side",
    description="Convenience endpoint — forces run_all=true regardless of request body.",
)
async def compare_methods(req: ConsensusRequest) -> ConsensusResponse:
    req = req.model_copy(update={"run_all": True})
    try:
        return consensus_service.process(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Consensus compare error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal fusion error — check server logs",
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/v1/consensus/methods
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/methods",
    summary="List available consensus methods",
)
async def list_methods() -> dict:
    return {
        "methods": [m.value for m in ConsensusMethod],
        "descriptions": {
            "bayesian": "Log-odds Bayesian fusion weighted by specialty prior reliability",
            "weighted": "Confidence × specialty-prior weighted average",
            "majority": "Quorum-based majority vote with confidence tie-breaking",
            "graph":    "Graph evidence propagation across specialty adjacency network",
        },
        "default": ConsensusMethod.bayesian.value,
    }
