"""
Consensus Service
=================
Orchestrates fusion engine calls, handles run_all mode,
computes risk level, and flags cases for doctor review.
"""

from __future__ import annotations

import logging
from app.ai.bayesian_fusion import run_fusion
from app.models.schemas import (
    ConsensusMethod,
    ConsensusRequest,
    ConsensusResponse,
    FusionResult,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# Low agreement or low confidence always triggers doctor review
DOCTOR_REVIEW_CONF_THRESHOLD      = 0.70
DOCTOR_REVIEW_AGREEMENT_THRESHOLD = 0.65


class ConsensusService:

    def process(self, req: ConsensusRequest) -> ConsensusResponse:
        if req.run_all:
            all_results  = self._run_all_methods(req)
            primary      = all_results[req.method.value]
        else:
            primary      = self._run_method(req, req.method)
            all_results  = None

        risk_level = self._compute_risk(primary)
        needs_review = self._needs_doctor_review(primary)

        logger.info(
            "patient=%s method=%s conf=%.3f risk=%s review=%s",
            req.patient_id, req.method.value,
            primary.fused_confidence, risk_level.value, needs_review,
        )

        return ConsensusResponse(
            patient_id             = req.patient_id,
            session_id             = req.session_id,
            result                 = primary,
            all_results            = all_results,
            risk_level             = risk_level,
            requires_doctor_review = needs_review,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_method(self, req: ConsensusRequest, method: ConsensusMethod) -> FusionResult:
        kwargs: dict = {}
        if method == ConsensusMethod.majority:
            kwargs["quorum"] = req.quorum
        elif method == ConsensusMethod.graph:
            kwargs["iterations"] = req.graph_iters
        return run_fusion(req.predictions, method=method, **kwargs)

    def _run_all_methods(self, req: ConsensusRequest) -> dict[str, FusionResult]:
        results: dict[str, FusionResult] = {}
        for method in ConsensusMethod:
            try:
                results[method.value] = self._run_method(req, method)
            except Exception as exc:
                logger.warning("Method %s failed: %s", method.value, exc)
        return results

    @staticmethod
    def _compute_risk(result: FusionResult) -> RiskLevel:
        return result.risk_level

    @staticmethod
    def _needs_doctor_review(result: FusionResult) -> bool:
        return (
            result.fused_confidence  < DOCTOR_REVIEW_CONF_THRESHOLD or
            result.agreement_score   < DOCTOR_REVIEW_AGREEMENT_THRESHOLD
        )


# Singleton
consensus_service = ConsensusService()
