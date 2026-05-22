# ── Consensus Engine — Usage Examples ────────────────────────────────────────

# 1. Health check
curl http://localhost:8000/health

# 2. Bayesian fusion (default)
curl -X POST http://localhost:8000/api/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT-2024-00042",
    "session_id": "SESS-20240521-001",
    "method": "bayesian",
    "predictions": [
      {"agent_id":"radiology-v2",    "specialty":"radiology","diagnosis":"Type 2 Diabetes","confidence":0.87,"icd10_code":"E11.3"},
      {"agent_id":"diabetes-xgb-v3", "specialty":"diabetes", "diagnosis":"Type 2 Diabetes","confidence":0.91,"icd10_code":"E11.9"},
      {"agent_id":"ecg-cnn-v1",      "specialty":"ecg",      "diagnosis":"Sinus tachycardia","confidence":0.79,"icd10_code":"R00.0"},
      {"agent_id":"lab-gbm-v2",      "specialty":"lab",      "diagnosis":"HbA1c elevated","confidence":0.94,"icd10_code":"R73.09"}
    ]
  }'

# 3. Compare all 4 methods at once
curl -X POST http://localhost:8000/api/v1/consensus/compare \
  -H "Content-Type: application/json" \
  -d '{ ... same body ... }'

# 4. List methods
curl http://localhost:8000/api/v1/consensus/methods

# ── JavaScript (Next.js frontend) ────────────────────────────────────────────

async function runConsensus(patientId, agentPredictions, method = "bayesian") {
  const res = await fetch("/api/v1/consensus", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      method,
      predictions: agentPredictions,
    }),
  });

  if (!res.ok) throw new Error(`Consensus API error: ${res.status}`);
  const data = await res.json();

  return {
    diagnosis:       data.result.diagnosis,
    confidence:      data.result.fused_confidence,
    agreement:       data.result.agreement_score,
    riskLevel:       data.risk_level,
    needsReview:     data.requires_doctor_review,
    recommendation:  data.result.recommendation,
    agentWeights:    data.result.agent_weights,
    allMethods:      data.all_results,    // populated when run_all=true
  };
}

// Usage in a Next.js page
const result = await runConsensus("PAT-001", [
  { agent_id: "radiology-v2",    specialty: "radiology", diagnosis: "Type 2 Diabetes", confidence: 0.87 },
  { agent_id: "diabetes-xgb-v3", specialty: "diabetes",  diagnosis: "Type 2 Diabetes", confidence: 0.91 },
  { agent_id: "lab-gbm-v2",      specialty: "lab",       diagnosis: "HbA1c elevated",  confidence: 0.94 },
]);

console.log(result.diagnosis);    // "Type 2 Diabetes — metabolic syndrome"
console.log(result.confidence);   // 0.8964
console.log(result.riskLevel);    // "high"
console.log(result.needsReview);  // false
