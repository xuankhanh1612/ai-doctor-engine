# Consensus Engine — FastAPI Server

> Multi-Agent Medical Consensus API cho AI Clinic  
> Fuses predictions từ các specialist AI agents thành một chẩn đoán hội đồng duy nhất.

---

## Cấu trúc project

```
consensus_engine/
├── app/
│   ├── main.py                  # FastAPI app, CORS, middleware, health check
│   ├── routes/
│   │   └── consensus.py         # 3 endpoints: /consensus, /compare, /methods
│   ├── services/
│   │   └── consensus_service.py # Orchestrator: gọi fusion, tính risk, flag review
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── ai/
│   │   └── bayesian_fusion.py   # 4 thuật toán: Bayesian, Weighted, Majority, Graph
│   └── tests/
│       └── test_consensus.py    # 21 pytest cases
├── Dockerfile
├── requirements.txt
└── USAGE.sh
```

---

## Cài đặt & chạy

```bash
# 1. Vào thư mục
cd consensus_engine

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Chạy server
uvicorn app.main:app --reload

# Server chạy tại: http://127.0.0.1:8000
# Swagger UI:       http://127.0.0.1:8000/docs
# ReDoc:            http://127.0.0.1:8000/redoc
```

---

## Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/v1/consensus` | Chạy một fusion method |
| `POST` | `/api/v1/consensus/compare` | Chạy tất cả 4 methods, trả về so sánh |
| `GET`  | `/api/v1/consensus/methods` | Liệt kê methods có sẵn |
| `GET`  | `/health` | Health check |

---

## 4 thuật toán Fusion

### 1. Bayesian (mặc định)
Chuyển confidence của mỗi agent thành **log-odds**, nhân với **specialty prior weight** (mức độ tin cậy lịch sử của chuyên khoa đó), sau đó tổng hợp và sigmoid ngược lại.

```
log_odds_i  = log(confidence_i / (1 - confidence_i))
weight_i    = log(specialty_prior_i / (1 - specialty_prior_i))
final       = sigmoid( Σ(log_odds_i × weight_i) / Σ(weight_i) )
```

Specialty priors được định nghĩa trong `bayesian_fusion.py`:

| Specialty | Prior |
|-----------|-------|
| lab | 0.93 |
| diabetes | 0.91 |
| radiology | 0.88 |
| pathology | 0.89 |
| oncology | 0.86 |
| ecg | 0.84 |

### 2. Weighted
Weighted average đơn giản: `confidence × specialty_prior`, normalize.  
Nhanh hơn Bayesian, phù hợp cho real-time inference.

### 3. Majority Vote
Agents bỏ phiếu cho diagnosis của mình. Cần đạt **quorum** (mặc định 60%) để kết quả được chấp nhận. Tie-breaking dựa trên tổng confidence.

### 4. Graph Propagation
Message-passing trên **specialty adjacency graph**. Agents liền kề (ví dụ lab ↔ diabetes ↔ cardiology) reinforce lẫn nhau qua 2 iterations. Phát hiện cluster bằng chứng xuyên chuyên khoa.

---

## Request schema

```json
POST /api/v1/consensus
{
  "patient_id": "PAT-2024-00042",
  "session_id": "SESS-001",
  "method": "bayesian",
  "run_all": false,
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
    }
  ]
}
```

Các field trong `predictions[]`:

| Field | Bắt buộc | Mô tả |
|-------|----------|-------|
| `agent_id` | ✓ | ID duy nhất của agent |
| `specialty` | ✓ | Chuyên khoa: `radiology`, `diabetes`, `ecg`, `lab`, `oncology`… |
| `diagnosis` | ✓ | Nhãn chẩn đoán dạng text |
| `confidence` | ✓ | Điểm tin cậy `0.0 – 1.0` |
| `icd10_code` | — | Mã ICD-10 nếu có |
| `metadata` | — | SHAP values, key findings, v.v. |

---

## Response schema

```json
{
  "patient_id": "PAT-2024-00042",
  "session_id": "SESS-001",
  "timestamp": "2024-05-21T10:30:00Z",
  "risk_level": "high",
  "requires_doctor_review": false,
  "result": {
    "method": "bayesian",
    "fused_confidence": 0.8964,
    "agreement_score": 0.9437,
    "dominant_agent": "lab-gbm-v2",
    "diagnosis": "Type 2 Diabetes — metabolic syndrome",
    "recommendation": "High confidence — recommend immediate clinical action. Lead specialist: lab (94% confidence).",
    "num_agents": 4,
    "agent_weights": [
      {
        "agent_id": "radiology-v2",
        "specialty": "radiology",
        "confidence": 0.87,
        "weight": 2.0001,
        "contribution": 1.7400
      }
    ]
  },
  "all_results": null
}
```

Khi `run_all: true`, `all_results` chứa kết quả của cả 4 methods:

```json
"all_results": {
  "bayesian": { ... },
  "weighted": { ... },
  "majority": { ... },
  "graph":    { ... }
}
```

---

## Risk levels

| Level | Điều kiện | Ý nghĩa |
|-------|-----------|---------|
| `low` | confidence < 55% | Cần thêm xét nghiệm |
| `moderate` | 55–75% | Review đa chuyên khoa |
| `high` | 75–90% | Xác nhận bằng test bổ sung |
| `critical` | ≥ 90% | Hành động lâm sàng ngay |

`requires_doctor_review: true` khi `fused_confidence < 0.70` **hoặc** `agreement_score < 0.65`.

---

## Test

```bash
pytest app/tests/ -v
# 21 tests — math helpers, 4 fusion algorithms, API endpoints
```

---

## Docker

```bash
# Build
docker build -t consensus-engine .

# Run
docker run -p 8000:8000 consensus-engine
```

`docker-compose.yml` (kết hợp với frontend):

```yaml
services:
  backend:
    build: ./consensus_engine
    ports:
      - "8000:8000"

  frontend:
    build: ./ai-doctor-admin
    ports:
      - "5173:5173"
    environment:
      - VITE_CONSENSUS_API_URL=http://backend:8000
```

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `PORT` | `8000` | Cổng server (uvicorn) |
| `WORKERS` | `2` | Số worker processes |

---

## Mở rộng thêm agent

Để thêm specialty mới (ví dụ `neurology`):

1. Thêm prior vào `SPECIALTY_PRIORS` trong `bayesian_fusion.py`:
   ```python
   SPECIALTY_PRIORS["neurology"] = 0.82
   ```
2. Thêm adjacency vào `SPECIALTY_GRAPH`:
   ```python
   SPECIALTY_GRAPH["neurology"] = ["radiology", "pathology"]
   ```
3. Gửi prediction với `"specialty": "neurology"` trong request — không cần thay đổi schema.
