# Production Vision OCR & Enterprise Batch Processing Platform

A high-performance, asynchronous FastAPI platform, SQLAlchemy persistence engine, and Streamlit dashboard for clinical document OCR, structured JSON extraction, and high-throughput multi-document batch pipelines powered directly by GPU-accelerated local Vision LLMs (**Ollama** / **ministral-3:latest**).

---

## 1. System Capabilities

- **Direct Vision Engine Connectivity**: Direct integration with local GPU inference backends:
  - **Ollama** (`http://localhost:11434`) – Default Production Model: `ministral-3:latest`
  - **llama.cpp / llama-server** (`http://localhost:8080` / `http://localhost:8081`)
- **Synchronous & Streaming Single OCR (`/ocr/api/ocr`, `/ocr/api/ocr/stream`)**: Direct single-document extraction with real-time Server-Sent Events (SSE) streaming and strict Pydantic V2 schema validation.
- **Enterprise Multi-Document Batch Processing (`/ocr/api/batch`)**:
  - Submits up to 100 PDF lab reports or image scans in a single multipart request.
  - Real-time batch progress tracking (`total_files`, `processed_files`, `failed_files`, `progress_percentage`).
  - Downloads results as a consolidated JSON or a ZIP archive containing individual report files.
- **Resilient Database Persistence**:
  - Backed by **SQLAlchemy 2.0 (Async)** with native **SQLite** (`aiosqlite`) and **PostgreSQL** (`asyncpg`) support.
  - Batches, individual document jobs, and webhook delivery audit logs are permanently stored and queryable.
- **HMAC-SHA256 Webhook Dispatcher**:
  - Emits `report.processed` and `batch.completed` events upon completion.
  - Cryptographically signed with `X-Signature-SHA256` for integrity and authenticity.
- **High-DPI Hybrid Ingestion Pipeline**:
  - Multi-page PDF rendering at 2.1x DPI scale via `pypdfium2` / `PyMuPDF`.
  - Automatic EXIF rotation, canvas stitching, and JPEG memory buffer optimization.
- **Interactive Streamlit Web Dashboard**:
  - Comprehensive UI for single-document streaming OCR, multi-file batch submissions, and historical report auditing.

---

## 2. Architecture & Documentation Links

- **API Guidance & Integration Reference**: [docs/API_GUIDANCE.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/API_GUIDANCE.md)
- **Data Flow Diagrams (Level 0, 1, 2)**: [docs/DATA_FLOW_DIAGRAMS.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/DATA_FLOW_DIAGRAMS.md)
- **Detailed System Architecture**: [docs/ARCHITECTURE.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/ARCHITECTURE.md)

---

## 3. Production Routing & Domain Architecture

All platform endpoints are mounted under the `/ocr` prefix and routed through Nginx on the production domain:

```text
http://aiagent.monoroc.com/ocr/...  -->  Nginx (Port 80)  -->  FastAPI Service (Port 8200)
```

### Core Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/ocr/health` | Service, database, and LLM backend health status |
| `POST` | `/ocr/api/ocr` | Synchronous direct OCR data extraction |
| `POST` | `/ocr/api/ocr/stream` | Real-time SSE token streaming OCR |
| `POST` | `/ocr/api/ocr/upload` | Multipart single-file upload OCR |
| `POST` | `/ocr/api/chat` | Multi-modal text and vision chat completion |
| `POST` | `/ocr/api/chat/upload` | Multipart multi-modal chat upload |
| `POST` | `/ocr/api/batch/upload` | Multipart multi-document batch submission |
| `POST` | `/ocr/api/batch` | JSON payload batch submission (Base64 / URLs) |
| `GET` | `/ocr/api/batches` | Paginated list of recent batches |
| `GET` | `/ocr/api/batch/{id}` | Real-time batch progress metrics |
| `GET` | `/ocr/api/batch/{id}/jobs` | Detailed job records and extracted JSON objects |
| `GET` | `/ocr/api/batch/{id}/download` | Download consolidated batch JSON or ZIP archive |
| `POST` | `/ocr/api/jobs` | Submit single asynchronous background job |
| `GET` | `/ocr/api/jobs/{id}` | Retrieve single job status and result |

---

## 4. Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ or Conda (`laiagent` / `stt` environment)
- Ollama with `ministral-3:latest` pulled on GPU

```bash
# Pull production model
ollama pull ministral-3:latest
```

### 2. Install Dependencies
```bash
cd ocr-api
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
```ini
PORT=8200
HOST=0.0.0.0
DEFAULT_BACKEND=ollama
DEFAULT_MODEL=ministral-3:latest
OLLAMA_URL=http://localhost:11434
LLAMA_CPP_URL=http://localhost:8080
DATABASE_URL=sqlite+aiosqlite:///./ocr.db
MAX_CONCURRENT_WORKERS=4
MAX_BATCH_SIZE=100
WEBHOOK_SECRET=ocr-webhook-secret-key-369
STREAMLIT_PORT=8600
```

### 4. Running Locally

#### Start FastAPI API Backend (Port 8200)
```bash
python3 app/main.py
# Or with uvicorn directly:
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

#### Start Streamlit Web Studio (Port 8600)
```bash
streamlit run ui/streamlit_app.py --server.port 8600
```

- Web Dashboard: `http://localhost:8600`
- Interactive API Documentation (Swagger): `http://localhost:8200/docs` or `http://aiagent.monoroc.com/ocr/docs`

---

## 5. Production Daemon Configuration (Systemd)

On Ubuntu/Debian Linux production hosts:

```ini
# /etc/systemd/system/ocr-api.service
[Unit]
Description=Production Vision OCR & Chat API Service
After=network.target ollama.service

[Service]
Type=simple
User=ml-dev-user
WorkingDirectory=/home/ml-dev-user/ocr-api
ExecStart=/home/ml-dev-user/miniconda3/envs/laiagent/bin/uvicorn app.main:app --host 0.0.0.0 --port 8200
Restart=always
RestartSec=3
EnvironmentFile=/home/ml-dev-user/ocr-api/.env

[Install]
WantedBy=multi-user.target
```

Control Commands:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ocr-api
sudo systemctl status ocr-api
sudo journalctl -u ocr-api -f
```

---

## 6. Automated Testing & Verification

Run the automated direct PDF OCR test suite to verify health, single PDF extractions, and multi-document batch pipelines:

```bash
conda run -n stt python3 scripts/test_vm_pdf_ocr.py
```

Run unit & router tests:
```bash
python3 -m unittest tests/test_production_vocr_suite.py
python3 -m unittest tests/test_api_endpoints.py
```

---

## 7. License
MIT License. Commercial and production deployment authorized.
