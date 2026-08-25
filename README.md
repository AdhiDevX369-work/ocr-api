# Vision OCR & Batch Processing Platform

A high-performance, asynchronous FastAPI platform, SQLAlchemy persistence engine, and Streamlit dashboard for document OCR, structured JSON extraction, and high-throughput multi-document batch pipelines powered directly by local Vision LLM inference engines (**Ollama**).

---

## 1. System Capabilities

- **Direct Vision Engine Connectivity**: Direct integration with local GPU inference backends:
  - **Ollama** (`http://localhost:11434`) – Default Model: `ministral-3:latest`
  - **llama.cpp / llama-server** (`http://localhost:8080` / `http://localhost:8081`)
- **Synchronous & Streaming Single OCR (`/ocr/api/ocr`, `/ocr/api/ocr/stream`)**: Direct single-document extraction with real-time Server-Sent Events (SSE) streaming and strict Pydantic V2 schema validation.
- **Enterprise Multi-Document Batch Processing (`/ocr/api/batch`)**:
  - Submits up to 100 PDF reports or image scans in a single multipart request.
  - Real-time batch progress tracking (`total_files`, `processed_files`, `failed_files`, `progress_percentage`).
  - Downloads results as a consolidated JSON or a ZIP archive containing individual report files.
- **Database Persistence**:
  - Backed by **SQLAlchemy 2.0 (Async)** with native **SQLite** (`aiosqlite`) and **PostgreSQL** (`asyncpg`) support.
  - Batches, individual document jobs, and webhook delivery audit logs are stored and queryable.
- **HMAC-SHA256 Webhook Dispatcher**:
  - Emits `report.processed` and `batch.completed` events upon completion.
  - Cryptographically signed with `X-Signature-SHA256` for integrity and authenticity.
- **High-DPI Hybrid Ingestion Pipeline**:
  - Multi-page PDF rendering at 2.1x DPI scale via `pypdfium2` / `PyMuPDF`.
  - Automatic EXIF rotation, canvas stitching, and JPEG memory buffer optimization.
- **Interactive Streamlit Web Dashboard**:
  - Comprehensive UI for single-document streaming OCR, multi-file batch submissions, and historical report auditing.

---

## 2. Documentation

- **API Guidance & Integration Reference**: [docs/API_GUIDANCE.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/API_GUIDANCE.md)
- **Data Flow Diagrams (Level 0, 1, 2)**: [docs/DATA_FLOW_DIAGRAMS.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/DATA_FLOW_DIAGRAMS.md)
- **Detailed System Architecture**: [docs/ARCHITECTURE.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/ARCHITECTURE.md)

---

## 3. Endpoints Overview

All platform endpoints are mounted under the `/ocr` prefix:

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

## 4. Quickstart

### 1. Prerequisites
- Python 3.10+
- Ollama with a vision model (e.g. `ministral-3:latest` or `qwen2.5vl:latest`)

```bash
ollama pull ministral-3:latest
```

### 2. Installation
```bash
git clone https://github.com/AdhiDevX369-work/ocr-api.git
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
WEBHOOK_SECRET=your-webhook-secret-key
STREAMLIT_PORT=8600
```

### 4. Running the Platform

#### Run API Backend (Port 8200)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload
```

#### Run Streamlit Web Dashboard (Port 8600)
```bash
streamlit run ui/streamlit_app.py --server.port 8600
```

- Web Dashboard: `http://localhost:8600`
- Interactive API Documentation (Swagger): `http://localhost:8200/docs` or `http://localhost:8200/ocr/docs`

---

## 5. Automated Testing

Run the automated direct PDF OCR test suite to verify endpoints, single PDF extractions, and multi-document batch pipelines:

```bash
python3 scripts/test_vm_pdf_ocr.py
```

Run unit & router tests:
```bash
python3 -m unittest tests/test_production_vocr_suite.py
python3 -m unittest tests/test_api_endpoints.py
```

---

## 6. License
MIT License.
