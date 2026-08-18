# 🩺 Enterprise Medical Vision OCR (vOCR) & Batch Processing Platform

A high-performance, asynchronous FastAPI backend, PostgreSQL/SQLite persistence engine, and interactive Streamlit web dashboard for medical report OCR, structured clinical JSON extraction, and high-throughput multi-document batch pipelines powered directly by local LLM engines (**Ollama** and **llama-server**).

---

## 🌟 Key Features

- **🚀 Direct Engine Connectivity**: Connects directly to high-throughput local LLM backends:
  - **Ollama** (`http://localhost:11434`) – Recommended: `qwen2.5vl:latest` / `gemma4:latest`
  - **llama.cpp / llama-server** (`http://localhost:8080`)
  - **Gateway llm-server** (`http://localhost:8100`)
- **⚡ Synchronous & Streaming Single OCR (`/api/v1/ocr/sync`, `/api/v1/ocr/stream`)**: Direct single-document extraction with real-time Server-Sent Events (SSE) streaming and Pydantic schema validation.
- **📦 Enterprise Multi-Document Batch Processing (`/api/v1/batches`)**:
  - Submit up to 100 PDF reports or image scans in a single batch API call.
  - Track parent batch progress in real-time (`total_files`, `processed_files`, `failed_files`, `progress_percentage`).
  - Download all results as a consolidated JSON or a ZIP archive containing individual JSON reports.
- **🗄️ Resilient Database Persistence**:
  - Backed by **SQLAlchemy 2.0 (Async)** with native **PostgreSQL** (`asyncpg`) and SQLite (`aiosqlite`) support.
  - All batches, individual document jobs, and webhook delivery audit logs are permanently stored and queryable.
- **📢 Resilient Webhook & Event Dispatcher**:
  - Automatically emits `report.processed` and `batch.completed` events upon completion.
  - Signed with **HMAC-SHA256** (`X-Signature-SHA256`) for security and authenticity.
  - Automatic exponential backoff retries (up to 5 attempts).
- **🧪 Structured JSON Auto-Repair & Schema Validator**:
  - Validates and auto-repairs clinical report JSON structures against strict Pydantic schemas (`MedicalReportExtraction`).
- **📄 High-DPI Hybrid Document Pipeline**:
  - Direct digital text layer extraction via PyMuPDF.
  - High-clarity multi-page rendering (150-200 DPI via `pypdfium2` / `PyMuPDF`).
- **🖥️ Modern Streamlit Web Studio**:
  - Dedicated tabs for Interactive Single Document OCR, Multi-Document Batch Studio with live progress bars, and Historical Batch Explorer.

---

## 📂 Project Structure

```text
ocr-api/
├── app/
│   ├── main.py                  # FastAPI Application Entrypoint (Port 8200)
│   ├── config.py                # Service, Database & Webhook Configuration
│   ├── db/
│   │   ├── __init__.py          # Database exports
│   │   ├── session.py           # Async Database Session & Engine Pool
│   │   └── models.py            # SQLAlchemy Models (Batches, Jobs, WebhookDeliveries)
│   ├── routers/
│   │   ├── ocr_router.py        # Synchronous & Streaming Single OCR Endpoints
│   │   ├── batch_router.py      # Enterprise Multi-Document Batch API
│   │   ├── job_router.py        # Single Async Jobs, Polling & Downloads
│   │   ├── chat_router.py       # Multi-modal Chat & OpenAI-Compatible Completions
│   │   └── health_router.py     # System, Database & LLM Engine Health Check
│   ├── services/
│   │   ├── llm_client.py        # Direct Ollama & llama-cpp Async Client with JSON mode
│   │   ├── image_processor.py   # Hybrid OCR & Multi-page PDF Rendering
│   │   ├── schema_validator.py  # JSON Auto-repair & Pydantic Schema Validator
│   │   ├── webhook_dispatcher.py # HMAC-SHA256 Signed Webhook Engine
│   │   ├── job_service.py       # Database-Backed Async Job Engine
│   │   └── batch_service.py     # Multi-Document Batch Engine & ZIP Exporter
│   └── schemas/                 # Pydantic Schemas (Medical, Batch, OCR, Job, Chat)
├── ui/
│   └── streamlit_app.py         # Streamlit Web Studio (Port 8600)
├── pdf/                         # Sample Medical Lab Reports (EGFR, FBC, Lipid, Bilirubin)
├── tests/                       # Unit & Integration Test Suites
├── .env                         # Environment Configuration File
├── requirements.txt             # Python Dependencies
└── README.md                    # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

```bash
# Clone repository and enter directory
cd ocr-api

# Install dependencies (using conda 'stt' environment)
conda run -n stt pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

```env
PORT=8200
HOST=0.0.0.0
DEFAULT_BACKEND=ollama
DEFAULT_MODEL=qwen2.5vl:latest
OLLAMA_URL=http://localhost:11434
LLAMA_CPP_URL=http://localhost:8080
DATABASE_URL=sqlite+aiosqlite:///./ocr.db # Or postgresql+asyncpg://user:pass@localhost:5432/ocr_db
MAX_CONCURRENT_WORKERS=4
MAX_BATCH_SIZE=100
WEBHOOK_SECRET=ocr-webhook-secret-key-369
STREAMLIT_PORT=8600
```

### 3. Launch Services

#### Start FastAPI API Backend (Port 8200)
```bash
conda run -n stt python app/main.py
```

#### Start Streamlit Web Studio (Port 8600)
```bash
conda run -n stt streamlit run ui/streamlit_app.py --server.port 8600
```

Access Web Dashboard: `http://localhost:8600`  
Access API Documentation (Swagger): `http://localhost:8200/docs`

---

## 📡 API Endpoints Overview

### 1. Direct Single Document OCR
- **`POST /api/v1/ocr/sync`**: Synchronous OCR extraction returning structured JSON.
- **`POST /api/v1/ocr/stream`**: Real-time SSE streaming of OCR transcriptions.
- **`POST /api/v1/ocr/upload`**: Multipart file upload direct OCR extraction.

### 2. Enterprise Multi-Document Batch Processing
- **`POST /api/v1/batches`**: Submit a batch of Base64 documents or URLs.
- **`POST /api/v1/batches/upload`**: Multipart upload of multiple PDF/image files.
- **`GET /api/v1/batches`**: Paginated list of recent batches.
- **`GET /api/v1/batches/{batch_id}`**: Real-time batch progress (% completed, processed/failed count).
- **`GET /api/v1/batches/{batch_id}/jobs`**: Detailed status and JSON extractions for every job in batch.
- **`GET /api/v1/batches/{batch_id}/download?format=json|zip`**: Consolidated batch JSON or ZIP download.

### 3. Single Background Jobs
- **`POST /api/v1/jobs`**: Submit a single document for background processing.
- **`GET /api/v1/jobs/{job_id}`**: Get job status and result.
- **`GET /api/v1/jobs/{job_id}/download`**: Download processed report output.

---

## 🧪 Testing & Verification

Run the comprehensive unit & integration test suites:

```bash
conda run -n stt python -m unittest tests/test_production_vocr_suite.py
conda run -n stt python -m unittest tests/test_api_endpoints.py
```

---

## 📜 License
MIT License. Free for development and production deployment.
