# 🩺 Medical OCR & Report Intelligence API

A high-performance, asynchronous FastAPI backend and interactive Streamlit web dashboard for medical report OCR, document extraction, and multimodal document chat powered directly by local LLM engines (**Ollama** and **llama-server**).

---

## 🌟 Key Features

- **Direct Engine Connectivity**: Bypasses gateway proxies to connect directly to local LLM backends:
  - **Ollama** (`http://localhost:11434`) – Default model: `gemma4:latest`
  - **llama.cpp** (`http://localhost:8080`)
- **Multimodal Document Chat (`/api/v1/image-chat`)**: Upload PDF lab reports or image scans and query them with real-time text streaming.
- **Asynchronous Batch Processing (`/api/v1/jobs`)**:
  - Submit multiple PDF/Image reports for background processing.
  - Poll real-time progress (`pending` -> `processing` -> `completed`).
  - Webhook callback support upon completion.
  - Download structured JSON / plain-text report extractions.
- **High-Quality PDF & Image Pipeline**:
  - High-clarity rendering of multipage PDFs (150 DPI via `pypdfium2` / `PyMuPDF`).
  - Vertical document page stitching for complete report analysis.
  - Dynamic aspect-ratio image optimization (max 1280px constraint).
- **Streamlit Web Dashboard**: Modern UI for image chat, PDF upload, batch job monitoring, and JSON data download.

---

## 📂 Project Structure

```text
ocr-api/
├── app/
│   ├── main.py                  # FastAPI Application Entrypoint (Port 8200)
│   ├── config.py                # Environment & Service Settings
│   ├── routers/
│   │   ├── chat_router.py       # Image Chat & Multimodal Streaming Endpoints
│   │   ├── job_router.py        # Async Batch Jobs, Status Polling & Webhooks
│   │   └── health_router.py     # System Healthcheck & LLM Discovery
│   ├── services/
│   │   ├── llm_client.py        # Direct Ollama & llama-cpp Async Client
│   │   ├── image_processor.py   # PDF Rendering & Image Optimization
│   │   └── job_service.py      # Async Background Job Engine
│   └── schemas/                 # Pydantic Schemas for Requests & Responses
├── ui/
│   └── streamlit_app.py         # Streamlit Interactive Web Dashboard (Port 8600)
├── pdf/                         # Sample Medical Lab Reports (EGFR, FBC, Lipid, Bilirubin)
├── .env                         # Environment Configuration File
├── requirements.txt             # Python Dependencies
└── README.md                    # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed along with local LLM server engines (Ollama or llama-server).

```bash
# Clone repository and enter directory
cd ocr-api

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Create or edit `.env` in the root directory:

```env
PORT=8200
HOST=0.0.0.0
DEFAULT_BACKEND=ollama
DEFAULT_MODEL=gemma4:latest
OLLAMA_URL=http://localhost:11434
LLAMA_CPP_URL=http://localhost:8080
STREAMLIT_PORT=8600
MAX_IMAGE_SIZE_PX=1280
IMAGE_JPEG_QUALITY=85
```

### 3. Launch Services

#### Start FastAPI API Backend (Port 8200)
```bash
python app/main.py
```

#### Start Streamlit Web UI (Port 8600)
```bash
streamlit run ui/streamlit_app.py --server.port 8600
```

Access the Web Dashboard at: `http://localhost:8600`  
Access API Documentation (Swagger) at: `http://localhost:8200/docs`

---

## 📡 API Usage & Endpoints

### 1. Interactive Image Chat (Upload File & Query)

**Endpoint**: `POST /api/v1/image-chat/upload`

```bash
curl -X POST "http://localhost:8200/api/v1/image-chat/upload" \
  -F "file=@pdf/FBC.pdf" \
  -F "prompt=Extract all test names, values, units, and reference ranges as JSON." \
  -F "backend=ollama" \
  -F "model=gemma4:latest"
```

---

### 2. Submit Async Batch Job

**Endpoint**: `POST /api/v1/jobs/upload`

```bash
curl -X POST "http://localhost:8200/api/v1/jobs/upload" \
  -F "file=@pdf/LIPID PROFILE.pdf" \
  -F "prompt=Extract patient details and lipid panel results." \
  -F "backend=ollama" \
  -F "model=gemma4:latest"
```

**Response**:
```json
{
  "job_id": "job_78d47f752be4",
  "status": "pending",
  "backend": "ollama",
  "model": "gemma4:latest",
  "created_at": "2026-08-14T10:40:00"
}
```

---

### 3. Poll Job Status & Extracted Data

**Endpoint**: `GET /api/v1/jobs/{job_id}`

```bash
curl -X GET "http://localhost:8200/api/v1/jobs/job_78d47f752be4"
```

---

### 4. Download Processed Report Result

**Endpoint**: `GET /api/v1/jobs/{job_id}/download?format=json`

```bash
curl -X GET "http://localhost:8200/api/v1/jobs/job_78d47f752be4/download?format=json" \
  -o report_result.json
```

---

## 🧪 Testing & Verification

Run automated test scripts from the scratch tools directory:

```bash
# Test direct Ollama connection & gemma4 completion
PYTHONPATH=. python /home/adhidevx369-work/.gemini/antigravity-ide/brain/c6ac56d0-0940-450b-aab1-ff42ddfc9a0e/scratch/test_fix_entirely.py

# Test batch processing on pdf/ directory reports
PYTHONPATH=. python /home/adhidevx369-work/.gemini/antigravity-ide/brain/c6ac56d0-0940-450b-aab1-ff42ddfc9a0e/scratch/test_batch_jobs.py
```

---

## 🩺 Supported Sample Lab Reports

The `pdf/` directory contains sample lab reports for testing:
- **`EGFR.pdf`**: Serum Creatinine & Estimated Glomerular Filtration Rate.
- **`FBC.pdf`**: Full Blood Count (WBC, Neutrophils, Lymphocytes, Hb, Platelets).
- **`LIPID PROFILE.pdf`**: Lipid Panel (Total Cholesterol, Triglycerides, HDL, LDL, VLDL).
- **`S.BILIRUBIN.pdf`**: Serum Bilirubin (Total, Direct, Indirect).

---

## 📜 License

MIT License. Free for development and deployment.
