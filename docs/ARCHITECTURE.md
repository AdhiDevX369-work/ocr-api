# Medical Report OCR & Vision AI Architecture

## Executive Overview

This platform provides an enterprise-grade **Medical Lab Report OCR & Vision AI Data Extraction Pipeline**. It is designed to ingest lab reports (PDF documents or image scans) directly from Base64 URIs, multipart file uploads, HTTP public endpoints, or Cloud Storage URIs (`gs://` or GCS Signed URLs).

It handles synchronous single-document extraction, real-time Server-Sent Events (SSE) token streaming, and high-throughput asynchronous multi-document batch pipelines powered directly by local Vision LLM engines (**Ollama** with `ministral-3:latest` / `qwen3.5`). All endpoints are unified under the `/ocr` prefix and exposed via Nginx Reverse Proxy on `http://aiagent.monoroc.com/ocr/...`.

---

## Architecture & Data Flow References

- **Data Flow Diagrams (Level 0, 1, 2)**: See [DATA_FLOW_DIAGRAMS.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/DATA_FLOW_DIAGRAMS.md)
- **API Specification & Guidance**: See [API_GUIDANCE.md](file:///Users/adithyabandara/ofiice/ocr-api/docs/API_GUIDANCE.md)

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client and Ingestion Layer"]
        A1["External Clients / EHR Systems"]
        A2["Streamlit Web Studio (Port 8600)"]
        A3["Direct Multipart PDF Uploads"]
    end

    subgraph Proxy ["Nginx Reverse Proxy (Port 80)"]
        N1["aiagent.monoroc.com/ocr/..."]
    end

    subgraph API ["FastAPI Gateway (Port 8200)"]
        B1["POST /ocr/api/ocr (Sync)"]
        B2["POST /ocr/api/ocr/stream (SSE)"]
        B3["POST /ocr/api/ocr/upload (Single Upload)"]
        B4["POST /ocr/api/batch/upload (Batch Upload)"]
        B5["GET /ocr/api/batch/id/jobs (Batch Details)"]
        B6["GET /ocr/api/batch/id/download (Export)"]
    end

    subgraph Engine ["Document Processing Engine"]
        C1["pypdfium2 High-DPI PDF Renderer"]
        C2["Multi-page Document Canvas Stitcher"]
        C3["EXIF Auto-Rotation & JPEG Optimizer"]
    end

    subgraph LLM ["Local Vision LLM Core"]
        D1["Ollama GPU Engine (Port 11434)"]
        D2["Model: ministral-3:latest (Deterministic Temp: 0.0)"]
        D3["Structured Clinical JSON Transcription"]
    end

    subgraph Persistence ["Persistence & Events"]
        E1["SQL Database (SQLite / PostgreSQL)"]
        E2["HMAC-SHA256 Webhook Dispatcher"]
    end

    A1 --> N1
    A2 --> N1
    A3 --> N1
    N1 --> API

    B1 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1

    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D2 --> D3

    D3 --> E1
    E1 --> E2
    E2 --> A1
```

    subgraph Engine ["Document Processing Engine"]
        C1["GCS HTTP Downloader"]
        C2["PDF Rendering Engine"]
        C3["Multi-page Document Stitching"]
        C4["EXIF Rotation and JPEG Optimization"]
    end

    subgraph LLM ["Vision LLM Core Gateway"]
        D1["Local Vision Gateway Ollama"]
        D2["Deterministic Vision Inference"]
        D3["Full Blood Count EGFR Lipid Extraction"]
    end

    subgraph Event ["Job Persistence and Event Dispatcher"]
        E1["Job State Store"]
        E2["PubSub Webhook Event Dispatcher"]
        E3["Event report.processed Payload"]
    end

    subgraph Sink ["Downstream Consumers"]
        F1["PubSub Subscriber Webhook Receiver"]
        F2["EHR Clinical Database System"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B1

    B1 --> C1
    B2 --> C1
    B3 --> C4

    C1 --> C2
    C2 --> C3
    C3 --> C4

    C4 --> D1
    D1 --> D2
    D2 --> D3

    D3 --> E1
    E1 --> E2
    E2 --> F1
    F1 --> B4
    B4 --> F2
```

---

## 🔄 2. Complete Dataflow Sequence (A-Z)

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Application / GCP Pipeline
    participant API as FastAPI Gateway
    participant Processor as Document & PDF Processor
    participant LLM as Vision LLM (Gemma 4)
    participant EventBus as PubSub / Webhook Listener

    Client->>API: POST /api/v1/jobs/batch (GCP Storage URLs: gs://my-bucket/report.pdf)
    API->>API: Generate Job ID & Return 202 Accepted
    API-->>Client: Return Job Response (status: pending, download_url)

    Note over API,Processor: Asynchronous Background Task Started
    API->>Processor: Fetch Document from GCP Storage / Signed URL
    Processor->>Processor: Render PDF (150 DPI) & Stitch Multi-page document
    Processor->>LLM: Send Vision Prompt + Data URI (Temperature: 0.0)
    LLM->>LLM: Transcribe Patient details, Observed values, Flags, Reference Ranges
    LLM-->>Processor: Return Structured JSON Response
    Processor->>API: Mark Job Status as COMPLETED & Save Result JSON
    API->>EventBus: POST Webhook Event ("report.processed") with download_url

    Note over EventBus,API: Event Processing & Download
    EventBus->>API: GET /api/v1/jobs/job_id/download
    API-->>EventBus: Download JSON Report Result Data
```

---

## ⚙️ 3. Detailed Component Architecture

### 📥 Component 1: Ingestion & GCP Cloud Storage Layer
- **GCP Cloud Storage Normalization**: Automatically converts `gs://bucket-name/object.pdf` into authenticated/signed or storage API URLs (`https://storage.googleapis.com/bucket-name/object.pdf`).
- **Batch Processing (`POST /api/v1/jobs/batch`)**: Accepts an array of GCS document URLs in a single request. Each item is queued into an asynchronous non-blocking job.

### 📄 Component 2: Document Rendering & Image Processor (`app/services/image_processor.py`)
- **PDF Detection**: Inspects magic bytes (`%PDF`).
- **Rendering Engine**: Renders PDF pages using `pypdfium2` or `PyMuPDF` at 2x scale (~150 DPI) for maximum numeric clarity.
- **Multi-page Stitching**: Stitches multi-page PDF documents vertically into a combined PIL Image stack.
- **Image Optimization**: Auto-corrects EXIF rotation, converts RGBA to RGB JPEG, and resizes large scans to optimal vision resolution (e.g. 1280px).

### 🧠 Component 3: Vision LLM Processing Engine (`app/services/llm_client.py`)
- Interfaced to local Vision Gateway on ports `11434` (Ollama) or `8100` (llama.cpp) running **Gemma 4 / Qwen 3.5**.
- Uses deterministic extraction (`temperature=0.0`) for exact number and decimal point precision.
- **1-to-1 Verification**:
  - Handles missing/unreported observed values (recording empty string `""` or `"N/A"`).
  - Captures complex reference ranges (e.g. multi-line Lipid risk classifications).
  - Extracts custom clinical tables (e.g. `AGE AVERAGE ESTIMATED GFR` grids in EGFR reports).

### 💾 Component 4: State Management & Result Persistence (`app/services/job_service.py`)
- Concurrent in-memory job registry with thread-safe asyncio locks.
- Assigns unique job identifiers (`job_a1b2c3d4e5f6`).
- Exposes direct download endpoints (`GET /api/v1/jobs/{job_id}/download?format=json`).

### 📢 Component 5: Event PubSub & Webhook Engine
- Once job processing completes (or fails), dispatches an HTTP POST event payload to the client's `webhook_url`.
- Downstream PubSub subscribers receive the notification containing the `job_id`, status, metadata, and direct result `download_url`.

---

## 📑 4. API Endpoints & Request Specifications

### A. Single Document Job (`POST /api/v1/jobs`)
```json
{
  "document": "gs://my-lab-bucket/reports/2026-08/patient_18353.pdf",
  "prompt": "Perform an exact line-by-line verification check of all values in this report against the printed document.",
  "system_prompt": "You are a High-Precision Medical Report Verification AI.",
  "webhook_url": "https://pubsub-listener.internal.company.com/events",
  "meta": {
    "patient_id": "18353",
    "facility": "Main Lab"
  }
}
```

### B. Batch Document URLs (`POST /api/v1/jobs/batch`)
```json
{
  "documents": [
    {
      "document": "https://storage.googleapis.com/my-lab-bucket/FBC.pdf",
      "meta": { "sample_id": "S101" }
    },
    {
      "document": "https://storage.googleapis.com/my-lab-bucket/EGFR.pdf",
      "meta": { "sample_id": "S102" }
    },
    {
      "document": "https://storage.googleapis.com/my-lab-bucket/LIPID_PROFILE.pdf",
      "meta": { "sample_id": "S103" }
    }
  ],
  "webhook_url": "https://pubsub-listener.internal.company.com/events"
}
```

### C. Webhook Event Payload (`report.processed`)
```json
{
  "event_type": "report.processed",
  "event_id": "evt_9a4f21b7c8e",
  "timestamp": "2026-08-13T10:35:00Z",
  "data": {
    "job_id": "job_3e981f2a4",
    "status": "completed",
    "created_at": "2026-08-13T10:34:45Z",
    "completed_at": "2026-08-13T10:35:00Z",
    "download_url": "http://localhost:8200/api/v1/jobs/job_3e981f2a4/download",
    "meta": {
      "batch_id": "batch_7a2b9c",
      "sample_id": "S101"
    },
    "result": {
      "patient_info": {
        "patient_name": "MISS TOPH",
        "pid_no": "18353",
        "age": "20 Years",
        "sex": "Female"
      },
      "report_title": "Full Blood Count",
      "investigations": [
        {
          "section": "Leucocytes",
          "investigation": "WBC Count",
          "observed_value": "3,000",
          "flag": "L",
          "unit": "cells / mm³",
          "reference_interval": "4,000 - 12,000"
        }
      ]
    },
    "error": null
  }
}
```

---

## 🛠️ 5. Running the Pipeline

1. **Start Ollama / Local LLM Server**:
   ```bash
   ollama serve
   ```
2. **Launch API & Streamlit Studio**:
   ```bash
   ./start_all.sh
   ```
3. **Run Batch Extraction Script on Sample PDFs**:
   ```bash
   python3 scripts/batch_extract.py
   ```
