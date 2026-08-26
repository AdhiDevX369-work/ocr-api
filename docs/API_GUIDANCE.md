# Enterprise Vision OCR & Batch API Guidance

Comprehensive API reference and integration guide for the Production Vision OCR and Batch Processing Platform.

Base URL (Production Domain): `http://aiagent.monoroc.com`  
Direct Service Port: `http://<HOST_IP>:8200`  
Prefix: `/ocr`

---

## 1. Authentication & Security

- **Network Routing**: Requests through `http://aiagent.monoroc.com/ocr/...` are proxied directly to the internal FastAPI service on port `8200`.
- **Webhook HMAC-SHA256**: Outbound webhooks include a signature header `X-Signature-SHA256` generated with the shared secret `WEBHOOK_SECRET`.
- **CORS**: Configured to permit cross-origin requests (`*`) from verified client origins and internal web dashboards.

---

## 2. Health & Telemetry Endpoint

### `GET /ocr/health`
Inspects system readiness, database connectivity, and configured local LLM vision inference engines.

#### Request Example (cURL)
```bash
curl -s http://aiagent.monoroc.com/ocr/health
```

#### Response Payload (`200 OK`)
```json
{
  "status": "healthy",
  "service": "Production Vision OCR & Batch API",
  "port": 8200,
  "database": {
    "status": "healthy",
    "url": "sqlite+aiosqlite:///./ocr.db"
  },
  "direct_backends": {
    "llama_cpp_url": "http://localhost:8080",
    "ollama_url": "http://localhost:11434",
    "default_backend": "ollama",
    "default_model": "ministral-3:latest",
    "health": {
      "status": "healthy",
      "backends": {
        "ollama": {
          "status": "healthy",
          "url": "http://localhost:11434",
          "models": ["ministral-3:latest"]
        }
      }
    }
  }
}
```

---

## 3. Direct Vision OCR Endpoints

### A. Direct Single Document OCR (`POST /ocr/api/ocr` or `POST /ocr/api/ocr/sync`)
Performs synchronous OCR data extraction from a Base64 data URI, HTTP URL, or Cloud Storage URI.

#### Request Schema
```json
{
  "document": "data:image/jpeg;base64,...",
  "prompt": "Extract all medical report data into structured clinical JSON.",
  "system_prompt": "You are a clinical OCR extraction specialist.",
  "format": "json",
  "backend": "ollama",
  "model": "ministral-3:latest",
  "temperature": 0.0,
  "max_tokens": 4096
}
```

#### Response Payload (`200 OK`)
```json
{
  "status": "success",
  "format": "json",
  "backend": "ollama",
  "model": "ministral-3:latest",
  "data": {
    "report_title": "FULL BLOOD COUNT",
    "patient_info": {
      "patient_name": "MISS TOPH",
      "pid_no": "18353",
      "tel_no": "077-1234567",
      "age": "20 Years",
      "sex": "Female",
      "reference_dr": "DR. SMITH",
      "registered_on": "2026-08-25 09:00",
      "collected_on": "2026-08-25 09:15",
      "reported_on": "2026-08-25 11:30"
    },
    "results": [
      {
        "type": "wbc_count",
        "name": "WBC Count",
        "value": "3000",
        "unit": "cells/mm³"
      },
      {
        "type": "hemoglobin",
        "name": "Haemoglobin",
        "value": "12.5",
        "unit": "g/dL"
      },
      {
        "type": "platelet_count",
        "name": "Platelet Count",
        "value": "110000",
        "unit": "/µl"
      }
    ]
  },
  "duration_seconds": 1.62,
  "tokens_used": 0,
  "created_at": "2026-08-25T07:10:00Z"
}
```

---

### B. Real-Time Streaming OCR (`POST /ocr/api/ocr/stream`)
Streams extraction tokens in real time via Server-Sent Events (SSE).

#### Request Example (cURL)
```bash
curl -N -X POST http://aiagent.monoroc.com/ocr/api/ocr/stream \
  -H "Content-Type: application/json" \
  -d '{
    "document": "data:image/jpeg;base64,...",
    "prompt": "Extract all report parameters.",
    "backend": "ollama",
    "model": "ministral-3:latest"
  }'
```

#### SSE Stream Format
```text
data: {"token": "{\n"}
data: {"token": "  \"report_title\": \"FULL BLOOD COUNT\",\n"}
...
data: {"done": true}
```

---

### C. Multipart File Upload OCR (`POST /ocr/api/ocr/upload`)
Uploads a single PDF file or image scan directly as `multipart/form-data`.

#### Request Example (cURL)
```bash
curl -X POST http://aiagent.monoroc.com/ocr/api/ocr/upload \
  -F "file=@/path/to/FBC.pdf;type=application/pdf" \
  -F "format=json" \
  -F "prompt=Extract all clinical values into structured JSON" \
  -F "temperature=0.0"
```

#### Python (`httpx` / `requests`) Example
```python
import httpx

with open("FBC.pdf", "rb") as f:
    files = {"file": ("FBC.pdf", f, "application/pdf")}
    data = {"format": "json", "temperature": 0.0}
    response = httpx.post("http://aiagent.monoroc.com/ocr/api/ocr/upload", files=files, data=data, timeout=120.0)
    print(response.json())
```

---

## 4. Multi-modal Chat Completions (`POST /ocr/api/chat`)

OpenAI-compatible multi-modal chat endpoint supporting vision images and system prompts.

#### Request Example
```bash
curl -X POST http://aiagent.monoroc.com/ocr/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Transcribe the printed table from this image.",
    "images": ["data:image/jpeg;base64,..."],
    "backend": "ollama",
    "model": "ministral-3:latest",
    "temperature": 0.0
  }'
```

---

## 5. Enterprise Multi-Document Batch Processing

### A. Multipart Batch Upload (`POST /ocr/api/batch/upload`)
Submit up to 100 PDF reports or image scans in a single request for asynchronous background extraction.

#### Request Example (cURL)
```bash
curl -X POST http://aiagent.monoroc.com/ocr/api/batch/upload \
  -F "files=@/path/to/EGFR.pdf;type=application/pdf" \
  -F "files=@/path/to/FBC.pdf;type=application/pdf" \
  -F "files=@/path/to/LIPID_PROFILE.pdf;type=application/pdf" \
  -F "name=Clinical_Batch_001" \
  -F "prompt=Extract all parameters into JSON" \
  -F "webhook_url=https://your-domain.com/api/webhooks/ocr"
```

#### Response Payload (`202 Accepted`)
```json
{
  "batch_id": "batch_d533c96410d6",
  "name": "Clinical_Batch_001",
  "status": "pending",
  "total_files": 3,
  "processed_files": 0,
  "failed_files": 0,
  "progress_percentage": 0.0,
  "webhook_url": "https://your-domain.com/api/webhooks/ocr",
  "meta": {},
  "created_at": "2026-08-25T07:09:24.940000",
  "completed_at": null
}
```

---

### B. Poll Batch Progress (`GET /ocr/api/batch/{batch_id}`)
Retrieve real-time processing status and completion percentage.

#### Request
```bash
curl -s http://aiagent.monoroc.com/ocr/api/batch/batch_d533c96410d6
```

#### Response
```json
{
  "batch_id": "batch_d533c96410d6",
  "name": "Clinical_Batch_001",
  "status": "completed",
  "total_files": 3,
  "processed_files": 3,
  "failed_files": 0,
  "progress_percentage": 100.0,
  "created_at": "2026-08-25T07:09:24.940000",
  "completed_at": "2026-08-25T07:10:31.554000"
}
```

---

### C. Retrieve Batch Document Results (`GET /ocr/api/batch/{batch_id}/jobs`)
Fetches all individual child job details, statuses, durations, and extracted clinical JSON objects.

#### Request
```bash
curl -s http://aiagent.monoroc.com/ocr/api/batch/batch_d533c96410d6/jobs
```

---

### D. Download Consolidated Results (`GET /ocr/api/batch/{batch_id}/download`)
Downloads batch results as a merged JSON file or a ZIP archive containing individual report files.

- **Download Merged JSON**:
  ```bash
  curl -O http://aiagent.monoroc.com/ocr/api/batch/batch_d533c96410d6/download?format=json
  ```
- **Download ZIP Archive**:
  ```bash
  curl -O http://aiagent.monoroc.com/ocr/api/batch/batch_d533c96410d6/download?format=zip
  ```

---

## 6. Webhook Signatures & Verification

When a job or batch completes, an HTTP POST request is sent to `webhook_url`.

### Webhook Headers
```http
Content-Type: application/json
X-Event-Type: batch.completed
X-Signature-SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

### Signature Verification (Python Example)
```python
import hmac
import hashlib

def verify_webhook(payload_bytes: bytes, received_signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_signature)
```
