# System Data Flow Diagrams (DFD)

This document provides formal Level 0, Level 1, and Level 2 Data Flow Diagrams (DFDs) for the Vision OCR and Batch Processing Platform.

---

## 1. DFD Level 0: Context Diagram

The Context Diagram defines the system boundaries, external entities, primary inputs, and output data flows.

```mermaid
flowchart LR
    %% External Entities
    User["Client Application / EHR System / Web Studio"]
    NginxProxy["Nginx Reverse Proxy (aiagent.monoroc.com)"]
    OllamaEngine["Local GPU LLM Engine (Ollama / ministral-3)"]
    Database["SQL Database (SQLite / PostgreSQL)"]
    WebhookReceiver["Downstream Webhook Subscriber"]

    %% Core System
    OCRSystem(("Vision OCR & Batch API Platform (Port 8200)"))

    %% Data Flows
    User -- "1. HTTP Request (PDF / Image / Prompt)" --> NginxProxy
    NginxProxy -- "2. Forward Request (/ocr/...)" --> OCRSystem
    
    OCRSystem -- "3. High-DPI Image Buffer + Prompt" --> OllamaEngine
    OllamaEngine -- "4. Raw Model Inference / Stream" --> OCRSystem

    OCRSystem -- "5. Store Batch, Job & Extraction State" --> Database
    Database -- "6. Query Status, History & Analytics" --> OCRSystem

    OCRSystem -- "7. Signed Webhook Event (report.processed / batch.completed)" --> WebhookReceiver
    OCRSystem -- "8. Structured JSON / SSE Stream / Export File" --> NginxProxy
    NginxProxy -- "9. HTTP Response" --> User
```

---

## 2. DFD Level 1: System Decomposition Diagram

The Level 1 Diagram illustrates the core sub-processes, internal data stores, and inter-component data exchanges.

```mermaid
flowchart TD
    %% Entities
    Client["Client / External Ingestion"]
    Webhooks["Webhook Receiver"]

    %% Processes
    P1["1.0 Ingestion & Request Dispatcher\n(FastAPI Gateway)"]
    P2["2.0 High-DPI Document Processor\n(pypdfium2 / PIL)"]
    P3["3.0 Vision LLM Client Engine\n(Ollama / llama-server)"]
    P4["4.0 Schema Validator & JSON Repair\n(Pydantic V2)"]
    P5["5.0 Async Batch & Job Worker Pool\n(Asyncio Semaphore Controller)"]
    P6["6.0 Webhook Dispatcher & Signer\n(HMAC-SHA256)"]
    P7["7.0 Archive & Export Generator\n(JSON / ZIP Builder)"]

    %% Data Stores
    D1[("D1: Batch Data Store")]
    D2[("D2: Job State Store")]
    D3[("D3: Webhook Delivery Log")]

    %% Flows
    Client -->|"Document Payload (Base64 / Multipart / URL)"| P1

    %% Process 1 Routing
    P1 -->|"Sync / Stream OCR"| P2
    P1 -->|"Batch Create Request"| P5
    P1 -->|"Download Request"| P7

    %% Process 2 Rendering
    P2 -->|"High-DPI Rendered Image URI"| P3

    %% Process 3 Inference
    P3 -->|"Raw Text / Stream Tokens"| P4
    P3 -->|"SSE Stream Tokens"| P1

    %% Process 4 Validation
    P4 -->|"Validated Clinical Schema"| P1
    P4 -->|"Job Result Object"| P5

    %% Process 5 Worker Controller
    P5 -->|"Persist Batch Metadata"| D1
    P5 -->|"Persist Job State"| D2
    P5 -->|"Queue Document"| P2
    P5 -->|"Trigger Completion Event"| P6

    %% Process 6 Webhook Dispatch
    P6 -->|"Log Delivery Status & Timestamp"| D3
    P6 -->|"Signed HTTP POST"| Webhooks

    %% Process 7 Export
    D1 -->|"Fetch Batch Record"| P7
    D2 -->|"Fetch Job Extractions"| P7
    P7 -->|"Consolidated JSON / ZIP Stream"| P1
```

---

## 3. DFD Level 2: Sub-Process Data Flow Diagrams

### 2.1 Sub-Process 2.0 & 3.0: Direct Vision OCR & Token Streaming Flow

```mermaid
flowchart TD
    subgraph Input ["Input Handling"]
        A["Incoming Document (PDF / Image)"] --> B{"File Type?"}
        B -->|PDF| C["pypdfium2 Renderer (2.1x DPI Scale)"]
        B -->|Image| D["PIL EXIF Auto-Rotation & RGB Conversion"]
        C --> E["Multi-Page Canvas Vertical Stitcher"]
        D --> F["JPEG Buffer Optimization"]
        E --> F
    end

    subgraph LLM_Inference ["Model Inference Engine"]
        F -->|"data:image/jpeg;base64,..."| G["LLM Client (Ollama ministral-3)"]
        H["Deterministic Prompt (Temp: 0.0)"] --> G
        G --> I{"Output Mode?"}
        I -->|Streaming| J["SSE Token Generator"]
        I -->|Synchronous| K["Raw JSON Buffer"]
    end

    subgraph Validation ["Validation & Repair"]
        K --> L["Markdown Fence Stripper"]
        L --> M["Pydantic MedicalReportExtraction Validator"]
        M -->|Valid| N["Structured Clinical JSON Object"]
        M -->|Warning / Soft-Repair| O["Fallback Key-Value Normalizer"]
        O --> N
    end
```

---

### 2.2 Sub-Process 5.0: Asynchronous Multi-Document Batch Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Application
    participant Gateway as FastAPI Router (/ocr/api/batch/upload)
    participant BatchService as Batch Service Controller
    participant DB as SQLite / PostgreSQL Database
    participant WorkerPool as Async Job Worker Pool
    participant Webhook as Webhook Dispatcher

    Client->>Gateway: POST /ocr/api/batch/upload (N files)
    Gateway->>BatchService: Initialize Batch Record
    BatchService->>DB: INSERT INTO batches (status: pending, total_files: N)
    loop For each file
        BatchService->>DB: INSERT INTO jobs (status: pending, parent_batch_id)
    end
    BatchService-->>Gateway: Return Batch ID & 202 Accepted
    Gateway-->>Client: 202 Accepted (batch_id, progress: 0%)

    par Parallel Job Workers (Max Concurrent: 4)
        WorkerPool->>DB: SELECT pending jobs LIMIT 4
        WorkerPool->>DB: UPDATE job status = 'processing'
        WorkerPool->>WorkerPool: Render PDF & Execute Vision LLM
        WorkerPool->>DB: UPDATE job status = 'completed', extraction_json, duration
        WorkerPool->>BatchService: Notify Child Job Finished
    end

    BatchService->>DB: Check if all child jobs finished
    DB-->>BatchService: All N jobs completed
    BatchService->>DB: UPDATE batch status = 'completed', progress = 100%
    BatchService->>Webhook: Dispatch Signed 'batch.completed' Event
    Webhook->>Client: POST Webhook Payload (HMAC-SHA256 Signed)
```
