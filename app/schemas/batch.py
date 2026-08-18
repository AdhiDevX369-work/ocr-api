from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from app.schemas.job import JobResponse

class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchDocumentInput(BaseModel):
    document: str = Field(..., description="Base64 Data URI or HTTP URL of PDF document or Image scan")
    name: Optional[str] = Field(None, description="Optional document name (e.g. 'FBC_Report_01.pdf')")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata for this specific document")

class BatchCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Descriptive batch name (e.g. 'Hospital_Ward_3_Daily_Labs')")
    documents: List[Union[str, BatchDocumentInput]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Array of documents (Base64/URLs or BatchDocumentInput objects) to process in this batch"
    )
    prompt: Optional[str] = Field(
        "Perform an exact line-by-line verification check and extract all values into structured JSON.",
        description="Custom prompt instruction for extraction"
    )
    system_prompt: Optional[str] = Field(
        "You are an expert Medical Report OCR and Clinical Data Extraction AI.",
        description="System instruction"
    )
    backend: Optional[str] = Field(None, description="Target backend ('ollama', 'llama-cpp', 'llm-server')")
    model: Optional[str] = Field(None, description="Target Vision model (e.g. 'qwen2.5vl:latest')")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=256, le=16384)
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL to receive 'batch.completed' event once all documents are processed"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional batch-level metadata (e.g. clinic_id, submission_source)"
    )

class BatchResponse(BaseModel):
    batch_id: str
    name: Optional[str] = None
    status: BatchStatus
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    progress_percentage: float = 0.0
    webhook_url: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None

class BatchDetailResponse(BatchResponse):
    jobs: List[JobResponse] = Field(default_factory=list, description="List of individual document jobs in this batch")

class BatchListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    batches: List[BatchResponse]
