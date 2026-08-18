from enum import Enum
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobCreateRequest(BaseModel):
    document: Optional[str] = Field(
        None,
        description="Base64 Data URI or HTTP URL of PDF document or Image report"
    )
    prompt: Optional[str] = Field(
        "Perform an exact line-by-line verification check of all values in this report against the printed document.",
        description="Text prompt instruction for extraction"
    )
    system_prompt: Optional[str] = Field(
        "You are an expert Medical Report OCR and Verification AI.",
        description="System instruction"
    )
    backend: Optional[str] = Field(None, description="LLM Backend ('ollama', 'llama-cpp', 'llm-server')")
    model: Optional[str] = Field(None, description="Vision LLM Model name (e.g. 'qwen2.5vl:latest')")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=256, le=16384)
    webhook_url: Optional[str] = Field(
        None,
        description="PubSub / Webhook callback HTTP URL for receiving 'report.processed' event when ready"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional custom metadata (e.g. client_id, sample_id) echoed back in webhook event"
    )

class JobResponse(BaseModel):
    job_id: str
    batch_id: Optional[str] = None
    status: JobStatus
    document_type: Optional[str] = None
    document_name: Optional[str] = None
    backend: Optional[str] = None
    model: Optional[str] = None
    download_url: Optional[str] = None
    webhook_url: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = 0
    duration_seconds: Optional[float] = 0.0
    meta: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class WebhookEventPayload(BaseModel):
    event_type: str = "report.processed"  # "report.processed" or "batch.completed"
    event_id: str
    timestamp: str
    signature: Optional[str] = None
    data: Dict[str, Any]
