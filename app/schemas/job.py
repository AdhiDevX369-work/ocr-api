from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

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
    backend: Optional[str] = Field(None, description="LLM Backend (ollama, llama-cpp)")
    model: Optional[str] = Field(None, description="Vision LLM Model name")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=256, le=4096)
    webhook_url: Optional[str] = Field(
        None,
        description="PubSub / Hook callback HTTP URL for receiving 'report.processed' event when ready"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional custom metadata (e.g. client_id, sample_id) echoed back in webhook event"
    )

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    webhook_url: Optional[str] = None
    download_url: str
    result: Optional[Any] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class WebhookEventPayload(BaseModel):
    event_type: str = "report.processed"
    event_id: str
    timestamp: str
    data: Dict[str, Any]
