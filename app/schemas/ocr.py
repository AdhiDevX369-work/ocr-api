from enum import Enum
from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field

class OCRFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCTAGS = "doctags"
    LATEX = "latex"
    TEXT = "text"

class OCRTaskType(str, Enum):
    GENERAL_OCR = "general_ocr"
    MEDICAL_EXTRACTION = "medical_extraction"
    TABLE_EXTRACTION = "table_extraction"
    DOCUMENT_RECONSTRUCTION = "document_reconstruction"
    CUSTOM = "custom"

class DocumentType(str, Enum):
    AUTO = "auto"
    PDF = "pdf"
    IMAGE = "image"

class PipelineMode(str, Enum):
    SINGLE = "single"   # Direct single-pass vision LLM
    DUAL = "dual"       # Stage 1 Vision OCR (DeepSeek) -> Stage 2 Text Structurer (Ministral)

class OCRRequest(BaseModel):
    document: str = Field(..., description="Base64 Data URI or HTTP URL of PDF or Image scan")
    format: OCRFormat = Field(OCRFormat.JSON, description="Desired output format: 'json', 'markdown', 'html', 'latex', or 'text'")
    task_type: OCRTaskType = Field(OCRTaskType.MEDICAL_EXTRACTION, description="Task preset: 'general_ocr', 'medical_extraction', 'table_extraction', 'document_reconstruction', 'custom'")
    pipeline: PipelineMode = Field(PipelineMode.SINGLE, description="Pipeline execution mode: 'single' (direct vision) or 'dual' (2-stage OCR + Structurer)")
    prompt: Optional[str] = Field(None, description="Custom prompt instructions. If omitted, task_type default is used.")
    system_prompt: Optional[str] = Field(None, description="System instruction. If omitted, task_type default is used.")
    backend: Optional[str] = Field(None, description="LLM backend ('ollama', 'vllm', 'llama-cpp', 'llm-server')")
    model: Optional[str] = Field(None, description="Vision LLM model (or Stage 2 model if not specified separately)")
    ocr_model: Optional[str] = Field(None, description="Stage 1 Vision OCR model for dual mode (defaults to 'deepseek-ocr:3b')")
    structurer_model: Optional[str] = Field(None, description="Stage 2 Text Structurer model for dual mode (defaults to 'Qwen3.5-4B-BF16.gguf' on port 8100)")
    structurer_backend: Optional[str] = Field(None, description="Stage 2 Text Structurer backend ('llm-server', 'llama-cpp', 'ollama', defaults to 'llm-server')")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(8192, ge=256, le=32768)
    strict_schema: bool = Field(True, description="Enforce strict JSON schema validation and repair (for medical JSON)")

class OCRResponse(BaseModel):
    status: str = "success"
    format: OCRFormat
    task_type: Optional[OCRTaskType] = None
    pipeline: Optional[PipelineMode] = None
    backend: str
    model: str
    data: Union[Dict[str, Any], str] = Field(..., description="Structured JSON object or formatted markdown/html text")
    raw_text: Optional[str] = Field(None, description="Raw transcription output")
    duration_seconds: float = Field(..., description="Processing time in seconds")
    stage_timings: Optional[Dict[str, float]] = Field(None, description="Per-stage latency breakdown in seconds (dual mode)")
    tokens_used: Optional[int] = Field(0, description="Tokens generated")
    page_count: Optional[int] = Field(1, description="Number of rendered document pages")
    created_at: str

class OCRStreamChunk(BaseModel):
    content: str
    done: bool = False
    tokens_generated: Optional[int] = None
    error: Optional[str] = None

