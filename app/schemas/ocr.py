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

class OCREngine(str, Enum):
    VISION_LLM = "vocr"     # Direct Vision LLM (Ministral-3)
    NATIVE = "native"       # Ultra-Fast Sub-Second Native OCR (Non-LLM)
    PADDLE_OCR = "paddle"   # Alias for Native OCR

class OCRRequest(BaseModel):
    document: str = Field(..., description="Base64 Data URI or HTTP URL of PDF or Image scan")
    format: OCRFormat = Field(OCRFormat.JSON, description="Desired output format: 'json', 'markdown', 'html', 'latex', or 'text'")
    task_type: OCRTaskType = Field(OCRTaskType.MEDICAL_EXTRACTION, description="Task preset: 'general_ocr', 'medical_extraction', 'table_extraction', 'document_reconstruction', 'custom'")
    engine: OCREngine = Field(OCREngine.VISION_LLM, description="OCR Engine: 'vocr' (Vision LLM) or 'paddle' (Native High-Speed OCR)")
    prompt: Optional[str] = Field(None, description="Custom prompt instructions. If omitted, task_type default is used.")
    system_prompt: Optional[str] = Field(None, description="System instruction. If omitted, task_type default is used.")
    backend: Optional[str] = Field(None, description="LLM backend ('ollama', 'vllm', 'llama-cpp', 'llm-server')")
    model: Optional[str] = Field(None, description="Vision LLM model (e.g. 'ministral-3:latest', 'qwen3-vl:4b')")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(8192, ge=256, le=32768)
    strict_schema: bool = Field(True, description="Enforce strict JSON schema validation and repair (for medical JSON)")

class OCRResponse(BaseModel):
    status: str = "success"
    format: OCRFormat
    task_type: Optional[OCRTaskType] = None
    engine: Optional[OCREngine] = None
    backend: str
    model: str
    data: Union[Dict[str, Any], str] = Field(..., description="Structured JSON object or formatted markdown/html text")
    raw_text: Optional[str] = Field(None, description="Raw transcription output")
    duration_seconds: float = Field(..., description="Processing time in seconds")
    tokens_used: Optional[int] = Field(0, description="Tokens generated")
    page_count: Optional[int] = Field(1, description="Number of rendered document pages")
    created_at: str

class OCRStreamChunk(BaseModel):
    content: str
    done: bool = False
    tokens_generated: Optional[int] = None
    error: Optional[str] = None

