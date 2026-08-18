from enum import Enum
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class OCRFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"

class DocumentType(str, Enum):
    AUTO = "auto"
    PDF = "pdf"
    IMAGE = "image"

class OCRRequest(BaseModel):
    document: str = Field(..., description="Base64 Data URI or HTTP URL of PDF or Image scan")
    format: OCRFormat = Field(OCRFormat.JSON, description="Desired output format: 'json', 'text', or 'markdown'")
    prompt: Optional[str] = Field(
        "Extract all medical report data, patient information, and test parameters with high precision.",
        description="Custom prompt instructions"
    )
    system_prompt: Optional[str] = Field(
        "You are an expert Medical Report OCR and Verification AI.",
        description="System instruction"
    )
    backend: Optional[str] = Field(None, description="LLM backend ('ollama', 'llama-cpp', 'llm-server')")
    model: Optional[str] = Field(None, description="Vision LLM model (e.g. 'qwen2.5vl:latest')")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(4096, ge=256, le=16384)
    strict_schema: bool = Field(True, description="Enforce strict JSON schema validation and repair")

class OCRResponse(BaseModel):
    status: str = "success"
    format: OCRFormat
    backend: str
    model: str
    data: Union[Dict[str, Any], str] = Field(..., description="Structured JSON object or formatted text/markdown")
    raw_text: Optional[str] = Field(None, description="Raw transcription output")
    duration_seconds: float = Field(..., description="Processing time in seconds")
    tokens_used: Optional[int] = Field(0, description="Tokens generated")
    created_at: str

class OCRStreamChunk(BaseModel):
    content: str
    done: bool = False
    tokens_generated: Optional[int] = None
    error: Optional[str] = None
