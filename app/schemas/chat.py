from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class MessageContentDetail(BaseModel):
    type: str = Field(..., description="Content type, e.g., 'text' or 'image_url'")
    text: Optional[str] = Field(None, description="Text content if type is 'text'")
    image_url: Optional[Dict[str, str]] = Field(None, description="Image URL dict e.g. {'url': 'data:image/jpeg;base64,...'}")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: Union[str, List[MessageContentDetail], List[Dict[str, Any]]] = Field(
        ..., description="Message text content or structured content array containing text & images"
    )
    images: Optional[List[str]] = Field(
        default=None, description="Optional list of base64 images (Ollama style)"
    )

class ImageChatRequest(BaseModel):
    image: Optional[str] = Field(
        None, description="Base64 encoded image string or image URL. Optional if message history includes images."
    )
    prompt: str = Field(
        "Describe this image in high detail.",
        description="Text prompt or query about the image."
    )
    system_prompt: Optional[str] = Field(
        "You are an expert AI vision assistant. Analyze images accurately, thoroughly, and concisely.",
        description="System instruction prompt for the model."
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Previous conversation turns."
    )
    model: Optional[str] = Field(
        None, description="Model name (e.g. qwen3.5-4b, qwen2.5-vl). Defaults to system config."
    )
    backend: Optional[str] = Field(
        None, description="Target backend ('llama-cpp' or 'ollama'). Defaults to system config."
    )
    stream: bool = Field(
        False, description="Whether to stream the response as Server-Sent Events (SSE)."
    )
    temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0 for deterministic & LRU cache)."
    )
    max_tokens: Optional[int] = Field(
        8192, description="Maximum tokens to generate (higher limits prevent cutoff on long documents & reasoning models)."
    )

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 8192
    stream: Optional[bool] = False
    backend: Optional[str] = None

class ChatResponse(BaseModel):
    model: str
    backend: str
    message: ChatMessage
    done: bool = True
    cached: Optional[bool] = False
    usage: Optional[Dict[str, Any]] = None
