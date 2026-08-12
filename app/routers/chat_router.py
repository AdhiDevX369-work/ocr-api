import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from app.schemas.chat import (
    ImageChatRequest,
    OpenAIChatCompletionRequest,
    ChatResponse,
    ChatMessage
)
from app.services.image_processor import ImageProcessor, ImageProcessingError
from app.services.llm_client import llm_client, LLMClientError
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Vision Chat"])

def format_openai_multimodal_message(prompt: str, image_uri: Optional[str] = None) -> Dict[str, Any]:
    """Helper to format text + optional image into standard OpenAI multimodal content structure."""
    if not image_uri:
        return {"role": "user", "content": prompt}
    
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri}}
        ]
    }

async def build_messages_payload(
    prompt: str,
    image_input: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[List[ChatMessage]] = None
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    # 1. System Prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. History turns
    if history:
        for turn in history:
            role = turn.role
            content = turn.content
            # If string content
            if isinstance(content, str):
                msg_dict = {"role": role, "content": content}
                # Check for legacy Ollama style images array
                if turn.images and len(turn.images) > 0:
                    processed_imgs = []
                    for img in turn.images:
                        processed_imgs.append(await ImageProcessor.process_image_input(img))
                    msg_dict = {
                        "role": role,
                        "content": [
                            {"type": "text", "text": content},
                            *[{"type": "image_url", "image_url": {"url": img}} for img in processed_imgs]
                        ]
                    }
                messages.append(msg_dict)
            else:
                messages.append({"role": role, "content": content})

    # 3. Current User Turn with Image
    processed_image_uri = None
    if image_input:
        processed_image_uri = await ImageProcessor.process_image_input(image_input)

    messages.append(format_openai_multimodal_message(prompt, processed_image_uri))
    return messages


@router.post(
    "/api/v1/image-chat",
    response_model=ChatResponse,
    summary="JSON Image & Text Multi-modal Chat Endpoint",
    description="Main endpoint for single or multi-turn chat with image inputs (Base64 or URL). Supports streaming."
)
async def image_chat(request: ImageChatRequest):
    try:
        messages = await build_messages_payload(
            prompt=request.prompt,
            image_input=request.image,
            system_prompt=request.system_prompt,
            history=request.history
        )
    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    target_backend = request.backend or settings.default_backend
    target_model = request.model or settings.default_model

    # Streaming mode
    if request.stream:
        async def event_generator():
            try:
                async for token in llm_client.chat_completion_stream(
                    messages=messages,
                    model=target_model,
                    backend=target_backend,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    chunk_data = json.dumps({"content": token, "done": False})
                    yield f"data: {chunk_data}\n\n"
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            except LLMClientError as err:
                err_data = json.dumps({"error": err.message, "done": True})
                yield f"data: {err_data}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming mode
    try:
        raw_res = await llm_client.chat_completion(
            messages=messages,
            model=target_model,
            backend=target_backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )

        choices = raw_res.get("choices", [])
        assistant_content = ""
        if choices:
            assistant_content = choices[0].get("message", {}).get("content", "")

        return ChatResponse(
            model=raw_res.get("model", target_model),
            backend=target_backend,
            message=ChatMessage(role="assistant", content=assistant_content),
            done=True,
            usage=raw_res.get("usage")
        )
    except LLMClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post(
    "/api/v1/image-chat/upload",
    summary="File Upload Image & Text Chat Endpoint",
    description="Endpoint for uploading an image file directly (multipart/form-data) along with text prompt."
)
async def image_chat_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Form("Describe this image in high detail."),
    system_prompt: Optional[str] = Form("You are an expert AI vision assistant."),
    model: Optional[str] = Form(None),
    backend: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    max_tokens: int = Form(2048),
    stream: bool = Form(False)
):
    try:
        file_bytes = await file.read()
        image_uri = ImageProcessor.process_image_bytes(file_bytes)
        messages = await build_messages_payload(
            prompt=prompt,
            image_input=image_uri,
            system_prompt=system_prompt
        )
    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    target_backend = backend or settings.default_backend
    target_model = model or settings.default_model

    if stream:
        async def event_generator():
            try:
                async for token in llm_client.chat_completion_stream(
                    messages=messages,
                    model=target_model,
                    backend=target_backend,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    yield f"data: {json.dumps({'content': token, 'done': False})}\n\n"
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            except LLMClientError as err:
                yield f"data: {json.dumps({'error': err.message, 'done': True})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        raw_res = await llm_client.chat_completion(
            messages=messages,
            model=target_model,
            backend=target_backend,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        choices = raw_res.get("choices", [])
        assistant_content = ""
        if choices:
            assistant_content = choices[0].get("message", {}).get("content", "")

        return ChatResponse(
            model=raw_res.get("model", target_model),
            backend=target_backend,
            message=ChatMessage(role="assistant", content=assistant_content),
            done=True,
            usage=raw_res.get("usage")
        )
    except LLMClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post(
    "/v1/chat/completions",
    summary="OpenAI Compatible Multimodal Chat Completions",
    description="Pass-through OpenAI standard chat completions endpoint."
)
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    target_backend = request.backend or settings.default_backend
    target_model = request.model or settings.default_model

    # Convert request messages array
    raw_messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        if isinstance(content, str):
            raw_messages.append({"role": role, "content": content})
        else:
            raw_messages.append({"role": role, "content": content})

    if request.stream:
        async def event_generator():
            try:
                async for token in llm_client.chat_completion_stream(
                    messages=raw_messages,
                    model=target_model,
                    backend=target_backend,
                    temperature=request.temperature or 0.0,
                    max_tokens=request.max_tokens or 2048
                ):
                    chunk = {
                        "choices": [{"delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except LLMClientError as err:
                yield f"data: {json.dumps({'error': err.message})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        return await llm_client.chat_completion(
            messages=raw_messages,
            model=target_model,
            backend=target_backend,
            temperature=request.temperature or 0.0,
            max_tokens=request.max_tokens or 2048,
            stream=False
        )
    except LLMClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
