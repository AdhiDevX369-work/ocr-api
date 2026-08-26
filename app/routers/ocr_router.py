import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from app.config import settings
from app.schemas.ocr import OCRRequest, OCRResponse, OCRFormat
from app.schemas.medical import MedicalReportExtraction
from app.services.image_processor import ImageProcessor, ImageProcessingError
from app.services.llm_client import llm_client, LLMClientError
from app.services.schema_validator import SchemaValidator

logger = logging.getLogger("ocr-router")

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream",
}

router = APIRouter(tags=["Direct Vision OCR (Single Processing)"])

@router.post(
    "",
    response_model=OCRResponse,
    summary="Synchronous Direct Vision OCR (Default)",
    description="Processes a single PDF document or image scan synchronously and returns structured JSON or Markdown text."
)
@router.post(
    "/sync",
    response_model=OCRResponse,
    summary="Synchronous Direct Vision OCR",
    description="Processes a single PDF document or image scan synchronously and returns structured JSON or Markdown text."
)
async def process_ocr_sync(request: OCRRequest):
    start_time = time.monotonic()
    target_backend = request.backend or settings.default_backend
    target_model = request.model or settings.default_model

    try:
        # 1. Normalize Document
        doc_uri = await ImageProcessor.process_image_input(request.document)

        # 2. Build Multi-modal Prompt
        messages = [
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image_url", "image_url": {"url": doc_uri}}
                ]
            }
        ]

        # 3. Call LLM
        raw_res = await llm_client.chat_completion(
            messages=messages,
            model=target_model,
            backend=target_backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            json_mode=(request.format == OCRFormat.JSON)
        )

        choices = raw_res.get("choices", [])
        raw_output = choices[0].get("message", {}).get("content", "") if choices else ""
        duration = round(time.monotonic() - start_time, 2)
        now_iso = datetime.now(timezone.utc).isoformat()

        # 4. Format Output
        if request.format == OCRFormat.JSON:
            parsed_data, err = SchemaValidator.parse_and_validate(raw_output, MedicalReportExtraction if request.strict_schema else None)
            return OCRResponse(
                status="success",
                format=request.format,
                backend=target_backend,
                model=target_model,
                data=parsed_data if parsed_data else {"raw": raw_output},
                raw_text=raw_output,
                duration_seconds=duration,
                created_at=now_iso
            )
        else:
            return OCRResponse(
                status="success",
                format=request.format,
                backend=target_backend,
                model=target_model,
                data=raw_output,
                raw_text=raw_output,
                duration_seconds=duration,
                created_at=now_iso
            )

    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except LLMClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OCR processing failed: {str(e)}")

@router.post(
    "/stream",
    summary="Real-time SSE Streaming Vision OCR",
    description="Streams OCR transcription tokens in real-time via Server-Sent Events (SSE)."
)
async def process_ocr_stream(request: OCRRequest):
    target_backend = request.backend or settings.default_backend
    target_model = request.model or settings.default_model

    try:
        doc_uri = await ImageProcessor.process_image_input(request.document)
        messages = [
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image_url", "image_url": {"url": doc_uri}}
                ]
            }
        ]
    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def event_generator():
        token_count = 0
        try:
            async for token in llm_client.chat_completion_stream(
                messages=messages,
                model=target_model,
                backend=target_backend,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                json_mode=(request.format == OCRFormat.JSON)
            ):
                token_count += 1
                chunk = {"content": token, "done": False, "tokens": token_count}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'content': '', 'done': True, 'tokens': token_count})}\n\n"
        except LLMClientError as err:
            yield f"data: {json.dumps({'error': err.message, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Clinical Laboratory Report Vision OCR AI. "
    "Your objective is to accurately transcribe patient demographics and lab test observations from any laboratory layout into strict, structured JSON."
)
DEFAULT_USER_PROMPT = (
    "Analyze this laboratory diagnostic report and extract all patient demographics and test observations into a JSON object matching this exact structure:\n"
    "{\n"
    '  "report_title": "Full Blood Count / Urine UPCR / Biochemistry",\n'
    '  "patient_info": {\n'
    '    "patient_name": "...",\n'
    '    "pid_no": "...",\n'
    '    "age": "...",\n'
    '    "sex": "...",\n'
    '    "tel_no": "",\n'
    '    "reference_dr": "",\n'
    '    "registered_on": "",\n'
    '    "collected_on": "",\n'
    '    "reported_on": ""\n'
    "  },\n"
    '  "results": [\n'
    "    {\n"
    '      "type": "fasting_blood_sugar",\n'
    '      "name": "Fasting Blood Sugar",\n'
    '      "value": "104",\n'
    '      "unit": "mg/dl"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "CRITICAL RULES:\n"
    "1. In the 'results' array, include ONLY actual patient test observations. Do NOT extract reference range tables, interpretation remark grids (e.g. '< 0.2 Normal'), or age guideline charts as results.\n"
    "2. If the value and unit are in the same column (e.g. '104 mg/dl'), separate them cleanly into value ('104') and unit ('mg/dl').\n"
    "3. Set 'type' as a lowercase snake_case identifier (e.g. 'fasting_blood_sugar', 'total_cholesterol', 'protein_total', 'creatinine', 'protein_creatinine_ratio').\n"
    "4. Return strictly valid JSON."
)

@router.post(
    "/upload",
    response_model=OCRResponse,
    summary="Direct File Upload Vision OCR",
    description="Upload a PDF file or Image scan directly via multipart/form-data for instant OCR extraction."
)
async def process_ocr_upload(
    file: UploadFile = File(..., description="PDF document or Image file to extract"),
    format: OCRFormat = Form(OCRFormat.JSON),
    prompt: Optional[str] = Form(DEFAULT_USER_PROMPT),
    system_prompt: Optional[str] = Form(DEFAULT_SYSTEM_PROMPT),
    backend: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    max_tokens: int = Form(1536)
):
    try:
        file_bytes = await file.read()
        doc_uri = ImageProcessor.process_image_bytes(file_bytes)
        req = OCRRequest(
            document=doc_uri,
            format=format,
            prompt=prompt,
            system_prompt=system_prompt,
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return await process_ocr_sync(req)
    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File OCR failed: {str(e)}")
