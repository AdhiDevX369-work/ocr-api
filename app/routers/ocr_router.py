import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from app.config import settings
from app.schemas.ocr import OCRRequest, OCRResponse, OCRFormat, OCRTaskType
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

PROMPT_PRESETS: Dict[OCRTaskType, Dict[str, str]] = {
    OCRTaskType.GENERAL_OCR: {
        "system": (
            "You are a cutting-edge Vision-Language OCR model. "
            "Your objective is to accurately transcribe all content from document pages into structured, high-fidelity Markdown."
        ),
        "user": (
            "Transcribe all text from this document naturally in exact reading order preserving structural hierarchy.\n"
            "Formatting Rules:\n"
            "- Represent tables using clean HTML (<table>...</table>) or Markdown tables.\n"
            "- Format mathematical expressions and chemical formulas in LaTeX ($...$ or $$...$$).\n"
            "- For charts or diagrams, provide descriptions inside <img>...</img> tags.\n"
            "- Preserve checkboxes using ☐ (unchecked) and ☑ (checked).\n"
            "- Wrap watermarks in <watermark>...</watermark> and page numbers in <page_number>...</page_number>.\n"
            "- Maintain original headings, bullet lists, and paragraphs faithfully without summarizing."
        )
    },
    OCRTaskType.MEDICAL_EXTRACTION: {
        "system": (
            "You are an expert Clinical Laboratory Report Vision OCR AI. "
            "Your objective is to accurately transcribe patient demographics and lab test observations from any laboratory layout into strict, structured JSON."
        ),
        "user": (
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
    },
    OCRTaskType.TABLE_EXTRACTION: {
        "system": "You are a specialized Document Table & Grid Extraction AI.",
        "user": "Extract all data tables and structured grids from this document into clean HTML <table> structures with exact column headers and row alignment."
    },
    OCRTaskType.DOCUMENT_RECONSTRUCTION: {
        "system": "You are an expert Document Layout and Semantic Reconstruction AI.",
        "user": "Faithfully reconstruct the full document layout, headings, text sections, tables, and figures into semantic, publication-grade Markdown."
    },
    OCRTaskType.CUSTOM: {
        "system": "You are an expert Vision-Language Document AI.",
        "user": "Extract and transcribe all content from this document accurately."
    }
}

def resolve_prompts(request: OCRRequest) -> tuple[str, str]:
    task = request.task_type or (OCRTaskType.MEDICAL_EXTRACTION if request.format == OCRFormat.JSON else OCRTaskType.GENERAL_OCR)
    preset = PROMPT_PRESETS.get(task, PROMPT_PRESETS[OCRTaskType.GENERAL_OCR])

    system_prompt = request.system_prompt or preset["system"]
    user_prompt = request.prompt or preset["user"]
    return system_prompt, user_prompt


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
        # 1. Process Document with Multi-Page extraction (no squashing)
        doc_res = await ImageProcessor.process_document_input(request.document)
        page_uris = doc_res.get("page_data_uris", [doc_res["primary_data_uri"]])
        page_count = len(page_uris)

        system_prompt, user_prompt = resolve_prompts(request)

        # 2. Build Multi-modal User Turn containing all pages
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for uri in page_uris:
            user_content.append({"type": "image_url", "image_url": {"url": uri}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
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
        used_model = raw_res.get("model", target_model)
        duration = round(time.monotonic() - start_time, 2)
        now_iso = datetime.now(timezone.utc).isoformat()

        # 4. Format Output
        if request.format == OCRFormat.JSON:
            target_schema = MedicalReportExtraction if (request.task_type == OCRTaskType.MEDICAL_EXTRACTION and request.strict_schema) else None
            parsed_data, err = SchemaValidator.parse_and_validate(raw_output, target_schema)
            return OCRResponse(
                status="success",
                format=request.format,
                task_type=request.task_type,
                backend=target_backend,
                model=used_model,
                data=parsed_data if parsed_data else {"raw": raw_output},
                raw_text=raw_output,
                duration_seconds=duration,
                page_count=page_count,
                created_at=now_iso
            )
        else:
            return OCRResponse(
                status="success",
                format=request.format,
                task_type=request.task_type,
                backend=target_backend,
                model=used_model,
                data=raw_output,
                raw_text=raw_output,
                duration_seconds=duration,
                page_count=page_count,
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
        doc_res = await ImageProcessor.process_document_input(request.document)
        page_uris = doc_res.get("page_data_uris", [doc_res["primary_data_uri"]])
        system_prompt, user_prompt = resolve_prompts(request)

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for uri in page_uris:
            user_content.append({"type": "image_url", "image_url": {"url": uri}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
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

@router.post(
    "/upload",
    response_model=OCRResponse,
    summary="Direct File Upload Vision OCR",
    description="Upload a PDF file or Image scan directly via multipart/form-data for instant OCR extraction."
)
async def process_ocr_upload(
    file: UploadFile = File(..., description="PDF document or Image file to extract"),
    format: OCRFormat = Form(OCRFormat.JSON),
    task_type: Optional[OCRTaskType] = Form(OCRTaskType.MEDICAL_EXTRACTION),
    prompt: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    backend: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    max_tokens: int = Form(8192)
):
    try:
        file_bytes = await file.read()
        doc_res = ImageProcessor.process_document(file_bytes)
        req = OCRRequest(
            document=doc_res["primary_data_uri"],
            format=format,
            task_type=task_type or OCRTaskType.MEDICAL_EXTRACTION,
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

