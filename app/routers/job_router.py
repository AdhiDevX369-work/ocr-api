import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from app.schemas.job import JobCreateRequest, JobResponse
from app.services.job_service import job_service
from app.services.image_processor import ImageProcessor, ImageProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Async Document Jobs & Webhooks"])

@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Async PDF / Image Report Processing Job",
    description="Submits a PDF document or Image report for background processing. Emits a webhook event once completed."
)
async def submit_job(request: JobCreateRequest):
    try:
        job_res = await job_service.create_job(request)
        return job_res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Job creation failed: {str(e)}")


@router.post(
    "/upload",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload PDF / Image File as Async Job",
    description="Upload a PDF file or Image scan (multipart/form-data) to process as an asynchronous background job with webhook callback."
)
async def submit_job_upload(
    file: UploadFile = File(..., description="PDF document or Image scan file"),
    prompt: Optional[str] = Form("Perform an exact line-by-line verification check of all values in this report against the printed document."),
    system_prompt: Optional[str] = Form("You are an expert Medical Report OCR and Verification AI."),
    backend: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    max_tokens: int = Form(2048),
    webhook_url: Optional[str] = Form(None),
    meta_json: Optional[str] = Form(None)
):
    try:
        file_bytes = await file.read()
        doc_uri = ImageProcessor.process_image_bytes(file_bytes)

        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {"raw_meta": meta_json}

        request = JobCreateRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            webhook_url=webhook_url,
            meta=meta
        )

        return await job_service.create_job(request, document_uri=doc_uri)

    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File job upload failed: {str(e)}")


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get Job Status and Extraction Result",
    description="Retrieve status, progress, download link, and extracted data for a specific background job."
)
async def get_job_status(job_id: str):
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job


@router.get(
    "/{job_id}/download",
    summary="Download Job Result",
    description="Download the processed report output as a JSON or text file for downstream processing."
)
async def download_job_result(job_id: str, format: Optional[str] = "json"):
    raw_job = await job_service.get_job_raw(job_id)
    if not raw_job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    if raw_job["status"] != "completed":
        return JSONResponse(
            status_code=status.HTTP_425_TOO_EARLY,
            content={
                "error": f"Job is currently in '{raw_job['status']}' state. Please try again when completed.",
                "job_id": job_id,
                "status": raw_job["status"]
            }
        )

    result_content = raw_job.get("result", "")

    if format.lower() == "json":
        if isinstance(result_content, dict) or isinstance(result_content, list):
            return JSONResponse(
                content=result_content,
                headers={"Content-Disposition": f"attachment; filename=report_result_{job_id}.json"}
            )
        try:
            parsed_json = json.loads(result_content)
            return JSONResponse(
                content=parsed_json,
                headers={"Content-Disposition": f"attachment; filename=report_result_{job_id}.json"}
            )
        except Exception:
            data_payload = {
                "job_id": job_id,
                "completed_at": raw_job.get("completed_at"),
                "meta": raw_job.get("meta"),
                "extracted_text": str(result_content)
            }
            return JSONResponse(
                content=data_payload,
                headers={"Content-Disposition": f"attachment; filename=report_result_{job_id}.json"}
            )
    else:
        text_str = json.dumps(result_content, indent=2) if isinstance(result_content, (dict, list)) else str(result_content)
        return PlainTextResponse(
            content=text_str,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=report_result_{job_id}.txt"}
        )

