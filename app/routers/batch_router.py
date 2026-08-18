import json
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, Query, status
from fastapi.responses import JSONResponse, Response
from app.schemas.batch import (
    BatchCreateRequest,
    BatchResponse,
    BatchDetailResponse,
    BatchListResponse,
    BatchDocumentInput
)
from app.services.batch_service import batch_service
from app.services.image_processor import ImageProcessor, ImageProcessingError

logger = logging.getLogger("batch-router")

router = APIRouter(prefix="/api/v1/batches", tags=["Enterprise Batch Processing"])

@router.post(
    "",
    response_model=BatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Multi-Document Batch Processing Job",
    description="Submits a batch of PDF documents or Image reports (up to 100 files/URLs) for asynchronous background processing."
)
async def submit_batch(request: BatchCreateRequest):
    try:
        batch_res = await batch_service.create_batch(request)
        return batch_res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch submission failed: {str(e)}")

@router.post(
    "/upload",
    response_model=BatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Multiple Files as a Batch",
    description="Upload multiple PDF files or Image scans in a single multipart request to process as an asynchronous batch."
)
async def submit_batch_upload(
    files: List[UploadFile] = File(..., description="Multiple PDF documents or Image files to process"),
    name: Optional[str] = Form(None, description="Optional batch title"),
    prompt: Optional[str] = Form("Perform an exact line-by-line verification check and extract all values into structured JSON."),
    system_prompt: Optional[str] = Form("You are an expert Medical Report OCR and Clinical Data Extraction AI."),
    backend: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    max_tokens: int = Form(2048),
    webhook_url: Optional[str] = Form(None),
    meta_json: Optional[str] = Form(None)
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    try:
        documents: List[BatchDocumentInput] = []
        for f in files:
            bytes_data = await f.read()
            doc_uri = ImageProcessor.process_image_bytes(bytes_data)
            documents.append(BatchDocumentInput(
                document=doc_uri,
                name=f.filename,
                meta={"content_type": f.content_type, "size_bytes": len(bytes_data)}
            ))

        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {"raw_meta": meta_json}

        batch_req = BatchCreateRequest(
            name=name or f"Upload_Batch_{len(files)}_Files",
            documents=documents,
            prompt=prompt,
            system_prompt=system_prompt,
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            webhook_url=webhook_url,
            meta=meta
        )

        return await batch_service.create_batch(batch_req)

    except ImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch file upload failed: {str(e)}")

@router.get(
    "",
    response_model=BatchListResponse,
    summary="List Recent Batches",
    description="Retrieves a paginated list of recent batch jobs and their overall statuses."
)
async def list_batches(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    return await batch_service.list_batches(page=page, page_size=page_size)

@router.get(
    "/{batch_id}",
    response_model=BatchResponse,
    summary="Get Batch Status & Progress Metrics",
    description="Retrieve real-time processing progress (% completed, processed/failed count, status) for a batch."
)
async def get_batch_status(batch_id: str):
    batch = await batch_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch '{batch_id}' not found.")
    return batch

@router.get(
    "/{batch_id}/jobs",
    response_model=BatchDetailResponse,
    summary="Get Batch Details & Individual Document Results",
    description="Retrieve batch metadata along with the detailed status and extracted JSON for every individual document job."
)
async def get_batch_jobs(batch_id: str):
    detail = await batch_service.get_batch_detail(batch_id)
    if not detail:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch '{batch_id}' not found.")
    return detail

@router.get(
    "/{batch_id}/download",
    summary="Download Complete Batch Extractions (ZIP or JSON)",
    description="Download all processed report extractions for this batch as a single consolidated JSON or a ZIP archive of individual files."
)
async def download_batch_results(batch_id: str, format: str = Query("json", description="Export format: 'json' or 'zip'")):
    try:
        content_bytes, media_type, filename = await batch_service.generate_batch_archive(batch_id, format_type=format)
        return Response(
            content=content_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Download generation failed: {str(e)}")
