import io
import zipfile
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy import select, update, func
from app.config import settings
from app.db.session import async_session_factory
from app.db.models import BatchModel, JobModel
from app.schemas.batch import (
    BatchStatus,
    BatchResponse,
    BatchDetailResponse,
    BatchListResponse,
    BatchCreateRequest,
    BatchDocumentInput
)
from app.schemas.job import JobResponse, JobStatus, JobCreateRequest
from app.services.job_service import job_service
from app.services.webhook_dispatcher import webhook_dispatcher

logger = logging.getLogger("batch-service")

class BatchService:
    async def create_batch(self, request: BatchCreateRequest) -> BatchResponse:
        total_docs = len(request.documents)
        if total_docs == 0:
            raise ValueError("No documents provided in batch request.")
        if total_docs > settings.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum limit of {settings.max_batch_size} documents.")

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        now_dt = datetime.now(timezone.utc)

        # 1. Create Batch record in DB
        async with async_session_factory() as session:
            batch_record = BatchModel(
                id=batch_id,
                name=request.name or f"Batch_{now_dt.strftime('%Y%m%d_%H%M%S')}",
                status=BatchStatus.PENDING.value,
                total_files=total_docs,
                processed_files=0,
                failed_files=0,
                webhook_url=request.webhook_url,
                meta_data=request.meta or {},
                created_at=now_dt
            )
            session.add(batch_record)
            await session.commit()

        logger.info(f"[Batch {batch_id}] Created batch with {total_docs} document(s)")

        # 2. Create individual child jobs for each document
        for idx, doc_item in enumerate(request.documents):
            doc_uri = doc_item.document if isinstance(doc_item, BatchDocumentInput) else str(doc_item)
            doc_name = (
                doc_item.name if isinstance(doc_item, BatchDocumentInput) and doc_item.name
                else f"document_{idx + 1}"
            )
            doc_meta = doc_item.meta if isinstance(doc_item, BatchDocumentInput) else {}

            job_req = JobCreateRequest(
                document=doc_uri,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                backend=request.backend,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                webhook_url=None,  # Parent batch will handle batch.completed event
                meta={"batch_index": idx, "document_name": doc_name, **doc_meta}
            )

            await job_service.create_job(
                request=job_req,
                document_uri=doc_uri,
                batch_id=batch_id,
                document_name=doc_name
            )

        # Update batch status to PROCESSING
        async with async_session_factory() as session:
            await session.execute(
                update(BatchModel)
                .where(BatchModel.id == batch_id)
                .values(status=BatchStatus.PROCESSING.value)
            )
            await session.commit()

        return await self.get_batch(batch_id)

    async def get_batch(self, batch_id: str) -> Optional[BatchResponse]:
        async with async_session_factory() as session:
            res = await session.execute(select(BatchModel).where(BatchModel.id == batch_id))
            batch = res.scalar_one_or_none()
            if not batch:
                return None

            progress_pct = (
                round((batch.processed_files / batch.total_files * 100), 1)
                if batch.total_files > 0 else 0.0
            )

            return BatchResponse(
                batch_id=batch.id,
                name=batch.name,
                status=BatchStatus(batch.status) if batch.status in [s.value for s in BatchStatus] else BatchStatus.PENDING,
                total_files=batch.total_files,
                processed_files=batch.processed_files,
                failed_files=batch.failed_files,
                progress_percentage=progress_pct,
                webhook_url=batch.webhook_url,
                meta=batch.meta_data,
                created_at=batch.created_at.isoformat() if batch.created_at else "",
                completed_at=batch.completed_at.isoformat() if batch.completed_at else None
            )

    async def get_batch_detail(self, batch_id: str) -> Optional[BatchDetailResponse]:
        async with async_session_factory() as session:
            res = await session.execute(select(BatchModel).where(BatchModel.id == batch_id))
            batch = res.scalar_one_or_none()
            if not batch:
                return None

            job_responses: List[JobResponse] = []
            for j in batch.jobs:
                job_responses.append(JobResponse(
                    job_id=j.id,
                    batch_id=j.batch_id,
                    status=JobStatus(j.status) if j.status in [s.value for s in JobStatus] else JobStatus.FAILED,
                    document_type=j.document_type,
                    document_name=j.document_name,
                    backend=j.backend,
                    model=j.model,
                    download_url=j.download_url,
                    webhook_url=j.webhook_url,
                    result=j.result_json if j.result_json is not None else j.result_raw,
                    error=j.error_message,
                    tokens_used=j.tokens_used,
                    duration_seconds=j.duration_seconds,
                    meta=j.meta_data,
                    created_at=j.created_at.isoformat() if j.created_at else "",
                    started_at=j.started_at.isoformat() if j.started_at else None,
                    completed_at=j.completed_at.isoformat() if j.completed_at else None
                ))

            progress_pct = (
                round((batch.processed_files / batch.total_files * 100), 1)
                if batch.total_files > 0 else 0.0
            )

            return BatchDetailResponse(
                batch_id=batch.id,
                name=batch.name,
                status=BatchStatus(batch.status) if batch.status in [s.value for s in BatchStatus] else BatchStatus.PENDING,
                total_files=batch.total_files,
                processed_files=batch.processed_files,
                failed_files=batch.failed_files,
                progress_percentage=progress_pct,
                webhook_url=batch.webhook_url,
                meta=batch.meta_data,
                created_at=batch.created_at.isoformat() if batch.created_at else "",
                completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
                jobs=job_responses
            )

    async def list_batches(self, page: int = 1, page_size: int = 20) -> BatchListResponse:
        offset = (page - 1) * page_size
        async with async_session_factory() as session:
            total_count = await session.scalar(select(func.count(BatchModel.id)))
            res = await session.execute(
                select(BatchModel)
                .order_by(BatchModel.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            batches = res.scalars().all()

            batch_list = []
            for b in batches:
                pct = round((b.processed_files / b.total_files * 100), 1) if b.total_files > 0 else 0.0
                batch_list.append(BatchResponse(
                    batch_id=b.id,
                    name=b.name,
                    status=BatchStatus(b.status) if b.status in [s.value for s in BatchStatus] else BatchStatus.PENDING,
                    total_files=b.total_files,
                    processed_files=b.processed_files,
                    failed_files=b.failed_files,
                    progress_percentage=pct,
                    webhook_url=b.webhook_url,
                    meta=b.meta_data,
                    created_at=b.created_at.isoformat() if b.created_at else "",
                    completed_at=b.completed_at.isoformat() if b.completed_at else None
                ))

            return BatchListResponse(
                total=total_count or 0,
                page=page,
                page_size=page_size,
                batches=batch_list
            )

    async def on_job_completed(self, batch_id: str, job_id: str, is_success: bool):
        """Called when a child job in the batch finishes processing."""
        webhook_to_fire = None
        event_payload = None

        async with async_session_factory() as session:
            res = await session.execute(select(BatchModel).where(BatchModel.id == batch_id))
            batch = res.scalar_one_or_none()
            if not batch:
                return

            batch.processed_files += 1
            if not is_success:
                batch.failed_files += 1

            # Check if all jobs are finished
            if batch.processed_files >= batch.total_files:
                batch.completed_at = datetime.now(timezone.utc)
                if batch.failed_files == 0:
                    batch.status = BatchStatus.COMPLETED.value
                elif batch.failed_files == batch.total_files:
                    batch.status = BatchStatus.FAILED.value
                else:
                    batch.status = BatchStatus.PARTIAL_FAILED.value

                logger.info(f"[Batch {batch_id}] Entire batch completed with status '{batch.status}'")

                if batch.webhook_url:
                    webhook_to_fire = batch.webhook_url
                    event_payload = {
                        "batch_id": batch.id,
                        "name": batch.name,
                        "status": batch.status,
                        "total_files": batch.total_files,
                        "processed_files": batch.processed_files,
                        "failed_files": batch.failed_files,
                        "created_at": batch.created_at.isoformat() if batch.created_at else "",
                        "completed_at": batch.completed_at.isoformat() if batch.completed_at else "",
                        "download_url": f"http://{settings.host}:{settings.port}/api/v1/batches/{batch.id}/download"
                    }

            await session.commit()

        if webhook_to_fire and event_payload:
            evt_id = f"evt_{uuid.uuid4().hex[:12]}"
            await webhook_dispatcher.dispatch_event(
                url=webhook_to_fire,
                event_type="batch.completed",
                event_id=evt_id,
                data=event_payload,
                batch_id=batch_id
            )

    async def generate_batch_archive(self, batch_id: str, format_type: str = "json") -> tuple[bytes, str, str]:
        """
        Generates downloadable export for entire batch.
        Returns: (file_bytes, media_type, filename)
        """
        detail = await self.get_batch_detail(batch_id)
        if not detail:
            raise ValueError(f"Batch '{batch_id}' not found.")

        if format_type.lower() == "zip":
            # Generate ZIP containing individual JSON report files
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Add summary manifest
                manifest = {
                    "batch_id": detail.batch_id,
                    "name": detail.name,
                    "status": detail.status.value,
                    "total_files": detail.total_files,
                    "processed_files": detail.processed_files,
                    "failed_files": detail.failed_files,
                    "created_at": detail.created_at,
                    "completed_at": detail.completed_at
                }
                zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

                for job in detail.jobs:
                    safe_name = (job.document_name or job.job_id).replace("/", "_").replace("\\", "_")
                    fname = f"{safe_name}_{job.job_id}.json"
                    content = {
                        "job_id": job.job_id,
                        "document_name": job.document_name,
                        "status": job.status.value,
                        "duration_seconds": job.duration_seconds,
                        "result": job.result,
                        "error": job.error
                    }
                    zip_file.writestr(fname, json.dumps(content, indent=2, ensure_ascii=False))

            zip_buffer.seek(0)
            return zip_buffer.getvalue(), "application/zip", f"batch_{batch_id}_results.zip"
        else:
            # Merged single JSON
            merged_data = {
                "batch_id": detail.batch_id,
                "name": detail.name,
                "status": detail.status.value,
                "total_files": detail.total_files,
                "processed_files": detail.processed_files,
                "failed_files": detail.failed_files,
                "created_at": detail.created_at,
                "completed_at": detail.completed_at,
                "results": [
                    {
                        "job_id": j.job_id,
                        "document_name": j.document_name,
                        "status": j.status.value,
                        "duration_seconds": j.duration_seconds,
                        "data": j.result,
                        "error": j.error
                    }
                    for j in detail.jobs
                ]
            }
            json_bytes = json.dumps(merged_data, indent=2, ensure_ascii=False).encode("utf-8")
            return json_bytes, "application/json", f"batch_{batch_id}_results.json"

batch_service = BatchService()
