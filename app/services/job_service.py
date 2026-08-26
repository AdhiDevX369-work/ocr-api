import uuid
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from sqlalchemy import select, update
from app.config import settings
from app.db.session import async_session_factory
from app.db.models import JobModel, BatchModel
from app.schemas.job import JobStatus, JobResponse, JobCreateRequest
from app.schemas.medical import MedicalReportExtraction
from app.services.image_processor import ImageProcessor, ImageProcessingError
from app.services.llm_client import llm_client, LLMClientError
from app.services.schema_validator import SchemaValidator
from app.services.webhook_dispatcher import webhook_dispatcher

logger = logging.getLogger("job-service")

class JobService:
    def __init__(self):
        # Concurrency semaphore to control maximum concurrent GPU/LLM tasks
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_workers)
        # Background task references to prevent garbage collection
        self._active_tasks = set()

    async def create_job(
        self,
        request: JobCreateRequest,
        document_uri: Optional[str] = None,
        batch_id: Optional[str] = None,
        document_name: Optional[str] = None
    ) -> JobResponse:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        doc_input = document_uri or request.document
        if not doc_input:
            raise ValueError("No PDF/Image document provided (Base64 data or URL required).")

        download_url = f"http://{settings.host}:{settings.port}/api/v1/jobs/{job_id}/download"

        doc_type = "pdf" if doc_input.startswith("data:application/pdf") or ".pdf" in doc_input.lower() else "image"

        target_backend = request.backend or settings.default_backend
        target_model = request.model or settings.default_model

        async with async_session_factory() as session:
            job_record = JobModel(
                id=job_id,
                batch_id=batch_id,
                status=JobStatus.PENDING.value,
                document_type=doc_type,
                document_name=document_name or "document",
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                backend=target_backend,
                model=target_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                download_url=download_url,
                webhook_url=request.webhook_url,
                meta_data=request.meta or {},
                created_at=datetime.now(timezone.utc)
            )
            session.add(job_record)
            await session.commit()
            await session.refresh(job_record)

        # Launch async processing in background task
        task = asyncio.create_task(self._process_job_worker(job_id, doc_input))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

        return JobResponse(
            job_id=job_id,
            batch_id=batch_id,
            status=JobStatus.PENDING,
            document_type=doc_type,
            document_name=document_name,
            backend=target_backend,
            model=target_model,
            download_url=download_url,
            webhook_url=request.webhook_url,
            meta=request.meta or {},
            created_at=datetime.now(timezone.utc).isoformat()
        )

    async def get_job(self, job_id: str) -> Optional[JobResponse]:
        async with async_session_factory() as session:
            result = await session.execute(select(JobModel).where(JobModel.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return None
            return JobResponse(
                job_id=job.id,
                batch_id=job.batch_id,
                status=JobStatus(job.status) if job.status in [s.value for s in JobStatus] else JobStatus.FAILED,
                document_type=job.document_type,
                document_name=job.document_name,
                backend=job.backend,
                model=job.model,
                download_url=job.download_url,
                webhook_url=job.webhook_url,
                result=job.result_json if job.result_json is not None else job.result_raw,
                error=job.error_message,
                tokens_used=job.tokens_used,
                duration_seconds=job.duration_seconds,
                meta=job.meta_data,
                created_at=job.created_at.isoformat() if job.created_at else "",
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None
            )

    async def get_job_raw(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with async_session_factory() as session:
            result = await session.execute(select(JobModel).where(JobModel.id == job_id))
            job = result.scalar_one_or_none()
            return job.to_dict() if job else None

    async def _process_job_worker(self, job_id: str, doc_input: str):
        async with self._semaphore:
            start_time = time.monotonic()
            started_at_dt = datetime.now(timezone.utc)
            logger.info(f"[Job {job_id}] Worker started processing")

            # 1. Update status to PROCESSING in DB
            async with async_session_factory() as session:
                await session.execute(
                    update(JobModel)
                    .where(JobModel.id == job_id)
                    .values(status=JobStatus.PROCESSING.value, started_at=started_at_dt)
                )
                await session.commit()

            # Retrieve job params
            async with async_session_factory() as session:
                res = await session.execute(select(JobModel).where(JobModel.id == job_id))
                job = res.scalar_one_or_none()
                if not job:
                    return
                prompt = job.prompt
                system_prompt = job.system_prompt
                backend = job.backend
                model = job.model
                temperature = job.temperature
                max_tokens = job.max_tokens
                webhook_url = job.webhook_url
                batch_id = job.batch_id

            try:
                # 2. Normalize and prepare document (multi-page support)
                doc_res = await ImageProcessor.process_document_input(doc_input)
                page_uris = doc_res.get("page_data_uris", [doc_res["primary_data_uri"]])

                # 3. Prepare Multi-modal Prompt with all document pages
                user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for uri in page_uris:
                    user_content.append({"type": "image_url", "image_url": {"url": uri}})

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]

                # 4. Stream LLM Vision completion
                content_parts = []
                async for chunk in llm_client.chat_completion_stream(
                    messages=messages,
                    model=model,
                    backend=backend,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True
                ):
                    content_parts.append(chunk)

                raw_output = "".join(content_parts)
                duration = round(time.monotonic() - start_time, 2)
                completed_at_dt = datetime.now(timezone.utc)

                # 5. Parse and Validate Structured JSON Output
                parsed_json, parse_err = SchemaValidator.parse_and_validate(raw_output, MedicalReportExtraction)

                # 6. Update Job in DB as COMPLETED
                async with async_session_factory() as session:
                    await session.execute(
                        update(JobModel)
                        .where(JobModel.id == job_id)
                        .values(
                            status=JobStatus.COMPLETED.value,
                            result_raw=raw_output,
                            result_json=parsed_json,
                            error_message=parse_err,
                            duration_seconds=duration,
                            completed_at=completed_at_dt
                        )
                    )
                    await session.commit()

                logger.info(f"[Job {job_id}] Successfully completed in {duration}s")

                # 7. Emit Webhook Event if configured
                if webhook_url:
                    evt_id = f"evt_{uuid.uuid4().hex[:12]}"
                    await webhook_dispatcher.dispatch_event(
                        url=webhook_url,
                        event_type="report.processed",
                        event_id=evt_id,
                        data={
                            "job_id": job_id,
                            "batch_id": batch_id,
                            "status": "completed",
                            "duration_seconds": duration,
                            "result": parsed_json if parsed_json else raw_output,
                            "download_url": f"http://{settings.host}:{settings.port}/api/v1/jobs/{job_id}/download"
                        },
                        job_id=job_id,
                        batch_id=batch_id
                    )

                # 8. Notify parent Batch if this job belongs to a batch
                if batch_id:
                    from app.services.batch_service import batch_service
                    await batch_service.on_job_completed(batch_id, job_id, is_success=True)

            except Exception as err:
                duration = round(time.monotonic() - start_time, 2)
                err_msg = str(err)
                logger.error(f"[Job {job_id}] Failed after {duration}s: {err_msg}")
                completed_at_dt = datetime.now(timezone.utc)

                async with async_session_factory() as session:
                    await session.execute(
                        update(JobModel)
                        .where(JobModel.id == job_id)
                        .values(
                            status=JobStatus.FAILED.value,
                            error_message=err_msg,
                            duration_seconds=duration,
                            completed_at=completed_at_dt
                        )
                    )
                    await session.commit()

                if webhook_url:
                    evt_id = f"evt_{uuid.uuid4().hex[:12]}"
                    await webhook_dispatcher.dispatch_event(
                        url=webhook_url,
                        event_type="report.processed",
                        event_id=evt_id,
                        data={
                            "job_id": job_id,
                            "batch_id": batch_id,
                            "status": "failed",
                            "error": err_msg,
                            "duration_seconds": duration
                        },
                        job_id=job_id,
                        batch_id=batch_id
                    )

                if batch_id:
                    from app.services.batch_service import batch_service
                    await batch_service.on_job_completed(batch_id, job_id, is_success=False)

job_service = JobService()
