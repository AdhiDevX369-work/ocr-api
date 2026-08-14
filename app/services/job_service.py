import uuid
import asyncio
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from app.schemas.job import JobStatus, JobResponse, JobCreateRequest, WebhookEventPayload
from app.services.image_processor import ImageProcessor, ImageProcessingError
from app.services.llm_client import llm_client, LLMClientError
from app.config import settings

logger = logging.getLogger(__name__)

class JobService:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        # Single worker semaphore so batch jobs execute sequentially without overloading local GPU LLM server
        self._gpu_semaphore = asyncio.Semaphore(1)

    async def create_job(self, request: JobCreateRequest, document_uri: Optional[str] = None) -> JobResponse:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        doc_input = document_uri or request.document
        if not doc_input:
            raise ValueError("No PDF/Image document provided (base64, file upload, or URL link required).")

        download_url = f"http://{settings.host}:{settings.port}/api/v1/jobs/{job_id}/download"

        job_record = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "created_at": now_iso,
            "completed_at": None,
            "document_input": doc_input,
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "backend": request.backend or settings.default_backend,
            "model": request.model or settings.default_model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "webhook_url": request.webhook_url,
            "meta": request.meta or {},
            "result": None,
            "error": None,
            "download_url": download_url
        }

        async with self._lock:
            self._jobs[job_id] = job_record

        # Launch async processing in background task
        asyncio.create_task(self._process_job_background(job_id))

        return self._to_job_response(job_record)

    async def get_job(self, job_id: str) -> Optional[JobResponse]:
        async with self._lock:
            record = self._jobs.get(job_id)
            if record:
                return self._to_job_response(record)
        return None

    async def get_job_raw(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _process_job_background(self, job_id: str):
        async with self._gpu_semaphore:
            logger.info(f"🔄 [Job {job_id}] Started processing background job...")
            
            async with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job["status"] = JobStatus.PROCESSING

        try:
            # 1. Normalize PDF or Image document into vision base64 URI
            doc_uri = await ImageProcessor.process_image_input(job["document_input"])

            # 2. Build LLM vision prompt payload
            messages = [
                {"role": "system", "content": job["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": job["prompt"]},
                        {"type": "image_url", "image_url": {"url": doc_uri}}
                    ]
                }
            ]

            # 3. Call LLM Vision Backend via streaming to maintain active connection and prevent timeouts
            content_parts = []
            async for chunk in llm_client.chat_completion_stream(
                messages=messages,
                model=job["model"],
                backend=job["backend"],
                temperature=job["temperature"],
                max_tokens=job["max_tokens"]
            ):
                content_parts.append(chunk)

            assistant_content = "".join(content_parts)

            finish_iso = datetime.now(timezone.utc).isoformat()

            async with self._lock:
                job["status"] = JobStatus.COMPLETED
                job["completed_at"] = finish_iso
                job["result"] = assistant_content
                job["error"] = None

            logger.info(f"✅ [Job {job_id}] Successfully processed report job!")

            # 4. Emit Webhook Event (PubSub / Hook)
            if job["webhook_url"]:
                await self._emit_webhook_event(job)

        except (ImageProcessingError, LLMClientError, Exception) as err:
            err_msg = str(err)
            logger.error(f"❌ [Job {job_id}] Job failed: {err_msg}")
            finish_iso = datetime.now(timezone.utc).isoformat()

            async with self._lock:
                job["status"] = JobStatus.FAILED
                job["completed_at"] = finish_iso
                job["error"] = err_msg

            if job["webhook_url"]:
                await self._emit_webhook_event(job)

    async def _emit_webhook_event(self, job: Dict[str, Any]):
        webhook_url = job["webhook_url"]
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        payload = WebhookEventPayload(
            event_type="report.processed",
            event_id=event_id,
            timestamp=now_iso,
            data={
                "job_id": job["job_id"],
                "status": job["status"].value if isinstance(job["status"], JobStatus) else str(job["status"]),
                "created_at": job["created_at"],
                "completed_at": job["completed_at"],
                "download_url": job["download_url"],
                "result": job["result"],
                "error": job["error"],
                "meta": job["meta"]
            }
        ).model_dump()

        logger.info(f"📢 [Job {job['job_id']}] Emitting event '{event_id}' to PubSub/Webhook endpoint: {webhook_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(webhook_url, json=payload)
                logger.info(f"📬 Webhook response status: {res.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to deliver webhook event to '{webhook_url}': {e}")

    def _to_job_response(self, record: Dict[str, Any]) -> JobResponse:
        return JobResponse(
            job_id=record["job_id"],
            status=record["status"],
            created_at=record["created_at"],
            completed_at=record["completed_at"],
            webhook_url=record["webhook_url"],
            download_url=record["download_url"],
            result=record["result"],
            error=record["error"],
            meta=record["meta"]
        )

job_service = JobService()
