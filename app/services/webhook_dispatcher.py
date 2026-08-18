import hmac
import hashlib
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import httpx
from app.config import settings
from app.db.session import async_session_factory
from app.db.models import WebhookDeliveryModel

logger = logging.getLogger("webhook-dispatcher")

class WebhookDispatcher:
    @staticmethod
    def calculate_hmac_signature(payload_bytes: bytes, secret: str = settings.webhook_secret) -> str:
        """Calculates HMAC-SHA256 hex digest signature for webhook payload."""
        return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

    @classmethod
    async def dispatch_event(
        cls,
        url: str,
        event_type: str,
        event_id: str,
        data: Dict[str, Any],
        job_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        max_retries: int = settings.webhook_max_retries,
        timeout: float = settings.webhook_timeout
    ) -> bool:
        """
        Asynchronously sends webhook event with HMAC-SHA256 signature and exponential backoff.
        Persists delivery audit log to database.
        """
        if not url or not url.strip().startswith(("http://", "https://")):
            logger.warning(f"Invalid or empty webhook URL: '{url}'")
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        raw_payload = {
            "event_type": event_type,
            "event_id": event_id,
            "timestamp": now_iso,
            "data": data
        }

        payload_json = json.dumps(raw_payload, ensure_ascii=False)
        payload_bytes = payload_json.encode("utf-8")
        signature = cls.calculate_hmac_signature(payload_bytes)

        # Include signature in payload for receiver convenience
        raw_payload["signature"] = signature
        payload_json = json.dumps(raw_payload, ensure_ascii=False)
        payload_bytes = payload_json.encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Event-ID": event_id,
            "X-Event-Type": event_type,
            "X-Timestamp": now_iso,
            "X-Signature-SHA256": signature,
            "User-Agent": "Vision-OCR-Webhook/1.0"
        }

        backoff_delays = [2.0, 8.0, 20.0, 60.0, 120.0]
        attempts = 0
        last_status_code = None
        last_response_text = None
        last_error = None
        delivered = False

        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                logger.info(f"📢 [Event {event_id}] Dispatching '{event_type}' to {url} (attempt {attempt}/{max_retries})...")
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, content=payload_bytes, headers=headers)
                    last_status_code = resp.status_code
                    last_response_text = resp.text[:1000]

                    if resp.is_success:
                        delivered = True
                        logger.info(f"✅ [Event {event_id}] Webhook successfully delivered to {url} (Status: {resp.status_code})")
                        break
                    else:
                        logger.warning(f"⚠️ [Event {event_id}] Webhook target returned HTTP {resp.status_code}: {last_response_text}")
                        last_error = f"HTTP {resp.status_code}"
            except Exception as ex:
                last_error = str(ex)
                logger.warning(f"⚠️ [Event {event_id}] Webhook delivery attempt {attempt} failed: {ex}")

            if attempt < max_retries:
                delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                logger.info(f"⏳ Backoff waiting {delay}s before retry...")
                await asyncio.sleep(delay)

        # Record delivery audit log in database
        try:
            async with async_session_factory() as db_session:
                delivery_record = WebhookDeliveryModel(
                    id=event_id,
                    job_id=job_id,
                    batch_id=batch_id,
                    event_type=event_type,
                    url=url,
                    status_code=last_status_code,
                    attempts=attempts,
                    payload=raw_payload,
                    response_body=last_response_text,
                    error=last_error if not delivered else None
                )
                db_session.add(delivery_record)
                await db_session.commit()
        except Exception as db_err:
            logger.error(f"Failed to record webhook delivery audit in DB: {db_err}")

        return delivered

webhook_dispatcher = WebhookDispatcher()
