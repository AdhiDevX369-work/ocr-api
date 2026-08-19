import sys
import os
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.session import init_db
from app.routers import health_router, chat_router, job_router, batch_router, ocr_router
from app.services.llm_client import llm_client

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vocr-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Initializing Production Vision OCR & Batch API on port {settings.port}...")
    logger.info(f"💾 Initializing Database ({settings.database_url})...")
    await init_db()
    logger.info(f"🔗 Direct Backends: llama-cpp ({settings.llama_cpp_url}), Ollama ({settings.ollama_url}) | Default: {settings.default_backend}")
    # Trigger background model pre-loading (warmup into VRAM)
    asyncio.create_task(llm_client.warmup_model())
    yield
    logger.info("🛑 Shutting down API service & closing active connections...")
    await llm_client.close()

app = FastAPI(
    title="Production Vision OCR & Batch Processing API",
    description=(
        "Enterprise-grade Medical & Document Vision OCR platform supporting synchronous extraction, "
        "real-time SSE streaming, asynchronous background jobs, multi-document batch pipelines, "
        "database persistence, and HMAC-signed webhook event dispatch."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# Correlation ID Middleware
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router.router)
app.include_router(ocr_router.router)
app.include_router(batch_router.router)
app.include_router(job_router.router)
app.include_router(chat_router.router)

@app.get("/", summary="Root API Info")
async def root():
    return {
        "service": "Production Vision OCR & Batch Processing Platform",
        "version": "2.0.0",
        "status": "online",
        "port": settings.port,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "ocr_sync": "/api/v1/ocr/sync",
            "ocr_stream": "/api/v1/ocr/stream",
            "ocr_upload": "/api/v1/ocr/upload",
            "batches": "/api/v1/batches",
            "batches_upload": "/api/v1/batches/upload",
            "jobs": "/api/v1/jobs",
            "chat": "/api/v1/image-chat"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
