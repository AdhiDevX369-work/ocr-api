import sys
import os
import logging
from contextlib import asynccontextmanager

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import health_router, chat_router
from app.services.llm_client import llm_client

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("image-chat-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting Image-Based Chat API on port {settings.port}")
    logger.info(f"🔗 Target LLM Gateway: {settings.llm_server_url} (Default Backend: {settings.default_backend}, Model: {settings.default_model})")
    yield
    logger.info("🛑 Shutting down API service & closing HTTP connections...")
    await llm_client.close()

app = FastAPI(
    title="Production Image-Based Vision Chat API",
    description=(
        "High-performance vision chat API interface supporting base64 images, direct file uploads, "
        "and image URLs with real-time SSE streaming. Interfaced to LLM Gateway Server running at port 8100 "
        "with llama.cpp backend and Qwen 3.5 4B model."
    ),
    version="1.0.0",
    lifespan=lifespan
)

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
app.include_router(chat_router.router)

@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "service": "Image-Based Vision Chat API",
        "version": "1.0.0",
        "status": "running",
        "port": settings.port,
        "docs_url": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
