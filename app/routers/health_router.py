from fastapi import APIRouter
from app.services.llm_client import llm_client
from app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Service & Direct LLM Engine Health Check")
async def health_check():
    llm_health = await llm_client.check_health()
    return {
        "status": llm_health.get("status", "unhealthy"),
        "service": "Image-Based Vision Chat API",
        "port": settings.port,
        "direct_backends": {
            "llama_cpp_url": settings.llama_cpp_url,
            "ollama_url": settings.ollama_url,
            "default_backend": settings.default_backend,
            "default_model": settings.default_model,
            "health": llm_health
        }
    }
