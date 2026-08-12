from fastapi import APIRouter
from app.services.llm_client import llm_client
from app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Service & LLM Server Health Check")
async def health_check():
    llm_health = await llm_client.check_health()
    return {
        "status": "healthy" if llm_health.get("status") == "healthy" else "degraded",
        "service": "Image-Based Vision Chat API",
        "port": settings.port,
        "llm_server_gateway": {
            "url": settings.llm_server_url,
            "configured_backend": settings.default_backend,
            "configured_model": settings.default_model,
            "health": llm_health
        }
    }
