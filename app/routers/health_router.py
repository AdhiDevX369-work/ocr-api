from fastapi import APIRouter
from sqlalchemy import text
from app.services.llm_client import llm_client
from app.db.session import async_session_factory
from app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Service, Database & LLM Engine Health Check")
async def health_check():
    llm_health = await llm_client.check_health()
    
    # DB health check
    db_status = "healthy"
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    overall_status = "healthy" if db_status == "healthy" and llm_health.get("status") == "healthy" else "degraded"

    return {
        "status": overall_status,
        "service": "Production Vision OCR & Batch API",
        "port": settings.port,
        "database": {
            "status": db_status,
            "url": settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url
        },
        "direct_backends": {
            "llama_cpp_url": settings.llama_cpp_url,
            "ollama_url": settings.ollama_url,
            "default_backend": settings.default_backend,
            "default_model": settings.default_model,
            "health": llm_health
        }
    }

