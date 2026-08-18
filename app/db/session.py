import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings

logger = logging.getLogger("ocr-db")

Base = declarative_base()

# Configure Async Engine
engine_kwargs = {"echo": settings.db_echo}

# SQLite doesn't support pool_size and max_overflow in the same way as Postgres
if "sqlite" not in settings.database_url:
    engine_kwargs.update({
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True
    })

engine = create_async_engine(settings.database_url, **engine_kwargs)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to yield an async database session per request."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initializes database tables on application startup."""
    try:
        async with engine.begin() as conn:
            # Import models to ensure they are registered with Base metadata
            from app.db import models  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables verified and initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
