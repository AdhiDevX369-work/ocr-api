import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    port: int = 8200
    host: str = "0.0.0.0"
    llama_cpp_url: str = "http://localhost:8080"
    ollama_url: str = "http://localhost:11434"
    llm_server_url: str = "http://localhost:8100"
    default_backend: str = "llm-server"
    default_model: str = "ministral-3:latest"
    llm_api_key: str = "sk_oWt_VA4WcX84xa18rjt-RovbNH7dwhmZjlrLeDVMIZo"
    streamlit_port: int = 8600
    max_image_size_px: int = 1024
    max_stitched_height_px: int = 1536
    image_jpeg_quality: int = 85
    llm_timeout: float = 300.0
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "2h"
    # Database & Storage
    database_url: str = "sqlite+aiosqlite:///./ocr.db"  # Defaults to async SQLite, override with postgresql+asyncpg://user:pass@host:5432/db in .env
    db_echo: bool = False
    db_pool_size: int = 20
    db_max_overflow: int = 10
    storage_dir: str = "./storage"

    # Batch & Job Execution Settings
    max_batch_size: int = 100
    max_concurrent_workers: int = 4
    webhook_timeout: float = 15.0
    webhook_max_retries: int = 5
    webhook_secret: str = "ocr-webhook-secret-key-369"
    api_key: str = "sk-vocr-prod-api-key-default"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()


