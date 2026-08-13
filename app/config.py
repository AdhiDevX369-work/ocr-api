import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    port: int = 8200
    host: str = "0.0.0.0"
    llm_server_url: str = "http://localhost:11434"
    default_backend: str = "ollama"
    default_model: str = "gemma4:latest"
    llm_api_key: str = ""
    streamlit_port: int = 8600
    max_image_size_px: int = 1280
    image_jpeg_quality: int = 85
    llm_timeout: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
