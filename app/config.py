import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    port: int = 8200
    host: str = "0.0.0.0"
    llama_cpp_url: str = "http://localhost:8080"
    ollama_url: str = "http://localhost:11434"
    llm_server_url: str = "http://localhost:8100"
    default_backend: str = "llm-server"
    default_model: str = "qwen2.5vl:latest"
    llm_api_key: str = "sk_oWt_VA4WcX84xa18rjt-RovbNH7dwhmZjlrLeDVMIZo"
    streamlit_port: int = 8600
    max_image_size_px: int = 1024
    max_stitched_height_px: int = 1536
    image_jpeg_quality: int = 85
    llm_timeout: float = 300.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

