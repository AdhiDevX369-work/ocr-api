import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from app.db.session import Base

def generate_uuid_str(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"

class BatchModel(Base):
    __tablename__ = "batches"

    id = Column(String(36), primary_key=True, default=lambda: generate_uuid_str("batch_"))
    name = Column(String(255), nullable=True)
    status = Column(String(32), default="pending", index=True)  # pending, processing, completed, partial_failed, failed, cancelled
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    webhook_url = Column(String(1024), nullable=True)
    meta_data = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    jobs = relationship("JobModel", back_populates="batch", cascade="all, delete-orphan", lazy="selectin")
    webhook_deliveries = relationship("WebhookDeliveryModel", back_populates="batch", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "batch_id": self.id,
            "name": self.name,
            "status": self.status,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "progress_percentage": round((self.processed_files / self.total_files * 100), 1) if self.total_files > 0 else 0.0,
            "webhook_url": self.webhook_url,
            "meta": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: generate_uuid_str("job_"))
    batch_id = Column(String(36), ForeignKey("batches.id", ondelete="CASCADE"), nullable=True, index=True)
    status = Column(String(32), default="pending", index=True)  # pending, processing, completed, failed, cancelled
    document_type = Column(String(64), default="unknown")  # pdf, image, url, base64
    document_name = Column(String(255), nullable=True)
    file_storage_path = Column(String(1024), nullable=True)
    
    prompt = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=True)
    backend = Column(String(64), default="ollama")
    model = Column(String(64), default="qwen2.5vl:latest")
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=2048)

    result_raw = Column(Text, nullable=True)
    result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    tokens_used = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)

    download_url = Column(String(1024), nullable=True)
    webhook_url = Column(String(1024), nullable=True)
    meta_data = Column(JSON, default=dict)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    batch = relationship("BatchModel", back_populates="jobs")
    webhook_deliveries = relationship("WebhookDeliveryModel", back_populates="job", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "job_id": self.id,
            "batch_id": self.batch_id,
            "status": self.status,
            "document_type": self.document_type,
            "document_name": self.document_name,
            "backend": self.backend,
            "model": self.model,
            "download_url": self.download_url,
            "webhook_url": self.webhook_url,
            "result": self.result_json if self.result_json is not None else self.result_raw,
            "error": self.error_message,
            "tokens_used": self.tokens_used,
            "duration_seconds": self.duration_seconds,
            "meta": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class WebhookDeliveryModel(Base):
    __tablename__ = "webhook_deliveries"

    id = Column(String(36), primary_key=True, default=lambda: generate_uuid_str("evt_"))
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    batch_id = Column(String(36), ForeignKey("batches.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = Column(String(64), default="report.processed")
    url = Column(String(1024), nullable=False)
    status_code = Column(Integer, nullable=True)
    attempts = Column(Integer, default=1)
    payload = Column(JSON, nullable=False)
    response_body = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    sent_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    job = relationship("JobModel", back_populates="webhook_deliveries")
    batch = relationship("BatchModel", back_populates="webhook_deliveries")
