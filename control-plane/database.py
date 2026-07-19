import datetime

def utcnow_naive():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./lora_orchestrator.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Worker(Base):
    __tablename__ = "workers"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="offline")  # idle/busy/offline
    gpu_vram_gb = Column(Float, nullable=False)
    last_seen = Column(DateTime, default=utcnow_naive)

    # Relationships
    jobs = relationship("Job", back_populates="worker")

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    base_model = Column(String, nullable=False)
    dataset_url = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    status = Column(String, default="PENDING")  # PENDING/TRAINING/COMPLETED/FAILED
    worker_id = Column(String, ForeignKey("workers.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    weights_path = Column(String, nullable=True)

    # Relationships
    worker = relationship("Worker", back_populates="jobs")
    metrics = relationship("Metric", back_populates="job", cascade="all, delete-orphan")

class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    step = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    epoch = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow_naive)

    # Relationships
    job = relationship("Job", back_populates="metrics")

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
