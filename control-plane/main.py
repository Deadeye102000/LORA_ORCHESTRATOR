import datetime
from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Response, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from database import engine, Base, get_db, SessionLocal, Worker, Job, Metric
from websocket_manager import ConnectionManager

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Distributed LoRA Training Orchestrator")
manager = ConnectionManager()

# ----------------- PYDANTIC SCHEMAS -----------------

class WorkerHeartbeatRequest(BaseModel):
    id: str = Field(..., description="Unique ID of the worker")
    status: str = Field("idle", description="Current status of the worker (idle, busy, offline)")
    gpu_vram_gb: float = Field(..., description="Worker GPU VRAM capacity in GB")

class WorkerResponse(BaseModel):
    id: str
    status: str
    gpu_vram_gb: float
    last_seen: datetime.datetime

    model_config = {"from_attributes": True}

class JobCreateRequest(BaseModel):
    base_model: str = Field(..., description="Base model name/path (e.g. meta-llama/Llama-3-8b)")
    dataset_url: str = Field(..., description="URL or path to training dataset")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="JSON dictionary of LoRA hyperparameters")

class JobResponse(BaseModel):
    id: int
    base_model: str
    dataset_url: str
    hyperparameters: Dict[str, Any]
    status: str
    worker_id: Optional[str] = None
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    weights_path: Optional[str] = None

    model_config = {"from_attributes": True}

class WorkerPollRequest(BaseModel):
    worker_id: str = Field(..., description="ID of the worker polling for jobs")

class MetricCreateRequest(BaseModel):
    step: Optional[int] = Field(None, description="Training step number")
    loss: Optional[float] = Field(None, description="Loss value")
    epoch: Optional[float] = Field(None, description="Epoch number")
    log_text: Optional[str] = Field(None, description="Raw log or output text")

class JobUpdateRequest(BaseModel):
    status: str = Field(..., description="Status (COMPLETED, FAILED, TRAINING)")
    weights_path: Optional[str] = Field(None, description="Path to trained adapter weights")

# ----------------- ROUTER SKELETONS -----------------

workers_router = APIRouter(prefix="/api/workers", tags=["Workers"])
jobs_router = APIRouter(prefix="/api/jobs", tags=["Jobs"])
metrics_router = APIRouter(prefix="/api/metrics", tags=["Metrics"])  # Skeleton for future phase

# --- Workers Router Endpoints ---

@workers_router.post("/heartbeat", response_model=WorkerResponse)
async def worker_heartbeat(payload: WorkerHeartbeatRequest, db: Session = Depends(get_db)):
    db_worker = db.query(Worker).filter(Worker.id == payload.id).first()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    
    if not db_worker:
        db_worker = Worker(
            id=payload.id,
            status=payload.status,
            gpu_vram_gb=payload.gpu_vram_gb,
            last_seen=now
        )
        db.add(db_worker)
    else:
        db_worker.status = payload.status
        db_worker.gpu_vram_gb = payload.gpu_vram_gb
        db_worker.last_seen = now
        
    db.commit()
    db.refresh(db_worker)
    
    # Broadcast status update to system room
    worker_data = {
        "id": db_worker.id,
        "status": db_worker.status,
        "gpu_vram_gb": db_worker.gpu_vram_gb,
        "last_seen": db_worker.last_seen.isoformat()
    }
    await manager.broadcast_to_room("system", {
        "event": "worker_update",
        "worker": worker_data
    })
    
    return db_worker

@workers_router.get("", response_model=List[WorkerResponse])
def list_workers(db: Session = Depends(get_db)):
    return db.query(Worker).all()

# --- Jobs Router Endpoints ---

@jobs_router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def submit_job(payload: JobCreateRequest, db: Session = Depends(get_db)):
    db_job = Job(
        base_model=payload.base_model,
        dataset_url=payload.dataset_url,
        hyperparameters=payload.hyperparameters,
        status="PENDING"
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

@jobs_router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    db_job = db.query(Job).filter(Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job

@jobs_router.post("/poll", response_model=Optional[JobResponse])
def poll_job(payload: WorkerPollRequest, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.status == "PENDING").order_by(Job.created_at.asc()).first()
    if not job:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    job.status = "TRAINING"
    job.worker_id = payload.worker_id
    job.started_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    db.commit()
    db.refresh(job)
    return job

@jobs_router.post("/{job_id}/metrics", status_code=status.HTTP_201_CREATED)
async def add_job_metric(job_id: int, payload: MetricCreateRequest, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Persist numerical metrics if present
    if payload.step is not None and payload.loss is not None and payload.epoch is not None:
        db_metric = Metric(
            job_id=job_id,
            step=payload.step,
            loss=payload.loss,
            epoch=payload.epoch,
            timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        )
        db.add(db_metric)
        db.commit()

    # Broadcast updates to job room
    await manager.broadcast_to_room(f"job_{job_id}", {
        "event": "metric_update",
        "job_id": job_id,
        "step": payload.step,
        "loss": payload.loss,
        "epoch": payload.epoch,
        "log_text": payload.log_text,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    })
    return {"message": "Metric recorded successfully"}

@jobs_router.patch("/{job_id}", response_model=JobResponse)
def update_job(job_id: int, payload: JobUpdateRequest, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job.status = payload.status
    if payload.weights_path is not None:
        job.weights_path = payload.weights_path
        
    if payload.status in ["COMPLETED", "FAILED"]:
        job.completed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        
    db.commit()
    db.refresh(job)
    return job

# --- Metrics Router Skeleton Endpoints ---

@metrics_router.get("/{job_id}")
def get_job_metrics(job_id: int, db: Session = Depends(get_db)):
    # Skeleton implementation for Phase 2/3
    db_metrics = db.query(Metric).filter(Metric.job_id == job_id).all()
    return {"job_id": job_id, "metrics": db_metrics}

# --- WebSocket Endpoints ---

@app.websocket("/ws/system")
async def websocket_system(websocket: WebSocket):
    await manager.connect(websocket, "system")
    with SessionLocal() as db:
        workers = db.query(Worker).all()
        workers_list = [
            {
                "id": w.id,
                "status": w.status,
                "gpu_vram_gb": w.gpu_vram_gb,
                "last_seen": w.last_seen.isoformat()
            } for w in workers
        ]
    try:
        await websocket.send_json({
            "event": "init_workers",
            "workers": workers_list
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "system")

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job(websocket: WebSocket, job_id: int):
    room_id = f"job_{job_id}"
    await manager.connect(websocket, room_id)
    with SessionLocal() as db:
        metrics = db.query(Metric).filter(Metric.job_id == job_id).order_by(Metric.step.asc()).all()
        history = [
            {
                "step": m.step,
                "loss": m.loss,
                "epoch": m.epoch,
                "timestamp": m.timestamp.isoformat()
            } for m in metrics
        ]
    try:
        await websocket.send_json({
            "event": "init_metrics",
            "job_id": job_id,
            "metrics": history
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, room_id)

# ----------------- APP REGISTRATION -----------------

app.include_router(workers_router)
app.include_router(jobs_router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "Distributed LoRA Training Orchestrator API is running"}
