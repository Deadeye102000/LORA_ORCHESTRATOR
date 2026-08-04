# Distributed LoRA Training Orchestrator & Model Registry
A lightweight, distributed orchestrator for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation) on demand. The system supports scheduling jobs across heterogeneous GPU/CPU worker nodes, real-time log and metrics streaming, worker heartbeat monitoring with automatic job failover, and a centralized Model Registry.
This project is built to demonstrate production-level software engineering applied to machine learning infrastructure, highlighting asynchronous task scheduling, distributed state management, live system monitoring, and clean UI dashboard design.
---
## 🏗️ System Architecture
The platform is split into three decoupled subsystems: **Control Plane**, **Compute Plane (Workers)**, and the **Data Plane (Model Registry & Database)**.
```
                  ┌──────────────────────────────────────────────┐
                  │                 CLIENT / UI                  │
                  │        React, Vite, WebSockets, Charts       │
                  └──────────────────────┬───────────────────────┘
                                         │ HTTP & WebSockets
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │                CONTROL PLANE                 │
                  │        FastAPI HTTP & WebSocket Server       │
                  │   Job Scheduler, Worker Heartbeat Registry   │
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
            JSON / TCP Polling   │                │ PostgreSQL Metadata &
            & Log Websockets     │                │ S3 Weight Uploads
                                 ▼                ▼
     ┌───────────────────────────────┐        ┌──────────────────────────┐
     │         COMPUTE PLANE         │        │        DATA PLANE        │
     │  ┌─────────────────────────┐  │        │  ┌────────────────────┐  │
     │  │      Worker Daemon      │  │        │  │   PostgreSQL DB    │  │
     │  └────────────┬────────────┘  │        │  │  (Job/Worker state)│  │
     │               │ Subprocess    │        │  └────────────────────┘  │
     │  ┌────────────▼────────────┐  │        │  ┌────────────────────┐  │
     │  │   PyTorch/PEFT Training │  │        │  │    MinIO / S3      │  │
     │  │    (GPU or CPU Mode)    │  │        │  │   (Model Registry) │  │
     │  └─────────────────────────┘  │        │  └────────────────────┘  │
     └───────────────────────────────┘        └──────────────────────────┘
```
### 1. Control Plane (FastAPI API Gateway)
*   **REST API**: Handles job submission (base model, dataset URL, hyperparameters like rank, alpha, learning rate), worker registration, and model registry queries.
*   **Scheduler**: Matches pending jobs to available workers based on system capability requirements (VRAM, CUDA capabilities).
*   **Heartbeat Monitor**: Regularly checks if workers have checked in. If a worker goes offline during a job, the job is automatically re-queued.
*   **WebSocket Hub**: Channels live logs and training metrics (loss, epoch progress) streamed from the worker daemon directly to the dashboard.
### 2. Compute Plane (Worker Daemon)
*   **Worker Daemon**: A lightweight daemon script running on any node (GPU cluster or CPU fallback mode). It registers with the control plane and polls for jobs.
*   **Trainer Subprocess**: Spawns an isolated training run using Hugging Face `peft` and `trl` library. In CPU mode, it falls back to a smaller model (e.g., `GPT-2` or `TinyLlama-1.1B`) to simulate training without needing expensive hardware.
*   **Log & Metric Streamer**: Monitored training loop logs are parsed and sent back to the control plane in real-time.
### 3. Data Plane (Storage & Registry)
*   **Relational DB (PostgreSQL)**: Holds the structural state of the system—jobs, worker capacity, and model metadata.
*   **Model Registry (S3/MinIO)**: Once training finishes, the worker uploads the resulting adapter weights (`adapter_model.bin`/`adapter_config.json`) to S3-compatible storage.
---
## 🛠️ Tech Stack
*   **Backend & Gateway**: Python 3.10+, FastAPI, SQLAlchemy, WebSockets, Uvicorn
*   **Database**: PostgreSQL / SQLite (for zero-configuration local runs)
*   **Frontend Dashboard**: React, Vite, Chart.js / Recharts (for live loss curves), Vanilla CSS (Custom clean UI)
*   **ML & Deep Learning**: PyTorch, Hugging Face `transformers`, `peft` (Parameter-Efficient Fine-Tuning), `trl`
*   **Infrastructure & DevOps**: Docker, Docker Compose, MinIO (S3 clone)
---
## 🚀 Getting Started (Local Run)
The entire cluster can be simulated locally on a single machine using Docker Compose.
### Prerequisites
*   Docker & Docker Compose
*   (Optional) NVIDIA Container Toolkit (for GPU support)
### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/lora-orchestrator.git
    cd lora-orchestrator
    ```
2.  **Start Services**:
    ```bash
    docker-compose up --build
    ```
    This command spins up:
    *   **PostgreSQL** database on port `5432`
    *   **MinIO (Local S3)** console on port `9001` (S3 API on `9000`)
    *   **FastAPI Control Plane** on port `8000`
    *   **React Dashboard** on port `3000`
    *   **Simulated Worker Node** (automatically polls the control plane and waits for jobs)
3.  **Access the Dashboard**:
    Open your browser and navigate to `http://localhost:3000` to view the live dashboard.
---
## 📋 API Specification Examples
### 1. Submit a Job
`POST /api/jobs`
```json
{
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "dataset_url": "https://huggingface.co/datasets/timdettmers/openassistant-guanaco",
  "hyperparameters": {
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 2
  }
}
```
### 2. Worker Register / Heartbeat
`POST /api/workers/heartbeat`
```json
{
  "worker_id": "worker-gpu-node-01",
  "gpu_vram_gb": 16.0,
  "status": "idle"
}
```
---
## 🎓 Key Engineering Achievements (Resume Bullet Points)
*   Designed and engineered an **asynchronous job scheduler** using FastAPI and SQLAlchemy to coordinate deep learning training runs across active nodes.
*   Built a **real-time metric streaming system** utilizing WebSockets, allowing instant visualization of training loss curves and console logs on a web dashboard.
*   Developed a **fault-tolerant heartbeat protocol** that detects node failure and automatically shifts active training runs back to the global job queue.
*   Optimized demo usability by incorporating a **CPU-fallback pipeline** utilizing lightweight target models (`TinyLlama`), enabling full end-to-end execution without hardware constraints.
