# ⚡ Distributed LoRA Training Orchestrator & Model Registry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5+-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?logo=docker)](https://www.docker.com/)

A lightweight, production-grade distributed orchestrator designed for fine-tuning Large Language Models (LLMs) on demand using **LoRA (Low-Rank Adaptation)**.

The system dynamically coordinates training tasks across heterogeneous compute nodes (GPU/CPU), streams live console logs and loss metrics over WebSockets, maintains fault isolation through worker heartbeat monitoring with automatic failover, and archives trained adapter weights into an S3-compatible Model Registry.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A: Fast Launch via Docker Compose (Recommended)](#option-a-fast-launch-via-docker-compose-recommended)
  - [Option B: Manual Local Development Setup](#option-b-manual-local-development-setup)
- [API & WebSocket Specifications](#-api--websocket-specifications)
  - [HTTP Endpoints](#http-endpoints)
  - [JSON Payload Reference](#json-payload-reference)
  - [WebSocket Channels](#websocket-channels)
- [Environment Configuration](#-environment-configuration)
  - [Control Plane Settings](#control-plane-settings)
  - [Worker Daemon Settings](#worker-daemon-settings)
- [Verification & Testing](#-verification--testing)
- [Engineering Highlights](#-engineering-highlights)
- [License](#-license)

---

## ✨ Key Features

- **🎯 Asynchronous Task Scheduling**: Intelligent matching of queued LoRA training requests to available worker nodes based on hardware resource availability and status.
- **⚡ Real-Time WebSockets Streaming**: Direct push of console logs, epoch progress, and continuous training loss curves (`loss`, `epoch`, `step`) from compute nodes to the React dashboard.
- **🛡️ Fault-Tolerant Heartbeat Protocol**: Background monitor thread automatically detects worker crashes or network timeouts (>30s) and safely reverts stuck `TRAINING` jobs back to `PENDING` for self-healing failover.
- **💻 CPU Fallback & Demo Mode**: Fully functional execution on standard laptops using lightweight base models (`TinyLlama-1.1B` or `GPT-2`) with CPU tensor ops, requiring zero GPU hardware to test end-to-end workflows.
- **📦 S3-Compatible Model Registry**: Automated packaging and S3 bucket upload (`adapter_model.bin`, `adapter_config.json`, training metrics summary) via MinIO or AWS S3.
- **🎨 Glassmorphic Real-Time Dashboard**: Modern responsive UI with live worker topology grid, job submission modal, interactive loss charts (Recharts), and embedded terminal output viewer.

---

## 🏗️ System Architecture

The orchestrator is built using a decoupled three-layer architecture: **Control Plane**, **Compute Plane**, and **Data Plane**.

```
                   ┌──────────────────────────────────────────────┐
                   │               REACT DASHBOARD                │
                   │        React 18, Vite, WebSockets, Charts     │
                   └──────────────────────┬───────────────────────┘
                                          │ HTTP REST & WebSockets
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │                CONTROL PLANE                 │
                   │        FastAPI Gateway & API Server          │
                   │   Job Scheduler & Heartbeat Monitor Loop     │
                   └──────────────┬────────────────┬──────────────┘
                                  │                │
             JSON / TCP Polling   │                │ Database Persistence &
             & Log Websockets     │                │ S3 Artifact Uploads
                                  ▼                ▼
      ┌───────────────────────────────┐        ┌──────────────────────────┐
      │         COMPUTE PLANE         │        │        DATA PLANE        │
      │  ┌─────────────────────────┐  │        │  ┌────────────────────┐  │
      │  │      Worker Daemon      │  │        │  │   PostgreSQL / DB  │  │
      │  └────────────┬────────────┘  │        │  │  (Job & Worker State)│  │
      │               │ Subprocess    │        │  └────────────────────┘  │
      │  ┌────────────▼────────────┐  │        │  ┌────────────────────┐  │
      │  │   PyTorch/PEFT Trainer  │  │        │  │    MinIO / S3      │  │
      │  │   (GPU / CPU Fallback)  │  │        │  │   (Model Registry) │  │
      │  └─────────────────────────┘  │        │  └────────────────────┘  │
      └───────────────────────────────┘        └──────────────────────────┘
```

### 1. Control Plane (`FastAPI API Gateway`)
- **Job Scheduler**: Exposes REST endpoints to receive job requests and dispatches them to active nodes via `/api/jobs/poll`.
- **Heartbeat Sweep**: Runs an asynchronous background loop checking worker active status every 10 seconds. Stale workers are set to `offline` and their pending work is re-queued.
- **WebSocket Room Manager**: Maintains dual WebSocket channels (`/ws/system` for cluster updates, `/ws/jobs/{job_id}` for per-job telemetry).

### 2. Compute Plane (`Worker Daemon & Trainer`)
- **Worker Daemon (`daemon.py`)**: Runs on compute nodes, sends periodic heartbeat pulses (`/api/workers/heartbeat`), and polls for work.
- **Isolated Subprocess Trainer (`trainer.py`)**: Spawns an isolated Python subprocess utilizing Hugging Face `peft`, `transformers`, and `trl` to execute fine-tuning runs.
- **Telemetry Exporter**: Captures stdout/stderr logs and streams formatted training metrics directly to the Control Plane.

### 3. Data Plane (`Metadata DB & Object Storage`)
- **Relational Storage**: Stores structural records for workers, jobs, execution history, and loss metrics (SQLite for local development, PostgreSQL for production).
- **Model Registry (MinIO / S3)**: Persists final LoRA adapter checkpoints and metadata for deployment.

---

## 📂 Repository Structure

```
.
├── control-plane/             # FastAPI Backend Service
│   ├── main.py                # REST endpoints, WebSocket routes & heartbeat monitor
│   ├── database.py            # SQLAlchemy models (Worker, Job, Metric) & DB connection
│   ├── websocket_manager.py   # WebSocket pub/sub connection manager
│   ├── test_main.py           # Pytest unit tests for Control Plane routes
│   ├── Dockerfile             # Container definition for Control Plane
│   └── requirements.txt       # Control Plane Python dependencies
│
├── worker/                    # Compute Node Daemon & Trainer
│   ├── daemon.py              # Worker heartbeat, polling daemon & log forwarder
│   ├── trainer.py             # PyTorch + PEFT/TRL LoRA fine-tuning subprocess
│   ├── weights/               # Local staging directory for generated adapter weights
│   ├── Dockerfile             # Container definition for Worker node
│   └── requirements.txt       # Compute plane dependencies (torch, peft, transformers)
│
├── frontend/                  # React Real-Time Operations Dashboard
│   ├── src/
│   │   ├── components/        # UI Components (WorkerGrid, JobList, LogTerminal, LossChart)
│   │   ├── hooks/             # Custom React Hooks (useSocket.js for WebSockets)
│   │   ├── App.jsx            # Main dashboard layout
│   │   ├── main.jsx           # React app entry point
│   │   └── index.css          # Modern dark-mode & Tailwind CSS design system
│   ├── package.json           # Frontend dependencies & build scripts
│   ├── tailwind.config.js     # Tailwind design system configuration
│   └── tsconfig.json          # TypeScript compiler options
│
├── verify_worker.py           # End-to-end integration test (Control Plane + Worker)
├── verify_websockets.py       # End-to-end WebSocket log & metric verification script
├── docker-compose.yml         # One-command multi-container cluster setup
├── pyrightconfig.json         # Python language server environment configuration
└── Readme.md                  # Project documentation
```

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn, WebSockets, Pydantic |
| **Database & ORM** | PostgreSQL, SQLite, SQLAlchemy 2.0 |
| **Machine Learning** | PyTorch, Hugging Face `transformers`, `peft` (LoRA), `trl` |
| **Frontend Framework** | React 18, Vite, Lucide React Icons, Recharts |
| **Styling & Design** | TailwindCSS, PostCSS, Vanilla CSS glassmorphism |
| **Object Storage** | MinIO S3 Object Store, AWS S3 SDK (`boto3`) |
| **Containerization** | Docker, Docker Compose |

---

## 🚀 Getting Started

### Prerequisites

- **Docker** (v20.10+) & **Docker Compose** (v2.0+)
- *(Optional for manual runs)* **Python 3.10+** and **Node.js 18+**

---

### Option A: Fast Launch via Docker Compose (Recommended)

Spin up the complete cluster—MinIO S3, Control Plane, React Dashboard, and Compute Worker—with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/Deadeye102000/LORA_ORCHESTRATOR.git
cd LORA_ORCHESTRATOR

# 2. Launch all services
docker-compose up --build
```

#### Services Endpoint Map

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **React Ops Dashboard** | `http://localhost:3000` | Real-time monitoring & job submission UI |
| **Control Plane API** | `http://localhost:8000` | REST API gateway & OpenAPI Swagger docs (`/docs`) |
| **MinIO Console** | `http://localhost:9001` | S3 Object Store web UI (`minioadmin` / `minioadmin`) |
| **MinIO S3 Endpoint** | `http://localhost:9000` | S3 API endpoint for model weight uploads |

---

### Option B: Manual Local Development Setup

#### 1. Setup Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r control-plane/requirements.txt
pip install -r worker/requirements.txt
```

#### 2. Start the Control Plane API Server
```bash
cd control-plane
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Start a Compute Worker Daemon
In a new terminal window:
```bash
source .venv/bin/activate
export CONTROL_PLANE_URL="http://127.0.0.1:8000"
export WORKER_ID="worker-local-01"
python worker/daemon.py
```

#### 4. Launch the React Frontend
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

---

## 📋 API & WebSocket Specifications

### HTTP Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health check |
| `POST` | `/api/workers/heartbeat` | Worker heartbeat registration & status update |
| `GET` | `/api/workers` | Retrieve list of all registered worker nodes |
| `POST` | `/api/jobs` | Submit a new LoRA fine-tuning job |
| `GET` | `/api/jobs/{job_id}` | Fetch detailed job status, parameters, and logs |
| `POST` | `/api/jobs/poll` | Polling endpoint used by workers to fetch queued jobs |
| `POST` | `/api/jobs/{job_id}/metrics` | Endpoint for workers to stream metric points & logs |
| `GET` | `/api/metrics/{job_id}` | Retrieve historical loss metrics for a job |

---

### JSON Payload Reference

#### 1. Submit Job (`POST /api/jobs`)
```json
{
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "dataset_url": "https://huggingface.co/datasets/timdettmers/openassistant-guanaco",
  "hyperparameters": {
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": 0.0002,
    "epochs": 3,
    "batch_size": 4
  }
}
```

#### 2. Worker Heartbeat (`POST /api/workers/heartbeat`)
```json
{
  "id": "worker-node-01",
  "status": "idle",
  "gpu_vram_gb": 16.0
}
```

#### 3. Stream Telemetry (`POST /api/jobs/{job_id}/metrics`)
```json
{
  "epoch": 1,
  "step": 50,
  "loss": 0.4215,
  "log_message": "Epoch 1/3 - Step 50/150 - Loss: 0.4215"
}
```

---

### WebSocket Channels

| Channel | Room | Payload Description |
| :--- | :--- | :--- |
| `ws://localhost:8000/ws/system` | `system` | Cluster events (`worker_registered`, `worker_offline`, `job_created`, `job_status_change`) |
| `ws://localhost:8000/ws/jobs/{job_id}` | `job_{id}` | Real-time console line append & live loss metric updates |

---

## ⚙️ Environment Configuration

### Control Plane Settings

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `sqlite:///./lora_orchestrator.db` | Connection string for metadata database |
| `PORT` | `8000` | Control Plane HTTP & WS port |

### Worker Daemon Settings

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `CONTROL_PLANE_URL` | `http://localhost:8000` | Target Control Plane URL |
| `WORKER_ID` | `worker-daemon-1` | Unique node identifier string |
| `POLL_INTERVAL` | `5` | Job polling interval in seconds |
| `MINIO_ENDPOINT` | `http://localhost:9000` | S3 API endpoint for model registry uploads |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | S3 Access Key |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` | S3 Secret Key |
| `S3_BUCKET` | `lora-models` | Target S3 bucket name for adapter weights |

---

## 🧪 Verification & Testing

The workspace includes automated integration tests to verify cluster orchestration end-to-end:

### 1. Control Plane Unit Tests
```bash
pytest control-plane/test_main.py
```

### 2. End-to-End Worker & Job Scheduling Verification
Spawns Control Plane and Worker Daemon processes locally, submits a test job, verifies polling, execution, heartbeat updates, and database state transitions:
```bash
python verify_worker.py
```

### 3. Real-Time WebSockets Telemetry Verification
Validates live connection handling, system broadcast events, and real-time loss curve data streaming:
```bash
python verify_websockets.py
```

---

## 🎓 Engineering Highlights

- **Robust Distributed State Machine**: Implemented clean job status transitions (`PENDING` ➔ `TRAINING` ➔ `COMPLETED` / `FAILED`) with transaction isolation to prevent race conditions during worker polling.
- **Failover & Self-Healing Workflows**: Designed background heartbeat sweep loops to ensure no job remains stuck indefinitely if a worker process crashes mid-training.
- **Scalable WebSockets Telemetry**: Created targeted channel rooms (`ConnectionManager`) to broadcast high-frequency training metrics only to active dashboard subscribers.
- **CPU Simulation & Zero-Cost Testing**: Added intelligent hardware fallback logic into `trainer.py` so full training loops, evaluation metrics, and weight packaging can be tested without requiring dedicated GPU instances.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
