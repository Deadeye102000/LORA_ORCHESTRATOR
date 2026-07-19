import os
import sys
import time
import asyncio
import subprocess
import requests
import websockets
import json

async def monitor_ws_system(uri, events_queue):
    print(f"[Client] Connecting to System WS: {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"[System WS Client Received] {data['event']}")
                await events_queue.put(data)
    except Exception as e:
        print(f"[System WS Client Error] {e}")

async def monitor_ws_job(uri, events_queue):
    print(f"[Client] Connecting to Job WS: {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"[Job WS Client Received] {data['event']} (step={data.get('step')}, log={data.get('log_text') is not None})")
                await events_queue.put(data)
    except Exception as e:
        print(f"[Job WS Client Error] {e}")

async def async_main():
    print("=== Starting End-to-End WebSocket Integration Verification ===")
    
    # 1. Clear DB
    for db_path in ["./lora_orchestrator.db", "./control-plane/lora_orchestrator.db"]:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"Cleared SQLite DB file: {db_path}")
            except Exception:
                pass

    # 2. Start control plane
    control_plane_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "control-plane")
    server_process = subprocess.Popen(
        ["../.venv/bin/uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=control_plane_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("FastAPI server spawned in background.")
    await asyncio.sleep(3)
    
    # 3. Create events queues
    system_queue = asyncio.Queue()
    job_queue = asyncio.Queue()
    
    # Start WS listener tasks
    system_ws_task = asyncio.create_task(monitor_ws_system("ws://127.0.0.1:8000/ws/system", system_queue))
    await asyncio.sleep(1)
    
    # 4. Submit Job
    job_payload = {
        "base_model": "prajjwal1/bert-tiny",
        "dataset_url": "s3://mock-bucket/dummy-dataset.json",
        "hyperparameters": {
            "lora_r": 4,
            "lora_alpha": 8,
            "learning_rate": 1e-4,
            "epochs": 1.0,
            "batch_size": 2
        }
    }
    submit_url = "http://127.0.0.1:8000/api/jobs"
    res = requests.post(submit_url, json=job_payload, timeout=3)
    job_data = res.json()
    job_id = job_data["id"]
    print(f"Submitted Job {job_id}.")
    
    # Start Job WS listener task
    job_ws_task = asyncio.create_task(monitor_ws_job(f"ws://127.0.0.1:8000/ws/jobs/{job_id}", job_queue))
    await asyncio.sleep(1)
    
    # 5. Start Worker Daemon
    worker_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "worker")
    env = os.environ.copy()
    env["CONTROL_PLANE_URL"] = "http://127.0.0.1:8000"
    env["WORKER_ID"] = "ws-worker-1"
    env["POLL_INTERVAL"] = "1"
    
    daemon_process = subprocess.Popen(
        ["../.venv/bin/python", "daemon.py"],
        cwd=worker_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Worker daemon spawned.")
    
    # 6. Wait for training completion
    timeout = 30
    start_time = time.time()
    completed = False
    status = "UNKNOWN"
    
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"http://127.0.0.1:8000/api/jobs/{job_id}", timeout=2)
            status = res.json()["status"]
            if status in ["COMPLETED", "FAILED"]:
                completed = True
                print(f"Job completed with status: {status}")
                break
        except Exception:
            pass
        await asyncio.sleep(2)
        
    # Shutdown processes
    print("Shutting down processes...")
    daemon_process.terminate()
    server_process.terminate()
    system_ws_task.cancel()
    job_ws_task.cancel()
    
    # Read outputs to clear buffers
    daemon_process.communicate()
    server_process.communicate()
    
    # Assertions on received events
    system_events = []
    while not system_queue.empty():
        system_events.append(await system_queue.get())
        
    job_events = []
    while not job_queue.empty():
        job_events.append(await job_queue.get())
        
    print("\n=== VERIFICATION ANALYSIS ===")
    
    init_workers_events = [e for e in system_events if e["event"] == "init_workers"]
    print(f"init_workers events received: {len(init_workers_events)}")
    
    worker_updates = [e for e in system_events if e["event"] == "worker_update"]
    print(f"worker_update events received: {len(worker_updates)}")
    for u in worker_updates:
        print(f"  Worker Status Update: {u['worker']['id']} status={u['worker']['status']}")
        
    init_metrics_events = [e for e in job_events if e["event"] == "init_metrics"]
    print(f"init_metrics events received: {len(init_metrics_events)}")
    
    metric_updates = [e for e in job_events if e["event"] == "metric_update"]
    print(f"metric_update events received: {len(metric_updates)}")
    
    structured_metrics = [m for m in metric_updates if m["step"] is not None]
    raw_logs = [m for m in metric_updates if m["step"] is None and m["log_text"] is not None]
    
    print(f"  Structured metric updates: {len(structured_metrics)}")
    print(f"  Raw console log updates: {len(raw_logs)}")
    
    if raw_logs:
        print(f"  First raw log sample: {raw_logs[0]['log_text']}")
    if structured_metrics:
        print(f"  First structured metric sample: step={structured_metrics[0]['step']}, loss={structured_metrics[0]['loss']}")

    success = completed and status == "COMPLETED" and len(init_workers_events) > 0 and len(worker_updates) > 0 and len(metric_updates) > 0 and len(raw_logs) > 0
    if success:
        print("\n=== WEBSOCKET VERIFICATION SUCCESSFUL ===")
        sys.exit(0)
    else:
        print("\n=== WEBSOCKET VERIFICATION FAILED ===")
        sys.exit(1)

def main():
    try:
        asyncio.run(async_main())
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    main()
