import os
import sys
import time
import subprocess
import requests

def main():
    print("=== Starting End-to-End Integration Verification ===")
    
    # 1. Clear existing database to ensure clean run
    db_file = "./lora_orchestrator.db"
    db_in_cp = "./control-plane/lora_orchestrator.db"
    for path in [db_file, db_in_cp]:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Cleared existing SQLite database file at: {path}")
            except Exception as e:
                print(f"Could not remove database file {path}: {e}")
            
    # 2. Start FastAPI Server
    control_plane_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "control-plane")
    server_process = subprocess.Popen(
        ["../.venv/bin/uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=control_plane_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("FastAPI server spawned in background.")
    
    # Wait for server to start up
    time.sleep(3)
    
    # Verify server is up
    url = "http://127.0.0.1:8000/"
    try:
        res = requests.get(url, timeout=3)
        print(f"Server response: {res.status_code} - {res.json()}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        server_process.terminate()
        sys.exit(1)
        
    # 3. Submit a training job
    job_payload = {
        "base_model": "prajjwal1/bert-tiny",
        "dataset_url": "s3://mock-bucket/dummy-dataset.json",
        "hyperparameters": {
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 3e-4,
            "epochs": 1.0,
            "batch_size": 2
        }
    }
    submit_url = "http://127.0.0.1:8000/api/jobs"
    try:
        res = requests.post(submit_url, json=job_payload, timeout=3)
        job_data = res.json()
        job_id = job_data["id"]
        print(f"Submitted Job {job_id}. Status: {job_data['status']}")
    except Exception as e:
        print(f"Failed to submit job: {e}")
        server_process.terminate()
        sys.exit(1)
        
    # 4. Start worker daemon
    worker_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "worker")
    env = os.environ.copy()
    env["CONTROL_PLANE_URL"] = "http://127.0.0.1:8000"
    env["WORKER_ID"] = "test-worker-1"
    env["POLL_INTERVAL"] = "1"
    
    daemon_process = subprocess.Popen(
        ["../.venv/bin/python", "daemon.py"],
        cwd=worker_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Worker daemon spawned in background.")
    
    # 5. Monitor the job status
    status_url = f"http://127.0.0.1:8000/api/jobs/{job_id}"
    metrics_url = f"http://127.0.0.1:8000/api/metrics/{job_id}"
    workers_url = "http://127.0.0.1:8000/api/workers"
    
    completed = False
    timeout = 30
    start_time = time.time()
    
    # Print worker list once registered
    time.sleep(2)
    try:
        res = requests.get(workers_url)
        print(f"Registered Workers: {res.json()}")
    except Exception:
        pass
        
    while time.time() - start_time < timeout:
        try:
            res = requests.get(status_url, timeout=3)
            job = res.json()
            status = job["status"]
            print(f"Checking Job {job_id} Status: {status}")
            
            if status in ["COMPLETED", "FAILED"]:
                completed = True
                print(f"Job finished with final status: {status}")
                if status == "COMPLETED":
                    print(f"Weights path reported: {job['weights_path']}")
                break
        except Exception as e:
            print(f"Error querying job status: {e}")
            
        time.sleep(2)
        
    # 6. Retrieve metrics
    if completed:
        try:
            res = requests.get(metrics_url, timeout=3)
            metrics = res.json()
            print(f"Job Metrics Recorded in DB: {len(metrics['metrics'])} steps logged.")
            for m in metrics['metrics']:
                print(f"  Step {m['step']}: loss={m['loss']}, epoch={m['epoch']}")
        except Exception as e:
            print(f"Error querying metrics: {e}")
            
    # 7. Shutdown
    print("Shutting down processes...")
    daemon_process.terminate()
    server_process.terminate()
    
    # Read outputs to clear buffers
    daemon_out, daemon_err = daemon_process.communicate()
    server_out, server_err = server_process.communicate()
    
    print("\n--- Worker Daemon Log Snippet ---")
    print(daemon_out)
    if daemon_err:
        print("Daemon Stderr:")
        print(daemon_err)
        
    if completed and status == "COMPLETED":
        print("\n=== VERIFICATION SUCCESSFUL ===")
        sys.exit(0)
    else:
        print("\n=== VERIFICATION FAILED or TIMED OUT ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
