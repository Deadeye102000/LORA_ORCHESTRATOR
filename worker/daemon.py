import os
import sys
import time
import json
import subprocess
import threading
import requests

class WorkerDaemon:
    def __init__(self):
        self.control_plane_url = os.getenv("CONTROL_PLANE_URL", "http://localhost:8000").rstrip("/")
        self.worker_id = os.getenv("WORKER_ID", "worker-1")
        self.poll_interval = float(os.getenv("POLL_INTERVAL", "5"))
        self.status = "idle"  # idle or busy
        self.stop_event = threading.Event()
        self.gpu_vram = self._detect_vram()
        
    def _detect_vram(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                return round(total_mem / (1024 ** 3), 2)
        except Exception:
            pass
        return 16.0  # Fallback 16 GB VRAM

    def heartbeat_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = {
                    "id": self.worker_id,
                    "status": self.status,
                    "gpu_vram_gb": self.gpu_vram
                }
                url = f"{self.control_plane_url}/api/workers/heartbeat"
                res = requests.post(url, json=payload, timeout=5)
                if res.status_code != 200:
                    print(f"[Heartbeat] Failed to send heartbeat (HTTP {res.status_code})", file=sys.stderr)
            except Exception as e:
                print(f"[Heartbeat] Connection error: {e}", file=sys.stderr)
            self.stop_event.wait(10)

    def poll_and_execute(self):
        if self.status == "busy":
            return
            
        try:
            url = f"{self.control_plane_url}/api/jobs/poll"
            res = requests.post(url, json={"worker_id": self.worker_id}, timeout=5)
            if res.status_code == 204:
                return  # No pending jobs
            if res.status_code != 200:
                print(f"[Poll] Warning: Poll request returned status code {res.status_code}", file=sys.stderr)
                return
                
            job = res.json()
            if job:
                print(f"[Poll] Received Job {job['id']}: model={job['base_model']}, dataset={job['dataset_url']}")
                self.execute_job(job)
        except Exception as e:
            print(f"[Poll] Error contacting control plane: {e}", file=sys.stderr)

    def execute_job(self, job):
        self.status = "busy"
        job_id = job["id"]
        hyperparams = job.get("hyperparameters", {})
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trainer_path = os.path.join(current_dir, "trainer.py")
        
        cmd = [
            sys.executable,
            trainer_path,
            "--job_id", str(job_id),
            "--base_model", job["base_model"],
            "--dataset_url", job["dataset_url"],
            "--lora_r", str(hyperparams.get("lora_r", 8)),
            "--lora_alpha", str(hyperparams.get("lora_alpha", 16)),
            "--learning_rate", str(hyperparams.get("learning_rate", 2e-4)),
            "--epochs", str(hyperparams.get("epochs", 3.0)),
            "--batch_size", str(hyperparams.get("batch_size", 4)),
        ]
        
        try:
            import torch
            if torch.cuda.is_available():
                cmd.extend(["--device", "cuda"])
            else:
                cmd.extend(["--device", "cpu"])
        except Exception:
            cmd.extend(["--device", "cpu"])
            
        print(f"[Job {job_id}] Spawning trainer process: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stripped_line = line.strip()
                    try:
                        metric_data = json.loads(stripped_line)
                        if "step" in metric_data and "loss" in metric_data:
                            metric_data["log_text"] = stripped_line
                            self.report_metric(job_id, metric_data)
                    except json.JSONDecodeError:
                        self.report_metric(job_id, {"log_text": stripped_line})
                        print(f"[Trainer Log] {stripped_line}")
            
            _, stderr_output = process.communicate()
            if stderr_output:
                print(f"[Trainer Stderr] {stderr_output.strip()}", file=sys.stderr)
                
            exit_code = process.returncode
            if exit_code == 0:
                print(f"[Job {job_id}] Training completed successfully.")
                self.update_job_status(job_id, "COMPLETED", f"./weights/job_{job_id}")
            else:
                print(f"[Job {job_id}] Training failed with exit code {exit_code}.", file=sys.stderr)
                self.update_job_status(job_id, "FAILED")
                
        except Exception as e:
            print(f"[Job {job_id}] Error running trainer: {e}", file=sys.stderr)
            self.update_job_status(job_id, "FAILED")
        finally:
            self.status = "idle"

    def report_metric(self, job_id, metric):
        try:
            url = f"{self.control_plane_url}/api/jobs/{job_id}/metrics"
            res = requests.post(url, json=metric, timeout=5)
            if res.status_code != 201:
                print(f"[Job {job_id}] Failed to report metric: HTTP {res.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"[Job {job_id}] Connection error reporting metric: {e}", file=sys.stderr)

    def update_job_status(self, job_id, status, weights_path=None):
        try:
            url = f"{self.control_plane_url}/api/jobs/{job_id}"
            payload = {"status": status}
            if weights_path:
                payload["weights_path"] = weights_path
            res = requests.patch(url, json=payload, timeout=5)
            if res.status_code != 200:
                print(f"[Job {job_id}] Failed to update status to {status}: HTTP {res.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"[Job {job_id}] Connection error updating status: {e}", file=sys.stderr)

    def start(self):
        print(f"Starting Worker Daemon '{self.worker_id}'...")
        print(f"Control Plane URL: {self.control_plane_url}")
        print(f"Detected GPU VRAM: {self.gpu_vram} GB")
        
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        try:
            while True:
                self.poll_and_execute()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("Shutting down worker daemon...")
        finally:
            self.stop_event.set()

def main():
    daemon = WorkerDaemon()
    daemon.start()

if __name__ == "__main__":
    main()
