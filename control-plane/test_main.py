import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure the control-plane directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app
from database import Base, engine, SessionLocal, Worker, Job

class TestControlPlane(unittest.TestCase):
    def setUp(self):
        Base.metadata.create_all(bind=engine)
        self.client = TestClient(app)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.query(Job).delete()
        self.db.query(Worker).delete()
        self.db.commit()
        self.db.close()

    def test_worker_heartbeat_and_list(self):
        # 1. Register a new worker via heartbeat
        response = self.client.post(
            "/api/workers/heartbeat",
            json={"id": "worker-1", "status": "idle", "gpu_vram_gb": 16.0}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "worker-1")
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["gpu_vram_gb"], 16.0)

        # 2. Verify worker is in list
        list_response = self.client.get("/api/workers")
        self.assertEqual(list_response.status_code, 200)
        workers = list_response.json()
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]["id"], "worker-1")

        # 3. Update heartbeat (change status to busy)
        update_response = self.client.post(
            "/api/workers/heartbeat",
            json={"id": "worker-1", "status": "busy", "gpu_vram_gb": 16.0}
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["status"], "busy")

    def test_job_submission_and_fetch(self):
        # 1. Submit a job
        job_data = {
            "base_model": "meta-llama/Llama-3-8b",
            "dataset_url": "s3://my-bucket/dataset.json",
            "hyperparameters": {"lr": 2e-4, "batch_size": 4, "lora_r": 8}
        }
        response = self.client.post("/api/jobs", json=job_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["base_model"], "meta-llama/Llama-3-8b")
        self.assertEqual(data["status"], "PENDING")

        job_id = data["id"]

        # 2. Fetch the job details
        fetch_response = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(fetch_response.status_code, 200)
        fetch_data = fetch_response.json()
        self.assertEqual(fetch_data["id"], job_id)
        self.assertEqual(fetch_data["hyperparameters"]["lora_r"], 8)

        # 3. Fetch non-existent job
        not_found_response = self.client.get("/api/jobs/99999")
        self.assertEqual(not_found_response.status_code, 404)

    def test_job_polling_and_updates(self):
        # 1. Poll when no jobs exist
        poll_response = self.client.post("/api/jobs/poll", json={"worker_id": "worker-1"})
        self.assertEqual(poll_response.status_code, 204)

        # 2. Submit a job
        job_data = {
            "base_model": "gpt2",
            "dataset_url": "dummy",
            "hyperparameters": {"lr": 1e-4}
        }
        self.client.post("/api/jobs", json=job_data)

        # 3. Poll for the job
        poll_response2 = self.client.post("/api/jobs/poll", json={"worker_id": "worker-1"})
        self.assertEqual(poll_response2.status_code, 200)
        job = poll_response2.json()
        self.assertEqual(job["status"], "TRAINING")
        self.assertEqual(job["worker_id"], "worker-1")
        self.assertIsNotNone(job["started_at"])
        
        job_id = job["id"]

        # 4. Report metric
        metric_response = self.client.post(
            f"/api/jobs/{job_id}/metrics",
            json={"step": 1, "loss": 1.25, "epoch": 0.1}
        )
        self.assertEqual(metric_response.status_code, 201)

        # 5. Complete job
        complete_response = self.client.patch(
            f"/api/jobs/{job_id}",
            json={"status": "COMPLETED", "weights_path": "/path/to/weights"}
        )
        self.assertEqual(complete_response.status_code, 200)
        updated_job = complete_response.json()
        self.assertEqual(updated_job["status"], "COMPLETED")
        self.assertEqual(updated_job["weights_path"], "/path/to/weights")
        self.assertIsNotNone(updated_job["completed_at"])

if __name__ == "__main__":
    unittest.main()
