import { useEffect, useState, useRef } from "react";
import { RefreshCw, Package } from "lucide-react";
import LossCurveChart from "./LossCurveChart";
import LogTerminal from "./LogTerminal";
import { useJobSocket } from "../hooks/useSocket";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function JobMeta({ job }) {
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs font-mono mb-4">
      <span className="text-gray-500">
        model: <span className="text-gray-200">{job.base_model}</span>
      </span>
      <span className="text-gray-500">
        worker: <span className="text-gray-200">{job.worker_id ?? "—"}</span>
      </span>
      {job.weights_path && (
        <span className="text-gray-500">
          weights:{" "}
          <span className="text-teal-400 break-all">{job.weights_path}</span>
        </span>
      )}
    </div>
  );
}

export default function JobViewer({ job, onJobUpdate }) {
  const { metrics, logs } = useJobSocket(job?.id);
  const pollRef = useRef(null);

  // Poll job status until it reaches a terminal state
  useEffect(() => {
    if (!job || ["COMPLETED", "FAILED"].includes(job.status)) return;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${job.id}`);
        if (res.ok) {
          const updated = await res.json();
          if (updated.status !== job.status) onJobUpdate(updated);
        }
      } catch (_) { /* ignore */ }
    };

    pollRef.current = setInterval(poll, 3000);
    return () => clearInterval(pollRef.current);
  }, [job?.id, job?.status, onJobUpdate]);

  if (!job) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-600">
        <Package size={40} className="opacity-20" />
        <p className="text-sm">Select a job from the list to view details</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Job header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-white">
            Job #{job.id}
            <span className="ml-2 text-xs font-normal text-gray-500">
              {job.status}
            </span>
          </h3>
          <JobMeta job={job} />
        </div>
        {job.status === "TRAINING" && (
          <RefreshCw size={14} className="text-teal-400 animate-spin flex-shrink-0 mt-1" />
        )}
      </div>

      {/* Loss curve (upper pane) */}
      <div className="card flex-1 min-h-0" style={{ minHeight: "200px" }}>
        <LossCurveChart metrics={metrics} />
      </div>

      {/* Log terminal (lower pane) */}
      <div className="flex flex-col" style={{ height: "240px" }}>
        <LogTerminal logs={logs} />
      </div>
    </div>
  );
}
