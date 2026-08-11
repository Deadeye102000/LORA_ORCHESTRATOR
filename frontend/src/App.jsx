import { useState, useCallback } from "react";
import { Plus, Wifi, WifiOff, Zap } from "lucide-react";
import { useSystemSocket } from "./hooks/useSocket";
import WorkerGrid from "./components/WorkerGrid";
import JobList from "./components/JobList";
import JobViewer from "./components/JobViewer";
import JobSubmitModal from "./components/JobSubmitModal";

export default function App() {
  const { workers, connected } = useSystemSocket();
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [showModal, setShowModal] = useState(false);

  // Called when a job is newly submitted via the modal
  const handleJobSubmitted = useCallback((job) => {
    setJobs((prev) => [job, ...prev]);
    setSelectedJob(job);
  }, []);

  // Called when JobViewer polls a status update
  const handleJobUpdate = useCallback((updated) => {
    setJobs((prev) =>
      prev.map((j) => (j.id === updated.id ? updated : j))
    );
    setSelectedJob((prev) => (prev?.id === updated.id ? updated : prev));
  }, []);

  const idleCount = workers.filter((w) => w.status === "idle").length;
  const busyCount = workers.filter((w) => w.status === "busy").length;
  const trainingCount = jobs.filter((j) => j.status === "TRAINING").length;

  return (
    <div className="flex flex-col h-screen max-h-screen overflow-hidden">
      {/* ── Top Nav ────────────────────────────────────────────── */}
      <header className="flex-shrink-0 border-b border-navy-700 px-6 py-3 flex items-center gap-4">
        {/* Brand */}
        <div className="flex items-center gap-2 mr-auto">
          <div className="w-7 h-7 rounded-lg bg-teal-400/10 border border-teal-400/30 flex items-center justify-center">
            <Zap size={14} className="text-teal-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white leading-none">
              LoRA Orchestrator
            </h1>
            <p className="text-[10px] text-gray-500 leading-none mt-0.5">
              Distributed Fine-Tuning Platform
            </p>
          </div>
        </div>

        {/* Stats pills */}
        <div className="hidden sm:flex items-center gap-2 text-xs font-mono">
          <span className="px-2.5 py-1 rounded-full bg-navy-800 border border-navy-600 text-gray-400">
            {idleCount} idle
          </span>
          <span className="px-2.5 py-1 rounded-full bg-navy-800 border border-navy-600 text-gray-400">
            {busyCount} busy
          </span>
          <span className="px-2.5 py-1 rounded-full bg-navy-800 border border-navy-600 text-gray-400">
            {trainingCount} training
          </span>
        </div>

        {/* Connection badge */}
        <div
          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${
            connected
              ? "text-status-idle border-green-500/30 bg-green-500/10"
              : "text-status-offline border-red-500/30 bg-red-500/10"
          }`}
        >
          {connected ? <Wifi size={11} /> : <WifiOff size={11} />}
          {connected ? "Live" : "Disconnected"}
        </div>

        {/* New Job button */}
        <button
          onClick={() => setShowModal(true)}
          className="btn-primary text-xs"
          id="submit-job-btn"
        >
          <Plus size={14} />
          New Job
        </button>
      </header>

      {/* ── Worker Grid ─────────────────────────────────────────── */}
      <section className="flex-shrink-0 border-b border-navy-700 px-6 py-4">
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
            Worker Nodes
          </h2>
          <span className="text-xs text-gray-600 font-mono">
            ({workers.length})
          </span>
        </div>
        <WorkerGrid workers={workers} />
      </section>

      {/* ── Main Content (Jobs + Viewer) ─────────────────────────── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Jobs sidebar */}
        <aside className="w-64 flex-shrink-0 border-r border-navy-700 flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-navy-700">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
              Jobs
            </h2>
            <span className="text-xs text-gray-600 font-mono">
              ({jobs.length})
            </span>
          </div>
          <div className="flex-1 overflow-y-auto px-2 py-2">
            <JobList
              jobs={jobs}
              selectedId={selectedJob?.id}
              onSelect={setSelectedJob}
            />
          </div>
        </aside>

        {/* Job viewer */}
        <main className="flex-1 overflow-y-auto px-6 py-5">
          <JobViewer
            job={selectedJob}
            onJobUpdate={handleJobUpdate}
          />
        </main>
      </div>

      {/* ── Modal ────────────────────────────────────────────────── */}
      {showModal && (
        <JobSubmitModal
          onClose={() => setShowModal(false)}
          onSubmitted={handleJobSubmitted}
        />
      )}
    </div>
  );
}
