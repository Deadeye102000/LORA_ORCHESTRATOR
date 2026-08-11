import { CheckCircle, XCircle, Loader, Clock, ChevronRight } from "lucide-react";

const STATUS_ICON = {
  PENDING:   { Icon: Clock,       color: "text-gray-400",       bg: "bg-gray-400/10"  },
  TRAINING:  { Icon: Loader,      color: "text-teal-400",       bg: "bg-teal-400/10"  },
  COMPLETED: { Icon: CheckCircle, color: "text-status-idle",    bg: "bg-green-500/10" },
  FAILED:    { Icon: XCircle,     color: "text-status-offline", bg: "bg-red-500/10"   },
};

function JobRow({ job, selected, onClick }) {
  const cfg = STATUS_ICON[job.status] ?? STATUS_ICON.PENDING;
  const { Icon, color, bg } = cfg;
  const isTraining = job.status === "TRAINING";

  return (
    <button
      onClick={onClick}
      className={`w-full text-left flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-150
        ${selected
          ? "bg-teal-400/10 border border-teal-400/40"
          : "hover:bg-navy-700 border border-transparent"
        }`}
    >
      {/* Status icon */}
      <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${bg}`}>
        <Icon
          size={14}
          className={`${color} ${isTraining ? "animate-spin" : ""}`}
        />
      </div>

      {/* Job info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-gray-400">#{job.id}</span>
          <span
            className="text-xs font-semibold truncate max-w-[120px]"
            style={{ color: getStatusColor(job.status) }}
          >
            {job.status}
          </span>
        </div>
        <p className="text-xs text-gray-500 truncate mt-0.5 font-mono">
          {job.base_model?.split("/").pop() ?? "—"}
        </p>
      </div>

      <ChevronRight
        size={14}
        className={`flex-shrink-0 transition-colors ${selected ? "text-teal-400" : "text-gray-600"}`}
      />
    </button>
  );
}

function getStatusColor(status) {
  const map = {
    PENDING:   "#8b949e",
    TRAINING:  "#00b4d8",
    COMPLETED: "#2ea043",
    FAILED:    "#f85149",
  };
  return map[status] ?? "#8b949e";
}

export default function JobList({ jobs, selectedId, onSelect }) {
  if (jobs.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500 text-sm">
        <p>No jobs submitted yet.</p>
        <p className="text-xs mt-1 text-gray-600">Use the &ldquo;+ New Job&rdquo; button above.</p>
      </div>
    );
  }

  // Show newest jobs first
  const sorted = [...jobs].sort((a, b) => b.id - a.id);

  return (
    <div className="space-y-1 overflow-y-auto max-h-full pr-0.5">
      {sorted.map((job) => (
        <JobRow
          key={job.id}
          job={job}
          selected={job.id === selectedId}
          onClick={() => onSelect(job)}
        />
      ))}
    </div>
  );
}
