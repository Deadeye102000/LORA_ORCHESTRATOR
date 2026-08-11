import { Cpu, Zap, WifiOff } from "lucide-react";

const STATUS_CONFIG = {
  idle:    { label: "Idle",    color: "bg-status-idle",    glow: "glow-idle",    icon: Cpu,   dot: "bg-status-idle"    },
  busy:    { label: "Busy",    color: "bg-status-busy",    glow: "glow-busy",    icon: Zap,   dot: "bg-status-busy"    },
  offline: { label: "Offline", color: "bg-status-offline", glow: "",             icon: WifiOff, dot: "bg-status-offline" },
};

function WorkerCard({ worker }) {
  const cfg = STATUS_CONFIG[worker.status] ?? STATUS_CONFIG.offline;
  const Icon = cfg.icon;
  const isBusy = worker.status === "busy";

  return (
    <div
      className={`card relative overflow-hidden transition-all duration-300 ${cfg.glow} hover:translate-y-[-2px]`}
    >
      {/* Animated top border stripe */}
      <div
        className={`absolute top-0 left-0 right-0 h-0.5 ${cfg.dot} ${isBusy ? "animate-pulse_slow" : ""}`}
      />

      <div className="flex items-start justify-between mb-3 pt-1">
        <div className="flex items-center gap-2">
          <div className={`status-dot ${cfg.dot} ${isBusy ? "animate-pulse_slow" : ""}`} />
          <span className="text-xs font-mono text-gray-400 truncate max-w-[120px]">
            {worker.id}
          </span>
        </div>
        <Icon size={14} className={`flex-shrink-0 ${isBusy ? "text-status-busy" : cfg.dot === "bg-status-idle" ? "text-status-idle" : "text-gray-500"}`} />
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Status</span>
          <span className={`badge ${cfg.color} bg-opacity-20 text-xs`} style={{ color: getStatusColor(worker.status) }}>
            {cfg.label}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">VRAM</span>
          <span className="text-xs font-mono text-gray-300">{worker.gpu_vram_gb} GB</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Last seen</span>
          <span className="text-xs text-gray-500">
            {formatRelative(worker.last_seen)}
          </span>
        </div>
      </div>
    </div>
  );
}

function getStatusColor(status) {
  const map = { idle: "#2ea043", busy: "#f0a500", offline: "#f85149" };
  return map[status] ?? "#8b949e";
}

function formatRelative(iso) {
  if (!iso) return "—";
  const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 10) return "just now";
  if (secs < 60) return `${secs}s ago`;
  return `${Math.floor(secs / 60)}m ago`;
}

export default function WorkerGrid({ workers }) {
  if (workers.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500 text-sm">
        <WifiOff size={32} className="mx-auto mb-2 opacity-40" />
        No workers registered yet. Start a worker daemon.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
      {workers.map((w) => (
        <WorkerCard key={w.id} worker={w} />
      ))}
    </div>
  );
}
