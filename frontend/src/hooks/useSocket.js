import { useEffect, useRef, useState, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

/**
 * useSystemSocket
 * Connects to /ws/system and keeps the workers list in sync.
 * Returns { workers, connected }
 */
export function useSystemSocket() {
  const [workers, setWorkers] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_BASE}/ws/system`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.event === "init_workers") {
        setWorkers(msg.workers);
      } else if (msg.event === "worker_update") {
        setWorkers((prev) => {
          const idx = prev.findIndex((w) => w.id === msg.worker.id);
          if (idx === -1) return [...prev, msg.worker];
          const next = [...prev];
          next[idx] = msg.worker;
          return next;
        });
      } else if (msg.event === "worker_offline") {
        setWorkers((prev) =>
          prev.map((w) =>
            w.id === msg.worker_id ? { ...w, status: "offline" } : w
          )
        );
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { workers, connected };
}

/**
 * useJobSocket
 * Connects to /ws/jobs/{jobId} and streams metrics + logs.
 * Returns { metrics, logs }
 */
export function useJobSocket(jobId) {
  const [metrics, setMetrics] = useState([]);
  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    // Close any previous connection
    wsRef.current?.close();
    setMetrics([]);
    setLogs([]);

    const ws = new WebSocket(`${WS_BASE}/ws/jobs/${jobId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.event === "init_metrics") {
        setMetrics(msg.metrics || []);
      } else if (msg.event === "metric_update") {
        if (msg.step !== null && msg.loss !== null) {
          setMetrics((prev) => [
            ...prev,
            { step: msg.step, loss: msg.loss, epoch: msg.epoch },
          ]);
        }
        if (msg.log_text) {
          setLogs((prev) => [...prev, msg.log_text]);
        }
      }
    };

    ws.onerror = () => ws.close();

    return () => ws.close();
  }, [jobId]);

  return { metrics, logs };
}

/**
 * fetchJobs — one-time REST fetch to hydrate initial jobs list
 */
export async function fetchJobs() {
  // The API doesn't expose GET /api/jobs (list), so we can't hydrate.
  // Jobs are tracked via state after submission in this UI.
  return [];
}
