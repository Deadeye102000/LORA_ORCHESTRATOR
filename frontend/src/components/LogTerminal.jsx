import { useEffect, useRef } from "react";
import { Terminal } from "lucide-react";

function colorize(line) {
  if (!line) return <span className="text-gray-600">—</span>;

  // Error lines
  if (/error|failed|exception/i.test(line))
    return <span className="text-red-400">{line}</span>;
  // Warning lines
  if (/warn/i.test(line))
    return <span className="text-yellow-400">{line}</span>;
  // Metric JSON lines (step/loss)
  if (line.startsWith("{") && line.includes("loss"))
    return <span className="text-teal-300">{line}</span>;
  // Info / upload lines
  if (/S3 upload|upload|completed|saved/i.test(line))
    return <span className="text-green-400">{line}</span>;
  // Heartbeat / system lines
  if (/heartbeat|monitor|starting|spawning/i.test(line))
    return <span className="text-gray-500">{line}</span>;

  return <span className="text-gray-300">{line}</span>;
}

export default function LogTerminal({ logs }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom whenever new logs arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="flex flex-col h-full">
      {/* Terminal header bar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-navy-900 rounded-t-lg border border-navy-600 border-b-0">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
        </div>
        <Terminal size={12} className="text-gray-500 ml-1" />
        <span className="text-xs text-gray-500 font-mono">worker output</span>
        <div className="ml-auto text-xs text-gray-600 font-mono">
          {logs.length} lines
        </div>
      </div>

      {/* Log content */}
      <div
        className="flex-1 bg-navy-900 border border-navy-600 rounded-b-lg
                   overflow-y-auto px-3 py-2 font-mono text-xs leading-relaxed"
      >
        {logs.length === 0 ? (
          <div className="flex items-center gap-2 text-gray-600 mt-2">
            <span className="text-teal-500">$</span>
            <span className="animate-pulse">Waiting for worker output…</span>
          </div>
        ) : (
          <>
            {logs.map((line, i) => (
              <div key={i} className="flex gap-2 hover:bg-navy-800/40 px-1 rounded">
                <span className="text-gray-700 select-none w-8 text-right flex-shrink-0">
                  {i + 1}
                </span>
                <pre className="whitespace-pre-wrap break-all leading-relaxed">
                  {colorize(line)}
                </pre>
              </div>
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>
    </div>
  );
}
