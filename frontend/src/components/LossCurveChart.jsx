import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { TrendingDown } from "lucide-react";

const TEAL = "#00b4d8";
const GRID_COLOR = "#21262d";
const AXIS_COLOR = "#8b949e";

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="card px-3 py-2 text-xs space-y-0.5 min-w-[120px] shadow-xl glow-teal">
      <div className="flex justify-between gap-4">
        <span className="text-gray-500">Step</span>
        <span className="font-mono text-white">{d.step}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-gray-500">Loss</span>
        <span className="font-mono text-teal-400">{d.loss?.toFixed(4)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-gray-500">Epoch</span>
        <span className="font-mono text-gray-300">{d.epoch}</span>
      </div>
    </div>
  );
}

export default function LossCurveChart({ metrics }) {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2 text-gray-600">
        <TrendingDown size={28} className="opacity-30" />
        <p className="text-xs">Waiting for training metrics…</p>
      </div>
    );
  }

  const minLoss = Math.min(...metrics.map((m) => m.loss));
  const minStep = metrics.find((m) => m.loss === minLoss)?.step;

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-teal-400" />
          <span className="text-xs text-gray-400">Training Loss</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-gray-500 font-mono">
          <span>steps: <span className="text-white">{metrics.length}</span></span>
          <span>min loss: <span className="text-teal-400">{minLoss.toFixed(4)}</span></span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={metrics} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
          <XAxis
            dataKey="step"
            tick={{ fill: AXIS_COLOR, fontSize: 10, fontFamily: "JetBrains Mono" }}
            label={{ value: "Step", position: "insideBottom", offset: -2, fill: AXIS_COLOR, fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fill: AXIS_COLOR, fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickLine={false}
            axisLine={false}
            width={50}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: TEAL, strokeWidth: 1, strokeDasharray: "3 3" }} />
          {minStep && (
            <ReferenceLine
              x={minStep}
              stroke="#2ea043"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{ value: "best", fill: "#2ea043", fontSize: 9, fontFamily: "JetBrains Mono" }}
            />
          )}
          <Line
            type="monotone"
            dataKey="loss"
            stroke={TEAL}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: TEAL, stroke: "#0d1117", strokeWidth: 2 }}
            animationDuration={300}
            isAnimationActive={metrics.length < 200} // disable heavy animation for large runs
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
