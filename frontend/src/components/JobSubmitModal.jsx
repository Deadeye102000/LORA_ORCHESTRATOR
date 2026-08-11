import { useState } from "react";
import { X, Send, ChevronDown, ChevronUp } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const DEFAULT_FORM = {
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  dataset_url: "timdettmers/openassistant-guanaco",
  lora_r: 8,
  lora_alpha: 16,
  learning_rate: 0.0002,
  epochs: 3,
  batch_size: 2,
  min_vram_gb: 0,
};

export default function JobSubmitModal({ onClose, onSubmitted }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const set = (field) => (e) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const payload = {
      base_model: form.base_model,
      dataset_url: form.dataset_url,
      min_vram_gb: parseFloat(form.min_vram_gb),
      hyperparameters: {
        lora_r: parseInt(form.lora_r),
        lora_alpha: parseInt(form.lora_alpha),
        learning_rate: parseFloat(form.learning_rate),
        epochs: parseFloat(form.epochs),
        batch_size: parseInt(form.batch_size),
      },
    };

    try {
      const res = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const job = await res.json();
      onSubmitted(job);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    /* Glass backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center glass-backdrop"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="card w-full max-w-lg mx-4 glow-teal animate-in fade-in slide-in-from-bottom-4 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-base font-semibold text-white">Submit Training Job</h2>
            <p className="text-xs text-gray-500 mt-0.5">Configure a new LoRA fine-tuning run</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-lg">
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Base model */}
          <div>
            <label className="label">Base Model</label>
            <input
              className="input font-mono text-xs"
              value={form.base_model}
              onChange={set("base_model")}
              placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0"
              required
            />
          </div>

          {/* Dataset */}
          <div>
            <label className="label">Dataset URL / HF Name</label>
            <input
              className="input font-mono text-xs"
              value={form.dataset_url}
              onChange={set("dataset_url")}
              placeholder="e.g. timdettmers/openassistant-guanaco"
              required
            />
          </div>

          {/* Min VRAM */}
          <div>
            <label className="label">Min VRAM Required (GB)</label>
            <input
              className="input"
              type="number"
              min={0}
              step={1}
              value={form.min_vram_gb}
              onChange={set("min_vram_gb")}
            />
            <p className="text-xs text-gray-600 mt-1">
              0 = any worker can pick this job
            </p>
          </div>

          {/* Advanced toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-teal-400 transition-colors"
          >
            {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            LoRA Hyperparameters
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 p-3 bg-navy-900 rounded-lg border border-navy-700">
              {[
                ["LoRA Rank (r)", "lora_r", "number"],
                ["LoRA Alpha",    "lora_alpha", "number"],
                ["Learning Rate", "learning_rate", "number"],
                ["Epochs",        "epochs", "number"],
                ["Batch Size",    "batch_size", "number"],
              ].map(([label, field, type]) => (
                <div key={field}>
                  <label className="label">{label}</label>
                  <input
                    className="input"
                    type={type}
                    step="any"
                    value={form[field]}
                    onChange={set(field)}
                  />
                </div>
              ))}
            </div>
          )}

          {error && (
            <div className="text-xs text-status-offline bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-1">
            <button type="button" onClick={onClose} className="btn-ghost">
              Cancel
            </button>
            <button type="submit" disabled={loading} className="btn-primary">
              <Send size={14} />
              {loading ? "Submitting…" : "Submit Job"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
