import React from "react";

export default function PredictionPanel({ result, virial, loading }) {
  if (loading) return (
    <div className="panel">
      <div className="loading-spinner">Running 30 MC-Dropout passes...</div>
    </div>
  );

  if (!result) return (
    <div className="panel empty">
      <p>Upload a cluster CSV and click <strong>Predict</strong> to begin.</p>
    </div>
  );

  const statusColor = { green: "#00c48c", amber: "#ffb800", red: "#ff4757" };

  return (
    <div className="panel prediction-panel">
      <h2>Prediction Results</h2>

      {/* GNN Prediction */}
      <div className="result-block gnn-block">
        <div className="result-label">GNN Predicted log₁₀(M_halo / M☉)</div>
        <div className="result-value">
          {result.gnn_log_mass_mean.toFixed(3)}
          <span className="result-unit"> dex</span>
        </div>
        <div className="confidence-bar-wrap">
          <div className="confidence-label">
            95% CI: [{result.gnn_log_mass_lower.toFixed(3)}, {result.gnn_log_mass_upper.toFixed(3)}]
          </div>
          <div className="confidence-label">
            σ = {result.gnn_log_mass_std.toFixed(4)} dex
          </div>
        </div>
      </div>

      {/* Symbolic Equation Prediction */}
      <div className="result-block sym-block">
        <div className="result-label">Symbolic Equation Prediction</div>
        <div className="result-value">
          {result.sym_log_mass.toFixed(3)}
          <span className="result-unit"> dex</span>
        </div>
        <div className="confidence-label">R² = {result.sym_r2.toFixed(3)}</div>
        <div className="confidence-label agreement">
          Δ = {Math.abs(result.gnn_log_mass_mean - result.sym_log_mass).toFixed(3)} dex from GNN
        </div>
      </div>

      {/* Virial Check */}
      {virial && (
        <div className="result-block virial-block">
          <div className="result-label">Virial Physics Check</div>
          <div className="virial-status" style={{ color: statusColor[virial.status] }}>
            ● {virial.status.toUpperCase()} — Q = {virial.virial_ratio.toFixed(3)}
          </div>
          <div className="virial-message">{virial.message}</div>
        </div>
      )}

      {/* Graph metadata */}
      <div className="meta-row">
        <span>{result.n_subhalos} subhalos</span>
        <span>{result.n_edges} edges</span>
      </div>
    </div>
  );
}
