import React, { useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import Plot from "react-plotly.js";
import katex from "katex";
import "katex/dist/katex.min.css";
import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";
const api = axios.create({ baseURL: BASE });

const S = {
  app: { background: "#080818", minHeight: "100vh", color: "#c8d6e5", fontFamily: "'Courier New', monospace" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 28px", background: "#0d0d1a", borderBottom: "1px solid #1e2a4a" },
  h1: { fontSize: "22px", color: "#4a9eff", letterSpacing: "2px", margin: 0 },
  headerSub: { fontSize: "11px", color: "#5a7a9f", display: "block", marginTop: "3px" },
  headerRight: { display: "flex", gap: "8px" },
  badge: { fontSize: "10px", background: "#1a2a4a", border: "1px solid #2a4a7a", color: "#4a9eff", padding: "4px 10px", borderRadius: "12px" },
  uploadBar: { display: "flex", alignItems: "center", gap: "14px", padding: "14px 28px", background: "#0d0d1a", borderBottom: "1px solid #1e2a4a", flexWrap: "wrap" },
  fileLabel: { cursor: "pointer", padding: "8px 18px", border: "1px dashed #2a4a7a", borderRadius: "6px", fontSize: "13px", color: "#4a9eff" },
  predictBtn: { padding: "8px 24px", background: "#4a9eff", color: "#080818", border: "none", borderRadius: "6px", fontFamily: "inherit", fontSize: "13px", fontWeight: "bold", cursor: "pointer" },
  predictBtnDisabled: { padding: "8px 24px", background: "#1a2a4a", color: "#3a5a7f", border: "none", borderRadius: "6px", fontFamily: "inherit", fontSize: "13px", cursor: "not-allowed" },
  errorMsg: { color: "#ff4757", fontSize: "12px" },
  grid: { display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: "16px", padding: "16px 28px" },
  leftCol: { display: "flex", flexDirection: "column", gap: "16px" },
  rightCol: { display: "flex", flexDirection: "column", gap: "16px" },
  panel: { background: "#0d0d1a", border: "1px solid #1e2a4a", borderRadius: "10px", padding: "20px" },
  panelH2: { fontSize: "13px", color: "#4a9eff", letterSpacing: "1px", marginBottom: "14px", textTransform: "uppercase" },
  emptyMsg: { color: "#3a5a7f", fontSize: "13px", textAlign: "center", padding: "30px 0" },
  graphHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" },
  whyBtn: { padding: "6px 16px", background: "transparent", border: "1px solid #ff4757", color: "#ff4757", borderRadius: "6px", fontFamily: "inherit", fontSize: "11px", cursor: "pointer" },
  resultBlock: { border: "1px solid #1e2a4a", borderRadius: "8px", padding: "14px", marginBottom: "12px" },
  resultLabel: { fontSize: "11px", color: "#5a7a9f", marginBottom: "6px" },
  resultValueGNN: { fontSize: "28px", fontWeight: "bold", color: "#4a9eff", letterSpacing: "1px" },
  resultValueSym: { fontSize: "28px", fontWeight: "bold", color: "#00c48c", letterSpacing: "1px" },
  resultUnit: { fontSize: "14px", color: "#5a7a9f" },
  confLabel: { fontSize: "11px", color: "#5a7a9f", marginTop: "4px" },
  metaRow: { display: "flex", gap: "20px", fontSize: "11px", color: "#3a5a7f", borderTop: "1px solid #1e2a4a", paddingTop: "10px" },
  copyBtn: { padding: "7px 18px", background: "transparent", border: "1px solid #2a4a7a", color: "#4a9eff", borderRadius: "6px", fontFamily: "inherit", fontSize: "12px", cursor: "pointer" },
  overviewGrid: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", marginBottom: "16px" },
  statsTable: { width: "100%", borderCollapse: "collapse", fontSize: "11px" },
  th: { padding: "6px 10px", borderBottom: "1px solid #1e2a4a", textAlign: "left", color: "#4a9eff" },
  td: { padding: "6px 10px", borderBottom: "1px solid #1e2a4a", textAlign: "left", color: "#8a9ab5" },
  katexBox: { background: "#080818", border: "1px solid #1e2a4a", borderRadius: "6px", padding: "16px", marginBottom: "12px", overflowX: "auto" },
  legend: { display: "flex", gap: "16px", fontSize: "11px", marginBottom: "10px" },
};

const virialColor = { green: "#00c48c", amber: "#ffb800", red: "#ff4757" };

export default function App() {
  const [file, setFile] = useState(null);
  const [csvText, setCsvText] = useState(null);
  const [overview, setOverview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [virial, setVirial] = useState(null);
  const [explainResult, setExplain] = useState(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f); setError(null); setPrediction(null);
    setVirial(null); setExplain(null);
    const text = await f.text();
    setCsvText(text);
    try {
      const form = new FormData(); form.append("file", f);
      const res = await api.post("/overview", form);
      setOverview(res.data);
    } catch { setOverview(null); }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form1 = new FormData(); form1.append("file", file);
      const form2 = new FormData(); form2.append("file", file);
      const [p, v] = await Promise.all([
        api.post("/predict", form1),
        api.post("/virial", form2),
      ]);
      setPrediction(p.data); setVirial(v.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed.");
    } finally { setLoading(false); }
  };

  const handleExplain = async () => {
    if (!file) return;
    setExplaining(true);
    try {
      const form = new FormData(); form.append("file", file);
      const res = await api.post("/explain", form);
      setExplain(res.data);
    } catch { setError("PGExplainer failed."); }
    finally { setExplaining(false); }
  };

  // Build Cytoscape elements from CSV
  const buildGraphElements = () => {
    if (!csvText) return [];
    const rows = csvText.trim().split("\n");
    const headers = rows[0].split(",").map(h => h.trim());
    const data = rows.slice(1).map(r => {
      const vals = r.split(",");
      return Object.fromEntries(headers.map((h, i) => [h, parseFloat(vals[i])]));
    });
    const masses = data.map(r => r.stellar_mass);
    const minM = Math.min(...masses), maxM = Math.max(...masses);
    const nodeSize = m => 20 + ((m - minM) / (maxM - minM + 1e-8)) * 30;

    const nodes = data.map((row, i) => ({
      data: { id: `n${i}`, label: `#${i}`, size: nodeSize(row.stellar_mass),
              stellar_mass: row.stellar_mass?.toFixed(2),
              vel_dispersion: row.vel_dispersion?.toFixed(1),
              metallicity: row.metallicity?.toFixed(4) },
      position: {
        x: (row.pos_x - Math.min(...data.map(r => r.pos_x))) * 80 + 40,
        y: (row.pos_y - Math.min(...data.map(r => r.pos_y))) * 80 + 40,
      },
    }));

    let edges = [];
    if (explainResult?.edge_importances) {
      edges = explainResult.edge_importances.map((e, i) => ({
        data: { id: `e${i}`, source: `n${e.source}`, target: `n${e.target}`,
                width: 1 + e.importance * 6,
                color: e.importance > 0.6 ? "#ff4757" : e.importance > 0.3 ? "#ffb800" : "#4a9eff" }
      }));
    } else {
      data.forEach((a, i) => {
        const dists = data.map((b, j) => ({ j, d: Math.hypot(a.pos_x - b.pos_x, a.pos_y - b.pos_y) }))
          .filter(x => x.j !== i).sort((a, b) => a.d - b.d).slice(0, 3);
        dists.forEach(({ j }) => edges.push({
          data: { id: `e${i}_${j}`, source: `n${i}`, target: `n${j}`, width: 1.5, color: "#4a9eff" }
        }));
      });
    }
    return [...nodes, ...edges];
  };

  const anchorSet = new Set(explainResult?.top_anchor_indices?.map(i => `n${i}`) || []);
  const cyStylesheet = [
    { selector: "node", style: { width: "data(size)", height: "data(size)", backgroundColor: "#4a9eff", borderWidth: 2, borderColor: "#1a1a2e", label: "data(label)", color: "#fff", fontSize: 9, textValign: "center" } },
    { selector: anchorSet.size > 0 ? [...anchorSet].map(id => `node[id="${id}"]`).join(",") : ".x", style: { backgroundColor: "#ff4757", borderColor: "#fff", borderWidth: 3 } },
    { selector: "edge", style: { width: "data(width)", lineColor: "data(color)", opacity: 0.7, curveStyle: "bezier" } },
  ];

  // KaTeX render
  const renderLatex = (latex) => {
    try { return katex.renderToString(latex, { throwOnError: false, displayMode: true }); }
    catch { return latex; }
  };

  const plotLayout = (title, xTitle, yTitle) => ({
    title: { text: title, font: { color: "#4a9eff", size: 12 } },
    xaxis: { title: xTitle, color: "#5a7a9f", gridcolor: "#1e2a4a" },
    yaxis: { title: yTitle, color: "#5a7a9f", gridcolor: "#1e2a4a" },
    paper_bgcolor: "#080818", plot_bgcolor: "#080818",
    font: { color: "#c8d6e5", size: 10 },
    margin: { t: 35, b: 45, l: 55, r: 15 },
  });

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <div>
          <h1 style={S.h1}>✦ Cosmic-Net</h1>
          <span style={S.headerSub}>Neural Surrogate for Dark Matter Halo Mass Prediction</span>
        </div>
        <div style={S.headerRight}>
          <span style={S.badge}>GNN + PGExplainer + Symbolic Regression</span>
          <span style={S.badge}>IllustrisTNG-100 · R² = 0.924</span>
        </div>
      </header>

      {/* Upload bar */}
      <div style={S.uploadBar}>
        <label style={S.fileLabel}>
          <input type="file" accept=".csv" onChange={handleFileChange} style={{ display: "none" }} />
          {file ? `📁 ${file.name}` : "📂 Upload Galaxy Cluster CSV"}
        </label>
        <button
          style={!file || loading ? S.predictBtnDisabled : S.predictBtn}
          onClick={handlePredict}
          disabled={!file || loading}
        >
          {loading ? "⟳ Predicting..." : "⚡ Predict Halo Mass"}
        </button>
        {error && <span style={S.errorMsg}>⚠ {error}</span>}
      </div>

      {/* Main grid */}
      <div style={S.grid}>

        {/* LEFT COLUMN */}
        <div style={S.leftCol}>

          {/* Graph Panel */}
          <div style={S.panel}>
            <div style={S.graphHeader}>
              <h2 style={{ ...S.panelH2, margin: 0 }}>Galaxy Cluster Graph</h2>
              <button
                style={S.whyBtn}
                onClick={handleExplain}
                disabled={!file || explaining}
              >
                {explaining ? "Running..." : "❓ Why? (PGExplainer)"}
              </button>
            </div>

            {explainResult && (
              <div style={S.legend}>
                <span style={{ color: "#ff4757" }}>● High importance</span>
                <span style={{ color: "#ffb800" }}>● Medium</span>
                <span style={{ color: "#4a9eff" }}>● Low</span>
                <span style={{ color: "#ff4757", marginLeft: 8 }}>★ Anchors</span>
              </div>
            )}

            {csvText ? (
              <CytoscapeComponent
                elements={buildGraphElements()}
                stylesheet={cyStylesheet}
                style={{ width: "100%", height: "380px", background: "#0d0d1a", borderRadius: "6px" }}
              />
            ) : (
              <p style={S.emptyMsg}>Upload a CSV to visualize the galaxy cluster graph.</p>
            )}
          </div>

          {/* Data Overview */}
          {overview && (
            <div style={S.panel}>
              <h2 style={S.panelH2}>Data Overview — {overview.n_subhalos} Subhalos</h2>
              <div style={S.overviewGrid}>
                <Plot
                  data={[{ x: overview.stellar_mass_log, type: "histogram", marker: { color: "#4a9eff" } }]}
                  layout={plotLayout("Stellar Mass", "log₁₀(M★)", "Count")}
                  style={{ width: "100%", height: "200px" }}
                  config={{ displayModeBar: false }}
                />
                <Plot
                  data={[{ x: overview.stellar_mass_log, y: overview.velocity_dispersion, mode: "markers", type: "scatter", marker: { color: "#00c48c", size: 5 } }]}
                  layout={plotLayout("σ vs M★", "log₁₀(M★)", "σ [km/s]")}
                  style={{ width: "100%", height: "200px" }}
                  config={{ displayModeBar: false }}
                />
                <Plot
                  data={[{ z: overview.correlation_matrix, x: overview.correlation_labels, y: overview.correlation_labels, type: "heatmap", colorscale: "RdBu", zmin: -1, zmax: 1 }]}
                  layout={plotLayout("Correlation", "", "")}
                  style={{ width: "100%", height: "200px" }}
                  config={{ displayModeBar: false }}
                />
              </div>
              <table style={S.statsTable}>
                <thead><tr><th style={S.th}>Feature</th><th style={S.th}>Mean</th><th style={S.th}>Std</th><th style={S.th}>Min</th><th style={S.th}>Max</th></tr></thead>
                <tbody>
                  {Object.entries(overview.feature_stats).map(([k, v]) => (
                    <tr key={k}>
                      <td style={S.td}>{k}</td>
                      <td style={S.td}>{v.mean.toFixed(3)}</td>
                      <td style={S.td}>{v.std.toFixed(3)}</td>
                      <td style={S.td}>{v.min.toFixed(3)}</td>
                      <td style={S.td}>{v.max.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN */}
        <div style={S.rightCol}>

          {/* Prediction Panel */}
          <div style={S.panel}>
            <h2 style={S.panelH2}>Prediction Results</h2>
            {loading && <p style={{ color: "#4a9eff", fontSize: "13px" }}>⟳ Running 30 MC-Dropout passes...</p>}
            {!prediction && !loading && <p style={S.emptyMsg}>Upload a CSV and click Predict to begin.</p>}
            {prediction && (
              <>
                {/* GNN */}
                <div style={{ ...S.resultBlock, borderColor: "#2a4a7a" }}>
                  <div style={S.resultLabel}>GNN Predicted log₁₀(M_halo / M☉)</div>
                  <div style={S.resultValueGNN}>
                    {prediction.gnn_log_mass_mean.toFixed(3)}
                    <span style={S.resultUnit}> dex</span>
                  </div>
                  <div style={S.confLabel}>95% CI: [{prediction.gnn_log_mass_lower.toFixed(3)}, {prediction.gnn_log_mass_upper.toFixed(3)}]</div>
                  <div style={S.confLabel}>σ = {prediction.gnn_log_mass_std.toFixed(4)} dex</div>
                </div>

                {/* Symbolic */}
                <div style={{ ...S.resultBlock, borderColor: "#1a4a2a" }}>
                  <div style={S.resultLabel}>Symbolic Equation Prediction</div>
                  <div style={S.resultValueSym}>
                    {prediction.sym_log_mass.toFixed(3)}
                    <span style={S.resultUnit}> dex</span>
                  </div>
                  <div style={S.confLabel}>R² = {prediction.sym_r2.toFixed(3)}</div>
                  <div style={{ ...S.confLabel, color: "#ffb800" }}>
                    Δ = {Math.abs(prediction.gnn_log_mass_mean - prediction.sym_log_mass).toFixed(3)} dex from GNN
                  </div>
                </div>

                {/* Virial */}
                {virial && (
                  <div style={{ ...S.resultBlock, borderColor: "#3a3a1a" }}>
                    <div style={S.resultLabel}>Virial Physics Check</div>
                    <div style={{ fontSize: "16px", fontWeight: "bold", color: virialColor[virial.status], margin: "6px 0" }}>
                      ● {virial.status.toUpperCase()} — Q = {virial.virial_ratio.toFixed(3)}
                    </div>
                    <div style={{ fontSize: "11px", color: "#5a7a9f", lineHeight: 1.5 }}>{virial.message}</div>
                  </div>
                )}

                <div style={S.metaRow}>
                  <span>{prediction.n_subhalos} subhalos</span>
                  <span>{prediction.n_edges} edges</span>
                </div>
              </>
            )}
          </div>

          {/* Equation Panel */}
          <div style={S.panel}>
            <h2 style={S.panelH2}>Symbolic Equation</h2>
            <p style={{ fontSize: "11px", color: "#5a7a9f", marginBottom: "16px" }}>
              Discovered by PySR — distilled from GNN learned behaviour
            </p>
            {prediction ? (
              <>
                <div
                  style={S.katexBox}
                  dangerouslySetInnerHTML={{ __html: renderLatex(prediction.sym_equation_latex) }}
                />
                <div style={{ fontSize: "11px", color: "#5a7a9f", marginBottom: "12px" }}>
                  R² = {prediction.sym_r2.toFixed(3)} on validation set
                </div>
                <button style={S.copyBtn} onClick={() => {
                  navigator.clipboard.writeText(prediction.sym_equation_latex);
                  alert("LaTeX copied!");
                }}>
                  Copy as LaTeX
                </button>
              </>
            ) : (
              <p style={{ fontSize: "13px", color: "#3a5a7f" }}>Run a prediction to see the equation.</p>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
