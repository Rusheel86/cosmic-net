import React from "react";

export default function NodeDetail({ node, edgeData, onClose }) {
  if (!node) return null;

  const S = {
    panel: { background: "#0a0a1a", border: "1px solid #ff4757", borderRadius: "10px", padding: "16px", position: "relative" },
    title: { fontSize: "11px", color: "#ff4757", letterSpacing: "2px", textTransform: "uppercase", marginBottom: "12px" },
    close: { position: "absolute", top: "12px", right: "12px", background: "transparent", border: "none", color: "#ff4757", cursor: "pointer", fontSize: "16px" },
    row: { display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #0d1525", fontSize: "12px", fontFamily: "monospace" },
    key: { color: "#5a7a9f" },
    val: { color: "#c8d6e5", fontWeight: "bold" },
    edgeTitle: { fontSize: "10px", color: "#ffb800", letterSpacing: "1px", textTransform: "uppercase", margin: "12px 0 6px" },
    edgeRow: { display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "11px", fontFamily: "monospace" },
  };

  const fields = [
    { key: "Subhalo Index", val: `#${node.index}` },
    { key: "log(M★)", val: node.data?.stellar_mass?.toFixed(4) || "—" },
    { key: "σ_vel (km/s)", val: node.data?.vel_dispersion?.toFixed(2) || "—" },
    { key: "R½ (kpc)", val: node.data?.half_mass_radius?.toFixed(2) || "—" },
    { key: "Metallicity Z", val: node.data?.metallicity?.toFixed(5) || "—" },
    { key: "pos (x,y,z)", val: `${node.data?.pos_x?.toFixed(1)}, ${node.data?.pos_y?.toFixed(1)}, ${node.data?.pos_z?.toFixed(1)}` },
  ];

  const connectedEdges = edgeData?.filter(e => e.source === node.index || e.target === node.index)
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 5) || [];

  return (
    <div style={S.panel}>
      <button style={S.close} onClick={onClose}>✕</button>
      <div style={S.title}>Subhalo Details</div>

      {fields.map((f, i) => (
        <div key={i} style={S.row}>
          <span style={S.key}>{f.key}</span>
          <span style={S.val}>{f.val}</span>
        </div>
      ))}

      {connectedEdges.length > 0 && (
        <>
          <div style={S.edgeTitle}>Connected Edges (PGExplainer)</div>
          {connectedEdges.map((e, i) => (
            <div key={i} style={S.edgeRow}>
              <span style={{ color: "#5a7a9f" }}>#{e.source} → #{e.target}</span>
              <span style={{
                color: e.importance > 0.6 ? "#ff4757" : e.importance > 0.3 ? "#ffb800" : "#4a9eff",
                fontWeight: "bold"
              }}>
                {(e.importance * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
