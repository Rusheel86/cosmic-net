import React, { useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

export default function GraphPanel({ csvData, explainResult, onExplain, explaining }) {
  const [tooltip, setTooltip] = useState(null);

  if (!csvData) return (
    <div className="panel graph-panel empty">
      <p>Upload a CSV to visualize the galaxy cluster graph.</p>
    </div>
  );

  // Build Cytoscape elements from CSV rows
  const rows = csvData.trim().split("\n");
  const headers = rows[0].split(",");
  const dataRows = rows.slice(1).map(r => {
    const vals = r.split(",");
    return Object.fromEntries(headers.map((h, i) => [h.trim(), parseFloat(vals[i])]));
  });

  // Nodes — sized by stellar_mass
  const minMass = Math.min(...dataRows.map(r => r.stellar_mass));
  const maxMass = Math.max(...dataRows.map(r => r.stellar_mass));
  const nodeSize = (m) => 20 + ((m - minMass) / (maxMass - minMass + 1e-8)) * 30;

  const nodes = dataRows.map((row, i) => ({
    data: {
      id: `n${i}`,
      label: `#${i}`,
      stellar_mass: row.stellar_mass?.toFixed(2),
      vel_dispersion: row.vel_dispersion?.toFixed(1),
      metallicity: row.metallicity?.toFixed(4),
      size: nodeSize(row.stellar_mass),
    },
    position: {
      x: (row.pos_x - Math.min(...dataRows.map(r => r.pos_x))) * 80,
      y: (row.pos_y - Math.min(...dataRows.map(r => r.pos_y))) * 80,
    },
  }));

  // Edges — from explainer if available, else k=3 nearest by position
  let edges = [];
  if (explainResult?.edge_importances) {
    edges = explainResult.edge_importances.map((e, i) => ({
      data: {
        id: `e${i}`,
        source: `n${e.source}`,
        target: `n${e.target}`,
        importance: e.importance,
        width: 1 + e.importance * 6,
        color: e.importance > 0.6 ? "#ff4757" : e.importance > 0.3 ? "#ffb800" : "#4a9eff",
      },
    }));
  } else {
    // Simple k-NN edges for display before explain runs
    dataRows.forEach((a, i) => {
      const dists = dataRows.map((b, j) => ({
        j,
        d: Math.sqrt((a.pos_x - b.pos_x) ** 2 + (a.pos_y - b.pos_y) ** 2),
      })).filter(x => x.j !== i).sort((a, b) => a.d - b.d).slice(0, 3);

      dists.forEach(({ j }, idx) => {
        edges.push({
          data: {
            id: `e${i}_${j}`,
            source: `n${i}`,
            target: `n${j}`,
            width: 1.5,
            color: "#4a9eff",
          },
        });
      });
    });
  }

  // Highlight anchor nodes
  const anchorSet = new Set(explainResult?.top_anchor_indices?.map(i => `n${i}`) || []);

  const elements = [...nodes, ...edges];

  const stylesheet = [
    {
      selector: "node",
      style: {
        width: "data(size)",
        height: "data(size)",
        backgroundColor: "#4a9eff",
        borderWidth: 2,
        borderColor: "#1a1a2e",
        label: "data(label)",
        color: "#fff",
        fontSize: 9,
        textValign: "center",
      },
    },
    {
      selector: anchorSet.size > 0
        ? [...anchorSet].map(id => `node[id="${id}"]`).join(", ")
        : ".never-matches",
      style: {
        backgroundColor: "#ff4757",
        borderColor: "#fff",
        borderWidth: 3,
      },
    },
    {
      selector: "edge",
      style: {
        width: "data(width)",
        lineColor: "data(color)",
        targetArrowShape: "none",
        opacity: 0.7,
        curveStyle: "bezier",
      },
    },
  ];

  return (
    <div className="panel graph-panel">
      <div className="graph-header">
        <h2>Galaxy Cluster Graph</h2>
        <button
          className="why-btn"
          onClick={onExplain}
          disabled={explaining}
        >
          {explaining ? "Running PGExplainer..." : "Why? (PGExplainer)"}
        </button>
      </div>

      {explainResult && (
        <div className="explain-legend">
          <span style={{ color: "#ff4757" }}>● High importance</span>
          <span style={{ color: "#ffb800" }}>● Medium</span>
          <span style={{ color: "#4a9eff" }}>● Low</span>
          <span style={{ color: "#ff4757", marginLeft: 12 }}>★ Gravitational anchors</span>
        </div>
      )}

      <CytoscapeComponent
        elements={elements}
        stylesheet={stylesheet}
        style={{ width: "100%", height: "400px", background: "#0d0d1a" }}
        cy={(cy) => {
          cy.on("mouseover", "node", (e) => {
            const d = e.target.data();
            setTooltip({
              x: e.renderedPosition.x,
              y: e.renderedPosition.y,
              content: `M★: ${d.stellar_mass} | σ: ${d.vel_dispersion} | Z: ${d.metallicity}`,
            });
          });
          cy.on("mouseout", "node", () => setTooltip(null));
        }}
      />

      {tooltip && (
        <div
          className="node-tooltip"
          style={{ left: tooltip.x + 10, top: tooltip.y - 30 }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
