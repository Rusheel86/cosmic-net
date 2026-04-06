import React from "react";
import Plot from "react-plotly.js";

export default function DataOverview({ overview }) {
  if (!overview) return null;

  const { stellar_mass_log, velocity_dispersion, metallicity,
          correlation_matrix, correlation_labels, n_subhalos, feature_stats } = overview;

  return (
    <div className="panel overview-panel">
      <h2>Data Overview — {n_subhalos} Subhalos</h2>

      <div className="overview-grid">
        {/* Stellar Mass Histogram */}
        <Plot
          data={[{
            x: stellar_mass_log,
            type: "histogram",
            marker: { color: "#4a9eff" },
            name: "Stellar Mass",
          }]}
          layout={{
            title: "Stellar Mass Distribution",
            xaxis: { title: "log₁₀(M★/M☉)" },
            yaxis: { title: "Count" },
            paper_bgcolor: "#0d0d1a",
            plot_bgcolor: "#0d0d1a",
            font: { color: "#fff" },
            margin: { t: 40, b: 40, l: 50, r: 20 },
          }}
          style={{ width: "100%", height: "220px" }}
          config={{ displayModeBar: false }}
        />

        {/* Velocity Dispersion vs Stellar Mass scatter */}
        <Plot
          data={[{
            x: stellar_mass_log,
            y: velocity_dispersion,
            mode: "markers",
            type: "scatter",
            marker: { color: "#00c48c", size: 6, opacity: 0.8 },
            name: "σ vs M★",
          }]}
          layout={{
            title: "Velocity Dispersion vs Stellar Mass",
            xaxis: { title: "log₁₀(M★/M☉)" },
            yaxis: { title: "σ_vel [km/s]" },
            paper_bgcolor: "#0d0d1a",
            plot_bgcolor: "#0d0d1a",
            font: { color: "#fff" },
            margin: { t: 40, b: 40, l: 50, r: 20 },
          }}
          style={{ width: "100%", height: "220px" }}
          config={{ displayModeBar: false }}
        />

        {/* Correlation Heatmap */}
        <Plot
          data={[{
            z: correlation_matrix,
            x: correlation_labels,
            y: correlation_labels,
            type: "heatmap",
            colorscale: "RdBu",
            zmin: -1,
            zmax: 1,
          }]}
          layout={{
            title: "Feature Correlation",
            paper_bgcolor: "#0d0d1a",
            plot_bgcolor: "#0d0d1a",
            font: { color: "#fff" },
            margin: { t: 40, b: 60, l: 80, r: 20 },
          }}
          style={{ width: "100%", height: "220px" }}
          config={{ displayModeBar: false }}
        />
      </div>

      {/* Feature stats table */}
      <table className="stats-table">
        <thead>
          <tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        </thead>
        <tbody>
          {Object.entries(feature_stats).map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v.mean.toFixed(3)}</td>
              <td>{v.std.toFixed(3)}</td>
              <td>{v.min.toFixed(3)}</td>
              <td>{v.max.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
