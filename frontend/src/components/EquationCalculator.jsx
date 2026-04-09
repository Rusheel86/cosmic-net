import React from "react";

// Full physics linear equation coefficients
const EQUATION = {
  intercept: 12.968,
  terms: [
    { coef: 0.022,  feature: "log_stellar_mass",    label: "log(M★_central)",      key: "stellar_mass",      dominant: false },
    { coef: 0.288,  feature: "log_vel_dispersion",  label: "log(σ_central)",        key: "vel_dispersion",    dominant: true  },
    { coef: 0.079,  feature: "log_half_mass_r",     label: "log(R½_central)",       key: "half_mass_radius",  dominant: false },
    { coef: 0.024,  feature: "log_sigma_sat",       label: "log(σ_satellites)",     key: "vel_dispersion",    dominant: false },
    { coef: 0.001,  feature: "log_n_sat",           label: "log(N_satellites)",     key: "n_subhalos",        dominant: false },
    { coef: -0.001, feature: "log_m_sat_total",     label: "log(M★_sat_total)",     key: "stellar_mass",      dominant: false },
  ]
};

function computeEquation(df, nSubhalos) {
  if (!df || df.length === 0) return null;

  // Central subhalo = highest stellar mass
  const sorted = [...df].sort((a, b) => b.stellar_mass - a.stellar_mass);
  const central = sorted[0];
  const satellites = sorted.slice(1);

  const logMcen   = central.stellar_mass;  // already log10
  const logSigCen = Math.log10(Math.max(central.vel_dispersion, 1e-8));
  const logRcen   = Math.log10(Math.max(central.half_mass_radius, 1e-8));
  const logSigSat = satellites.length > 0
    ? Math.log10(satellites.reduce((s, r) => s + r.vel_dispersion, 0) / satellites.length)
    : 0;
  const logNsat   = Math.log10(Math.max(nSubhalos - 1, 1));
  const mSatTotal = satellites.reduce((s, r) => s + Math.pow(10, r.stellar_mass), 0);
  const logMsat   = mSatTotal > 0 ? Math.log10(mSatTotal) : 0;

  const values = [logMcen, logSigCen, logRcen, logSigSat, logNsat, logMsat];

  const termContributions = EQUATION.terms.map((t, i) => ({
    ...t,
    rawValue: values[i],
    contribution: t.coef * values[i],
  }));

  const total = EQUATION.intercept + termContributions.reduce((s, t) => s + t.contribution, 0);

  // Faber-Jackson: log(M) ≈ 4 * log(σ) + const (using σ_central)
  const faberJackson = 4 * logSigCen + 8.5;

  return { termContributions, total, faberJackson, central, logSigCen };
}

export default function EquationCalculator({ df, nSubhalos, gnnMean }) {
  const result = df ? computeEquation(df, nSubhalos) : null;

  const S = {
    panel: { background: "#0a0a1a", border: "1px solid #1e2a4a", borderRadius: "10px", padding: "20px" },
    title: { fontSize: "12px", color: "#4a9eff", letterSpacing: "2px", textTransform: "uppercase", marginBottom: "4px" },
    subtitle: { fontSize: "10px", color: "#3a5a7f", marginBottom: "16px" },
    divider: { border: "none", borderTop: "1px solid #1e2a4a", margin: "10px 0" },
    row: { display: "flex", alignItems: "center", padding: "6px 8px", borderRadius: "4px", marginBottom: "3px", fontFamily: "monospace" },
    intercept: { display: "flex", justifyContent: "space-between", padding: "6px 8px", fontFamily: "monospace", color: "#8a9ab5", fontSize: "12px" },
    label: { flex: 2, fontSize: "11px" },
    coef: { flex: 1, textAlign: "right", fontSize: "11px" },
    times: { flex: 0, padding: "0 6px", color: "#3a5a7f", fontSize: "11px" },
    rawVal: { flex: 1.2, textAlign: "right", fontSize: "10px", color: "#5a7a9f" },
    eq: { flex: 0, padding: "0 4px", color: "#3a5a7f", fontSize: "11px" },
    contrib: { flex: 1, textAlign: "right", fontSize: "11px", fontWeight: "bold" },
    totalRow: { display: "flex", justifyContent: "space-between", padding: "8px", background: "#0d1a2e", borderRadius: "6px", marginTop: "8px", fontFamily: "monospace" },
    totalLabel: { fontSize: "12px", color: "#8a9ab5" },
    totalVal: { fontSize: "16px", color: "#4a9eff", fontWeight: "bold" },
    gnnRow: { display: "flex", justifyContent: "space-between", padding: "6px 8px", fontFamily: "monospace", marginTop: "4px" },
    diffRow: { display: "flex", justifyContent: "space-between", padding: "6px 8px", fontFamily: "monospace" },
    fj: { background: "#0d1520", border: "1px solid #1e3a2a", borderRadius: "6px", padding: "10px", marginTop: "12px", fontSize: "11px", color: "#5a9a7f", fontFamily: "monospace" },
    empty: { color: "#3a5a7f", fontSize: "12px", textAlign: "center", padding: "20px 0", fontFamily: "monospace" },
  };

  if (!result) {
    return (
      <div style={S.panel}>
        <div style={S.title}>Symbolic Equation</div>
        <div style={S.subtitle}>full_physics linear model — PySR discovered</div>
        <div style={S.empty}>Upload a cluster to see the live equation breakdown.</div>
      </div>
    );
  }

  const diff = gnnMean != null ? Math.abs(result.total - gnnMean) : null;
  const diffColor = diff == null ? "#8a9ab5" : diff < 0.1 ? "#00c48c" : diff < 0.2 ? "#ffb800" : "#ff4757";

  return (
    <div style={S.panel}>
      <div style={S.title}>Symbolic Equation — Live Calculator</div>
      <div style={S.subtitle}>full_physics linear model · PySR discovered · per-term breakdown for this cluster</div>

      {/* Header */}
      <div style={{ ...S.row, color: "#3a5a7f", fontSize: "10px", letterSpacing: "1px" }}>
        <span style={S.label}>TERM</span>
        <span style={S.coef}>COEF</span>
        <span style={S.times}> </span>
        <span style={S.rawVal}>log VALUE</span>
        <span style={S.eq}> </span>
        <span style={S.contrib}>CONTRIB</span>
      </div>

      <hr style={S.divider} />

      {/* Intercept */}
      <div style={S.intercept}>
        <span>intercept</span>
        <span style={{ color: "#8a9ab5" }}>{EQUATION.intercept.toFixed(3)}</span>
      </div>

      {/* Terms */}
      {result.termContributions.map((t, i) => (
        <div
          key={i}
          style={{
            ...S.row,
            background: t.dominant ? "rgba(74,158,255,0.08)" : "transparent",
            border: t.dominant ? "1px solid rgba(74,158,255,0.2)" : "1px solid transparent",
          }}
        >
          <span style={{ ...S.label, color: t.dominant ? "#4a9eff" : "#8a9ab5" }}>
            {t.dominant && "★ "}{t.label}
          </span>
          <span style={{ ...S.coef, color: t.coef >= 0 ? "#00c48c" : "#ff4757" }}>
            {t.coef >= 0 ? "+" : ""}{t.coef.toFixed(3)}
          </span>
          <span style={S.times}>×</span>
          <span style={S.rawVal}>{t.rawValue.toFixed(3)}</span>
          <span style={S.eq}>=</span>
          <span style={{ ...S.contrib, color: t.dominant ? "#4a9eff" : t.contribution >= 0 ? "#c8d6e5" : "#ff8a80" }}>
            {t.contribution >= 0 ? "+" : ""}{t.contribution.toFixed(4)}
          </span>
        </div>
      ))}

      <hr style={S.divider} />

      {/* Equation total */}
      <div style={S.totalRow}>
        <span style={S.totalLabel}>Equation total</span>
        <span style={S.totalVal}>{result.total.toFixed(3)} log(M☉)</span>
      </div>

      {/* GNN prediction */}
      {gnnMean != null && (
        <>
          <div style={S.gnnRow}>
            <span style={{ fontSize: "12px", color: "#8a9ab5" }}>GNN prediction</span>
            <span style={{ fontSize: "12px", color: "#00c48c", fontWeight: "bold" }}>{gnnMean.toFixed(3)} log(M☉)</span>
          </div>
          <div style={S.diffRow}>
            <span style={{ fontSize: "11px", color: "#5a7a9f" }}>
              Difference {diff < 0.1 ? "✓ equation is a good summary" : diff < 0.2 ? "⚠ marginal" : "✗ cluster has unusual properties"}
            </span>
            <span style={{ fontSize: "12px", color: diffColor, fontWeight: "bold" }}>
              {diff.toFixed(3)} dex
            </span>
          </div>
        </>
      )}

      {/* Faber-Jackson context */}
      <div style={S.fj}>
        <div style={{ color: "#3a7a5f", marginBottom: "4px", letterSpacing: "1px", fontSize: "10px" }}>FABER-JACKSON CONTEXT</div>
        <div>σ_central = {Math.pow(10, result.logSigCen).toFixed(1)} km/s</div>
        <div>Faber-Jackson alone → M̂ = <strong>{result.faberJackson.toFixed(3)}</strong> log(M☉)</div>
        <div>Full equation gives → M̂ = <strong>{result.total.toFixed(3)}</strong> log(M☉)</div>
        <div style={{ color: "#4a9eff", marginTop: "4px" }}>
          Satellite correction = {(result.total - result.faberJackson).toFixed(3)} dex
          {Math.abs(result.total - result.faberJackson) < 0.05
            ? " — satellites negligible for this cluster"
            : " — satellites meaningfully shift the prediction"}
        </div>
      </div>
    </div>
  );
}
