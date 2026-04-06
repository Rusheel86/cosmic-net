import React, { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

export default function EquationPanel({ result }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current && result?.sym_equation_latex) {
      try {
        katex.render(result.sym_equation_latex, ref.current, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        ref.current.textContent = result.sym_equation_latex;
      }
    }
  }, [result]);

  const handleCopy = () => {
    if (result?.sym_equation_latex) {
      navigator.clipboard.writeText(result.sym_equation_latex);
      alert("LaTeX copied to clipboard!");
    }
  };

  return (
    <div className="panel equation-panel">
      <h2>Symbolic Equation</h2>
      <p className="eq-subtitle">
        Discovered by PySR — distilled from GNN learned behaviour
      </p>

      {result ? (
        <>
          <div className="katex-display" ref={ref} />
          <div className="eq-meta">
            <span>R² = {result.sym_r2.toFixed(3)} vs GNN on validation set</span>
          </div>
          <button className="copy-btn" onClick={handleCopy}>
            Copy as LaTeX
          </button>
        </>
      ) : (
        <p className="empty-msg">Run a prediction to see the equation.</p>
      )}
    </div>
  );
}
