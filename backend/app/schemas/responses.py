from pydantic import BaseModel, Field
from typing import Literal


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Returned by POST /predict"""

    # GNN output
    gnn_log_mass_mean: float = Field(
        ..., description="Mean predicted log10(M_halo / M_sun) from MC-Dropout"
    )
    gnn_log_mass_std: float = Field(
        ..., description="Std dev across 50 MC-Dropout forward passes"
    )
    gnn_log_mass_lower: float = Field(..., description="Mean − 2σ (95% band)")
    gnn_log_mass_upper: float = Field(..., description="Mean + 2σ (95% band)")

    # Symbolic equation output (for side-by-side comparison)
    sym_log_mass: float = Field(
        ..., description="log10(M_halo) from symbolic equation"
    )
    sym_equation_latex: str = Field(..., description="LaTeX string of the equation")
    sym_r2: float = Field(..., description="R² of symbolic eq vs GNN on this cluster")

    # Metadata
    n_subhalos: int = Field(..., description="Number of subhalos in the cluster")
    n_edges: int = Field(..., description="Number of k-NN edges in the graph")


# ── Virial Check ──────────────────────────────────────────────────────────────

class VirialResponse(BaseModel):
    """Returned by POST /virial"""

    virial_ratio: float = Field(
        ..., description="|2·KE / PE| — should be ≈ 1.0 for a virialized cluster"
    )
    status: Literal["green", "amber", "red"] = Field(
        ..., description="green: virialized | amber: marginal | red: not virialized"
    )
    message: str = Field(..., description="Human-readable interpretation")


# ── Explainer ─────────────────────────────────────────────────────────────────

class EdgeImportance(BaseModel):
    source: int = Field(..., description="Source subhalo index")
    target: int = Field(..., description="Target subhalo index")
    importance: float = Field(..., description="PGExplainer importance score [0, 1]")


class ExplainerResponse(BaseModel):
    """Returned by POST /explain"""

    edge_importances: list[EdgeImportance]
    top_anchor_indices: list[int] = Field(
        ..., description="Node indices of the top gravitational anchors"
    )
    explanation_note: str = Field(
        default=(
            "Edge importances normalised to [0,1]. "
            "Nodes with highest mean incoming importance are gravitational anchors."
        )
    )


# ── Data Overview ─────────────────────────────────────────────────────────────

class FeatureStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class DataOverviewResponse(BaseModel):
    """Returned by POST /overview — auto-render on CSV upload"""

    n_subhalos: int
    feature_stats: dict[str, FeatureStats]
    # Raw arrays for Plotly rendering on the frontend
    stellar_mass_log: list[float]
    velocity_dispersion: list[float]
    metallicity: list[float]
    # Correlation matrix as nested list (row-major)
    correlation_matrix: list[list[float]]
    correlation_labels: list[str]
