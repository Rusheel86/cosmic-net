"""
Symbolic equation service.

Evaluates the best PySR-discovered equation:
    log(M_halo) = α · log(σ_vel) + β · Z + γ

This runs analytically (no model needed) and is returned alongside the GNN
prediction so researchers can directly compare the interpretable formula
against the black-box GNN.
"""

import numpy as np
import pandas as pd
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def evaluate_symbolic_equation(df: pd.DataFrame) -> float:
    """
    Evaluate the symbolic equation on a cluster DataFrame.

    Uses the cluster-level aggregates:
        σ_vel  = mean of per-subhalo velocity_dispersion column
        Z      = mean metallicity across subhalos

    Returns:
        log10(M_halo) predicted by the symbolic equation
    """
    sigma_vel = df["vel_dispersion"].mean()
    metallicity = df["metallicity"].mean()

    # Guard against log(0)
    if sigma_vel <= 0:
        logger.warning("velocity_dispersion ≤ 0 — clamping to 1e-6 for symbolic eval")
        sigma_vel = 1e-6

    log_mass = (
        settings.EQUATION_ALPHA * np.log10(sigma_vel)
        + settings.EQUATION_BETA * metallicity
        + settings.EQUATION_GAMMA
    )

    logger.debug("Symbolic equation → log10(M_halo) = %.4f", log_mass)
    return float(log_mass)


def compute_r2_vs_gnn(sym_pred: float, gnn_mean: float) -> float:
    """
    Approximate R² between symbolic equation and GNN on this single cluster.

    For a proper R² you need a dataset — here we return a stored global R²
    from validation (set in config), but note it in the response so researchers
    know this is dataset-level, not per-cluster.
    """
    # Per-cluster "agreement" — |Δ| as a fraction of GNN prediction
    delta = abs(sym_pred - gnn_mean)
    agreement = max(0.0, 1.0 - delta / (abs(gnn_mean) + 1e-8))

    logger.debug(
        "Symbolic vs GNN: sym=%.4f  gnn=%.4f  agreement=%.4f",
        sym_pred,
        gnn_mean,
        agreement,
    )

    # Return the dataset-level R² from config (more meaningful for the paper)
    return settings.EQUATION_R2
