"""
Data overview service — computes summary statistics and raw arrays
for Plotly rendering on the frontend when a CSV is first uploaded.
"""

import numpy as np
import pandas as pd
from app.core.logging import get_logger

logger = get_logger(__name__)

OVERVIEW_FEATURES = ["stellar_mass", "velocity_dispersion", "metallicity"]


def compute_overview(df: pd.DataFrame) -> dict:
    """
    Compute feature statistics and arrays needed by the frontend overview panel.

    Returns:
        dict matching DataOverviewResponse schema
    """
    n = len(df)

    # ── Per-feature statistics ────────────────────────────────────────────────
    feature_stats = {}
    for col in OVERVIEW_FEATURES:
        if col in df.columns:
            series = df[col].dropna()
            feature_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    # ── Raw arrays for Plotly ─────────────────────────────────────────────────
    stellar_mass_log = df["stellar_mass"].tolist() if "stellar_mass" in df.columns else []
    velocity_dispersion = df["velocity_dispersion"].tolist() if "velocity_dispersion" in df.columns else []
    metallicity = df["metallicity"].tolist() if "metallicity" in df.columns else []

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr_cols = [c for c in OVERVIEW_FEATURES if c in df.columns]
    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr().values.tolist()
    else:
        corr_matrix = [[1.0]]

    logger.info("Data overview computed: n_subhalos=%d  features=%s", n, corr_cols)

    return {
        "n_subhalos": n,
        "feature_stats": feature_stats,
        "stellar_mass_log": stellar_mass_log,
        "velocity_dispersion": velocity_dispersion,
        "metallicity": metallicity,
        "correlation_matrix": corr_matrix,
        "correlation_labels": corr_cols,
    }
