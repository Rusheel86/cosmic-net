"""
Virial physics sanity check.

The Virial Theorem states that for a gravitationally bound, virialized system:
    2 · KE + PE = 0   →   Q = |2·KE / PE| ≈ 1

Where:
    KE = (1/2) Σ m_i · |v_i - v_cm|²          (kinetic energy)
    PE = -G Σ_{i<j} m_i · m_j / r_ij           (potential energy)

This is a one-function physics check that makes the tool credible
to astrophysicists — it verifies that the uploaded cluster is
actually in dynamical equilibrium before trusting the mass prediction.

References:
    Zwicky (1933) — original Virial mass estimator
    Carlberg et al. (1997) — modern cluster Virial analysis
"""

import numpy as np
import pandas as pd
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Gravitational constant in units consistent with:
#   mass in M_sun, distance in kpc, velocity in km/s
# G = 4.302e-3 pc M_sun^-1 (km/s)^2  → convert to kpc: × 1e-3
G_KPC = 4.302e-6   # kpc M_sun^-1 (km/s)^2


def compute_virial_ratio(df: pd.DataFrame) -> dict:
    """
    Compute the Virial ratio Q = |2·KE / PE| from the cluster DataFrame.

    Required columns: pos_x, pos_y, pos_z [kpc], vel_x, vel_y, vel_z [km/s],
                      stellar_mass [log10 M_sun]

    Returns dict with: virial_ratio, status, message
    """
    # ── Check velocity columns exist ──────────────────────────────────────────
    vel_cols = ["vel_x", "vel_y", "vel_z"]
    if not all(c in df.columns for c in vel_cols):
        return {
            "virial_ratio": -1.0,
            "status": "amber",
            "message": (
                "Velocity columns (vel_x, vel_y, vel_z) not found in CSV. "
                "Virial check skipped — add velocity data for physics validation."
            ),
        }

    # ── Extract arrays ────────────────────────────────────────────────────────
    pos = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)     # kpc
    vel = df[["vel_x", "vel_y", "vel_z"]].values.astype(np.float64)     # km/s
    mass = 10 ** df["stellar_mass"].values.astype(np.float64)             # M_sun

    n = len(df)

    # ── Kinetic Energy ────────────────────────────────────────────────────────
    # Centre-of-mass velocity
    v_cm = np.average(vel, weights=mass, axis=0)
    v_rel = vel - v_cm                               # peculiar velocities
    KE = 0.5 * np.sum(mass * np.sum(v_rel ** 2, axis=1))

    # ── Potential Energy (O(N²) — fine for cluster sizes ≤ ~10^4) ────────────
    PE = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(pos[i] - pos[j])   # kpc
            if r_ij < 1e-6:                            # avoid division by zero
                continue
            PE -= G_KPC * mass[i] * mass[j] / r_ij

    # ── Virial ratio ──────────────────────────────────────────────────────────
    if abs(PE) < 1e-30:
        return {
            "virial_ratio": -1.0,
            "status": "amber",
            "message": "Potential energy is effectively zero — cluster may be too sparse.",
        }

    Q = abs(2 * KE / PE)

    logger.info("Virial check: KE=%.3e  PE=%.3e  Q=%.4f", KE, PE, Q)

    # ── Status classification ─────────────────────────────────────────────────
    deviation = abs(Q - 1.0)

    if deviation < settings.VIRIAL_GREEN_THRESHOLD:
        status = "green"
        message = (
            f"Q = {Q:.3f} — cluster is well virialized. "
            "Mass prediction is physically reliable."
        )
    elif deviation < settings.VIRIAL_AMBER_THRESHOLD:
        status = "amber"
        message = (
            f"Q = {Q:.3f} — cluster shows marginal virialization. "
            "Prediction may have elevated uncertainty (merging/infalling substructure?)."
        )
    else:
        status = "red"
        message = (
            f"Q = {Q:.3f} — cluster does not appear virialized. "
            "Treat mass prediction with caution; system may be dynamically young."
        )

    return {"virial_ratio": float(Q), "status": status, "message": message}
