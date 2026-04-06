"""
Preprocessing service: CSV → PyTorch Geometric Data object.

Node features (4 — must match training order exactly):
    log_stellar_mass    log10(M_* / M_sun)         — column: stellar_mass
    log_vel_disp        log10(sigma_vel [km/s])     — column: vel_dispersion
    log_half_mass_r     log10(R_half [kpc])         — column: half_mass_radius
    log_metallicity     log10(Z)                    — column: metallicity

Edge features (5 — computed from positions and velocities):
    distance     Euclidean 3D separation [Mpc]
    delta_v      Relative velocity magnitude [km/s]
    cos_theta    Angle between delta_pos and delta_vel
    mass_ratio   log(m_src) - log(m_dst)
    proj_sep     Projected XY separation [Mpc]

Required CSV columns:
    pos_x, pos_y, pos_z          — positions [Mpc]
    vel_x, vel_y, vel_z          — velocities [km/s]
    stellar_mass                  — log10(M_* / M_sun)
    vel_dispersion                — sigma_vel [km/s]
    half_mass_radius              — R_half [kpc]
    metallicity                   — Z (NOT log — raw value)
"""

import io
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Columns the CSV must contain
REQUIRED_COLS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "stellar_mass",
    "vel_dispersion",
    "half_mass_radius",
    "metallicity",
]


def parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a validated DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Required: {REQUIRED_COLS}"
        )

    if len(df) < 3:
        raise ValueError("Cluster must contain at least 3 subhalos.")

    return df


def _safe_log10(arr: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """log10 with a floor to avoid log(0)."""
    return np.log10(np.clip(arr, floor, None))


def build_graph(df: pd.DataFrame) -> Data:
    """
    Build a k-NN graph matching the training pipeline exactly.

    Steps:
      1. Compute 4 log-transformed node features
      2. Build k-NN edges from 3D positions (k=8, matching training)
      3. Compute 5 edge features from positions and velocities
      4. Return torch_geometric.data.Data object
    """
    n = len(df)
    k = min(settings.KNN_K, n - 1)

    # ── Node features (4, log-transformed, matching training) ─────────────────
    log_stellar_mass = df["stellar_mass"].values.astype(np.float32)       # already log10
    log_vel_disp     = _safe_log10(df["vel_dispersion"].values).astype(np.float32)
    log_half_mass_r  = _safe_log10(df["half_mass_radius"].values).astype(np.float32)
    log_metallicity  = _safe_log10(df["metallicity"].values).astype(np.float32)

    # Stack into [N, 4] — exact training order
    node_features = np.stack(
        [log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity],
        axis=1
    )
    x = torch.tensor(node_features, dtype=torch.float)

    # ── Positions and velocities ───────────────────────────────────────────────
    pos = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)   # Mpc
    vel = df[["vel_x", "vel_y", "vel_z"]].values.astype(np.float64)   # km/s

    # ── k-NN edges from 3D positions ──────────────────────────────────────────
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(pos)
    _, indices = nbrs.kneighbors(pos)

    src_list, dst_list = [], []
    for i, nbr_idx in enumerate(indices):
        for j in nbr_idx[1:]:    # skip self (index 0)
            src_list.append(i)
            dst_list.append(int(j))

    src = np.array(src_list)
    dst = np.array(dst_list)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ── Edge features (5, matching training augment_graph logic) ─────────────
    diff_pos  = pos[dst] - pos[src]                           # [E, 3]
    diff_vel  = vel[dst] - vel[src]                           # [E, 3]

    # 1. distance — Euclidean 3D [Mpc]
    distance  = np.linalg.norm(diff_pos, axis=1, keepdims=True)

    # 2. delta_v — relative velocity magnitude [km/s]
    delta_v   = np.linalg.norm(diff_vel, axis=1, keepdims=True)

    # 3. cos_theta — angle between position and velocity difference vectors
    eps       = 1e-8
    cos_theta = (
        (diff_pos * diff_vel).sum(axis=1, keepdims=True)
        / (distance * delta_v + eps)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 4. mass_ratio — log(m_src) - log(m_dst)
    mass_ratio = (
        log_stellar_mass[src, None] - log_stellar_mass[dst, None]
    )

    # 5. proj_sep — projected XY separation [Mpc]
    proj_sep  = np.linalg.norm(diff_pos[:, :2], axis=1, keepdims=True)

    edge_attr = np.concatenate(
        [distance, delta_v, cos_theta, mass_ratio, proj_sep], axis=1
    ).astype(np.float32)

    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)

    # ── Assemble Data object ──────────────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    data.raw_df = df   # stash for Virial check and overview

    logger.debug(
        "Graph built: %d nodes, %d edges, k=%d", data.num_nodes, data.num_edges, k
    )

    return data
