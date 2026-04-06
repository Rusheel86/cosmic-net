"""
Endpoint tests — run with:
    pytest tests/ -v

These tests use TestClient (no running server needed).
The model loads in dev/fallback mode (random weights) when best_model.pt is absent.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Ensure the app can be imported from the backend root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)
SAMPLE_CSV = Path(__file__).parent / "sample_cluster.csv"


def _csv_file():
    """Helper: returns an open file tuple for multipart upload."""
    return ("file", (SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), "text/csv"))


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.post("/api/v1/predict", files=[_csv_file()])
    assert r.status_code == 200, r.text


def test_predict_response_schema():
    r = client.post("/api/v1/predict", files=[_csv_file()])
    body = r.json()

    # GNN fields
    assert "gnn_log_mass_mean" in body
    assert "gnn_log_mass_std" in body
    assert "gnn_log_mass_lower" in body
    assert "gnn_log_mass_upper" in body

    # Confidence interval is correctly ordered
    assert body["gnn_log_mass_lower"] < body["gnn_log_mass_upper"]

    # Symbolic fields
    assert "sym_log_mass" in body
    assert "sym_equation_latex" in body
    assert "sym_r2" in body
    assert 0.0 <= body["sym_r2"] <= 1.0

    # Metadata
    assert body["n_subhalos"] == 15
    assert body["n_edges"] > 0


def test_predict_bad_csv():
    bad_csv = b"col_a,col_b\n1,2\n3,4"
    r = client.post("/api/v1/predict", files=[("file", ("bad.csv", bad_csv, "text/csv"))])
    assert r.status_code == 422
    assert "missing required columns" in r.json()["detail"]


def test_predict_too_few_rows():
    tiny = b"pos_x,pos_y,pos_z,stellar_mass,velocity_dispersion,metallicity\n1,2,3,10,400,0.01"
    r = client.post("/api/v1/predict", files=[("file", ("tiny.csv", tiny, "text/csv"))])
    assert r.status_code == 422


# ── /virial ───────────────────────────────────────────────────────────────────

def test_virial_returns_200():
    r = client.post("/api/v1/virial", files=[_csv_file()])
    assert r.status_code == 200, r.text


def test_virial_response_schema():
    r = client.post("/api/v1/virial", files=[_csv_file()])
    body = r.json()
    assert "virial_ratio" in body
    assert body["status"] in ("green", "amber", "red")
    assert "message" in body
    assert body["virial_ratio"] > 0


def test_virial_no_velocity_cols():
    """CSV without vel columns should return amber + helpful message."""
    no_vel = SAMPLE_CSV.read_text()
    # Strip velocity columns
    lines = no_vel.split("\n")
    header = "pos_x,pos_y,pos_z,stellar_mass,velocity_dispersion,metallicity"
    rows = [",".join(l.split(",")[:6]) for l in lines[1:] if l.strip()]
    csv_bytes = (header + "\n" + "\n".join(rows)).encode()
    r = client.post("/api/v1/virial", files=[("file", ("no_vel.csv", csv_bytes, "text/csv"))])
    assert r.status_code == 200
    assert r.json()["status"] == "amber"
    assert "vel" in r.json()["message"].lower()


# ── /overview ─────────────────────────────────────────────────────────────────

def test_overview_returns_200():
    r = client.post("/api/v1/overview", files=[_csv_file()])
    assert r.status_code == 200, r.text


def test_overview_response_schema():
    r = client.post("/api/v1/overview", files=[_csv_file()])
    body = r.json()
    assert body["n_subhalos"] == 15
    assert "stellar_mass" in body["feature_stats"]
    assert len(body["stellar_mass_log"]) == 15
    assert len(body["correlation_matrix"]) > 0
    assert len(body["correlation_labels"]) >= 2


# ── /explain (lightweight smoke test — full PGExplainer is slow) ──────────────

def test_explain_returns_200():
    r = client.post("/api/v1/explain", files=[_csv_file()])
    # May be slow in CI — just check it doesn't crash
    assert r.status_code == 200, r.text


def test_explain_response_schema():
    r = client.post("/api/v1/explain", files=[_csv_file()])
    body = r.json()
    assert "edge_importances" in body
    assert "top_anchor_indices" in body
    assert len(body["edge_importances"]) > 0

    # Each edge has source, target, importance in [0,1]
    for edge in body["edge_importances"]:
        assert "source" in edge
        assert "target" in edge
        assert 0.0 <= edge["importance"] <= 1.0
