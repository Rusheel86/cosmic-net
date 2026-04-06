from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.model_loader import get_model
from app.core.model import CosmicGNN
from app.services.preprocessing import parse_csv, build_graph
from app.services.inference import mc_dropout_predict
from app.services.symbolic import evaluate_symbolic_equation, compute_r2_vs_gnn
from app.schemas.responses import PredictionResponse
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict dark matter halo mass from a galaxy cluster CSV",
    description=(
        "Upload a CSV of subhalo features. Returns the GNN's predicted log10(M_halo) "
        "with Monte Carlo Dropout uncertainty bounds, plus the symbolic equation's "
        "prediction for side-by-side comparison."
    ),
)
async def predict(
    file: UploadFile = File(..., description="CSV with subhalo features"),
    model: CosmicGNN = Depends(get_model),
):
    # ── Parse & validate CSV ──────────────────────────────────────────────────
    raw = await file.read()
    try:
        df = parse_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Build graph ───────────────────────────────────────────────────────────
    try:
        data = build_graph(df)
    except Exception as exc:
        logger.exception("Graph construction failed")
        raise HTTPException(status_code=500, detail=f"Graph construction error: {exc}")

    # ── GNN inference with MC-Dropout ─────────────────────────────────────────
    gnn_result = mc_dropout_predict(model, data)

    # ── Symbolic equation ─────────────────────────────────────────────────────
    sym_pred = evaluate_symbolic_equation(df)
    r2 = compute_r2_vs_gnn(sym_pred, gnn_result["mean"])

    logger.info(
        "Prediction complete: gnn_mean=%.4f  gnn_std=%.4f  sym=%.4f",
        gnn_result["mean"],
        gnn_result["std"],
        sym_pred,
    )

    return PredictionResponse(
        gnn_log_mass_mean=gnn_result["mean"],
        gnn_log_mass_std=gnn_result["std"],
        gnn_log_mass_lower=gnn_result["lower"],
        gnn_log_mass_upper=gnn_result["upper"],
        sym_log_mass=sym_pred,
        sym_equation_latex=settings.EQUATION_LATEX,
        sym_r2=r2,
        n_subhalos=data.num_nodes,
        n_edges=data.num_edges,
    )
