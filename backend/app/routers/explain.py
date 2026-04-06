from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.model_loader import get_model
from app.core.model import CosmicGNN
from app.services.preprocessing import parse_csv, build_graph
from app.services.explainer import explain_prediction
from app.schemas.responses import ExplainerResponse, EdgeImportance
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainerResponse,
    summary="On-demand PGExplainer — returns edge importance weights",
    description=(
        "Runs PGExplainer on the uploaded cluster to identify which galaxy-galaxy "
        "connections most influence the halo mass prediction. Expensive — only call "
        "on user request (the 'Why?' button). Returns edge importances [0,1] and "
        "the top gravitational anchor node indices."
    ),
)
async def explain(
    file: UploadFile = File(..., description="Same CSV used for /predict"),
    model: CosmicGNN = Depends(get_model),
):
    raw = await file.read()
    try:
        df = parse_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        data = build_graph(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph construction error: {exc}")

    try:
        result = explain_prediction(model, data)
    except Exception as exc:
        logger.exception("PGExplainer failed")
        raise HTTPException(status_code=500, detail=f"Explainer error: {exc}")

    edge_importances = [EdgeImportance(**e) for e in result["edge_importances"]]

    return ExplainerResponse(
        edge_importances=edge_importances,
        top_anchor_indices=result["top_anchor_indices"],
    )
