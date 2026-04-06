from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.preprocessing import parse_csv
from app.services.overview import compute_overview
from app.schemas.responses import DataOverviewResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/overview",
    response_model=DataOverviewResponse,
    summary="Auto data overview on CSV upload",
    description=(
        "Call immediately after a CSV is selected — before running prediction. "
        "Returns feature statistics and raw arrays for Plotly to render: "
        "stellar mass histogram, velocity dispersion distribution, "
        "metallicity scatter, and a feature correlation heatmap."
    ),
)
async def overview(
    file: UploadFile = File(...),
):
    raw = await file.read()
    try:
        df = parse_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = compute_overview(df)
    return DataOverviewResponse(**result)