from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.preprocessing import parse_csv
from app.services.virial import compute_virial_ratio
from app.schemas.responses import VirialResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/virial",
    response_model=VirialResponse,
    summary="Physics Virial ratio sanity check",
    description=(
        "Computes Q = |2·KE / PE| from subhalo positions, velocities, and masses. "
        "Q ≈ 1 indicates a virialized cluster — a pre-condition for the mass "
        "estimator to be physically meaningful. "
        "Returns green / amber / red status. "
        "Requires vel_x, vel_y, vel_z columns in the CSV."
    ),
)
async def virial_check(
    file: UploadFile = File(..., description="CSV — must include vel_x, vel_y, vel_z"),
):
    raw = await file.read()
    try:
        df = parse_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = compute_virial_ratio(df)

    return VirialResponse(**result)