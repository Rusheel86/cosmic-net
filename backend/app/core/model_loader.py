"""
Model loader — loads best_model_augmented.pt exactly once at startup.
"""

import torch
from pathlib import Path
from app.core.model import CosmicGNN
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model: CosmicGNN | None = None


def load_model() -> CosmicGNN:
    global _model

    model_path = Path(settings.MODEL_PATH)

    _model = CosmicGNN(
        num_node_features=4,
        edge_features=5,
        hidden_dim=64,
        num_layers=3,
        dropout=0.05,
    )

    if not model_path.exists():
        logger.warning(
            "Model file not found at %s — using random weights for dev", model_path
        )
        return _model

    logger.info("Loading model from %s", model_path)

    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle full checkpoint dict (our format) or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Checkpoint epoch=%s  val_R2=%.4f  val_RMSE=%.4f",
            checkpoint.get("epoch", "?"),
            checkpoint.get("metrics", {}).get("r2", 0),
            checkpoint.get("metrics", {}).get("rmse", 0),
        )
    else:
        state_dict = checkpoint

    _model.load_state_dict(state_dict)
    logger.info(
        "Model loaded — %d parameters", sum(p.numel() for p in _model.parameters())
    )

    return _model


def get_model() -> CosmicGNN:
    if _model is None:
        raise RuntimeError("Model not loaded. Was load_model() called at startup?")
    return _model
