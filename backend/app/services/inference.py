"""
Inference service — MC-Dropout uncertainty estimation.
Uses 30 forward passes (matching training config mc_samples=30).
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from app.core.model import CosmicGNN
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def mc_dropout_predict(model: CosmicGNN, data: Data) -> dict:
    """
    Run MC-Dropout inference and return mean, std, and 95% interval.
    """
    batch = Batch.from_data_list([data])

    # train() mode keeps Dropout active → stochastic passes
    model.train()

    predictions = []
    with torch.no_grad():
        for _ in range(settings.MC_DROPOUT_PASSES):
            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,   # edge features required by NNConv
                batch.batch,
            )
            predictions.append(out.item())

    preds = np.array(predictions)
    mean  = float(preds.mean())
    std   = float(preds.std())

    logger.info(
        "MC-Dropout: passes=%d  mean=%.4f  std=%.4f",
        settings.MC_DROPOUT_PASSES, mean, std,
    )

    return {
        "mean":  mean,
        "std":   std,
        "lower": mean - 2 * std,
        "upper": mean + 2 * std,
    }
