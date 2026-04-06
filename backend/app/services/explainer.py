"""
Explainability service — wraps PGExplainer for on-demand edge attribution.

PGExplainer (Luo et al., 2020) trains a small MLP to predict edge masks
that maximally explain the GNN's output. It is expensive (requires a
short fine-tuning loop), so we only run it when the user clicks "Why?".

For a production deployment, consider pre-training PGExplainer on the
training set and saving the explainer weights alongside best_model.pt.
Here we run the lightweight on-the-fly version.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, PGExplainer

from app.core.model import CosmicGNN
from app.core.logging import get_logger

logger = get_logger(__name__)

# How many epochs to train the PGExplainer MLP on-the-fly
EXPLAINER_EPOCHS = 30


def explain_prediction(model: CosmicGNN, data: Data) -> dict:
    """
    Run PGExplainer on a single cluster graph and return edge importances.

    Args:
        model:  Trained CosmicGNN
        data:   PyG Data object for the cluster

    Returns:
        dict with edge_importances (list of {source, target, importance})
        and top_anchor_indices (list of node indices)
    """
    model.eval()
    batch = Batch.from_data_list([data])

    logger.info(
        "Running PGExplainer — nodes=%d  edges=%d  epochs=%d",
        data.num_nodes,
        data.num_edges,
        EXPLAINER_EPOCHS,
    )

    try:
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=EXPLAINER_EPOCHS, lr=0.003),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

        # PGExplainer requires a training phase — feed the same graph
        for epoch in range(EXPLAINER_EPOCHS):
            loss = explainer.algorithm.train(
                epoch, model, batch.x, batch.edge_index, target=None, batch=batch.batch
            )

        explanation = explainer(
            batch.x,
            batch.edge_index,
            target=None,
            batch=batch.batch,
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()

    except Exception as exc:
        # Graceful fallback: uniform importance if PGExplainer fails
        logger.warning("PGExplainer failed (%s) — returning uniform importances", exc)
        n_edges = data.edge_index.shape[1]
        edge_mask = np.ones(n_edges) / n_edges

    # Normalise to [0, 1]
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()

    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()

    edge_importances = [
        {"source": int(s), "target": int(d), "importance": float(w)}
        for s, d, w in zip(src, dst, edge_mask)
    ]

    # Gravitational anchors: nodes with highest mean incoming importance
    node_importance = np.zeros(data.num_nodes)
    for s, d, w in zip(src, dst, edge_mask):
        node_importance[d] += w
    node_importance /= (node_importance.max() + 1e-8)

    top_k = min(5, data.num_nodes)
    top_anchors = np.argsort(node_importance)[-top_k:][::-1].tolist()

    logger.info("PGExplainer done — top anchors: %s", top_anchors)

    return {
        "edge_importances": edge_importances,
        "top_anchor_indices": top_anchors,
    }
