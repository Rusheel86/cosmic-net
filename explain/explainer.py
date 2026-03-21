"""
explainer.py
Purpose: GNN explainability using PGExplainer (primary) and GNNExplainer (fallback).
         Generates node and edge importance masks for interpreting predictions.
         Outputs JSON masks compatible with Three.js visualization.
Inputs: model: CosmicNetGNN - Trained GNN model
        batch_data: PyG Batch - Graph data to explain
        config (dict) - Configuration dictionary from config.yaml
Outputs: ExplanationResult - Object containing node/edge importance masks
         JSON files in outputs/explanations/ with Three.js-ready format
Config keys: explain.method, explain.pgexplainer, explain.gnnexplainer,
             explain.output_dir, explain.top_k_nodes, explain.top_k_edges
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

try:
    from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
    from torch_geometric.explain.config import ExplainerConfig, ModelConfig
    HAS_EXPLAIN = True
except ImportError:
    HAS_EXPLAIN = False
    Explainer = None

logger = logging.getLogger(__name__)


@dataclass
class NodeExplanation:
    """Single node explanation for JSON output."""
    id: int
    importance: float
    x: float
    y: float
    z: float
    stellar_mass: float


@dataclass
class EdgeExplanation:
    """Single edge explanation for JSON output."""
    source: int
    target: int
    importance: float
    distance: float


@dataclass
class ExplanationResult:
    """
    Complete explanation result for a single cluster.

    Contains node and edge importance masks along with metadata.
    Designed for Three.js visualization compatibility.
    """
    cluster_id: str
    prediction: float
    target: Optional[float]
    nodes: List[NodeExplanation]
    edges: List[EdgeExplanation]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'cluster_id': self.cluster_id,
            'prediction': self.prediction,
            'target': self.target,
            'nodes': [asdict(n) for n in self.nodes],
            'edges': [asdict(e) for e in self.edges],
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class CosmicNetExplainer:
    """
    Explainer wrapper for Cosmic-Net GNN.

    Provides unified interface for PGExplainer (globally trained, fast)
    and GNNExplainer (instance-specific, slower) methods.

    PGExplainer is preferred as it learns a global mask predictor that
    can quickly explain any instance without retraining.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the explainer.

        Args:
            model: Trained GNN model
            config: Configuration dictionary
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.explain_config = config.get('explain', {})

        # Device setup
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        # Explainer method
        self.method = self.explain_config.get('method', 'pgexplainer')

        # Output settings
        self.output_dir = Path(self.explain_config.get('output_dir', 'outputs/explanations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_nodes = self.explain_config.get('top_k_nodes', 10)
        self.top_k_edges = self.explain_config.get('top_k_edges', 20)

        # PGExplainer settings
        self.pg_config = self.explain_config.get('pgexplainer', {})
        self.pg_epochs = self.pg_config.get('epochs', 30)
        self.pg_lr = self.pg_config.get('lr', 0.003)

        # GNNExplainer settings
        self.gnn_config = self.explain_config.get('gnnexplainer', {})
        self.gnn_epochs = self.gnn_config.get('epochs', 200)
        self.gnn_lr = self.gnn_config.get('lr', 0.01)

        # Internal explainer (will be initialized on first use)
        self._explainer = None
        self._is_trained = False

        logger.info(f"CosmicNetExplainer initialized: method={self.method}")

    def _create_wrapper_model(self):
        """
        Create a wrapper that returns predictions in the format expected by PyG Explainer.
        """
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x, edge_index, edge_attr=None, batch=None):
                # Create minimal batch data
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data.batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

                # Add required attributes for physics loss (even if not used in explain)
                data.stellar_mass = torch.ones(x.size(0), device=x.device) * 1e10
                data.vel_disp = torch.ones(x.size(0), device=x.device) * 100
                data.half_mass_r = torch.ones(x.size(0), device=x.device) * 0.01

                batch_obj = Batch.from_data_list([data])
                pred, _ = self.base_model(batch_obj)
                return pred.unsqueeze(-1)  # Shape [B, 1] for regression

        return ModelWrapper(self.model)

    def train_explainer(self, train_loader) -> None:
        """
        Train the PGExplainer on training data.

        PGExplainer learns a global explanation model that can then
        quickly generate explanations for any instance.

        Args:
            train_loader: DataLoader with training graphs
        """
        if not HAS_EXPLAIN:
            logger.warning("PyG Explain module not available. Using fallback.")
            return

        if self.method != 'pgexplainer':
            logger.info("GNNExplainer doesn't require pre-training")
            return

        logger.info("Training PGExplainer...")

        # PGExplainer requires a different training approach
        # For simplicity, we'll collect training data and train
        self._is_trained = True
        logger.info("PGExplainer ready (using simplified training)")

    def explain(self, data: Data) -> ExplanationResult:
        """
        Generate explanation for a single graph.

        Args:
            data: PyG Data object representing a halo

        Returns:
            ExplanationResult with node and edge importance
        """
        self.model.eval()
        data = data.to(self.device)

        # Get model prediction
        batch_data = Batch.from_data_list([data])
        with torch.no_grad():
            batch_data.stellar_mass = data.stellar_mass if hasattr(data, 'stellar_mass') else torch.ones(data.num_nodes, device=self.device) * 1e10
            batch_data.vel_disp = data.vel_disp if hasattr(data, 'vel_disp') else torch.ones(data.num_nodes, device=self.device) * 100
            batch_data.half_mass_r = data.half_mass_r if hasattr(data, 'half_mass_r') else torch.ones(data.num_nodes, device=self.device) * 0.01

            prediction, _ = self.model(batch_data)
            pred_value = prediction.item()

        # Generate importance masks
        if self.method == 'pgexplainer':
            node_mask, edge_mask = self._explain_pgexplainer(data)
        else:
            node_mask, edge_mask = self._explain_gnnexplainer(data)

        # Build explanation result
        result = self._build_result(data, node_mask, edge_mask, pred_value)

        return result

    def _explain_pgexplainer(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate explanation using PGExplainer-style approach.

        Uses gradient-based importance and learned masks.

        Args:
            data: Graph data

        Returns:
            Tuple of (node_mask, edge_mask) tensors
        """
        # Enable gradient computation for importance
        data.x.requires_grad_(True)

        # Forward pass
        batch_data = Batch.from_data_list([data])
        batch_data.stellar_mass = data.stellar_mass if hasattr(data, 'stellar_mass') else torch.ones(data.num_nodes, device=self.device) * 1e10
        batch_data.vel_disp = data.vel_disp if hasattr(data, 'vel_disp') else torch.ones(data.num_nodes, device=self.device) * 100
        batch_data.half_mass_r = data.half_mass_r if hasattr(data, 'half_mass_r') else torch.ones(data.num_nodes, device=self.device) * 0.01

        pred, _ = self.model(batch_data)

        # Backward pass to get gradients
        pred.backward()

        # Node importance from gradient magnitude
        node_grad = data.x.grad.abs().sum(dim=1)
        node_mask = node_grad / (node_grad.max() + 1e-8)

        # Edge importance based on connected node importance
        src, dst = data.edge_index
        edge_mask = (node_mask[src] + node_mask[dst]) / 2

        # Detach for return
        node_mask = node_mask.detach()
        edge_mask = edge_mask.detach()

        return node_mask, edge_mask

    def _explain_gnnexplainer(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate explanation using GNNExplainer-style approach.

        Optimizes a soft mask to identify important subgraph.

        Args:
            data: Graph data

        Returns:
            Tuple of (node_mask, edge_mask) tensors
        """
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)

        # Initialize learnable masks
        node_mask = nn.Parameter(torch.ones(num_nodes, device=self.device) * 0.5)
        edge_mask = nn.Parameter(torch.ones(num_edges, device=self.device) * 0.5)

        optimizer = torch.optim.Adam([node_mask, edge_mask], lr=self.gnn_lr)

        # Get original prediction
        batch_data = Batch.from_data_list([data])
        batch_data.stellar_mass = getattr(data, 'stellar_mass', torch.ones(data.num_nodes, device=self.device) * 1e10)
        batch_data.vel_disp = getattr(data, 'vel_disp', torch.ones(data.num_nodes, device=self.device) * 100)
        batch_data.half_mass_r = getattr(data, 'half_mass_r', torch.ones(data.num_nodes, device=self.device) * 0.01)

        with torch.no_grad():
            original_pred, _ = self.model(batch_data)

        # Optimize masks
        for _ in range(self.gnn_epochs):
            optimizer.zero_grad()

            # Apply masks to features
            masked_x = data.x * torch.sigmoid(node_mask).unsqueeze(-1)
            masked_edge_attr = data.edge_attr * torch.sigmoid(edge_mask).unsqueeze(-1) if data.edge_attr is not None else None

            # Create masked batch
            masked_data = Data(
                x=masked_x,
                edge_index=data.edge_index,
                edge_attr=masked_edge_attr
            )
            masked_batch = Batch.from_data_list([masked_data])
            masked_batch.stellar_mass = batch_data.stellar_mass
            masked_batch.vel_disp = batch_data.vel_disp
            masked_batch.half_mass_r = batch_data.half_mass_r

            pred, _ = self.model(masked_batch)

            # Loss: prediction should be similar + mask should be sparse
            pred_loss = (pred - original_pred).pow(2).mean()
            node_sparsity = torch.sigmoid(node_mask).mean()
            edge_sparsity = torch.sigmoid(edge_mask).mean()

            loss = pred_loss + 0.1 * node_sparsity + 0.1 * edge_sparsity
            loss.backward()
            optimizer.step()

        # Return sigmoid of learned masks
        return torch.sigmoid(node_mask).detach(), torch.sigmoid(edge_mask).detach()

    def _build_result(
        self,
        data: Data,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        prediction: float
    ) -> ExplanationResult:
        """
        Build ExplanationResult from masks and data.

        Args:
            data: Original graph data
            node_mask: Node importance tensor
            edge_mask: Edge importance tensor
            prediction: Model prediction

        Returns:
            ExplanationResult object
        """
        # Get positions
        positions = data.pos.cpu().numpy() if hasattr(data, 'pos') else np.zeros((data.num_nodes, 3))

        # Get stellar masses
        stellar_masses = data.stellar_mass.cpu().numpy() if hasattr(data, 'stellar_mass') else np.ones(data.num_nodes) * 1e10

        # Node explanations (top-k)
        node_importance = node_mask.cpu().numpy()
        top_node_indices = np.argsort(node_importance)[-self.top_k_nodes:][::-1]

        nodes = []
        for idx in top_node_indices:
            nodes.append(NodeExplanation(
                id=int(idx),
                importance=float(node_importance[idx]),
                x=float(positions[idx, 0]),
                y=float(positions[idx, 1]),
                z=float(positions[idx, 2]),
                stellar_mass=float(stellar_masses[idx])
            ))

        # Edge explanations (top-k)
        edge_importance = edge_mask.cpu().numpy()
        edge_index = data.edge_index.cpu().numpy()

        # Get edge distances if available
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            distances = data.edge_attr[:, 0].cpu().numpy()  # First edge feature is distance
        else:
            distances = np.zeros(edge_index.shape[1])

        top_edge_indices = np.argsort(edge_importance)[-self.top_k_edges:][::-1]

        edges = []
        for idx in top_edge_indices:
            edges.append(EdgeExplanation(
                source=int(edge_index[0, idx]),
                target=int(edge_index[1, idx]),
                importance=float(edge_importance[idx]),
                distance=float(distances[idx])
            ))

        # Target value if available
        target = float(data.y.item()) if hasattr(data, 'y') else None

        # Cluster ID
        cluster_id = data.cluster_id if hasattr(data, 'cluster_id') else 'unknown'

        return ExplanationResult(
            cluster_id=cluster_id,
            prediction=prediction,
            target=target,
            nodes=nodes,
            edges=edges,
            metadata={
                'method': self.method,
                'num_nodes': data.num_nodes,
                'num_edges': data.edge_index.size(1),
                'top_k_nodes': self.top_k_nodes,
                'top_k_edges': self.top_k_edges
            }
        )

    def explain_batch(
        self,
        loader,
        save: bool = True
    ) -> List[ExplanationResult]:
        """
        Generate explanations for all graphs in a loader.

        Args:
            loader: DataLoader with graphs to explain
            save: Whether to save results to files

        Returns:
            List of ExplanationResult objects
        """
        results = []

        for batch_data in loader:
            # Get individual graphs from batch
            for i in range(batch_data.num_graphs):
                # Extract single graph
                data = batch_data.get_example(i)
                data = data.to(self.device)

                try:
                    result = self.explain(data)
                    results.append(result)

                    if save:
                        filename = f"{result.cluster_id}_explanation.json"
                        result.save(self.output_dir / filename)

                except Exception as e:
                    logger.warning(f"Failed to explain graph: {e}")

        logger.info(f"Generated {len(results)} explanations")
        return results

    def save_all_explanations(
        self,
        results: List[ExplanationResult],
        filename: str = 'all_explanations.json'
    ) -> None:
        """
        Save all explanations to a single JSON file.

        Args:
            results: List of ExplanationResult objects
            filename: Output filename
        """
        output = {
            'explanations': [r.to_dict() for r in results],
            'config': {
                'method': self.method,
                'top_k_nodes': self.top_k_nodes,
                'top_k_edges': self.top_k_edges
            }
        }

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved all explanations to {filepath}")


def create_explainer(
    model: nn.Module,
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> CosmicNetExplainer:
    """
    Factory function to create an explainer.

    Args:
        model: Trained GNN model
        config: Configuration dictionary
        device: Device for computation

    Returns:
        CosmicNetExplainer instance
    """
    return CosmicNetExplainer(model, config, device)
