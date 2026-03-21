"""
visualize.py
Purpose: Matplotlib 3D visualization of galaxy cluster graphs and Pareto frontiers.
         Creates publication-ready figures for node/edge importance and equation discovery.
Inputs: data: PyG Data object with pos Tensor[N,3], edge_index Tensor[2,E]
        masks: dict with node_importance and edge_importance arrays
        config (dict) - Configuration dictionary from config.yaml
Outputs: PNG files saved to outputs/explanations/ and outputs/equations/
Config keys: explain.output_dir, symbolic.output_dir
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


def plot_cluster(
    data: Any,
    masks: Dict[str, Any],
    cluster_id: str,
    config: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Create 3D visualization of a galaxy cluster graph with importance masks.

    Args:
        data: PyG Data object with pos Tensor[N,3], edge_index Tensor[2,E]
        masks: Dict with 'nodes' and 'edges' lists containing importance scores
        cluster_id: Unique identifier for the cluster
        config: Configuration dictionary from config.yaml
        save_path: Optional override for save path

    Returns:
        Path to saved figure
    """
    # Get output directory from config
    explain_config = config.get('explain', {})
    output_dir = Path(explain_config.get('output_dir', 'outputs/explanations'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path = output_dir / f"{cluster_id}_plot.png"
    else:
        save_path = Path(save_path)

    # Extract positions
    if hasattr(data, 'pos') and data.pos is not None:
        positions = data.pos.cpu().numpy()
    else:
        logger.warning("No positions in data, using random layout")
        num_nodes = data.x.shape[0] if hasattr(data, 'x') else 10
        positions = np.random.randn(num_nodes, 3)

    # Extract edge index
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edge_index = data.edge_index.cpu().numpy()
    else:
        edge_index = np.array([[], []])

    # Extract node importance scores
    node_importance = np.zeros(len(positions))
    if 'nodes' in masks:
        for node_info in masks['nodes']:
            node_id = node_info.get('id', 0)
            importance = node_info.get('importance', 0.0)
            if node_id < len(node_importance):
                node_importance[node_id] = importance

    # Normalize node importance to [0, 1]
    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()

    # Extract edge importance scores
    num_edges = edge_index.shape[1] if edge_index.shape[1] > 0 else 0
    edge_importance = np.zeros(num_edges)
    if 'edges' in masks:
        edge_map = {}  # Map (src, tgt) to importance
        for edge_info in masks['edges']:
            src = edge_info.get('source', 0)
            tgt = edge_info.get('target', 0)
            importance = edge_info.get('importance', 0.0)
            edge_map[(src, tgt)] = importance
            edge_map[(tgt, src)] = importance  # Symmetric

        for i in range(num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            edge_importance[i] = edge_map.get((src, tgt), 0.0)

    # Normalize edge importance to [0, 1]
    if edge_importance.max() > 0:
        edge_importance = edge_importance / edge_importance.max()

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot edges with importance coloring
    if num_edges > 0:
        edge_segments = []
        edge_colors = []
        cmap = cm.plasma

        for i in range(num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            if src < len(positions) and tgt < len(positions):
                p1 = positions[src]
                p2 = positions[tgt]
                edge_segments.append([p1, p2])
                edge_colors.append(cmap(edge_importance[i]))

        if edge_segments:
            edge_collection = Line3DCollection(
                edge_segments,
                colors=edge_colors,
                linewidths=1.0 + 2.0 * edge_importance,
                alpha=0.3 + 0.5 * edge_importance
            )
            ax.add_collection3d(edge_collection)

    # Plot nodes with importance coloring
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=node_importance,
        cmap='plasma',
        s=50 + 150 * node_importance,
        alpha=0.8,
        edgecolors='white',
        linewidths=0.5
    )

    # Colorbar for node importance
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Node Importance', fontsize=12)

    # Labels and title
    ax.set_xlabel('X (Mpc)', fontsize=11)
    ax.set_ylabel('Y (Mpc)', fontsize=11)
    ax.set_zlabel('Z (Mpc)', fontsize=11)
    ax.set_title(f'Galaxy Cluster: {cluster_id}\n'
                 f'Nodes: {len(positions)}, Edges: {num_edges}',
                 fontsize=14, fontweight='bold')

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved cluster visualization to {save_path}")
    return str(save_path)


def plot_pareto(
    pareto_df: pd.DataFrame,
    config: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Plot Pareto frontier of symbolic regression results.

    Args:
        pareto_df: DataFrame with columns [complexity, rmse, equation]
        config: Configuration dictionary from config.yaml
        save_path: Optional override for save path

    Returns:
        Path to saved figure
    """
    # Get output directory from config
    symbolic_config = config.get('symbolic', {})
    output_dir = Path(symbolic_config.get('output_dir', 'outputs/equations'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path = output_dir / 'pareto_front.png'
    else:
        save_path = Path(save_path)

    # Validate DataFrame columns
    required_cols = ['complexity', 'rmse']
    if not all(col in pareto_df.columns for col in required_cols):
        logger.error(f"DataFrame must have columns: {required_cols}")
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    # Sort by complexity for Pareto front line
    df_sorted = pareto_df.sort_values('complexity')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all points
    scatter = ax.scatter(
        df_sorted['complexity'],
        df_sorted['rmse'],
        c=df_sorted['rmse'],
        cmap='viridis_r',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    # Plot Pareto front line
    pareto_points = _compute_pareto_front(df_sorted[['complexity', 'rmse']].values)
    if len(pareto_points) > 1:
        pareto_sorted = pareto_points[pareto_points[:, 0].argsort()]
        ax.plot(
            pareto_sorted[:, 0],
            pareto_sorted[:, 1],
            'r--',
            linewidth=2,
            alpha=0.8,
            label='Pareto Front'
        )

    # Annotate top 3 equations (lowest RMSE)
    top_3 = df_sorted.nsmallest(3, 'rmse')
    for i, (idx, row) in enumerate(top_3.iterrows()):
        equation = row.get('equation', f'Eq {i+1}')
        # Truncate long equations
        if len(str(equation)) > 40:
            equation = str(equation)[:37] + "..."

        ax.annotate(
            f"#{i+1}: {equation}",
            xy=(row['complexity'], row['rmse']),
            xytext=(10 + i * 5, -20 - i * 15),
            textcoords='offset points',
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
        )

    # Highlight best equation (lowest RMSE)
    best = df_sorted.loc[df_sorted['rmse'].idxmin()]
    ax.scatter(
        [best['complexity']],
        [best['rmse']],
        c='red',
        s=200,
        marker='*',
        edgecolors='black',
        linewidths=1,
        zorder=5,
        label=f"Best (RMSE={best['rmse']:.4f})"
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('RMSE', fontsize=12)

    # Labels and title
    ax.set_xlabel('Equation Complexity (number of nodes)', fontsize=12)
    ax.set_ylabel('RMSE (log₁₀ M_halo)', fontsize=12)
    ax.set_title('Symbolic Regression: Pareto Frontier\n'
                 'Complexity vs Accuracy Trade-off',
                 fontsize=14, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(0, df_sorted['complexity'].max() * 1.1)
    ax.set_ylim(0, df_sorted['rmse'].max() * 1.1)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved Pareto front plot to {save_path}")
    return str(save_path)


def _compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Compute Pareto front from 2D points (minimizing both dimensions).

    Args:
        points: Array of shape [N, 2] with (complexity, rmse)

    Returns:
        Array of Pareto-optimal points
    """
    pareto_points = []
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    min_rmse = float('inf')
    for point in sorted_points:
        if point[1] < min_rmse:
            pareto_points.append(point)
            min_rmse = point[1]

    return np.array(pareto_points) if pareto_points else points[:1]


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    lambda_values: List[float],
    config: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Plot training curves with loss and lambda annealing.

    Args:
        train_losses: List of training MSE per epoch
        val_losses: List of validation MSE per epoch
        lambda_values: List of lambda values per epoch
        config: Configuration dictionary
        save_path: Optional override for save path

    Returns:
        Path to saved figure
    """
    output_dir = Path(config.get('training', {}).get('checkpoint_dir', 'outputs/checkpoints'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path = output_dir / 'training_curves.png'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train MSE', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val MSE', linewidth=2)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Lambda annealing
    ax2.plot(epochs, lambda_values, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Lambda (λ)', fontsize=12)
    ax2.set_title('Physics Loss Weight Annealing', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(lambda_values) * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved training curves to {save_path}")
    return str(save_path)


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    config: Dict[str, Any] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Plot predicted vs true halo masses with uncertainty.

    Args:
        predictions: Predicted log10(M_halo) [N]
        targets: True log10(M_halo) [N]
        uncertainties: Optional standard deviations [N]
        config: Configuration dictionary
        save_path: Optional override for save path

    Returns:
        Path to saved figure
    """
    config = config or {}
    output_dir = Path(config.get('training', {}).get('checkpoint_dir', 'outputs/checkpoints'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path = output_dir / 'prediction_scatter.png'

    fig, ax = plt.subplots(figsize=(10, 10))

    # Identity line
    min_val = min(predictions.min(), targets.min())
    max_val = max(predictions.max(), targets.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')

    # Scatter with optional error bars
    if uncertainties is not None:
        ax.errorbar(
            targets, predictions,
            yerr=1.96 * uncertainties,
            fmt='o',
            markersize=6,
            alpha=0.6,
            elinewidth=1,
            capsize=2,
            label='Predictions with 95% CI'
        )
    else:
        ax.scatter(targets, predictions, alpha=0.6, s=50, label='Predictions')

    # Labels
    ax.set_xlabel('True log₁₀(M_halo / M☉)', fontsize=12)
    ax.set_ylabel('Predicted log₁₀(M_halo / M☉)', fontsize=12)
    ax.set_title('Halo Mass Prediction: Predicted vs True', fontsize=14, fontweight='bold')

    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    scatter = np.std(predictions - targets)

    # Add metrics text box
    textstr = f'RMSE = {rmse:.4f}\nR² = {r2:.4f}\nScatter = {scatter:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved prediction scatter to {save_path}")
    return str(save_path)
