"""
CosmicGNN — exact architecture matching best_model_augmented.pt

Node features (4): log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity
Edge features (5): distance, delta_v, cos_theta, mass_ratio, proj_sep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool


class EdgeMLP(nn.Module):
    def __init__(self, edge_features: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class EdgeConditionedBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_features: int, dropout: float):
        super().__init__()
        self.edge_mlp = EdgeMLP(edge_features, hidden_dim)
        self.conv = NNConv(
            hidden_dim,
            hidden_dim,
            self.edge_mlp,
            aggr="mean",
        )
        self.norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual


class CosmicGNN(nn.Module):
    def __init__(
        self,
        num_node_features: int = 4,
        edge_features: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.conv_blocks = nn.ModuleList([
            EdgeConditionedBlock(hidden_dim, edge_features, dropout)
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)
        for block in self.conv_blocks:
            x = block(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.pred_head(x).squeeze(-1)
