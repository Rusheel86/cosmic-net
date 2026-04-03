import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
import torch.nn.functional as F

# PyG imports
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.tng_loader import TNGLoader
from graph.graph_builder import GraphBuilder
from model.model import build_model

def load_config():
    config = {
        'seed': 42,
        'data': {
            'tng': {
                'clustered_file': 'data/raw/tng100_clustered.csv',
                'min_subhalos_per_halo': 3,
                'snapshot': 99,
                'n_halos': 600,
                'cache_dir': 'data/raw/tng_cache',
            }
        },
        'graph': {
            'method': 'radius',
            'radius_mpc': 2.0,
            'k_neighbors': 8,
            'self_loops': True,
            'edge_features': ['distance','delta_v','cos_theta','mass_ratio','proj_sep'],
            'hierarchical': False,
        },
        'model': {
            'node_features': 4,
            'edge_features': 5,
            'hidden_dim': 64,
            'output_dim': 64,
            'num_layers': 3,
            'dropout': 0.05,
            'pooling': 'mean',
            'residual': True,
            'mc_dropout': True,
            'mc_samples': 30,
            'activation': 'leaky_relu',
        }
    }
    return config

def setup_plotting():
    # Set publication-ready plot style
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.5,
    })

def custom_train_pgexplainer(model, explainer, data_loader, device, epochs=10, lr=0.003):
    """
    Standard PyG 2.0+ explainer training loop for Saliency Mapping.
    """
    print(f"Training Saliency Mapping for {epochs} epochs...")
    for epoch in range(epochs):
        loss_total = 0.0
        for batch in data_loader:
            batch = batch.to(device)
            # Some versions of Saliency Mapping support train on batch directly
            # Alternatively we iterate through graph items
            for i in range(batch.num_graphs):
                g = batch[i]
                loss = explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=g.x,
                    edge_index=g.edge_index,
                    target=g.y,
                    edge_attr=g.edge_attr
                )
                loss_total += loss
    print(f"Done Training Saliency Mapping.")
    

class WrapModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, edge_index, **kwargs):
        from torch_geometric.data import Data, Batch
        import torch
        data = Data(x=x, edge_index=edge_index)
        if 'edge_attr' in kwargs:
            data.edge_attr = kwargs['edge_attr']
        
        batch_idx = kwargs.get('batch', None)
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data.batch = batch_idx
        
        # Saliency Mapping passes edge_weight sometimes?
        
        batch_obj = Batch.from_data_list([data]) if batch_idx.max() == 0 else data
        
        # Add physics loss required attributes (fake) if they are accessed
        batch_obj.stellar_mass = torch.ones(x.size(0), device=x.device) * 1e10
        batch_obj.vel_disp = torch.ones(x.size(0), device=x.device) * 100
        batch_obj.half_mass_r = torch.ones(x.size(0), device=x.device) * 0.01
        
        out, _ = self.m(batch_obj, return_embeddings=False)
        return out.view(-1)


def generate_saliency_explanations(model_path, model_name, cfg, test_loader, device, out_dir):
    base_model = build_model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        base_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        base_model.load_state_dict(checkpoint)
    base_model.eval()

    print(f"--- Generating Integrated Saliency for {model_name} ---")
    edge_importance = np.zeros(5) # 5 edge features
    total_edges = 0

    for batch in tqdm(test_loader, desc=f"Explaining target test batch"):
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            g = batch[i].clone().detach().to(device)
            
            # We track gradients specifically on edge features!
            test_x = g.x.clone().detach().to(device)
            test_edge_idx = g.edge_index.clone().detach().to(device)
            test_edge_attr = g.edge_attr.clone().detach().to(device).requires_grad_(True)
            
            base_model.zero_grad()
            batch_obj = Batch.from_data_list([Data(x=test_x, edge_index=test_edge_idx, edge_attr=test_edge_attr)])
            
            # Dummy physics constants
            batch_obj.stellar_mass = torch.ones(test_x.size(0), device=device) * 1e10
            batch_obj.vel_disp = torch.ones(test_x.size(0), device=device) * 100
            batch_obj.half_mass_r = torch.ones(test_x.size(0), device=device) * 0.01

            out, _ = base_model(batch_obj, return_embeddings=False)
            
            # Propagate gradients backwards from halo prediction
            out.sum().backward()

            if test_edge_attr.grad is not None:
                # Saliency strategy (gradient absolute magnitude)
                saliency = test_edge_attr.grad.abs()
                edge_importance += saliency.sum(dim=0).cpu().numpy()
                total_edges += test_edge_attr.size(0)

    return edge_importance / (total_edges + 1e-9)


def main():
    print("="*50)
    print("Research Paper XAI Generator (Saliency Mapping)")
    print("="*50)
    
    out_dir = "outputs/xai_results"
    os.makedirs(out_dir, exist_ok=True)
    setup_plotting()
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = TNGLoader(cfg)
    halos = loader.load()
    train_h, val_h, test_h = loader.split_data(halos)
    
    builder = GraphBuilder(cfg)
    train_graphs = builder.build_graphs(train_h)
    test_graphs = builder.build_graphs(test_h)
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    model_paths = {
        'Baseline': 'kaggle/best_model (2).pt',
        'Augmented': 'kaggle/best_model_augmented.pt'
    }
    
    results = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            e_imp = generate_saliency_explanations(path, name, cfg, test_loader, device, out_dir)
            results[name] = {"edge_importance": e_imp}
        else:
            print(f"Skipping {name}: model not found.")

    # Print / Save Table
    edge_labels = ['Distance', 'Delta_v', 'Cos_theta', 'Mass_Ratio', 'Proj_Sep']
    
    if results:
        # Save XAI Table as LaTeX and Markdown
        headers = ["Model"] + edge_labels
        table_data = []
        for name, data in results.items():
            row = [name]
            edge_vals = data["edge_importance"]
            # Normalize to sum to 100% for readability
            edge_vals = (np.abs(edge_vals) / (np.sum(np.abs(edge_vals)) + 1e-9)) * 100
            row.extend([f"{v:.2f}%" for v in edge_vals])
            table_data.append(row)
            
        md_table = tabulate(table_data, headers, tablefmt="github")
        latex_table = tabulate(table_data, headers, tablefmt="latex_booktabs")
        
        with open(f"{out_dir}/saliency_analysis.md", "w") as f:
            f.write(f"## Subhalo Kinematics Importance (Saliency Mapping)\n\n{md_table}\n\n### LaTeX Format\n```latex\n{latex_table}\n```")
            
        print("\nGenerated XAI Analysis Table:")
        print(md_table)

        # Plot XAI Bar chart for Edges
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar_width = 0.35
        x = np.arange(len(edge_labels))
        
        base_vals = np.abs(results['Baseline']['edge_importance'])
        base_vals = (base_vals / (base_vals.sum() + 1e-9)) * 100
        aug_vals = np.abs(results['Augmented']['edge_importance'])
        aug_vals = (aug_vals / (aug_vals.sum() + 1e-9)) * 100
        
        ax.bar(x - bar_width/2, base_vals, bar_width, label='Baseline Model', color='steelblue', edgecolor='black')
        ax.bar(x + bar_width/2, aug_vals, bar_width, label='Augmented Model', color='coral', edgecolor='black')
        
        ax.set_ylabel('Edge Feature Influence via Mask (%)')
        ax.set_title('Subhalo Interaction Importance (Saliency Mapping)')
        ax.set_xticks(x)
        ax.set_xticklabels(edge_labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/saliency_edge_importance.png", dpi=300)
        plt.close()

        
        print(f"\nSUCCESS: XAI Visualizations saved to '{out_dir}/'")

if __name__ == "__main__":
    main()