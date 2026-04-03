import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loaders.tng_loader import TNGLoader
from graph.graph_builder import GraphBuilder
from model.model import build_model

def load_config():
    config = {
        'seed': 42,
        'data': {
            'source': 'tng',
            'grouping': 'fof',
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'batch_size': 16,
            'num_workers': 2,
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
        },
    }
    return config

def compute_scatter(y_true, y_pred):
    return np.std(y_pred - y_true)

def plot_model_comparison(preds_dict, targets, output_dir="outputs/paper_figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # 1. Prediction vs True Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
    
    for idx, (model_name, preds) in enumerate(preds_dict.items()):
        ax = axes[idx]
        
        # Calculate metrics
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        scatter = compute_scatter(targets, preds)
        
        # Plot density scatter
        sns.kdeplot(x=targets, y=preds, fill=True, cmap="mako" if idx==0 else "rocket", 
                    ax=ax, alpha=0.8, thresh=0.05)
        ax.scatter(targets, preds, s=10, color="navy" if idx==0 else "darkred", alpha=0.3)
        
        # Ideal line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        ax.set_title(f"{model_name}\n$R^2={r2:.3f}$, $RMSE={rmse:.3f}$ dex\nScatter $\sigma={scatter:.3f}$ dex")
        ax.set_xlabel(r"True $\log_{10}(M_{\rm halo} / M_\odot)$")
        if idx == 0:
            ax.set_ylabel(r"Predicted $\log_{10}(M_{\rm halo} / M_\odot)$")
            
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_scatter.pdf", dpi=300)
    plt.close()
    
    # 2. Residual Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e"]
    for idx, (model_name, preds) in enumerate(preds_dict.items()):
        res = preds - targets
        sns.histplot(res, bins=40, kde=True, stat="density", 
                     label=f"{model_name} ($\mu={np.mean(res):.3f}$)",
                     color=colors[idx], alpha=0.6, ax=ax)
    
    ax.axvline(0, color="black", linestyle="--", lw=2)
    ax.set_title("Residual Distribution: Predicted - True Halo Mass")
    ax.set_xlabel("Residual (dex)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_distribution.pdf", dpi=300)
    plt.close()

def generate_eda(df, output_dir="outputs/paper_figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    
    centrals = df[df['is_central'] == 1]
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Subhalo multiplicity
    ax1 = plt.subplot(2, 3, 1)
    nodes_per_group = df.groupby('group_id').size()
    sns.histplot(nodes_per_group, bins=range(0, max(nodes_per_group)+2), color="cadetblue", ax=ax1, discrete=True)
    ax1.set_xlabel("Galaxies per Halo")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Halo Multiplicity vs Graph Size")

    # 2. SHMR (Stellar to Halo Mass Relation)
    ax2 = plt.subplot(2, 3, 2)
    sns.kdeplot(x=centrals['stellar_mass'], y=centrals['halo_mass_log'], 
                cmap="flare", fill=True, ax=ax2, alpha=0.8)
    sns.scatterplot(x=centrals['stellar_mass'], y=centrals['halo_mass_log'], 
                    color="indianred", s=15, alpha=0.5, ax=ax2)
    ax2.set_xlabel(r"Central $\log_{10}(M_* / M_\odot)$")
    ax2.set_ylabel(r"True $\log_{10}(M_{\rm halo} / M_\odot)$")
    ax2.set_title("Stellar-to-Halo Mass Relation")
    
    # 3. Kinematics (Velocity Dispersion)
    ax3 = plt.subplot(2, 3, 3)
    sns.scatterplot(x=centrals['log_vel_dispersion'], y=centrals['halo_mass_log'], 
                    hue=centrals['stellar_mass'], palette="viridis", 
                    s=20, alpha=0.7, ax=ax3)
    ax3.set_xlabel(r"Central $\log_{10}(\sigma_v)$")
    ax3.set_ylabel(r"True $\log_{10}(M_{\rm halo} / M_\odot)$")
    ax3.set_title("Kinematics vs Halo Mass")

    # 4. Feature Correlations
    ax4 = plt.subplot(2, 3, 4)
    cols = ['stellar_mass', 'vel_dispersion', 'half_mass_radius', 'metallicity', 'halo_mass_log']
    labels = [r'$\log_{10}(M_*)$', r'$\sigma_v$', r'$R_{1/2}$', r'$Z$', r'$\log_{10}(M_{\rm halo})$']
    corr = centrals[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels, ax=ax4, vmin=-1, vmax=1)
    ax4.set_title("Feature Linear Correlation")
    
    # 5. Architecture placeholder (Mock plot for illustration)
    ax5 = plt.subplot(2, 3, 5)
    ax5.text(0.5, 0.5, "GNN Architecture\n"
                      r"$N_{layers} = 3, H = 64$""\n"
                      r"Inputs: $M_*, \sigma_v, R_{1/2}, Z$""\n"
                      r"Physics Loss: Virial Theorem ($\lambda_{max}=0.25$)", 
             fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_eda_figure.pdf", dpi=300)
    plt.close()

def main():
    print("Starting Paper Figures Generation Pipeline...")
    
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. EDA
    print("Generating EDA...")
    csv_path = cfg['data']['tng']['clustered_file']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        generate_eda(df)
        print("EDA generation complete!")
    else:
        print(f"Warning: CSV file not found at {csv_path}. Skipping EDA.")

    # 2. Graph Building & Loading
    print("Loading proper input features using Cosmic-Net Loader...")
    loader = TNGLoader(cfg)
    halos = loader.load()
    _, _, test_h = loader.split_data(halos)
    
    builder = GraphBuilder(cfg)
    test_graphs = builder.build_graphs(test_h)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    print(f"Created {len(test_graphs)} test graphs with exact sequence.")

    # 3. Evaluate both models
    model_paths = {
        'Baseline Best Model': 'kaggle/best_model (2).pt',
        'Augmented Best Model': 'kaggle/best_model_augmented.pt'
    }
    
    preds_dict = {}
    targets = None
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Model {path} missing, skipping...")
            continue
            
        print(f"Evaluating {name}...")
        try:
            model = build_model(cfg).to(device)
            checkpoint = torch.load(path, map_location=device)
            # The checkpoint might contain model_state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint) # Fallback if raw weights
                
            model.eval()
            
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    # Support output tuples from cosmic net model
                    out = model(batch)
                    if isinstance(out, tuple):
                        p = out[0]
                    else:
                        p = out
                    all_preds.append(p.cpu().numpy())
                    all_targets.append(batch.y.squeeze(-1).cpu().numpy())
            
            p_cat = np.concatenate(all_preds)
            t_cat = np.concatenate(all_targets)
            
            preds_dict[name] = p_cat
            targets = t_cat
            print(f" -> Result for {name}: R2 = {r2_score(t_cat, p_cat):.4f}")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    if preds_dict and targets is not None:
        plot_model_comparison(preds_dict, targets)
        print("Paper comparison graphs generated successfully in 'outputs/paper_figures'!")
    else:
        print("Failed to run model comparisons.")

if __name__ == "__main__":
    main()
