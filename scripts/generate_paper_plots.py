import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.tng_loader import TNGLoader
from graph.graph_builder import GraphBuilder
from model.model import build_model

def load_config():
    # Use exact dictionary to bypass yaml dependency issues if they arise
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

def setup_plotting():
    # Set publication-ready plot style
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
    })

def plot_dataset_eda(df, out_dir):
    print("Generating Dataset EDA Plots...")
    centrals = df[df['is_central'] == 1]
    
    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    features = [
        ('stellar_mass', r'$\log_{10}(M_* / M_\odot)$', 'teal'),
        ('log_vel_dispersion', r'$\log_{10}(\sigma_v / \mathrm{km\,s}^{-1})$', 'coral'),
        ('log_half_mass_radius', r'$\log_{10}(R_{1/2} / \mathrm{kpc})$', 'mediumseagreen'),
        ('log_metallicity', r'$\log_{10}(Z)$', 'orchid')
    ]
    for ax, (col, label, color) in zip(axes.flatten(), features):
        if col in df.columns:
            sns.histplot(df[col], bins=40, kde=True, color=color, ax=ax, edgecolor="black", alpha=0.7)
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_feature_distributions.png", dpi=300)
    plt.close()

    # 2. Target Distribution (Halo Mass)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(centrals['halo_mass_log'], bins=30, kde=True, color="darkslateblue", ax=ax, edgecolor="black")
    ax.set_xlabel(r"Target Value: $\log_{10}(M_{\rm halo} / M_\odot)$")
    ax.set_ylabel("Number of Halos")
    ax.set_title("Halo Mass Function (Dataset Targets)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_target_distribution.png", dpi=300)
    plt.close()

    # 3. SHMR (Stellar to Halo Mass Relation) with Density
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hexbin plot to handle density of points well
    hb = ax.hexbin(centrals['stellar_mass'], centrals['halo_mass_log'], 
                   gridsize=30, cmap='YlOrRd', mincnt=1)
    cb = fig.colorbar(hb, ax=ax, label='Count in bin')
    
    # Binned median line
    bins = np.linspace(centrals['stellar_mass'].min(), centrals['stellar_mass'].max(), 12)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    means, _, _ = stats.binned_statistic(centrals['stellar_mass'], centrals['halo_mass_log'], statistic='mean', bins=bins)
    stds, _, _ = stats.binned_statistic(centrals['stellar_mass'], centrals['halo_mass_log'], statistic='std', bins=bins)
    ax.errorbar(bin_centers, means, yerr=stds, fmt='o-', color='black', capsize=4, linewidth=2, label='Binned Mean $\pm 1\sigma$')
    
    ax.set_xlabel(r"Central Galaxy $\log_{10}(M_* / M_\odot)$")
    ax.set_ylabel(r"True Halo Mass $\log_{10}(M_{\rm halo} / M_\odot)$")
    ax.set_title("Stellar-to-Halo Mass Relation (SHMR)")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_shmr_relation.png", dpi=300)
    plt.close()

    # 4. Multiplicity & Subhalos Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    nodes_per_group = df.groupby('group_id').size()
    sns.histplot(nodes_per_group, discrete=True, color="steelblue", edgecolor="black", alpha=0.8, ax=ax)
    ax.set_xlabel("Number of Subhalos per Halo (Graph Size)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Graph Sizes")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_graph_sizes.png", dpi=300)
    plt.close()
    
    # 5. Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cols_corr = ['stellar_mass', 'vel_dispersion', 'half_mass_radius', 'metallicity', 'halo_mass_log']
    labels_corr = [r'$\log_{10}(M_*)$', r'$\sigma_v$', r'$R_{1/2}$', r'$Z$', r'$\log_{10}(M_{\rm halo})$']
    if all(c in centrals.columns for c in cols_corr):
        corr_matrix = centrals[cols_corr].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="vlag", fmt=".2f",
                    xticklabels=labels_corr, yticklabels=labels_corr, ax=ax,
                    square=True, cbar_kws={'shrink': .8})
        ax.set_title("Feature Correlations (Central Galaxies)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/5_correlation_matrix.png", dpi=300)
    plt.close()

def plot_model_evaluations(preds_dict, targets, out_dir):
    print("Generating Model Evaluation Plots...")
    # 6. Scatter Pred vs True Hexbins
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(7 * len(preds_dict), 6), sharey=True, sharex=True)
    if len(preds_dict) == 1:
        axes = [axes]
        
    for ax, (model_name, preds) in zip(axes, preds_dict.items()):
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        scatter = np.std(preds - targets)
        
        hb = ax.hexbin(targets, preds, gridsize=30, cmap='mako_r', mincnt=1)
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, zorder=3)
        
        ax.set_title(f"{model_name}\n$R^2={r2:.3f}$ | $RMSE={rmse:.3f}$ dex | Scatter={scatter:.3f} dex")
        ax.set_xlabel(r"True Halo Mass $\log_{10}(M_{\rm halo} / M_\odot)$")
        if ax == axes[0]:
            ax.set_ylabel(r"Predicted Halo Mass $\log_{10}(M_{\rm halo} / M_\odot)$")
            
    fig.colorbar(hb, ax=axes, label='Density', pad=0.02)
    plt.savefig(f"{out_dir}/6_predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Residuals Histogram
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2a9d8f", "#e76f51", "#e9c46a"]
    for idx, (model_name, preds) in enumerate(preds_dict.items()):
        res = preds - targets
        sns.kdeplot(res, fill=True, label=f"{model_name} ($\mu={np.mean(res):.3f}$)",
                    color=colors[idx % len(colors)], alpha=0.5, ax=ax, lw=2)
    
    ax.axvline(0, color="black", linestyle="--", lw=2)
    ax.set_title("Prediction Residuals Distribution")
    ax.set_xlabel("Residual (dex) [Predicted - True]")
    ax.set_ylabel("Probability Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/7_residuals_kde.png", dpi=300)
    plt.close()
    
    # 8. Binned Bias and Scatter (Crucial for astrophysical scaling relations)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    bins = np.linspace(targets.min(), targets.max(), 10)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    for idx, (model_name, preds) in enumerate(preds_dict.items()):
        res = preds - targets
        
        bias, _, _ = stats.binned_statistic(targets, res, statistic='mean', bins=bins)
        scatter, _, _ = stats.binned_statistic(targets, res, statistic='std', bins=bins)
        
        # Bias Plot
        axes[0].plot(bin_centers, bias, marker='o', lw=2, color=colors[idx % len(colors)], label=model_name)
        # Scatter Plot
        axes[1].plot(bin_centers, scatter, marker='s', lw=2, color=colors[idx % len(colors)], label=model_name)
        
    axes[0].axhline(0, color='black', linestyle='--', lw=1.5)
    axes[0].set_ylabel(r"Bias ($\mu_{\rm pred} - \mu_{\rm true}$) [dex]")
    axes[0].set_title("Binned Error Analysis")
    axes[0].legend()
    
    axes[1].set_ylabel(r"Scatter ($\sigma_{\rm residual}$) [dex]")
    axes[1].set_xlabel(r"True Halo Mass $\log_{10}(M_{\rm halo} / M_\odot)$")
    axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/8_binned_bias_scatter.png", dpi=300)
    plt.close()

def main():
    print("="*50)
    print("Cosmic-Net Paper Graph Generator")
    print("="*50)
    
    out_dir = "outputs/research_paper_plots"
    os.makedirs(out_dir, exist_ok=True)
    setup_plotting()
    cfg = load_config()
    
    csv_path = cfg['data']['tng']['clustered_file']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        plot_dataset_eda(df, out_dir)
    else:
        print(f"ERROR: Dataset not found at {csv_path}. Skipping EDA.")
        
    print("\nLoading test set to evaluate models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = TNGLoader(cfg)
    halos = loader.load()
    _, _, test_h = loader.split_data(halos)
    
    builder = GraphBuilder(cfg)
    test_graphs = builder.build_graphs(test_h)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    model_paths = {
        'Baseline Model': 'kaggle/best_model (2).pt',
        'Augmented Model': 'kaggle/best_model_augmented.pt'
    }
    
    preds_dict = {}
    targets = None
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"> Evaluating {name} from {path}...")
            model = build_model(cfg).to(device)
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint) 
            model.eval()
            
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    p = out[0] if isinstance(out, tuple) else out
                    all_preds.append(p.cpu().numpy())
                    all_targets.append(batch.y.squeeze(-1).cpu().numpy())
            
            preds_dict[name] = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
        else:
            print(f"> Warning: {path} not found.")
            
    if preds_dict and targets is not None:
        plot_model_evaluations(preds_dict, targets, out_dir)
        
    print(f"\nSUCCESS: All publication plots have been saved to '{out_dir}/'")

if __name__ == "__main__":
    main()