"""
TASK 8: PCA / t-SNE / UMAP Visualization

Visualize geometry of hidden states using dimensionality reduction.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, compute_lds, compute_cds, compute_rmsc, cosine_similarity_tokens,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS,
    GRID_H, GRID_W, NUM_PATCHES, HIDDEN_DIM, TIMESTEPS,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_all_shards(shard_dir):
    merged = {}
    for fp in sorted(Path(shard_dir).glob('*.pt')):
        data = load_shard(fp)
        merged.update(data['tensors'])
    return merged


def run_pca_rgb(raw, img_ids, output_dir, variant_name='raw'):
    """PCA-RGB visualization: project tokens to 3 PCA dims, reshape to grid, render as RGB."""
    print(f"\n  PCA-RGB for {variant_name}...")
    
    # Collect all tokens from all layers/timesteps/images
    all_tokens = []
    token_meta = []  # (img_id, layer, ts, patch_row, patch_col)
    
    for img_id in img_ids:
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                key = (img_id, layer_idx, ts_idx)
                if key in raw:
                    tokens = raw[key].float().numpy()  # (P, D)
                    all_tokens.append(tokens)
                    for p in range(NUM_PATCHES):
                        r, c = divmod(p, GRID_W)
                        token_meta.append((img_id, layer_idx, ts_idx, r, c))
    
    all_tokens = np.concatenate(all_tokens, axis=0)  # (N_total, D)
    print(f"  Total tokens: {all_tokens.shape[0]}")
    
    # Fit PCA on all tokens (use subsample if too many)
    if all_tokens.shape[0] > 500000:
        subsample_idx = np.random.choice(all_tokens.shape[0], 500000, replace=False)
        pca = PCA(n_components=3)
        pca.fit(all_tokens[subsample_idx])
    else:
        pca = PCA(n_components=3)
        pca.fit(all_tokens)
    
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Project and visualize for selected layers/timesteps
    layer_indices = [0, 13, 27]
    ts_indices = [0, 4, 9]
    
    for img_id in img_ids[:4]:  # First 4 images
        fig, axes = plt.subplots(len(layer_indices), len(ts_indices), 
                                  figsize=(4*len(ts_indices), 4*len(layer_indices)))
        fig.suptitle(f'PCA-RGB (img={img_id}, {variant_name})', fontsize=14)
        
        for r, li in enumerate(layer_indices):
            for c, ti in enumerate(ts_indices):
                key = (img_id, li, ti)
                if key in raw:
                    tokens = raw[key].float().numpy()
                    projected = pca.transform(tokens)  # (P, 3)
                    
                    # Normalize to [0, 1]
                    for ch in range(3):
                        vmin, vmax = projected[:, ch].min(), projected[:, ch].max()
                        if vmax > vmin:
                            projected[:, ch] = (projected[:, ch] - vmin) / (vmax - vmin)
                        else:
                            projected[:, ch] = 0.5
                    
                    rgb = projected.reshape(GRID_H, GRID_W, 3)
                    ax = axes[r, c]
                    ax.imshow(rgb)
                    ax.set_title(f'L{li+1}, t{ti}')
                    ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'pca_rgb_{variant_name}_img{img_id}.png', dpi=150)
        plt.close(fig)


def run_tsne(raw, img_ids, output_dir, variant_name='raw'):
    """t-SNE visualization at selected layers/timesteps."""
    print(f"\n  t-SNE for {variant_name}...")
    
    layer_indices = [0, 13, 27]
    ts_indices = [0, 4, 9]
    
    for li in layer_indices:
        for ti in ts_indices:
            all_tokens = []
            all_colors = []  # color by image
            all_positions = []  # color by patch position
            
            for img_id in img_ids[:8]:  # Use 8 images for speed
                key = (img_id, li, ti)
                if key in raw:
                    tokens = raw[key].float().numpy()
                    all_tokens.append(tokens)
                    all_colors.extend([img_id] * NUM_PATCHES)
                    for p in range(NUM_PATCHES):
                        r, c = divmod(p, GRID_W)
                        all_positions.append(r * GRID_W + c)
            
            if len(all_tokens) == 0:
                continue
            
            all_tokens = np.concatenate(all_tokens, axis=0)
            
            # PCA to 50 dims
            pca = PCA(n_components=min(50, all_tokens.shape[1]))
            tokens_pca = pca.fit_transform(all_tokens)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tokens_2d = tsne.fit_transform(tokens_pca)
            
            # Plot colored by position
            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(tokens_2d[:, 0], tokens_2d[:, 1], 
                                c=all_positions, cmap='viridis', s=5, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='Patch Position')
            ax.set_title(f't-SNE (L{li+1}, t{ti}, {variant_name})')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.tight_layout()
            fig.savefig(output_dir / f'tsne_{variant_name}_L{li+1}_t{ti}.png', dpi=150)
            plt.close(fig)


def run_umap_hidden_state(raw, common, img_ids, output_dir, variant_name='raw', is_mean=False):
    """UMAP on hidden-state-level descriptors."""
    print(f"\n  UMAP hidden-state for {variant_name}...")
    
    try:
        import umap
    except ImportError:
        print("  UMAP not available, skipping")
        return
    
    descriptors = []
    labels_layer = []
    labels_time = []
    
    for img_id in img_ids:
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                key = (img_id, layer_idx, ts_idx)
                common_key = (img_id, ts_idx) if is_mean else key
                
                if key not in raw:
                    continue
                
                A = raw[key]
                
                # Build descriptor: [LDS, CDS, RMSC, cos_to_common, mean, var]
                lds_val = compute_lds(A)
                cds_val = compute_cds(A)
                rmsc_val = compute_rmsc(A)
                
                if common_key in common:
                    cos_val = cosine_similarity_tokens(A.float(), common[common_key].float())
                else:
                    cos_val = 0.0
                
                A_f = A.float()
                mean_val = A_f.mean().item()
                var_val = A_f.var().item()
                
                descriptors.append([lds_val, cds_val, rmsc_val, cos_val, mean_val, var_val])
                labels_layer.append(layer_idx)
                labels_time.append(ts_idx)
    
    if len(descriptors) == 0:
        return
    
    descriptors = np.array(descriptors)
    labels_layer = np.array(labels_layer)
    labels_time = np.array(labels_time)
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(descriptors)
    
    # Plot colored by layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                              c=labels_layer, cmap='tab20', s=3, alpha=0.5)
    plt.colorbar(scatter, ax=axes[0], label='Layer')
    axes[0].set_title(f'UMAP Hidden-State (colored by layer, {variant_name})')
    
    scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                               c=labels_time, cmap='tab10', s=3, alpha=0.5)
    plt.colorbar(scatter2, ax=axes[1], label='Timestep')
    axes[1].set_title(f'UMAP Hidden-State (colored by timestep, {variant_name})')
    
    plt.tight_layout()
    fig.savefig(output_dir / f'umap_hiddenstate_{variant_name}.png', dpi=150)
    fig.savefig(output_dir / f'umap_hiddenstate_{variant_name}.pdf')
    plt.close(fig)


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task8'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 8: PCA / t-SNE / UMAP VISUALIZATION")
    print("="*60)
    
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    img_ids = sorted(set(k[0] for k in raw.keys()))[:16]  # 16 fixed images
    
    # PCA-RGB
    run_pca_rgb(raw, img_ids, output_dir, 'raw')
    
    # t-SNE
    run_tsne(raw, img_ids, output_dir, 'raw')
    
    # UMAP hidden-state (need common for descriptors)
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    run_umap_hidden_state(raw, common_mean, img_ids, output_dir, 'mean', is_mean=True)
    del common_mean
    
    # Residual visualizations
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    run_pca_rgb(res_mean, img_ids, output_dir, 'res_mean')
    run_tsne(res_mean, img_ids, output_dir, 'res_mean')
    del res_mean
    
    # tSVD variants
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        run_pca_rgb(res_k, img_ids, output_dir, f'res_tsvd_K{K}')
        run_umap_hidden_state(raw, common_k, img_ids, output_dir, f'tsvd_K{K}')
        del common_k, res_k
    
    print("\n  ★ Task 8 complete")


if __name__ == '__main__':
    main()
