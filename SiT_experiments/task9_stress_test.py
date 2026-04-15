"""
TASK 9: Spatial Stress Test

Test whether common representation depends on spatial layout 
using patch shuffle and block shuffle.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_latents_from_arrayrecord, load_sit_model,
    extract_activations, l2_normalize_tokens,
    build_mean_common, build_tsvd_common,
    compute_lds, compute_cds, compute_rmsc,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, TIMESTEPS, SVD_RANKS,
    GRID_H, GRID_W, NUM_PATCHES, SEED, INPUT_SIZE, HIDDEN_DIM,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def shuffle_patches_in_latent(x, mode='patch', seed=42):
    """
    Shuffle patches in the latent space.
    x: (4, 32, 32) latent tensor
    mode: 'patch' for full random shuffle, 'block' for block shuffle
    """
    rng = np.random.RandomState(seed)
    C, H, W = x.shape
    patch_size = 2  # SiT patch size
    
    if mode == 'patch':
        # Reshape to patches, shuffle, reshape back
        n_h = H // patch_size
        n_w = W // patch_size
        # (C, n_h, patch_size, n_w, patch_size) -> (n_h*n_w, C*patch_size*patch_size)
        patches = x.reshape(C, n_h, patch_size, n_w, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(n_h * n_w, C * patch_size * patch_size)
        # Shuffle
        idx = rng.permutation(n_h * n_w)
        patches = patches[idx]
        # Reshape back
        patches = patches.reshape(n_h, n_w, C, patch_size, patch_size)
        x_shuffled = patches.permute(2, 0, 3, 1, 4).reshape(C, H, W)
        return x_shuffled
    
    elif mode == 'block':
        # Block shuffle with 4x4 grid
        block_h = H // 4
        block_w = W // 4
        blocks = []
        for i in range(4):
            for j in range(4):
                block = x[:, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w].clone()
                blocks.append(block)
        
        idx = rng.permutation(16)
        x_shuffled = x.clone()
        for new_pos, old_pos in enumerate(idx):
            ni, nj = divmod(new_pos, 4)
            x_shuffled[:, ni*block_h:(ni+1)*block_h, nj*block_w:(nj+1)*block_w] = blocks[old_pos]
        return x_shuffled


def extract_and_compute_metrics(model, x, label, device='cpu'):
    """Extract activations and compute spatial metrics for all layers/timesteps."""
    metrics = {}
    atilde_all = {}
    
    for ts_idx, t_val in enumerate(TIMESTEPS):
        A_patch, block_tokens = extract_activations(model, x, t_val, label, device)
        
        for layer_idx in range(NUM_LAYERS):
            A = block_tokens[layer_idx]
            A_tilde = l2_normalize_tokens(A)
            atilde_all[(layer_idx, ts_idx)] = A_tilde
            
            lds = compute_lds(A)
            cds = compute_cds(A)
            rmsc = compute_rmsc(A)
            metrics[(layer_idx, ts_idx)] = {'lds': lds, 'cds': cds, 'rmsc': rmsc}
    
    # Build mean-common
    for ts_idx in range(NUM_TIMESTEPS):
        layers = [atilde_all[(l, ts_idx)] for l in range(NUM_LAYERS)]
        A_common, B_res = build_mean_common(layers)
        
        lds_c = compute_lds(A_common)
        cds_c = compute_cds(A_common)
        rmsc_c = compute_rmsc(A_common)
        metrics[('common_mean', ts_idx)] = {'lds': lds_c, 'cds': cds_c, 'rmsc': rmsc_c}
    
    return metrics


def main():
    output_dir = RESULTS_DIR / 'task9'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 9: SPATIAL STRESS TEST")
    print("="*60)
    
    # Load model
    model = load_sit_model(device='cpu')
    
    # Load 100 control images
    data_dir = '/home/thanhnamngo26/gcs-bucket/imagenet_moments'
    latents, labels = load_latents_from_arrayrecord(data_dir, 100, seed=SEED + 1000)
    
    rows = []
    
    for img_idx in tqdm(range(len(latents)), desc="Stress test"):
        x = latents[img_idx]
        label = labels[img_idx].item()
        
        # Original
        metrics_orig = extract_and_compute_metrics(model, x, label)
        
        # Patch shuffle
        x_ps = shuffle_patches_in_latent(x, mode='patch', seed=SEED + img_idx)
        metrics_ps = extract_and_compute_metrics(model, x_ps, label)
        
        # Block shuffle
        x_bs = shuffle_patches_in_latent(x, mode='block', seed=SEED + img_idx + 10000)
        metrics_bs = extract_and_compute_metrics(model, x_bs, label)
        
        # Compute deltas (averaged over all layers and timesteps)
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                key = (layer_idx, ts_idx)
                rows.append({
                    'image_id': img_idx,
                    'layer': layer_idx + 1,
                    'timestep_idx': ts_idx,
                    # Original
                    'orig_lds': metrics_orig[key]['lds'],
                    'orig_cds': metrics_orig[key]['cds'],
                    'orig_rmsc': metrics_orig[key]['rmsc'],
                    # Patch shuffle
                    'ps_lds': metrics_ps[key]['lds'],
                    'ps_cds': metrics_ps[key]['cds'],
                    'ps_rmsc': metrics_ps[key]['rmsc'],
                    # Block shuffle
                    'bs_lds': metrics_bs[key]['lds'],
                    'bs_cds': metrics_bs[key]['cds'],
                    'bs_rmsc': metrics_bs[key]['rmsc'],
                    # Deltas
                    'delta_ps_lds': metrics_orig[key]['lds'] - metrics_ps[key]['lds'],
                    'delta_bs_lds': metrics_orig[key]['lds'] - metrics_bs[key]['lds'],
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'task9_stress_results.csv', index=False)
    
    # Summary
    summary = df.groupby('layer').agg({
        'orig_lds': 'mean', 'ps_lds': 'mean', 'bs_lds': 'mean',
        'delta_ps_lds': 'mean', 'delta_bs_lds': 'mean',
    }).reset_index()
    summary.to_csv(output_dir / 'task9_delta.csv', index=False)
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(summary['layer'], summary['orig_lds'], 'b-o', label='Original', markersize=3)
    axes[0].plot(summary['layer'], summary['ps_lds'], 'r-s', label='Patch Shuffle', markersize=3)
    axes[0].plot(summary['layer'], summary['bs_lds'], 'g-^', label='Block Shuffle', markersize=3)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('LDS')
    axes[0].set_title('LDS: Original vs Shuffled')
    axes[0].legend()
    
    axes[1].plot(summary['layer'], summary['delta_ps_lds'], 'r-s', label='Patch Shuffle Δ', markersize=3)
    axes[1].plot(summary['layer'], summary['delta_bs_lds'], 'g-^', label='Block Shuffle Δ', markersize=3)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('LDS Drop')
    axes[1].set_title('Metric Drop from Shuffling')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'figure_T22_stress.png', dpi=150)
    fig.savefig(output_dir / 'figure_T22_stress.pdf')
    plt.close(fig)
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    mean_delta_ps = df['delta_ps_lds'].mean()
    mean_delta_bs = df['delta_bs_lds'].mean()
    
    if mean_delta_ps > 0:
        print(f"  ✓ Patch shuffle reduces LDS (mean Δ = {mean_delta_ps:.4f})")
    else:
        print(f"  ✗ Patch shuffle does not reduce LDS — FAIL")
    
    if mean_delta_bs > 0:
        print(f"  ✓ Block shuffle reduces LDS (mean Δ = {mean_delta_bs:.4f})")
    else:
        print(f"  ✗ Block shuffle does not reduce LDS — FAIL")
    
    if abs(mean_delta_ps - mean_delta_bs) > 0.001:
        print(f"  ✓ Patch and block shuffle differ (Δ_ps={mean_delta_ps:.4f}, Δ_bs={mean_delta_bs:.4f})")
    else:
        print(f"  ⚠ Patch and block shuffle behave identically — inspect")
    
    print("\n  ★ Task 9 complete")


if __name__ == '__main__':
    main()
