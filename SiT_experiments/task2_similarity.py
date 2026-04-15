"""
TASK 2: Similarity Heatmaps

Construct similarity heatmaps (token-mean cosine, flattened cosine, CKA)
for raw/common/residual across layers and timesteps.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, cosine_similarity_tokens, cosine_flat, linear_cka,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS, TIMESTEPS,
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


def compute_layer_layer_heatmap(tensors, img_ids, ts_idx, metric_fn):
    """Compute 28x28 layer-layer heatmap at a given timestep, averaged over images."""
    H = np.zeros((NUM_LAYERS, NUM_LAYERS))
    count = 0
    for img_id in img_ids:
        for i in range(NUM_LAYERS):
            for j in range(i, NUM_LAYERS):
                key_i = (img_id, i, ts_idx)
                key_j = (img_id, j, ts_idx)
                if key_i in tensors and key_j in tensors:
                    val = metric_fn(tensors[key_i].float(), tensors[key_j].float())
                    H[i, j] += val
                    H[j, i] += val
        count += 1
    if count > 0:
        H /= count
    # Fix diagonal double-counting
    for i in range(NUM_LAYERS):
        H[i, i] /= 2
    return H


def compute_time_time_heatmap(tensors, img_ids, layer_idx, metric_fn):
    """Compute 10x10 timestep-timestep heatmap at a given layer, averaged over images."""
    H = np.zeros((NUM_TIMESTEPS, NUM_TIMESTEPS))
    count = 0
    for img_id in img_ids:
        for s in range(NUM_TIMESTEPS):
            for t in range(s, NUM_TIMESTEPS):
                key_s = (img_id, layer_idx, s)
                key_t = (img_id, layer_idx, t)
                if key_s in tensors and key_t in tensors:
                    val = metric_fn(tensors[key_s].float(), tensors[key_t].float())
                    H[s, t] += val
                    H[t, s] += val
        count += 1
    if count > 0:
        H /= count
    for i in range(NUM_TIMESTEPS):
        H[i, i] /= 2
    return H


def plot_heatmap(H, title, xlabel, ylabel, filepath, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(H, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def run_heatmaps_for_variant(raw, common, residual, variant_name, output_dir, is_mean=False):
    """Generate all heatmaps for one common variant."""
    print(f"\n  Computing heatmaps for {variant_name}...")
    
    img_ids = sorted(set(k[0] for k in raw.keys()))
    
    # Use token-mean cosine as the primary metric
    metric = cosine_similarity_tokens
    
    # 1. Layer x layer heatmaps for raw at selected timesteps
    for ts_idx in [0, 4, 9]:  # t_1, t_5, t_10
        H_raw = compute_layer_layer_heatmap(raw, img_ids, ts_idx, metric)
        np.save(output_dir / f'{variant_name}_raw_LxL_t{ts_idx}.npy', H_raw)
        plot_heatmap(H_raw, f'Raw Layer×Layer (t={ts_idx}, {variant_name})',
                     'Layer', 'Layer',
                     output_dir / f'{variant_name}_raw_LxL_t{ts_idx}.png')
    
    # 2. Layer x layer for residual
    for ts_idx in [0, 4, 9]:
        H_res = compute_layer_layer_heatmap(residual, img_ids, ts_idx, metric)
        np.save(output_dir / f'{variant_name}_res_LxL_t{ts_idx}.npy', H_res)
        plot_heatmap(H_res, f'Residual Layer×Layer (t={ts_idx}, {variant_name})',
                     'Layer', 'Layer',
                     output_dir / f'{variant_name}_res_LxL_t{ts_idx}.png')
    
    # 3. Timestep x timestep for raw at selected layers
    for layer_idx in [0, 13, 27]:  # layer 1, 14, 28
        H_raw_tt = compute_time_time_heatmap(raw, img_ids, layer_idx, metric)
        np.save(output_dir / f'{variant_name}_raw_TxT_l{layer_idx}.npy', H_raw_tt)
        plot_heatmap(H_raw_tt, f'Raw Time×Time (layer={layer_idx+1}, {variant_name})',
                     'Timestep', 'Timestep',
                     output_dir / f'{variant_name}_raw_TxT_l{layer_idx}.png')
    
    # 4. Full 280x280 heatmap (all layer-timestep combinations)
    # This is a combined (layer,timestep) similarity matrix
    N = NUM_LAYERS * NUM_TIMESTEPS  # 280
    H_full = np.zeros((N, N))
    print(f"  Computing full {N}x{N} heatmap (averaged over {len(img_ids)} images)...")
    
    for img_id in tqdm(img_ids[:50], desc="Full heatmap"):  # subsample for speed
        for idx_a in range(N):
            la, ta = divmod(idx_a, NUM_TIMESTEPS)
            key_a = (img_id, la, ta)
            if key_a not in raw:
                continue
            for idx_b in range(idx_a, N):
                lb, tb = divmod(idx_b, NUM_TIMESTEPS)
                key_b = (img_id, lb, tb)
                if key_b not in raw:
                    continue
                val = metric(raw[key_a].float(), raw[key_b].float())
                H_full[idx_a, idx_b] += val
                H_full[idx_b, idx_a] += val
    
    H_full /= min(len(img_ids), 50)
    for i in range(N):
        H_full[i, i] /= 2
    
    np.save(output_dir / f'{variant_name}_tokenmean_heatmap.npy', H_full)
    plot_heatmap(H_full, f'Full Similarity ({variant_name})',
                 'Layer×Timestep', 'Layer×Timestep',
                 output_dir / f'{variant_name}_full_heatmap.png')
    
    # Summary statistics
    summary = {
        'variant': variant_name,
        'raw_diag_mean': np.diag(H_full).mean(),
        'raw_offdiag_mean': (H_full.sum() - np.trace(H_full)) / (N*N - N),
    }
    
    return summary


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 2: SIMILARITY HEATMAPS")
    print("="*60)
    
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    summaries = []
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    s = run_heatmaps_for_variant(raw, common_mean, res_mean, 'mean', output_dir, is_mean=True)
    summaries.append(s)
    del common_mean, res_mean
    
    # tSVD variants
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        s = run_heatmaps_for_variant(raw, common_k, res_k, f'tsvd_K{K}', output_dir)
        summaries.append(s)
        del common_k, res_k
    
    # Table T3
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / 'table_T3_summary.csv', index=False)
    print("\nTable T3 saved.")
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    for s in summaries:
        if s['raw_diag_mean'] > s['raw_offdiag_mean']:
            print(f"  ✓ {s['variant']}: diagonal > off-diagonal")
        else:
            print(f"  ✗ {s['variant']}: diagonal ≤ off-diagonal — FAIL")
    
    print("\n  ★ Task 2 complete")


if __name__ == '__main__':
    main()
