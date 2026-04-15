"""
TASK 1: Basic Statistics

Compute min, max, mean, var for raw/common/residual tensors.
Run for all common variants: mean-common, tSVD K=16,32,64.
"""

import os
import sys
import glob
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, l2_normalize_tokens, setup_plotting,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, HIDDEN_DIM, NUM_PATCHES,
    SVD_RANKS, EPS, TIMESTEPS,
)


def compute_stats(Z):
    """Compute basic statistics of tensor Z (P, D)."""
    Z = Z.float()
    Z_tilde = l2_normalize_tokens(Z)
    return {
        'min': Z.min().item(),
        'max': Z.max().item(),
        'mean': Z_tilde.mean().item(),
        'var': Z_tilde.var().item(),
    }


def load_all_shards(shard_dir):
    """Load and merge all shards from a directory."""
    merged = {}
    for fp in sorted(Path(shard_dir).glob('*.pt')):
        data = load_shard(fp)
        merged.update(data['tensors'])
    return merged


def run_stats_for_variant(raw_dir, common_dir, residual_dir, variant_name, output_dir):
    """Run statistics for one common variant."""
    print(f"\n  Loading shards for {variant_name}...")
    raw = load_all_shards(raw_dir)
    common = load_all_shards(common_dir)
    residual = load_all_shards(residual_dir)
    
    rows = []
    
    # Identify unique images
    if variant_name == 'mean':
        # raw keys: (img_id, layer_idx, ts_idx)
        # common keys: (img_id, ts_idx)
        # residual keys: (img_id, layer_idx, ts_idx)
        img_ids = sorted(set(k[0] for k in raw.keys()))
    else:
        img_ids = sorted(set(k[0] for k in raw.keys()))
    
    print(f"  Processing {len(img_ids)} images...")
    for img_id in tqdm(img_ids, desc=f"Stats ({variant_name})"):
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                raw_key = (img_id, layer_idx, ts_idx)
                
                if raw_key not in raw:
                    continue
                
                raw_stats = compute_stats(raw[raw_key])
                
                if variant_name == 'mean':
                    common_key = (img_id, ts_idx)
                else:
                    common_key = (img_id, layer_idx, ts_idx)
                
                common_stats = compute_stats(common[common_key])
                res_stats = compute_stats(residual[raw_key])
                
                rows.append({
                    'image_id': img_id,
                    'layer': layer_idx + 1,
                    'timestep_idx': ts_idx,
                    'timestep_val': TIMESTEPS[ts_idx],
                    'variant': variant_name,
                    # Raw
                    'raw_min': raw_stats['min'],
                    'raw_max': raw_stats['max'],
                    'raw_mean': raw_stats['mean'],
                    'raw_var': raw_stats['var'],
                    # Common
                    'common_min': common_stats['min'],
                    'common_max': common_stats['max'],
                    'common_mean': common_stats['mean'],
                    'common_var': common_stats['var'],
                    # Residual
                    'res_min': res_stats['min'],
                    'res_max': res_stats['max'],
                    'res_mean': res_stats['mean'],
                    'res_var': res_stats['var'],
                })
    
    df = pd.DataFrame(rows)
    
    # Per-image stats file
    df.to_csv(output_dir / f'task1_{variant_name}_stats.csv', index=False)
    
    # Aggregate by layer (avg over images and timesteps)
    by_layer = df.groupby('layer').agg({
        'raw_mean': 'mean', 'raw_var': 'mean',
        'common_mean': 'mean', 'common_var': 'mean',
        'res_mean': 'mean', 'res_var': 'mean',
    }).reset_index()
    by_layer.to_csv(output_dir / f'task1_{variant_name}_stats_by_layer.csv', index=False)
    
    # Aggregate by timestep (avg over images and layers)
    by_time = df.groupby('timestep_idx').agg({
        'raw_mean': 'mean', 'raw_var': 'mean',
        'common_mean': 'mean', 'common_var': 'mean',
        'res_mean': 'mean', 'res_var': 'mean',
    }).reset_index()
    by_time.to_csv(output_dir / f'task1_{variant_name}_stats_by_time.csv', index=False)
    
    return df, by_layer, by_time


def plot_stats(by_layer_dict, by_time_dict, output_dir):
    """Generate line plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Figure T1: Line plot by layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure T1: Statistics by Layer', fontsize=16)
    
    for variant_name, by_layer in by_layer_dict.items():
        for idx, (metric, title) in enumerate([
            ('raw_mean', 'Raw Mean'), ('raw_var', 'Raw Var'),
            ('common_mean', 'Common Mean'), ('res_var', 'Residual Var')
        ]):
            ax = axes[idx // 2, idx % 2]
            ax.plot(by_layer['layer'], by_layer[metric], label=variant_name, marker='o', markersize=3)
            ax.set_xlabel('Layer')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'figure_T1_by_layer.pdf')
    fig.savefig(output_dir / 'figure_T1_by_layer.png')
    plt.close(fig)
    
    # Figure T2: Line plot by timestep
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure T2: Statistics by Timestep', fontsize=16)
    
    for variant_name, by_time in by_time_dict.items():
        for idx, (metric, title) in enumerate([
            ('raw_mean', 'Raw Mean'), ('raw_var', 'Raw Var'),
            ('common_mean', 'Common Mean'), ('res_var', 'Residual Var')
        ]):
            ax = axes[idx // 2, idx % 2]
            ax.plot(by_time['timestep_idx'], by_time[metric], label=variant_name, marker='o', markersize=3)
            ax.set_xlabel('Timestep Index')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'figure_T2_by_timestep.pdf')
    fig.savefig(output_dir / 'figure_T2_by_timestep.png')
    plt.close(fig)


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task1'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 1: BASIC STATISTICS")
    print("="*60)
    
    by_layer_dict = {}
    by_time_dict = {}
    
    # Mean-common
    df, by_layer, by_time = run_stats_for_variant(
        raw_dir=task0_dir / 'task0_Araw_shards',
        common_dir=task0_dir / 'task0_Acommon_mean_shards',
        residual_dir=task0_dir / 'task0_Bres_mean_shards',
        variant_name='mean',
        output_dir=output_dir,
    )
    by_layer_dict['mean'] = by_layer
    by_time_dict['mean'] = by_time
    
    # tSVD variants
    for K in SVD_RANKS:
        df_k, by_layer_k, by_time_k = run_stats_for_variant(
            raw_dir=task0_dir / 'task0_Araw_shards',
            common_dir=task0_dir / f'task0_Acommon_tsvd_K{K}_shards',
            residual_dir=task0_dir / f'task0_Bres_tsvd_K{K}_shards',
            variant_name=f'tsvd_K{K}',
            output_dir=output_dir,
        )
        by_layer_dict[f'tsvd_K{K}'] = by_layer_k
        by_time_dict[f'tsvd_K{K}'] = by_time_k
    
    # Generate plots
    print("\nGenerating plots...")
    try:
        plot_stats(by_layer_dict, by_time_dict, output_dir)
        print("  Plots saved.")
    except Exception as e:
        print(f"  Plot generation failed: {e}")
        # Try simpler plotting
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        for variant_name, by_layer in by_layer_dict.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Statistics by Layer ({variant_name})')
            for idx, (metric, title) in enumerate([
                ('raw_mean', 'Raw Mean'), ('raw_var', 'Raw Var'),
                ('common_mean', 'Common Mean'), ('res_var', 'Residual Var')
            ]):
                ax = axes[idx // 2, idx % 2]
                ax.plot(by_layer['layer'], by_layer[metric], marker='o', markersize=3)
                ax.set_xlabel('Layer')
                ax.set_ylabel(title)
                ax.set_title(title)
            plt.tight_layout()
            fig.savefig(output_dir / f'figure_T1_{variant_name}.png')
            plt.close(fig)
        
        for variant_name, by_time in by_time_dict.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Statistics by Timestep ({variant_name})')
            for idx, (metric, title) in enumerate([
                ('raw_mean', 'Raw Mean'), ('raw_var', 'Raw Var'),
                ('common_mean', 'Common Mean'), ('res_var', 'Residual Var')
            ]):
                ax = axes[idx // 2, idx % 2]
                ax.plot(by_time['timestep_idx'], by_time[metric], marker='o', markersize=3)
                ax.set_xlabel('Timestep')
                ax.set_ylabel(title)
                ax.set_title(title)
            plt.tight_layout()
            fig.savefig(output_dir / f'figure_T2_{variant_name}.png')
            plt.close(fig)
        
        print("  Fallback plots saved.")
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    # Check raw/common/residual stats differ
    mean_stats = pd.read_csv(output_dir / 'task1_mean_stats_by_layer.csv')
    if not np.allclose(mean_stats['raw_var'].values, mean_stats['common_var'].values):
        print("  ✓ Raw and common variance differ")
    else:
        print("  ✗ Raw and common variance are identical — FAIL")
    
    if not mean_stats['raw_var'].isna().any():
        print("  ✓ No NaN in variance")
    else:
        print("  ✗ NaN found in variance — FAIL")
    
    print("\n  ★ Task 1 complete")


if __name__ == '__main__':
    main()
