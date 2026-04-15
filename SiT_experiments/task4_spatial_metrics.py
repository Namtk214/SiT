"""
TASK 4: Spatial Structure Metrics

Compute LDS, CDS, RMSC for raw/common/residual.
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
    load_shard, compute_lds, compute_cds, compute_rmsc,
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


def compute_spatial_metrics(tensor):
    """Compute LDS, CDS, RMSC for a (P, D) tensor."""
    return {
        'lds': compute_lds(tensor),
        'cds': compute_cds(tensor),
        'rmsc': compute_rmsc(tensor),
    }


def run_spatial_for_variant(raw, common, residual, variant_name, output_dir, is_mean=False):
    """Compute spatial metrics for one variant."""
    print(f"\n  Computing spatial metrics for {variant_name}...")
    
    img_ids = sorted(set(k[0] for k in raw.keys()))
    rows = []
    
    for img_id in tqdm(img_ids, desc=f"Spatial ({variant_name})"):
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                raw_key = (img_id, layer_idx, ts_idx)
                common_key = (img_id, ts_idx) if is_mean else (img_id, layer_idx, ts_idx)
                
                if raw_key not in raw:
                    continue
                
                raw_m = compute_spatial_metrics(raw[raw_key])
                common_m = compute_spatial_metrics(common[common_key])
                res_m = compute_spatial_metrics(residual[raw_key])
                
                rows.append({
                    'image_id': img_id,
                    'layer': layer_idx + 1,
                    'timestep_idx': ts_idx,
                    'variant': variant_name,
                    'raw_lds': raw_m['lds'],
                    'raw_cds': raw_m['cds'],
                    'raw_rmsc': raw_m['rmsc'],
                    'common_lds': common_m['lds'],
                    'common_cds': common_m['cds'],
                    'common_rmsc': common_m['rmsc'],
                    'res_lds': res_m['lds'],
                    'res_cds': res_m['cds'],
                    'res_rmsc': res_m['rmsc'],
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f'task4_{variant_name}_spatial.csv', index=False)
    
    # Aggregate by layer
    by_layer = df.groupby('layer').mean(numeric_only=True).reset_index()
    by_layer.to_csv(output_dir / f'task4_{variant_name}_by_layer.csv', index=False)
    
    # Aggregate by timestep
    by_time = df.groupby('timestep_idx').mean(numeric_only=True).reset_index()
    by_time.to_csv(output_dir / f'task4_{variant_name}_by_time.csv', index=False)
    
    # Plot layer-wise curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('lds', 'LDS'), ('cds', 'CDS'), ('rmsc', 'RMSC')]
    
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(by_layer['layer'], by_layer[f'raw_{metric}'], 'b-o', label='Raw', markersize=3)
        ax.plot(by_layer['layer'], by_layer[f'common_{metric}'], 'g-s', label='Common', markersize=3)
        ax.plot(by_layer['layer'], by_layer[f'res_{metric}'], 'r-^', label='Residual', markersize=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Layer ({variant_name})')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / f'task4_{variant_name}_by_layer.png', dpi=150)
    plt.close(fig)
    
    # Plot timestep-wise curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(by_time['timestep_idx'], by_time[f'raw_{metric}'], 'b-o', label='Raw', markersize=3)
        ax.plot(by_time['timestep_idx'], by_time[f'common_{metric}'], 'g-s', label='Common', markersize=3)
        ax.plot(by_time['timestep_idx'], by_time[f'res_{metric}'], 'r-^', label='Residual', markersize=3)
        ax.set_xlabel('Timestep Index')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Timestep ({variant_name})')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / f'task4_{variant_name}_by_time.png', dpi=150)
    plt.close(fig)
    
    # Summary
    summary = {
        'variant': variant_name,
        'raw_lds_mean': df['raw_lds'].mean(),
        'common_lds_mean': df['common_lds'].mean(),
        'res_lds_mean': df['res_lds'].mean(),
        'raw_cds_mean': df['raw_cds'].mean(),
        'common_cds_mean': df['common_cds'].mean(),
        'res_cds_mean': df['res_cds'].mean(),
        'raw_rmsc_mean': df['raw_rmsc'].mean(),
        'common_rmsc_mean': df['common_rmsc'].mean(),
        'res_rmsc_mean': df['res_rmsc'].mean(),
    }
    return summary


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task4'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 4: SPATIAL STRUCTURE METRICS")
    print("="*60)
    
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    summaries = []
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    s = run_spatial_for_variant(raw, common_mean, res_mean, 'mean', output_dir, is_mean=True)
    summaries.append(s)
    del common_mean, res_mean
    
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        s = run_spatial_for_variant(raw, common_k, res_k, f'tsvd_K{K}', output_dir)
        summaries.append(s)
        del common_k, res_k
    
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / 'task4_summary.csv', index=False)
    
    # LaTeX table
    with open(output_dir / 'task4_summary.tex', 'w') as f:
        f.write(summary_df.to_latex(index=False, float_format='%.4f'))
    
    print("\nSANITY CHECKS:")
    for s in summaries:
        has_nan = any(np.isnan(v) for v in s.values() if isinstance(v, float))
        if not has_nan:
            print(f"  ✓ {s['variant']}: no NaN in metrics")
        else:
            print(f"  ✗ {s['variant']}: NaN found — FAIL")
    
    print("\n  ★ Task 4 complete")


if __name__ == '__main__':
    main()
