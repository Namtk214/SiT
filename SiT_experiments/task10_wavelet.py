"""
TASK 10/11: Wavelet Decomposition

Analyze common and residual under a multiscale/frequency view using 2D Haar wavelets.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS,
    GRID_H, GRID_W, HIDDEN_DIM, TIMESTEPS,
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


def compute_wavelet_energy(Z, wavelet='haar', levels=2):
    """
    Compute wavelet decomposition energies for a (P, D) tensor.
    Reshape to (H, W, D), apply 2D DWT per channel.
    
    Returns dict with energies per level.
    """
    Z = Z.float().numpy()
    Z_3d = Z.reshape(GRID_H, GRID_W, HIDDEN_DIM)  # (H, W, D)
    
    results = {}
    for level in range(1, levels + 1):
        E_A = 0.0
        E_H = 0.0
        E_V = 0.0
        E_D_coeff = 0.0
        
        for d in range(HIDDEN_DIM):
            channel = Z_3d[:, :, d]
            
            try:
                coeffs = pywt.wavedec2(channel, wavelet, level=level)
                # coeffs[0] = approximation at deepest level
                # coeffs[1:] = (cH, cV, cD) at each level from deepest to shallowest
                
                approx = coeffs[0]
                E_A += np.sum(approx ** 2)
                
                for l_coeffs in coeffs[1:]:
                    cH, cV, cD = l_coeffs
                    E_H += np.sum(cH ** 2)
                    E_V += np.sum(cV ** 2)
                    E_D_coeff += np.sum(cD ** 2)
            except Exception:
                continue
        
        E_detail = E_H + E_V + E_D_coeff
        R_wave = E_A / (E_detail + 1e-10)
        
        results[level] = {
            'E_A': E_A,
            'E_H': E_H,
            'E_V': E_V,
            'E_D': E_D_coeff,
            'E_detail': E_detail,
            'R_wave': R_wave,
        }
    
    return results


def run_wavelet_for_variant(common, residual, variant_name, output_dir, is_mean=False):
    """Run wavelet analysis for one variant."""
    print(f"\n  Computing wavelets for {variant_name}...")
    
    rows = []
    
    # Wavelet on common
    if is_mean:
        common_keys = sorted(common.keys())  # (img_id, ts_idx)
    else:
        common_keys = sorted(common.keys())  # (img_id, layer, ts_idx)
    
    print(f"  Processing {len(common_keys)} common tensors...")
    for key in tqdm(common_keys[:1000], desc=f"Wavelet common ({variant_name})"):
        tensor = common[key]
        energies = compute_wavelet_energy(tensor)
        
        if is_mean and len(key) == 2:
            img_id, ts_idx = key
            layer = 'common'
        elif len(key) == 3:
            img_id, layer_idx, ts_idx = key
            layer = layer_idx + 1
        else:
            continue
        
        for level, e in energies.items():
            rows.append({
                'image_id': img_id,
                'layer': layer,
                'timestep_idx': ts_idx,
                'tensor_type': 'common',
                'variant': variant_name,
                'level': level,
                'E_A': e['E_A'],
                'E_detail': e['E_detail'],
                'R_wave': e['R_wave'],
            })
    
    # Wavelet on residual
    res_keys = sorted(residual.keys())
    print(f"  Processing {len(res_keys)} residual tensors...")
    for key in tqdm(res_keys[:5000], desc=f"Wavelet residual ({variant_name})"):
        tensor = residual[key]
        energies = compute_wavelet_energy(tensor)
        
        img_id, layer_idx, ts_idx = key
        
        for level, e in energies.items():
            rows.append({
                'image_id': img_id,
                'layer': layer_idx + 1,
                'timestep_idx': ts_idx,
                'tensor_type': 'residual',
                'variant': variant_name,
                'level': level,
                'E_A': e['E_A'],
                'E_detail': e['E_detail'],
                'R_wave': e['R_wave'],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f'task11_wavelet_{variant_name}.csv', index=False)
    
    return df


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task10'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 10/11: WAVELET DECOMPOSITION")
    print("="*60)
    
    all_dfs = []
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    df = run_wavelet_for_variant(common_mean, res_mean, 'mean', output_dir, is_mean=True)
    all_dfs.append(df)
    del common_mean, res_mean
    
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        df = run_wavelet_for_variant(common_k, res_k, f'tsvd_K{K}', output_dir)
        all_dfs.append(df)
        del common_k, res_k
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Summary table
    summary = combined.groupby(['variant', 'tensor_type', 'level']).agg({
        'E_A': 'mean',
        'E_detail': 'mean',
        'R_wave': 'mean',
    }).reset_index()
    summary.to_csv(output_dir / 'task11_wavelet_ratio.csv', index=False)
    
    # Plots
    for variant in combined['variant'].unique():
        vdf = combined[combined['variant'] == variant]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Wavelet Decomposition ({variant})', fontsize=14)
        
        for level in [1, 2]:
            ax = axes[level - 1]
            for ttype in ['common', 'residual']:
                subset = vdf[(vdf['tensor_type'] == ttype) & (vdf['level'] == level)]
                if len(subset) > 0:
                    ax.bar(ttype, subset['R_wave'].mean(), label=ttype, alpha=0.7)
            ax.set_title(f'R_wave (Level {level})')
            ax.set_ylabel('Approx/Detail Energy Ratio')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'figure_T24_{variant}.png', dpi=150)
        plt.close(fig)
    
    # Layer-wise residual wavelet profile
    res_df = combined[(combined['tensor_type'] == 'residual') & (combined['level'] == 1)]
    if len(res_df) > 0:
        res_by_layer = res_df.groupby(['variant', 'layer']).agg({'R_wave': 'mean'}).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for variant in res_by_layer['variant'].unique():
            vdf = res_by_layer[res_by_layer['variant'] == variant]
            ax.plot(vdf['layer'], vdf['R_wave'], marker='o', markersize=3, label=variant)
        ax.set_xlabel('Layer')
        ax.set_ylabel('R_wave (Approx/Detail)')
        ax.set_title('Residual Wavelet Profile by Layer')
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / 'figure_T25_residual_wavelet.png', dpi=150)
        plt.close(fig)
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    for variant in summary['variant'].unique():
        vsummary = summary[summary['variant'] == variant]
        
        com_r = vsummary[(vsummary['tensor_type'] == 'common') & (vsummary['level'] == 1)]['R_wave'].values
        res_r = vsummary[(vsummary['tensor_type'] == 'residual') & (vsummary['level'] == 1)]['R_wave'].values
        
        if len(com_r) > 0 and len(res_r) > 0:
            if not np.isnan(com_r[0]) and not np.isnan(res_r[0]):
                print(f"  ✓ {variant}: energies are finite (common R={com_r[0]:.4f}, res R={res_r[0]:.4f})")
                if abs(com_r[0] - res_r[0]) > 0.01:
                    print(f"  ✓ {variant}: common and residual wavelet profiles differ")
                else:
                    print(f"  ⚠ {variant}: profiles are very similar — inspect")
            else:
                print(f"  ✗ {variant}: NaN energy — FAIL")
    
    print("\n  ★ Task 10/11 complete")


if __name__ == '__main__':
    main()
