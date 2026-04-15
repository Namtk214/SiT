"""
TASK 3: Layer-Timestep Map

Build 28×10 maps for LDS, cosine-to-common, and probe score on residual.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, compute_lds, cosine_similarity_tokens,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS, TIMESTEPS, NUM_PATCHES, HIDDEN_DIM,
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


def run_maps_for_variant(raw, common, residual, variant_name, output_dir, labels_dict, is_mean=False):
    """Build 28×10 maps for one variant."""
    print(f"\n  Building maps for {variant_name}...")
    
    img_ids = sorted(set(k[0] for k in raw.keys()))
    
    # 1. LDS map
    lds_map = np.zeros((NUM_LAYERS, NUM_TIMESTEPS))
    lds_count = np.zeros((NUM_LAYERS, NUM_TIMESTEPS))
    
    print("  Computing LDS map...")
    for img_id in tqdm(img_ids, desc="LDS"):
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                key = (img_id, layer_idx, ts_idx)
                if key in raw:
                    lds_val = compute_lds(raw[key])
                    lds_map[layer_idx, ts_idx] += lds_val
                    lds_count[layer_idx, ts_idx] += 1
    
    lds_map = np.where(lds_count > 0, lds_map / lds_count, 0)
    np.save(output_dir / f'task3_{variant_name}_lds_map.npy', lds_map)
    
    # 2. Cosine-to-common map
    cos_map = np.zeros((NUM_LAYERS, NUM_TIMESTEPS))
    cos_count = np.zeros((NUM_LAYERS, NUM_TIMESTEPS))
    
    print("  Computing cosine-to-common map...")
    for img_id in tqdm(img_ids, desc="Cos"):
        for layer_idx in range(NUM_LAYERS):
            for ts_idx in range(NUM_TIMESTEPS):
                raw_key = (img_id, layer_idx, ts_idx)
                common_key = (img_id, ts_idx) if is_mean else (img_id, layer_idx, ts_idx)
                if raw_key in raw and common_key in common:
                    cos_val = cosine_similarity_tokens(raw[raw_key].float(), common[common_key].float())
                    cos_map[layer_idx, ts_idx] += cos_val
                    cos_count[layer_idx, ts_idx] += 1
    
    cos_map = np.where(cos_count > 0, cos_map / cos_count, 0)
    np.save(output_dir / f'task3_{variant_name}_cos_map.npy', cos_map)
    
    # 3. Probe score map (quick layer-id accuracy per cell)
    print("  Computing probe score map...")
    probe_map = np.zeros((NUM_LAYERS, NUM_TIMESTEPS))
    
    for ts_idx in range(NUM_TIMESTEPS):
        # Build descriptors from residual for this timestep
        X_all = []
        y_all = []
        for img_id in img_ids:
            for layer_idx in range(NUM_LAYERS):
                key = (img_id, layer_idx, ts_idx)
                if key in residual:
                    desc = residual[key].float().mean(dim=0)  # (D,) mean pooling
                    X_all.append(desc.numpy())
                    y_all.append(layer_idx)
        
        if len(X_all) > 0:
            X_all = np.stack(X_all)
            y_all = np.array(y_all)
            
            # Quick logistic regression
            try:
                clf = LogisticRegression(max_iter=200, random_state=42, solver='lbfgs')
                clf.fit(X_all, y_all)
                preds = clf.predict(X_all)
                acc = (preds == y_all).mean()
                
                # Assign same accuracy to all layers at this timestep
                for layer_idx in range(NUM_LAYERS):
                    probe_map[layer_idx, ts_idx] = acc
            except Exception as e:
                print(f"    Probe failed at ts={ts_idx}: {e}")
    
    np.save(output_dir / f'task3_{variant_name}_probe_map.npy', probe_map)
    
    # Plot all three maps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    titles = ['LDS on Raw', 'Cosine to Common', 'Probe Score on Residual']
    maps = [lds_map, cos_map, probe_map]
    
    for ax, m, title in zip(axes, maps, titles):
        im = ax.imshow(m, aspect='auto', cmap='viridis')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Layer')
        ax.set_title(f'{title}\n({variant_name})')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    fig.savefig(output_dir / f'task3_{variant_name}_maps.png', dpi=150)
    fig.savefig(output_dir / f'task3_{variant_name}_maps.pdf')
    plt.close(fig)
    
    return lds_map, cos_map, probe_map


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 3: LAYER-TIMESTEP MAP")
    print("="*60)
    
    # Load raw
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    
    # Load labels
    meta = pd.read_csv(task0_dir / 'task0_metadata.csv')
    labels_dict = {}
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    run_maps_for_variant(raw, common_mean, res_mean, 'mean', output_dir, labels_dict, is_mean=True)
    del common_mean, res_mean
    
    # tSVD variants
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        run_maps_for_variant(raw, common_k, res_k, f'tsvd_K{K}', output_dir, labels_dict)
        del common_k, res_k
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    for variant in ['mean'] + [f'tsvd_K{K}' for K in SVD_RANKS]:
        lds = np.load(output_dir / f'task3_{variant}_lds_map.npy')
        if lds.shape == (NUM_LAYERS, NUM_TIMESTEPS):
            print(f"  ✓ {variant} LDS map shape: {lds.shape}")
        else:
            print(f"  ✗ {variant} LDS map shape: {lds.shape} — FAIL")
        if lds.std() > 1e-6:
            print(f"  ✓ {variant} LDS map is not flat (std={lds.std():.6f})")
        else:
            print(f"  ✗ {variant} LDS map is flat — FAIL")
    
    print("\n  ★ Task 3 complete")


if __name__ == '__main__':
    main()
