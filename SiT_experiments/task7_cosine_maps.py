"""
TASK 7: Patch-wise Cosine Maps

Visualize self-similarity at patch level for 16 fixed images.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS,
    GRID_H, GRID_W, NUM_PATCHES, EPS,
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


def compute_cosine_map(Z, anchor_idx, eps=EPS):
    """
    Compute cosine similarity map from anchor patch to all other patches.
    Z: (P, D), anchor_idx: int
    Returns: (P,) cosine similarities
    """
    Z = Z.float()
    anchor = Z[anchor_idx]  # (D,)
    norms = Z.norm(dim=-1) + eps  # (P,)
    anchor_norm = anchor.norm() + eps
    cos_map = (Z @ anchor) / (norms * anchor_norm)
    return cos_map.numpy()


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task7'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 7: PATCH-WISE COSINE MAPS")
    print("="*60)
    
    # Use first 16 images
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    img_ids = sorted(set(k[0] for k in raw.keys()))[:16]
    
    # Fixed anchor patches
    p1 = (GRID_H // 4) * GRID_W + (GRID_W // 4)       # top-left quadrant
    p2 = (GRID_H // 2) * GRID_W + (GRID_W // 2)       # center
    p3 = (3 * GRID_H // 4) * GRID_W + (3 * GRID_W // 4)  # bottom-right quadrant
    anchors = [p1, p2, p3]
    anchor_names = ['top-left', 'center', 'bottom-right']
    
    # Layer subset: {1, 14, 28} -> indices {0, 13, 27}
    layer_indices = [0, 13, 27]
    # Timestep subset: {t_1, t_5, t_10} -> indices {0, 4, 9}
    ts_indices = [0, 4, 9]
    
    # Load common variants
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    
    # For each image, create a panel
    for img_idx, img_id in enumerate(tqdm(img_ids, desc="CosMap images")):
        for anchor_idx, anchor_name in zip(anchors, anchor_names):
            n_rows = len(layer_indices)
            n_cols = len(ts_indices)
            
            # Raw panels
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            fig.suptitle(f'Raw Cosine Map (img={img_id}, anchor={anchor_name})', fontsize=14)
            
            for r, li in enumerate(layer_indices):
                for c, ti in enumerate(ts_indices):
                    key = (img_id, li, ti)
                    if key in raw:
                        cmap = compute_cosine_map(raw[key], anchor_idx)
                        cmap_2d = cmap.reshape(GRID_H, GRID_W)
                        ax = axes[r, c] if n_rows > 1 else axes[c]
                        im = ax.imshow(cmap_2d, cmap='RdBu_r', vmin=-1, vmax=1)
                        ax.set_title(f'L{li+1}, t{ti}')
                        ax.axis('off')
                        # Mark anchor
                        ay, ax_pos = divmod(anchor_idx, GRID_W)
                        ax.plot(ax_pos, ay, 'k*', markersize=10)
            
            plt.tight_layout()
            fig.savefig(output_dir / f'cosmap_raw_img{img_id}_anchor{anchor_name}.png', dpi=100)
            plt.close(fig)
            
            # Residual panels
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            fig.suptitle(f'Residual Cosine Map (img={img_id}, anchor={anchor_name})', fontsize=14)
            
            for r, li in enumerate(layer_indices):
                for c, ti in enumerate(ts_indices):
                    key = (img_id, li, ti)
                    if key in res_mean:
                        cmap = compute_cosine_map(res_mean[key], anchor_idx)
                        cmap_2d = cmap.reshape(GRID_H, GRID_W)
                        ax = axes[r, c] if n_rows > 1 else axes[c]
                        im = ax.imshow(cmap_2d, cmap='RdBu_r', vmin=-1, vmax=1)
                        ax.set_title(f'L{li+1}, t{ti}')
                        ax.axis('off')
                        ay, ax_pos = divmod(anchor_idx, GRID_W)
                        ax.plot(ax_pos, ay, 'k*', markersize=10)
            
            plt.tight_layout()
            fig.savefig(output_dir / f'cosmap_res_img{img_id}_anchor{anchor_name}.png', dpi=100)
            plt.close(fig)
    
    # Sanity check: cosine at anchor with itself == 1
    print("\nSANITY CHECKS:")
    sample_key = (img_ids[0], 0, 0)
    if sample_key in raw:
        cmap = compute_cosine_map(raw[sample_key], anchors[0])
        self_cos = cmap[anchors[0]]
        if abs(self_cos - 1.0) < 0.01:
            print(f"  ✓ Self-cosine at anchor: {self_cos:.6f} ≈ 1.0")
        else:
            print(f"  ✗ Self-cosine at anchor: {self_cos:.6f} ≠ 1.0 — FAIL")
    
    print("\n  ★ Task 7 complete")


if __name__ == '__main__':
    main()
