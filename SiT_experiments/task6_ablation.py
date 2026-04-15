"""
TASK 6: Causal Ablation

Test functional role of full/common/residual by running probes + spatial metrics
on each component separately.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_shard, compute_lds, compute_cds, compute_rmsc,
    RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS, SEED,
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


def run_ablation_for_variant(raw, common, residual, variant_name, output_dir, is_mean=False):
    """Run ablation: compare full/common/residual on probes + spatial metrics."""
    print(f"\n  Running ablation for {variant_name}...")
    
    img_ids = sorted(set(k[0] for k in raw.keys()))
    train_imgs, test_imgs = train_test_split(img_ids, test_size=0.2, random_state=SEED)
    train_set = set(train_imgs)
    test_set = set(test_imgs)
    
    results = []
    
    for comp_name, comp_data in [('full', raw), ('common', common), ('residual', residual)]:
        print(f"    Processing {comp_name}...")
        
        # Spatial metrics (per image, averaged)
        lds_vals, cds_vals, rmsc_vals = [], [], []
        X_train_layer, y_train_layer = [], []
        X_test_layer, y_test_layer = [], []
        X_train_time, y_train_time = [], []
        X_test_time, y_test_time = [], []
        
        for key, tensor in comp_data.items():
            if len(key) == 3:
                img_id, layer_idx, ts_idx = key
            else:
                img_id, ts_idx = key
                layer_idx = 0
            
            # Spatial metrics
            lds_vals.append(compute_lds(tensor))
            cds_vals.append(compute_cds(tensor))
            rmsc_vals.append(compute_rmsc(tensor))
            
            # Descriptors for probes
            desc = tensor.float().mean(dim=0).numpy()
            if img_id in train_set:
                X_train_layer.append(desc); y_train_layer.append(layer_idx)
                X_train_time.append(desc); y_train_time.append(ts_idx)
            elif img_id in test_set:
                X_test_layer.append(desc); y_test_layer.append(layer_idx)
                X_test_time.append(desc); y_test_time.append(ts_idx)
        
        # Probes
        X_train_layer = np.stack(X_train_layer)
        y_train_layer = np.array(y_train_layer)
        X_test_layer = np.stack(X_test_layer)
        y_test_layer = np.array(y_test_layer)
        
        if len(np.unique(y_train_layer)) > 1:
            clf = LogisticRegression(max_iter=500, random_state=SEED, solver='lbfgs', multi_class='multinomial')
            clf.fit(X_train_layer, y_train_layer)
            layer_acc = clf.score(X_test_layer, y_test_layer)
        else:
            layer_acc = 0.0
        
        X_train_time = np.stack(X_train_time)
        y_train_time = np.array(y_train_time)
        X_test_time = np.stack(X_test_time)
        y_test_time = np.array(y_test_time)
        
        if len(np.unique(y_train_time)) > 1:
            clf = LogisticRegression(max_iter=500, random_state=SEED, solver='lbfgs', multi_class='multinomial')
            clf.fit(X_train_time, y_train_time)
            time_acc = clf.score(X_test_time, y_test_time)
        else:
            time_acc = 0.0
        
        results.append({
            'variant': variant_name,
            'component': comp_name,
            'lds_mean': np.mean(lds_vals),
            'cds_mean': np.mean(cds_vals),
            'rmsc_mean': np.mean(rmsc_vals),
            'layer_acc': layer_acc,
            'time_acc': time_acc,
        })
        
        print(f"      LDS={np.mean(lds_vals):.4f}, CDS={np.mean(cds_vals):.4f}, "
              f"Layer={layer_acc:.4f}, Time={time_acc:.4f}")
    
    return results


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task6'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 6: CAUSAL ABLATION")
    print("="*60)
    
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    all_results = []
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    results = run_ablation_for_variant(raw, common_mean, res_mean, 'mean', output_dir, is_mean=True)
    all_results.extend(results)
    del common_mean, res_mean
    
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        results = run_ablation_for_variant(raw, common_k, res_k, f'tsvd_K{K}', output_dir)
        all_results.extend(results)
        del common_k, res_k
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'task6_ablation_results.csv', index=False)
    
    with open(output_dir / 'task6_ablation_summary.tex', 'w') as f:
        f.write(df.to_latex(index=False, float_format='%.4f'))
    
    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, metric, title in [(axes[0], 'lds_mean', 'LDS (Spatial)'),
                               (axes[1], 'layer_acc', 'Layer-ID Accuracy')]:
        variants = df['variant'].unique()
        x = np.arange(len(variants))
        width = 0.25
        for i, comp in enumerate(['full', 'common', 'residual']):
            vals = [df[(df['variant'] == v) & (df['component'] == comp)][metric].values[0]
                    for v in variants]
            ax.bar(x + i*width, vals, width, label=comp)
        ax.set_xticks(x + width)
        ax.set_xticklabels(variants, rotation=45)
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'figure_T12_ablation.png', dpi=150)
    fig.savefig(output_dir / 'figure_T12_ablation.pdf')
    plt.close(fig)
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    for variant in df['variant'].unique():
        vdf = df[df['variant'] == variant]
        com_lds = vdf[vdf['component'] == 'common']['lds_mean'].values[0]
        res_lds = vdf[vdf['component'] == 'residual']['lds_mean'].values[0]
        res_layer = vdf[vdf['component'] == 'residual']['layer_acc'].values[0]
        com_layer = vdf[vdf['component'] == 'common']['layer_acc'].values[0]
        
        if com_lds > res_lds:
            print(f"  ✓ {variant}: common LDS ({com_lds:.4f}) > residual LDS ({res_lds:.4f})")
        else:
            print(f"  ⚠ {variant}: common LDS ({com_lds:.4f}) ≤ residual LDS ({res_lds:.4f})")
        
        if res_layer > com_layer:
            print(f"  ✓ {variant}: residual layer-id ({res_layer:.4f}) > common ({com_layer:.4f})")
        else:
            print(f"  ⚠ {variant}: residual layer-id ({res_layer:.4f}) ≤ common ({com_layer:.4f})")
    
    print("\n  ★ Task 6 complete")


if __name__ == '__main__':
    main()
