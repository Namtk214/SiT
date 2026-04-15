"""
TASK 5: Layer-ID / Timestep-ID Probes

Train linear classifiers to predict layer identity and timestep identity
from raw/common/residual descriptors (mean-pooled tokens).
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
    load_shard, RESULTS_DIR, NUM_LAYERS, NUM_TIMESTEPS, SVD_RANKS, TIMESTEPS, SEED,
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


def build_descriptors(tensors, is_common_mean=False):
    """
    Build mean-pooled descriptors.
    Returns: descriptors dict[(img_id, layer, ts)] -> (D,) numpy array
    """
    descriptors = {}
    for key, tensor in tensors.items():
        desc = tensor.float().mean(dim=0).numpy()  # (D,)
        descriptors[key] = desc
    return descriptors


def run_probes_for_variant(raw, common, residual, variant_name, output_dir, is_mean=False):
    """Run layer-id and timestep-id probes for one variant."""
    print(f"\n  Running probes for {variant_name}...")
    
    # Build descriptors
    raw_desc = build_descriptors(raw)
    common_desc = build_descriptors(common, is_common_mean=is_mean)
    res_desc = build_descriptors(residual)
    
    # Get all image IDs
    img_ids = sorted(set(k[0] for k in raw.keys()))
    
    # 80/20 split by image
    train_imgs, test_imgs = train_test_split(img_ids, test_size=0.2, random_state=SEED)
    train_imgs = set(train_imgs)
    test_imgs = set(test_imgs)
    
    results = []
    
    for tensor_name, desc_dict, label_type in [
        ('raw', raw_desc, 'raw'),
        ('common', common_desc, 'common'),
        ('residual', res_desc, 'residual'),
    ]:
        # === Layer-ID probe ===
        X_train, y_train, X_test, y_test = [], [], [], []
        
        for key, desc in desc_dict.items():
            if len(key) == 3:
                img_id, layer_idx, ts_idx = key
            else:
                img_id, ts_idx = key
                layer_idx = 0  # common has no layer
            
            if img_id in train_imgs:
                X_train.append(desc)
                y_train.append(layer_idx)
            elif img_id in test_imgs:
                X_test.append(desc)
                y_test.append(layer_idx)
        
        X_train = np.stack(X_train)
        y_train = np.array(y_train)
        X_test = np.stack(X_test)
        y_test = np.array(y_test)
        
        # Check if there's enough variety in labels
        if len(np.unique(y_train)) > 1:
            clf_layer = LogisticRegression(max_iter=500, random_state=SEED, solver='lbfgs', 
                                           multi_class='multinomial')
            clf_layer.fit(X_train, y_train)
            layer_acc_train = clf_layer.score(X_train, y_train)
            layer_acc_test = clf_layer.score(X_test, y_test)
        else:
            layer_acc_train = 0.0
            layer_acc_test = 0.0
        
        # === Timestep-ID probe ===
        X_train_t, y_train_t, X_test_t, y_test_t = [], [], [], []
        
        for key, desc in desc_dict.items():
            if len(key) == 3:
                img_id, layer_idx, ts_idx = key
            else:
                img_id, ts_idx = key
            
            if img_id in train_imgs:
                X_train_t.append(desc)
                y_train_t.append(ts_idx)
            elif img_id in test_imgs:
                X_test_t.append(desc)
                y_test_t.append(ts_idx)
        
        X_train_t = np.stack(X_train_t)
        y_train_t = np.array(y_train_t)
        X_test_t = np.stack(X_test_t)
        y_test_t = np.array(y_test_t)
        
        if len(np.unique(y_train_t)) > 1:
            clf_time = LogisticRegression(max_iter=500, random_state=SEED, solver='lbfgs',
                                          multi_class='multinomial')
            clf_time.fit(X_train_t, y_train_t)
            time_acc_train = clf_time.score(X_train_t, y_train_t)
            time_acc_test = clf_time.score(X_test_t, y_test_t)
        else:
            time_acc_train = 0.0
            time_acc_test = 0.0
        
        results.append({
            'variant': variant_name,
            'tensor': tensor_name,
            'layer_acc_train': layer_acc_train,
            'layer_acc_test': layer_acc_test,
            'time_acc_train': time_acc_train,
            'time_acc_test': time_acc_test,
            'n_train': len(X_train),
            'n_test': len(X_test),
        })
        
        print(f"    {tensor_name}: layer_acc={layer_acc_test:.4f}, time_acc={time_acc_test:.4f}")
    
    return results


def main():
    task0_dir = RESULTS_DIR / 'task0'
    output_dir = RESULTS_DIR / 'task5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TASK 5: LAYER-ID / TIMESTEP-ID PROBES")
    print("="*60)
    
    raw = load_all_shards(task0_dir / 'task0_Araw_shards')
    all_results = []
    
    # Mean-common
    common_mean = load_all_shards(task0_dir / 'task0_Acommon_mean_shards')
    res_mean = load_all_shards(task0_dir / 'task0_Bres_mean_shards')
    results = run_probes_for_variant(raw, common_mean, res_mean, 'mean', output_dir, is_mean=True)
    all_results.extend(results)
    del common_mean, res_mean
    
    for K in SVD_RANKS:
        common_k = load_all_shards(task0_dir / f'task0_Acommon_tsvd_K{K}_shards')
        res_k = load_all_shards(task0_dir / f'task0_Bres_tsvd_K{K}_shards')
        results = run_probes_for_variant(raw, common_k, res_k, f'tsvd_K{K}', output_dir)
        all_results.extend(results)
        del common_k, res_k
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'task5_probe_results.csv', index=False)
    
    # Tables
    layer_table = df.pivot_table(index='tensor', columns='variant', values='layer_acc_test')
    layer_table.to_csv(output_dir / 'task5_layerprobe_summary.csv')
    
    time_table = df.pivot_table(index='tensor', columns='variant', values='time_acc_test')
    time_table.to_csv(output_dir / 'task5_timeprobe_summary.csv')
    
    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, metric, title in [(axes[0], 'layer_acc_test', 'Layer-ID Accuracy'),
                               (axes[1], 'time_acc_test', 'Timestep-ID Accuracy')]:
        variants = df['variant'].unique()
        x = np.arange(len(variants))
        width = 0.25
        
        for i, tensor_type in enumerate(['raw', 'common', 'residual']):
            vals = [df[(df['variant'] == v) & (df['tensor'] == tensor_type)][metric].values[0] 
                    for v in variants]
            ax.bar(x + i*width, vals, width, label=tensor_type)
        
        ax.set_xlabel('Variant')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(variants, rotation=45)
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'figure_T11_probes.png', dpi=150)
    fig.savefig(output_dir / 'figure_T11_probes.pdf')
    plt.close(fig)
    
    # LaTeX
    with open(output_dir / 'task5_summary.tex', 'w') as f:
        f.write("% Table T6: Layer-ID accuracy\n")
        f.write(layer_table.to_latex(float_format='%.4f'))
        f.write("\n% Table T7: Timestep-ID accuracy\n")
        f.write(time_table.to_latex(float_format='%.4f'))
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    for variant in df['variant'].unique():
        vdf = df[df['variant'] == variant]
        res_layer = vdf[vdf['tensor'] == 'residual']['layer_acc_test'].values[0]
        com_layer = vdf[vdf['tensor'] == 'common']['layer_acc_test'].values[0]
        
        if res_layer > com_layer:
            print(f"  ✓ {variant}: residual ({res_layer:.4f}) > common ({com_layer:.4f}) on layer-id")
        else:
            print(f"  ⚠ {variant}: residual ({res_layer:.4f}) ≤ common ({com_layer:.4f}) on layer-id")
        
        raw_layer = vdf[vdf['tensor'] == 'raw']['layer_acc_test'].values[0]
        random_chance_layer = 1.0 / NUM_LAYERS
        if raw_layer > random_chance_layer * 2:
            print(f"  ✓ {variant}: raw layer-id ({raw_layer:.4f}) >> random ({random_chance_layer:.4f})")
        else:
            print(f"  ⚠ {variant}: raw layer-id ({raw_layer:.4f}) near random — inspect")
    
    print("\n  ★ Task 5 complete")


if __name__ == '__main__':
    main()
