"""
TASK 0: Extract Activations and Build Common / Residual

This is the foundation task. All subsequent tasks depend on its output.

Steps:
1. Load SiT-XL/2 pretrained model
2. Load N images from ImageNet VAE latent moments
3. For each image × 10 timesteps:
   - Forward pass with return_block_tokens=True
   - Extract A_patch and A_{1..28,t}
4. L2-normalize → A_tilde
5. Build mean-common + residual
6. Build truncated-SVD common (K=16,32,64) + residual
7. Save all tensors + metadata
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    load_latents_from_arrayrecord,
    load_sit_model,
    extract_activations,
    l2_normalize_tokens,
    build_mean_common,
    build_tsvd_common,
    save_shard,
    RESULTS_DIR,
    NUM_LAYERS,
    HIDDEN_DIM,
    NUM_PATCHES,
    NUM_TIMESTEPS,
    TIMESTEPS,
    SVD_RANKS,
    SEED,
    EPS,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Task 0: Activation Extraction')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to process (100 for pilot, 500 for main)')
    parser.add_argument('--data-dir', type=str,
                        default='/home/thanhnamngo26/gcs-bucket/imagenet_moments',
                        help='Path to ImageNet moments ArrayRecord shards')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'task0'),
                        help='Output directory for task0 results')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of images to process before saving a shard')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for shards
    shard_dirs = {
        'Araw': output_dir / 'task0_Araw_shards',
        'Atilde': output_dir / 'task0_Atilde_shards',
        'Acommon_mean': output_dir / 'task0_Acommon_mean_shards',
        'Bres_mean': output_dir / 'task0_Bres_mean_shards',
    }
    for K in SVD_RANKS:
        shard_dirs[f'Acommon_tsvd_K{K}'] = output_dir / f'task0_Acommon_tsvd_K{K}_shards'
        shard_dirs[f'Bres_tsvd_K{K}'] = output_dir / f'task0_Bres_tsvd_K{K}_shards'
    
    for d in shard_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load model
    print("="*60)
    print("TASK 0: EXTRACT ACTIVATIONS AND BUILD COMMON / RESIDUAL")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Images: {args.num_images}")
    print(f"  Timesteps: {NUM_TIMESTEPS} ({TIMESTEPS})")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  SVD Ranks: {SVD_RANKS}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {args.device}")
    print()
    
    model = load_sit_model(device=args.device)
    
    # 2. Load latents
    print("\nLoading ImageNet latent moments...")
    latents, labels = load_latents_from_arrayrecord(
        args.data_dir, args.num_images, seed=args.seed
    )
    num_images = len(latents)
    print(f"Loaded {num_images} images")
    
    # 3. Metadata tracking
    metadata_rows = []
    
    # 4. Process images in batches
    patch_embeddings = {}  # img_id -> A_patch
    all_timings = []
    
    num_batches = (num_images + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_img = batch_idx * args.batch_size
        end_img = min(start_img + args.batch_size, num_images)
        batch_range = range(start_img, end_img)
        
        print(f"\n--- Batch {batch_idx+1}/{num_batches} (images {start_img}-{end_img-1}) ---")
        batch_start = time.time()
        
        # Storage for this batch
        batch_raw = {}       # (img_id, layer, timestep_idx) -> (P, D) float16
        batch_atilde = {}    # same
        batch_acommon_mean = {}   # (img_id, timestep_idx) -> (P, D) float16
        batch_bres_mean = {}     # (img_id, layer, timestep_idx) -> (P, D) float16
        # For tSVD: collect all A_tilde per image
        batch_atilde_per_image = {}  # img_id -> dict[(layer, ts_idx)] -> (P, D)
        
        for img_id in tqdm(batch_range, desc="Images"):
            x = latents[img_id]
            label = labels[img_id].item()
            
            img_atilde_all = {}  # (layer, ts_idx) -> (P, D) float32 for SVD
            
            for ts_idx, t_val in enumerate(TIMESTEPS):
                img_start = time.time()
                
                # Extract activations
                A_patch, block_tokens = extract_activations(
                    model, x, t_val, label, device=args.device
                )
                
                # Save patch embedding (only once per image, same across timesteps in practice)
                if ts_idx == 0:
                    patch_embeddings[img_id] = A_patch.half()
                
                for layer_idx in range(NUM_LAYERS):
                    A_raw = block_tokens[layer_idx]  # (P, D)
                    A_tilde = l2_normalize_tokens(A_raw)  # (P, D)
                    
                    batch_raw[(img_id, layer_idx, ts_idx)] = A_raw.half()
                    batch_atilde[(img_id, layer_idx, ts_idx)] = A_tilde.half()
                    img_atilde_all[(layer_idx, ts_idx)] = A_tilde  # keep float32 for SVD
                    
                    metadata_rows.append({
                        'image_id': img_id,
                        'label': label,
                        'layer': layer_idx + 1,  # 1-indexed as per protocol
                        'timestep_idx': ts_idx,
                        'timestep_val': t_val,
                        'seed': args.seed,
                        'dtype': 'float16',
                    })
                
                elapsed = time.time() - img_start
                all_timings.append(elapsed)
            
            # Build mean-common for this image (per timestep)
            for ts_idx in range(NUM_TIMESTEPS):
                layers_tilde = [img_atilde_all[(l, ts_idx)] for l in range(NUM_LAYERS)]
                A_common_mean, B_res_mean_list = build_mean_common(layers_tilde)
                
                batch_acommon_mean[(img_id, ts_idx)] = A_common_mean.half()
                for layer_idx in range(NUM_LAYERS):
                    batch_bres_mean[(img_id, layer_idx, ts_idx)] = B_res_mean_list[layer_idx].half()
            
            # Store for tSVD
            batch_atilde_per_image[img_id] = img_atilde_all
        
        # Build truncated-SVD common for each image in the batch
        batch_tsvd = {K: {'common': {}, 'residual': {}} for K in SVD_RANKS}
        
        print("  Computing truncated SVD...")
        for img_id in tqdm(batch_range, desc="tSVD"):
            atilde_all = batch_atilde_per_image[img_id]
            tsvd_results = build_tsvd_common(atilde_all, ranks=SVD_RANKS)
            
            for K in SVD_RANKS:
                for key, tensor in tsvd_results[K]['common'].items():
                    layer_idx, ts_idx = key
                    batch_tsvd[K]['common'][(img_id, layer_idx, ts_idx)] = tensor
                for key, tensor in tsvd_results[K]['residual'].items():
                    layer_idx, ts_idx = key
                    batch_tsvd[K]['residual'][(img_id, layer_idx, ts_idx)] = tensor
        
        # Save shards for this batch
        shard_name = f'shard_{batch_idx:03d}.pt'
        
        save_shard(batch_raw, shard_dirs['Araw'] / shard_name)
        save_shard(batch_atilde, shard_dirs['Atilde'] / shard_name)
        save_shard(batch_acommon_mean, shard_dirs['Acommon_mean'] / shard_name)
        save_shard(batch_bres_mean, shard_dirs['Bres_mean'] / shard_name)
        
        for K in SVD_RANKS:
            save_shard(batch_tsvd[K]['common'], 
                      shard_dirs[f'Acommon_tsvd_K{K}'] / shard_name)
            save_shard(batch_tsvd[K]['residual'], 
                      shard_dirs[f'Bres_tsvd_K{K}'] / shard_name)
        
        # Free memory
        del batch_raw, batch_atilde, batch_acommon_mean, batch_bres_mean
        del batch_atilde_per_image, batch_tsvd
        
        batch_time = time.time() - batch_start
        avg_time = np.mean(all_timings)
        remaining = (num_batches - batch_idx - 1) * batch_time
        print(f"  Batch time: {batch_time:.1f}s, Avg per img×ts: {avg_time:.2f}s")
        print(f"  Estimated remaining: {remaining/60:.1f} min")
    
    # 5. Save patch embeddings
    torch.save(patch_embeddings, output_dir / 'task0_Apatch.pt')
    
    # 6. Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / 'task0_metadata.csv', index=False)
    
    # 7. Table T0: Summary
    print("\n" + "="*60)
    print("TABLE T0: Task 0 Summary")
    print("="*60)
    
    summary = {
        'Total images': num_images,
        'Timesteps': NUM_TIMESTEPS,
        'Layers': NUM_LAYERS,
        'Patches (P)': NUM_PATCHES,
        'Hidden dim (D)': HIDDEN_DIM,
        'Tensor shape': f'({NUM_PATCHES}, {HIDDEN_DIM})',
        'Storage dtype': 'float16',
        'SVD compute dtype': 'float32',
        'Common variants': 'mean, tSVD-K16, tSVD-K32, tSVD-K64',
        'Seed': args.seed,
    }
    
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Save Table T0
    with open(output_dir / 'table_T0.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Property', 'Value'])
        for k, v in summary.items():
            writer.writerow([k, v])
    
    # 8. Sanity Checks
    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)
    
    passed = True
    
    # Check shapes
    shard0 = torch.load(shard_dirs['Araw'] / 'shard_000.pt', map_location='cpu', weights_only=False)
    sample_key = list(shard0['tensors'].keys())[0]
    sample_tensor = shard0['tensors'][sample_key]
    shape = sample_tensor.shape
    expected_shape = (NUM_PATCHES, HIDDEN_DIM)
    if shape == expected_shape:
        print(f"  ✓ Tensor shape: {shape} == {expected_shape}")
    else:
        print(f"  ✗ Tensor shape: {shape} != {expected_shape}")
        passed = False
    
    # Check completeness - verify metadata
    expected_entries = num_images * NUM_LAYERS * NUM_TIMESTEPS
    actual_entries = len(metadata_df)
    if actual_entries == expected_entries:
        print(f"  ✓ Metadata entries: {actual_entries} == {expected_entries}")
    else:
        print(f"  ✗ Metadata entries: {actual_entries} != {expected_entries}")
        passed = False
    
    # Check residuals are not zero
    res_shard = torch.load(shard_dirs['Bres_mean'] / 'shard_000.pt', map_location='cpu', weights_only=False)
    sample_res = list(res_shard['tensors'].values())[0]
    if sample_res.abs().sum() > 0:
        print(f"  ✓ Mean residual is non-zero (L1={sample_res.abs().mean():.6f})")
    else:
        print(f"  ✗ Mean residual is zero!")
        passed = False
    
    for K in SVD_RANKS:
        res_shard_k = torch.load(shard_dirs[f'Bres_tsvd_K{K}'] / 'shard_000.pt', map_location='cpu', weights_only=False)
        sample_res_k = list(res_shard_k['tensors'].values())[0]
        if sample_res_k.abs().sum() > 0:
            print(f"  ✓ tSVD-K{K} residual is non-zero (L1={sample_res_k.abs().mean():.6f})")
        else:
            print(f"  ✗ tSVD-K{K} residual is zero!")
            passed = False
    
    # Check all shards exist
    total_shards = num_batches
    for name, d in shard_dirs.items():
        shards = list(d.glob('*.pt'))
        if len(shards) == total_shards:
            print(f"  ✓ {name}: {len(shards)} shards")
        else:
            print(f"  ✗ {name}: {len(shards)} shards (expected {total_shards})")
            passed = False
    
    if passed:
        print("\n  ★ ALL SANITY CHECKS PASSED ★")
    else:
        print("\n  ✗ SOME SANITY CHECKS FAILED — review before continuing")
    
    total_time = sum(all_timings)
    print(f"\nTotal extraction time: {total_time/60:.1f} min")
    print(f"Average per image×timestep: {np.mean(all_timings):.2f}s")


# Need pandas for metadata
import pandas as pd

if __name__ == '__main__':
    main()
