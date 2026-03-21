#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow diffusion model.

Usage:
    # Single GPU
    python sample.py --ckpt path/to/checkpoint.pt --output-dir ./samples

    # Multi-GPU with torchrun
    torchrun --nnodes=1 --nproc_per_node=8 sample.py --ckpt path/to/checkpoint.pt

This script generates images for FID evaluation, outputting an NPZ file
compatible with the ADM evaluation suite.
"""

import os
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns

# Import from local src/ folder
from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop, create_transport, FixedSampler, vanilla_guidance
from src.utils import batched_prc_img, scattercat

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Similarity matrices will not be logged to WandB.")


#################################################################################
#                       Block-wise Cosine Similarity                            #
#################################################################################

def compute_block_cosine_matrix(block_tokens: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between all blocks.

    Args:
        block_tokens: List of hidden tokens from each block.
                     Each element has shape [B, N, D] where:
                     - B: batch size
                     - N: number of patches/tokens
                     - D: hidden dimension

    Returns:
        sim_mat: Cosine similarity matrix of shape [L, L] where L is number of blocks.
                 sim_mat[a, b] = average cosine similarity between block a and block b.
    """
    if not block_tokens:
        return None

    # Stack all block outputs: [L, B, N, D]
    H = torch.stack(block_tokens, dim=0)
    L, B, N, D = H.shape

    # Normalize along the hidden dimension
    H_norm = F.normalize(H, p=2, dim=-1)  # [L, B, N, D]

    # Compute pairwise cosine similarity: [L, L, B, N]
    # Use correct einsum notation
    sim = torch.einsum('ibnd,jbnd->ijbn',
                       H_norm,  # [L, B, N, D]
                       H_norm)  # [L, B, N, D]

    # Average over batch and patches: [L, L]
    sim_mat = sim.mean(dim=(-1, -2))

    return sim_mat


def plot_similarity_matrix(sim_mat: np.ndarray, timestep: int, noise_level: float, save_path: Path):
    """Plot a single similarity matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_mat, annot=False, cmap='viridis', vmin=0, vmax=1, square=True)
    plt.title(f'Block-wise Cosine Similarity\nNoise={noise_level:.2f}, Timestep={timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_noise_comparison_simple(
    results_by_noise: Dict[float, np.ndarray],
    save_path: Path,
):
    """Plot similarity matrices for different noise levels in a grid."""
    noise_levels = sorted(results_by_noise.keys())
    n_levels = len(noise_levels)

    ncols = min(5, n_levels)
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3))
    if n_levels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, noise_level in enumerate(noise_levels):
        sim_mat = results_by_noise[noise_level]
        ax = axes[idx]
        sns.heatmap(sim_mat, annot=False, cmap='viridis', vmin=0, vmax=1,
                   square=True, ax=ax, cbar=True)
        ax.set_title(f'Noise={noise_level:.2f}')
        ax.set_xlabel('Block')
        ax.set_ylabel('Block')

    # Hide unused subplots
    for idx in range(n_levels, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Block-wise Cosine Similarity Across Noise Levels')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def sample_with_similarity_tracking(
    model,
    x,
    x_ids,
    vector,
    num_steps,
    cfg_scale=None,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="SDE",
    device="cuda",
):
    """
    Custom sampling with block-wise similarity tracking at mid-point.

    Returns:
        samples: Final generated samples
        similarity_matrix: Single similarity matrix computed at mid-point
    """
    # Prepare transport and sampler
    transport = create_transport(
        path_type='Linear',
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )

    # Get time interval
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        diffusion_form="sigma",
        sde=True,
        eval=True,
        reverse=False,
        last_step_size=0.04,
    )

    # Create time steps
    t = torch.linspace(t0, t1, num_steps).to(device)
    dt = t[1] - t[0]

    # Track at mid-point
    mid_step = num_steps // 2
    similarity_matrix = None

    # Sampling loop
    reverse = True

    for step_idx, ti in enumerate(t[:-1]):
        # Apply CFG: duplicate second half for noise consistency
        if cfg_scale is not None and cfg_scale > 1.0:
            bs = x.shape[0]
            assert bs % 2 == 0
            x = torch.concat((x[bs // 2:], x[bs // 2:]))

        t_cur = torch.ones(x.size(0)).to(x) * ti

        # Track similarity at mid-point only
        should_track = (step_idx == mid_step)

        with torch.no_grad():
            # Prepare model inputs
            t_model = 1 - t_cur if reverse else t_cur

            # Check if we should apply CFG
            if cfg_scale is not None and cfg_scale > 1.0:
                apply_cfg = torch.all((guidance_low <= t_cur) & (t_cur <= guidance_high)).item()
            else:
                apply_cfg = False

            # Model forward with optional block tokens
            if should_track:
                pred, block_tokens = model(
                    x,
                    timesteps=t_model,
                    x_ids=x_ids,
                    vector=vector,
                    return_block_tokens=True,
                )

                print(f"[Similarity] Tracking at step {step_idx}/{num_steps}")
                print(f"[Similarity] Number of blocks = {len(block_tokens)}")
                if len(block_tokens) > 0:
                    print(f"[Similarity] Block token shape = {block_tokens[0].shape}")

                # Compute similarity matrix
                # If CFG is used, only take conditional branch (second half)
                if apply_cfg:
                    bs_half = len(block_tokens[0]) // 2
                    block_tokens_cond = [bt[bs_half:] for bt in block_tokens]
                    sim_mat = compute_block_cosine_matrix(block_tokens_cond)
                else:
                    sim_mat = compute_block_cosine_matrix(block_tokens)

                print(f"[Similarity] Similarity matrix shape = {sim_mat.shape}")

                # Convert to float32 and numpy
                similarity_matrix = sim_mat.float().cpu().numpy()
            else:
                pred = model(
                    x,
                    timesteps=t_model,
                    x_ids=x_ids,
                    vector=vector,
                )

            pred = pred.to(torch.float32)

            # Apply CFG guidance
            if apply_cfg:
                pred = vanilla_guidance(pred, cfg_val=cfg_scale)
                pred = torch.cat((pred, pred))

            # Reverse if needed
            model_output = -pred if reverse else pred

            # SDE step (Euler-Maruyama)
            w_cur = torch.randn(x.size()).to(x)
            dw = w_cur * torch.sqrt(dt)

            # Compute drift (velocity)
            drift = model_output

            # Compute diffusion coefficient
            diffusion = transport.path_sampler.compute_diffusion(
                x, t_cur, form="sigma", norm=1.0
            )

            # Get score from velocity
            score = transport.path_sampler.get_score_from_velocity(
                model_output, x, t_cur
            )

            # Combine drift with score term
            drift = drift + diffusion * score

            # Update
            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * diffusion) * dw

    # Last step
    t_last = torch.ones(x.size(0)).to(x) * t1
    t_last_model = 1 - t_last if reverse else t_last

    with torch.no_grad():
        pred_last = model(x, timesteps=t_last_model, x_ids=x_ids, vector=vector)
        pred_last = -pred_last if reverse else pred_last

    x = x + pred_last * 0.04

    return x, similarity_matrix


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = 0
        torch.cuda.set_device(device)
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_vae(device, dtype=torch.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    
    # VAE scaling factors
    scale_factor = 0.18215
    shift_factor = 0.0
    
    return vae, scale_factor, shift_factor


def decode_latents(vae, latents, scale_factor, shift_factor):
    """Decode latents to images using the VAE."""
    latents = latents / scale_factor + shift_factor
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)
    
    with torch.no_grad():
        images = vae.decode(latents).sample
    
    images = (images.float() + 1) / 2
    images = images.clamp(0, 1)
    
    return images


def load_model(ckpt_path, device):
    """Load the Self-Flow model from checkpoint."""
    print(f"Loading model from {ckpt_path}")
    
    # Create model with DiT-XL/2 settings
    config = dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )
    model = SelfFlowPerTokenDiT(**config)
    
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    # Keep model in float32 for weights, autocast handles precision
    model = model.to(device=device)
    model.eval()
    
    return model


@torch.no_grad()
def sample_batch(
    model,
    vae,
    scale_factor,
    shift_factor,
    batch_size,
    class_labels,
    num_steps=250,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="SDE",
    device="cuda",
    seed=None,
    noise_scale=1.0,
    track_similarity=False,
):
    """Sample a batch of images with optional similarity tracking at mid-point."""
    if seed is not None:
        torch.manual_seed(seed)

    latent_size = 32
    latent_channels = 4
    patch_size = 2

    # Sample noise in bfloat16 for mixed precision with noise scaling
    noise = torch.randn(
        batch_size, latent_channels, latent_size, latent_size,
        device=device, dtype=torch.bfloat16
    ) * noise_scale

    # Patchify noise: (B, C, H, W) -> (B, C*P*P, H/P, W/P)
    noise_patched = rearrange(
        noise,
        "b c (h p1) (w p2) -> b (c p1 p2) h w",
        p1=patch_size, p2=patch_size
    )

    # Convert to token format
    x, x_ids = batched_prc_img(noise_patched.cpu())
    x = x.to(device=device)
    x_ids = x_ids.to(device)

    # Prepare for CFG if needed
    if cfg_scale > 1.0:
        x = torch.cat([x, x], dim=0)
        x_ids = torch.cat([x_ids, x_ids], dim=0)
        null_labels = torch.full_like(class_labels, 1000)
        class_labels = torch.cat([null_labels, class_labels], dim=0)

    # Run denoising with optional similarity tracking
    similarity_matrix = None

    if track_similarity:
        # Use custom sampling with similarity tracking
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            samples, similarity_matrix = sample_with_similarity_tracking(
                model=model,
                x=x,
                x_ids=x_ids,
                vector=class_labels,
                num_steps=num_steps,
                cfg_scale=cfg_scale if cfg_scale > 1.0 else None,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                mode=mode,
                device=device,
            )
    else:
        # Standard sampling without tracking
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            samples = denoise_loop(
                model=model,
                num_steps=num_steps,
                cfg_scale=cfg_scale if cfg_scale > 1.0 else None,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                mode=mode,
                x=x,
                x_ids=x_ids,
                vector=class_labels,
            )

    # Extract conditional output if CFG was used
    if cfg_scale > 1.0:
        samples = samples[batch_size:]
        x_ids = x_ids[batch_size:]

    # Convert back to image format
    samples = scattercat(samples, x_ids)

    # Unpatchify: (B, C*P*P, H/P, W/P) -> (B, C, H, W)
    samples = rearrange(
        samples,
        "b (c p1 p2) h w -> b c (h p1) (w p2)",
        p1=patch_size, p2=patch_size, c=latent_channels
    )

    # Decode with VAE
    images = decode_latents(vae, samples, scale_factor, shift_factor)

    # Convert to numpy [0, 255]
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 255).astype(np.uint8)

    if track_similarity:
        return images, similarity_matrix
    else:
        return images


def main():
    parser = argparse.ArgumentParser(description="Sample images from Self-Flow model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE", "ODE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save individual PNG images")
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    # CFG options (not used in paper results)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--guidance-low", type=float, default=0.0, help="Lower guidance bound")
    parser.add_argument("--guidance-high", type=float, default=0.7, help="Upper guidance bound")
    # Similarity tracking options
    parser.add_argument("--track-similarity", action="store_true", help="Enable block-wise similarity tracking")
    parser.add_argument("--noise-levels", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                       help="Comma-separated noise levels for similarity tracking (e.g., '0.1,0.5,1.0')")
    parser.add_argument("--similarity-samples", type=int, default=1,
                       help="Number of samples for similarity analysis per noise level")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging for similarity")
    parser.add_argument("--wandb-project", type=str, default="selfflow-similarity", help="WandB project name")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    device = f"cuda:{device}"
    
    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Generating {args.num_fid_samples} samples")
        print(f"Mode: {args.mode}, Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    
    # Set seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            (output_dir / "images").mkdir(exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Load models
    model = load_model(args.ckpt, device)
    vae, scale_factor, shift_factor = load_vae(device)

    # Similarity tracking mode
    if args.track_similarity and rank == 0:
        print(f"\n{'='*60}")
        print("Running in SIMILARITY TRACKING mode")
        print(f"{'='*60}")

        # Parse noise levels
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]

        print(f"Noise levels: {noise_levels}")
        print(f"Samples per noise level: {args.similarity_samples}")
        print(f"Similarity computed at mid-point of sampling (step {args.num_steps//2})")

        # Initialize WandB if requested
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                config={
                    "noise_levels": noise_levels,
                    "num_samples": args.similarity_samples,
                    "num_steps": args.num_steps,
                    "cfg_scale": args.cfg_scale,
                }
            )

        # Create similarity output directory
        sim_output_dir = output_dir / "similarity_analysis"
        sim_output_dir.mkdir(exist_ok=True)

        # Store results for each noise level
        results_by_noise = {}

        # Process each noise level
        for noise_scale in noise_levels:
            print(f"\n{'='*60}")
            print(f"Processing noise level: {noise_scale:.2f}")
            print(f"{'='*60}")

            # Random class labels
            class_labels = torch.randint(0, 1000, (args.similarity_samples,), device=device)

            # Sample with similarity tracking
            images, sim_matrix = sample_batch(
                model=model,
                vae=vae,
                scale_factor=scale_factor,
                shift_factor=shift_factor,
                batch_size=args.similarity_samples,
                class_labels=class_labels,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                mode=args.mode,
                device=device,
                seed=args.seed,
                noise_scale=noise_scale,
                track_similarity=True,
            )

            # Store similarity matrix for this noise level
            results_by_noise[noise_scale] = sim_matrix

            print(f"Similarity matrix shape: {sim_matrix.shape}")

            # Save results
            noise_dir = sim_output_dir / f"noise_{noise_scale:.2f}"
            noise_dir.mkdir(exist_ok=True)

            np.save(noise_dir / "similarity_matrix.npy", sim_matrix)

            # Save sample images
            if args.save_images:
                for i, img in enumerate(images):
                    Image.fromarray(img).save(noise_dir / f"sample_{i:02d}.png")

            # Plot similarity matrix
            plot_path = noise_dir / "similarity_matrix.png"
            plot_similarity_matrix(sim_matrix, args.num_steps//2, noise_scale, plot_path)

            # Log to WandB
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"similarity/noise_{noise_scale:.2f}": wandb.Image(plot_path),
                    f"similarity_matrix/noise_{noise_scale:.2f}": sim_matrix.tolist(),
                })

        # Create cross-noise-level comparison
        print(f"\n{'='*60}")
        print("Creating cross-noise-level comparison")
        print(f"{'='*60}")

        if len(noise_levels) > 1:
            comparison_path = sim_output_dir / "noise_level_comparison.png"
            plot_noise_comparison_simple(results_by_noise, comparison_path)

            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "comparison/all_noise_levels": wandb.Image(comparison_path)
                })

        print(f"\n{'='*60}")
        print(f"Similarity analysis complete! Results saved to {sim_output_dir}")
        print(f"{'='*60}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        cleanup_distributed()
        return

    # Standard sampling mode (no similarity tracking)
    # Calculate samples per GPU
    total_samples = args.num_fid_samples
    samples_per_gpu = math.ceil(total_samples / world_size)
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, total_samples)
    my_samples = end_idx - start_idx

    if rank == 0:
        print(f"Each GPU will generate ~{samples_per_gpu} samples")

    # Generate samples
    all_samples = []
    all_labels = []

    num_batches = math.ceil(my_samples / args.batch_size)
    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", disable=rank != 0)
    
    for batch_idx in pbar:
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, my_samples)
        batch_size = batch_end - batch_start
        
        # Random class labels
        class_labels = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Unique seed for this batch
        batch_seed = args.seed + rank * 100000 + batch_idx
        
        images = sample_batch(
            model=model,
            vae=vae,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            batch_size=batch_size,
            class_labels=class_labels,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            mode=args.mode,
            device=device,
            seed=batch_seed,
            noise_scale=1.0,
            track_similarity=False,
        )
        
        all_samples.append(images)
        all_labels.extend(class_labels.cpu().numpy())
        
        if args.save_images and rank == 0:
            for i, img in enumerate(images):
                global_idx = start_idx + batch_start + i
                Image.fromarray(img).save(output_dir / "images" / f"{global_idx:06d}.png")
    
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Gather from all GPUs using file-based approach
    if world_size > 1:
        rank_npz = output_dir / f"samples_rank{rank}.npz"
        np.savez(rank_npz, arr_0=all_samples)
        dist.barrier()
        
        if rank == 0:
            gathered = []
            for r in range(world_size):
                r_path = output_dir / f"samples_rank{r}.npz"
                r_data = np.load(r_path)['arr_0']
                gathered.append(r_data)
                r_path.unlink()
            all_samples = np.concatenate(gathered, axis=0)
    
    # Save NPZ
    if rank == 0:
        all_samples = all_samples[:args.num_fid_samples]
        npz_path = output_dir / f"samples_{len(all_samples)}.npz"
        create_npz_from_samples(list(all_samples), npz_path)
        
        print(f"\nDone! Generated {len(all_samples)} samples")
        print(f"NPZ: {npz_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
