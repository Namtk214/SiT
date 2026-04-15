"""
Shared utilities for the SiT experimental protocol.
Includes: data loading, tensor I/O, metric functions, plotting helpers.
"""

import os
import sys
import glob
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# ============================================================================
# Global constants from protocol
# ============================================================================
NUM_LAYERS = 28
HIDDEN_DIM = 1152
PATCH_SIZE = 2
INPUT_SIZE = 32  # latent space size (256x256 / 8)
NUM_PATCHES = (INPUT_SIZE // PATCH_SIZE) ** 2  # = 256
GRID_H = INPUT_SIZE // PATCH_SIZE  # = 16
GRID_W = INPUT_SIZE // PATCH_SIZE  # = 16
EPS = 1e-6
SEED = 42

# 10 uniformly spaced timesteps in [0, 1] for SiT continuous time
NUM_TIMESTEPS = 10
TIMESTEPS = [k / (NUM_TIMESTEPS - 1) for k in range(NUM_TIMESTEPS)]
# [0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0]

# SVD ranks
SVD_RANKS = [16, 32, 64]

# Dataset sizes
PILOT_SIZE = 100
MAIN_SIZE = 500
CONTROL_SIZE = 100

# ============================================================================
# Data loading
# ============================================================================

def load_latents_from_arrayrecord(data_dir, num_images, seed=SEED):
    """
    Load pre-computed VAE latent moments from ArrayRecord format.
    
    Each record is a pickled dict: {'moments': (8,32,32) float32, 'label': int}
    The first 4 channels = mean, last 4 = logvar.
    We extract the mean latent (4,32,32).
    
    Returns:
        latents: torch.Tensor of shape (num_images, 4, 32, 32) float32
        labels: torch.Tensor of shape (num_images,) int64
    """
    from array_record.python.array_record_module import ArrayRecordReader
    
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "train-*-of-*.ar")))
    assert len(shard_paths) > 0, f"No ArrayRecord shards found in {data_dir}"
    
    # Collect all indices across shards
    rng = np.random.RandomState(seed)
    
    # First pass: count total records
    total_records = 0
    shard_counts = []
    for sp in shard_paths:
        reader = ArrayRecordReader(sp)
        n = reader.num_records()
        shard_counts.append(n)
        total_records += n
        reader.close()
    
    print(f"Total records across {len(shard_paths)} shards: {total_records}")
    
    # Sample indices
    selected_global = rng.choice(total_records, size=min(num_images, total_records), replace=False)
    selected_global.sort()
    
    # Map global indices to (shard, local_index)
    cumulative = np.cumsum([0] + shard_counts)
    shard_local_indices = {}  # shard_idx -> list of local indices
    for gidx in selected_global:
        shard_idx = np.searchsorted(cumulative[1:], gidx, side='right')
        local_idx = gidx - cumulative[shard_idx]
        if shard_idx not in shard_local_indices:
            shard_local_indices[shard_idx] = []
        shard_local_indices[shard_idx].append(int(local_idx))
    
    latents = []
    labels = []
    
    for shard_idx in sorted(shard_local_indices.keys()):
        sp = shard_paths[shard_idx]
        local_indices = shard_local_indices[shard_idx]
        reader = ArrayRecordReader(sp)
        records = reader.read(local_indices)
        reader.close()
        
        for rec in records:
            obj = pickle.loads(rec)
            moments = obj['moments']  # (8, 32, 32) float32
            label = obj['label']
            mean_latent = moments[:4]  # first 4 channels = mean
            latents.append(mean_latent)
            labels.append(label)
    
    latents = torch.tensor(np.stack(latents), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    print(f"Loaded {len(latents)} latents: shape {latents.shape}, labels shape {labels.shape}")
    return latents, labels


# ============================================================================
# Model loading
# ============================================================================

def load_sit_model(device='cpu'):
    """Load SiT-XL/2 with pre-trained weights."""
    from models import SiT_XL_2
    from download import find_model
    
    model = SiT_XL_2(input_size=INPUT_SIZE).to(device)
    state_dict = find_model('SiT-XL-2-256x256.pt')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"SiT-XL/2 loaded on {device}")
    print(f"  Blocks: {len(model.blocks)}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Num patches: {NUM_PATCHES}")
    return model


# ============================================================================
# Activation extraction
# ============================================================================

@torch.no_grad()
def extract_activations(model, x, t_scalar, class_label, device='cpu'):
    """
    Extract patch embedding + all 28 block outputs for one image at one timestep.
    
    Args:
        model: SiT model
        x: (4, 32, 32) latent tensor
        t_scalar: float in [0, 1]
        class_label: int class label
        device: torch device
    
    Returns:
        A_patch: (P, D) tensor — patch embedding output (before blocks)
        block_tokens: list of 28 tensors, each (P, D)
    """
    x = x.unsqueeze(0).to(device)  # (1, 4, 32, 32)
    t = torch.tensor([t_scalar], dtype=torch.float32, device=device)
    y = torch.tensor([class_label], dtype=torch.long, device=device)
    
    # Extract patch embedding (before entering blocks)
    A_patch = model.x_embedder(x) + model.pos_embed  # (1, P, D)
    
    # Full forward with block token extraction
    _, block_tokens = model.forward(x, t, y, return_block_tokens=True)
    
    # Remove batch dimension
    A_patch = A_patch.squeeze(0).cpu()  # (P, D)
    block_tokens = [bt.squeeze(0).cpu() for bt in block_tokens]  # list of (P, D)
    
    return A_patch, block_tokens


# ============================================================================
# L2 normalization
# ============================================================================

def l2_normalize_tokens(A, eps=EPS):
    """
    Per-token L2 normalization.
    A: (..., P, D) tensor
    Returns: A_tilde with each token normalized to unit length.
    """
    norms = A.norm(dim=-1, keepdim=True)  # (..., P, 1)
    return A / (norms + eps)


# ============================================================================
# Common representation construction
# ============================================================================

def build_mean_common(A_tilde_layers):
    """
    Build mean-common for a fixed timestep.
    
    Args:
        A_tilde_layers: list of 28 tensors, each (P, D) — normalized activations
    
    Returns:
        A_common_mean: (P, D)
        B_res_mean: list of 28 (P, D) residual tensors
    """
    stacked = torch.stack(A_tilde_layers, dim=0)  # (28, P, D)
    A_common_mean = stacked.mean(dim=0)  # (P, D)
    B_res_mean = [A_tilde_layers[i] - A_common_mean for i in range(len(A_tilde_layers))]
    return A_common_mean, B_res_mean


def build_tsvd_common(A_tilde_all, ranks=SVD_RANKS):
    """
    Build truncated-SVD common for one image across all layers and timesteps.
    
    Args:
        A_tilde_all: dict[(layer_idx, timestep_idx)] -> (P, D) normalized tensor
        ranks: list of K values
    
    Returns:
        dict[K] -> {
            'V_K': (D, K) right singular vectors,
            'common': dict[(i,t)] -> (P, D),
            'residual': dict[(i,t)] -> (P, D)
        }
    """
    # Stack all activations: M(x) in R^{(28*10*P) x D}
    keys = sorted(A_tilde_all.keys())
    M_parts = [A_tilde_all[k] for k in keys]
    M = torch.cat(M_parts, dim=0).float()  # (28*10*P, D)
    
    # Truncated SVD
    max_K = max(ranks)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # Vh: (min(rows,D), D)
    
    results = {}
    for K in ranks:
        V_K = Vh[:K].T  # (D, K)
        
        common = {}
        residual = {}
        for key in keys:
            A = A_tilde_all[key].float()
            proj = A @ V_K @ V_K.T  # (P, D)
            common[key] = proj.half()
            residual[key] = (A - proj).half()
        
        results[K] = {
            'V_K': V_K,
            'common': common,
            'residual': residual
        }
    
    return results


# ============================================================================
# Tensor I/O
# ============================================================================

def save_shard(tensor_dict, filepath, metadata=None):
    """Save a dict of tensors + metadata to a .pt file."""
    data = {'tensors': tensor_dict}
    if metadata is not None:
        data['metadata'] = metadata
    torch.save(data, filepath)


def load_shard(filepath):
    """Load a shard .pt file."""
    return torch.load(filepath, map_location='cpu', weights_only=False)


# ============================================================================
# Metrics
# ============================================================================

def cosine_similarity_tokens(X, Y, eps=EPS):
    """
    Token-mean cosine similarity.
    X, Y: (P, D) tensors
    Returns: scalar
    """
    # Per-token cosine
    X_norm = X / (X.norm(dim=-1, keepdim=True) + eps)
    Y_norm = Y / (Y.norm(dim=-1, keepdim=True) + eps)
    cos_per_token = (X_norm * Y_norm).sum(dim=-1)  # (P,)
    return cos_per_token.mean().item()


def cosine_flat(X, Y, eps=EPS):
    """
    Flattened cosine similarity.
    X, Y: (P, D) tensors
    Returns: scalar
    """
    x = X.reshape(-1).float()
    y = Y.reshape(-1).float()
    return (torch.dot(x, y) / (x.norm() * y.norm() + eps)).item()


def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.
    X, Y: (P, D) tensors (each column is a feature)
    Returns: scalar
    """
    X = X.float()
    Y = Y.float()
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + EPS)).item()


def compute_self_similarity(X, eps=EPS):
    """
    Compute self-similarity matrix using cosine similarity.
    X: (P, D) tensor
    Returns: (P, P) similarity matrix
    """
    X_norm = X.float() / (X.float().norm(dim=-1, keepdim=True) + eps)
    return X_norm @ X_norm.T


def compute_spatial_distances(H, W, metric='manhattan'):
    """
    Compute pairwise spatial distances on a H x W grid.
    Returns: (H*W, H*W) distance matrix
    """
    coords = np.array([(i, j) for i in range(H) for j in range(W)])
    if metric == 'manhattan':
        dists = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=-1)
    else:
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    return dists


def compute_lds(X, H=GRID_H, W=GRID_W, r_near=None, r_far=None, eps=EPS):
    """
    Local Dispersion Score.
    LDS(X) = E[K(t,t') | d(t,t') < r_near] - E[K(t,t') | d(t,t') >= r_far]
    """
    if r_near is None:
        r_near = H // 2
    if r_far is None:
        r_far = H // 2
    
    K = compute_self_similarity(X, eps)
    D = compute_spatial_distances(H, W)
    
    near_mask = D < r_near
    np.fill_diagonal(near_mask, False)  # exclude self
    far_mask = D >= r_far
    
    K_np = K.detach().cpu().numpy()
    
    near_sim = K_np[near_mask].mean() if near_mask.any() else 0.0
    far_sim = K_np[far_mask].mean() if far_mask.any() else 0.0
    
    return near_sim - far_sim


def compute_cds(X, H=GRID_H, W=GRID_W, eps=EPS):
    """
    Cosine Decay Score.
    Fit g(delta) = alpha + beta * delta, return CDS = -beta.
    """
    K = compute_self_similarity(X, eps).detach().cpu().numpy()
    D = compute_spatial_distances(H, W)
    
    max_dist = int(D.max())
    deltas = list(range(1, max_dist + 1))
    g_values = []
    
    for delta in deltas:
        mask = D == delta
        if mask.any():
            g_values.append(K[mask].mean())
        else:
            g_values.append(np.nan)
    
    # Remove NaN
    valid = [(d, g) for d, g in zip(deltas, g_values) if not np.isnan(g)]
    if len(valid) < 2:
        return 0.0
    
    ds, gs = zip(*valid)
    ds = np.array(ds, dtype=np.float64)
    gs = np.array(gs, dtype=np.float64)
    
    # Linear fit: g = alpha + beta * d
    A_mat = np.vstack([np.ones_like(ds), ds]).T
    result = np.linalg.lstsq(A_mat, gs, rcond=None)
    alpha, beta = result[0]
    
    return -beta


def compute_rmsc(X, eps=EPS):
    """
    Root Mean Square Contrast.
    RMSC(X) = sqrt( (1/P) * sum_p ||x_hat_p - x_bar||^2 )
    """
    X = X.float()
    X_hat = X / (X.norm(dim=-1, keepdim=True) + eps)
    X_bar = X_hat.mean(dim=0, keepdim=True)
    diffs = X_hat - X_bar
    return torch.sqrt((diffs ** 2).sum(dim=-1).mean()).item()


# ============================================================================
# Plotting
# ============================================================================

def setup_plotting():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
    return plt
