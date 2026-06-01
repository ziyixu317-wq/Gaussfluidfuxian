#
# GaussFluids: Density Optimization & SPH Density Estimation
# Paper Section 3.3, Equations (5)-(6)
#
# KNN backends (auto-select): pytorch3d > faiss > torch.cdist
# torch.cdist is pure PyTorch — always works, zero extra deps.
#

import torch
import math


# ---------------------------------------------------------------------------
# Multi-backend KNN
# ---------------------------------------------------------------------------

def _knn_faiss(query, ref, k):
    """KNN via faiss-gpu."""
    import numpy as np
    import faiss

    query_np = query.detach().cpu().numpy().astype(np.float32)
    ref_np = ref.detach().cpu().numpy().astype(np.float32)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(ref_np.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(ref_np)

    dists_np, idx_np = gpu_index.search(query_np, k)
    dists = torch.tensor(dists_np, dtype=torch.float32, device=query.device)
    idx = torch.tensor(idx_np, dtype=torch.long, device=query.device)
    return dists, idx


def _knn_torch_cdist(query, ref, k):
    """
    KNN via batched torch.cdist + topk.
    Pure PyTorch — always available, zero dependencies.
    """
    N = query.shape[0]
    chunk = 2048  # balance speed vs memory
    all_dists, all_idx = [], []

    for i in range(0, N, chunk):
        c = query[i:i + chunk]
        d = torch.cdist(c.unsqueeze(0), ref.unsqueeze(0)).squeeze(0)
        top_d, top_i = torch.topk(d, k=k, dim=-1, largest=False)
        all_dists.append(top_d)
        all_idx.append(top_i)

    return torch.cat(all_dists, 0), torch.cat(all_idx, 0)


def knn_points(query, ref, k):
    """
    KNN with auto-backend: pytorch3d → faiss → torch.cdist.

    Args:
        query: (N_q, D) tensor
        ref:   (N_r, D) tensor
        k:     number of neighbors
    Returns:
        dists: (N_q, K) **squared** L2 distances
        idx:   (N_q, K) neighbor indices
    """
    # 1) pytorch3d (fastest GPU-native)
    try:
        from pytorch3d.ops import knn_points as _knn
        with torch.no_grad():
            d, i, _ = _knn(query.unsqueeze(0), ref.unsqueeze(0), K=k,
                           return_nn=False, return_sorted=True)
        return d.squeeze(0), i.squeeze(0)
    except ImportError:
        pass

    # 2) faiss
    try:
        return _knn_faiss(query, ref, k)
    except (ImportError, AttributeError):
        pass

    # 3) pure PyTorch fallback (always works)
    return _knn_torch_cdist(query, ref, k)


# ---------------------------------------------------------------------------
# Poly6 SPH smoothing kernel (Equation 6)
# ---------------------------------------------------------------------------

def poly6_kernel(r, h):
    """
    W(r, h) = (315 / (64π·h⁹)) · (h² − r²)³   if 0 ≤ r ≤ h, else 0

    Args:
        r: (N, K) Euclidean distances
        h: smoothing length
    """
    h_sq = h * h
    coeff = 315.0 / (64.0 * math.pi * (h ** 9))
    mask = (r <= h).float()
    return coeff * ((h_sq - r * r) ** 3) * mask


# ---------------------------------------------------------------------------
# SPH density (Equation 5)
# ---------------------------------------------------------------------------

def compute_sph_density(positions, h=0.3, k=64):
    """
    ρ_i = Σ_j W(‖p_i − p_j‖₂, h)

    Args:
        positions: (N, 3) — MUST be deformed positions p_t
        h: smoothing length (paper: 0.3)
        k: KNN K (paper: 64)
    Returns:
        densities: (N,)
    """
    N = positions.shape[0]
    actual_k = min(k, N)

    sq_dists, _ = knn_points(positions, positions, actual_k)  # (N, K)
    dists = torch.sqrt(sq_dists + 1e-10)
    weights = poly6_kernel(dists, h)  # (N, K)
    return weights.sum(dim=-1)  # (N,)


# ---------------------------------------------------------------------------
# Scale initialization helper (replaces simple-knn with K=4)
# ---------------------------------------------------------------------------

def knn_avg_neighbor_dist(positions, k=4):
    """
    Average distance to k-1 nearest neighbors (excluding self).
    Used for Gaussian scale initialization, replacing simple-knn.

    Args:
        positions: (N, 3)
        k: number of neighbors to query (default 4)
    Returns:
        avg_dist: (N,)
    """
    N = positions.shape[0]
    actual_k = min(k, N)
    sq_dists, _ = knn_points(positions, positions, actual_k)
    dists = torch.sqrt(sq_dists[:, 1:] + 1e-10)  # skip self
    return dists.mean(dim=-1)
