#
# GaussFluids: Density Optimization & SPH Density Estimation
# Paper Section 3.3, Equations (5)-(6)
#

import torch
import math


# ---------------------------------------------------------------------------
# Poly6 SPH smoothing kernel
# ---------------------------------------------------------------------------

def poly6_kernel(r, h):
    """
    Poly6 SPH smoothing kernel (Equation 6).
    W(r, h) = (315 / (64π · h⁹)) · (h² − r²)³  if 0 ≤ r ≤ h, else 0

    Args:
        r: (N, K) tensor of distances between particles and their K neighbors
        h: smoothing length (scalar)
    Returns:
        (N, K) tensor of kernel weights
    """
    h_sq = h * h
    # Polynomial kernel coefficient: 315 / (64π · h⁹)
    coeff = 315.0 / (64.0 * math.pi * (h ** 9))

    mask = (r <= h).float()
    kernel_vals = coeff * ((h_sq - r * r) ** 3) * mask
    return kernel_vals


# ---------------------------------------------------------------------------
# SPH density computation using pytorch3d KNN
# ---------------------------------------------------------------------------

def compute_sph_density(positions, h=0.3, k=64):
    """
    Compute SPH density at each particle using KNN and Poly6 kernel.
    Density at particle i: ρ_i = Σ_j W(‖p_i − p_j‖₂, h)

    Uses pytorch3d.ops.knn_points for KNN search (K=64 per paper).

    Args:
        positions: (N, 3) tensor — MUST be deformed positions p_t
        h: smoothing length (default 0.3 per paper)
        k: number of neighbors (default 64 per paper)
    Returns:
        densities: (N,) tensor of ρ_i values
    """
    try:
        from pytorch3d.ops import knn_points
    except ImportError:
        raise ImportError(
            "pytorch3d is required for KNN-based SPH density. "
            "Install with: pip install pytorch3d"
        )

    N = positions.shape[0]

    # Handle small particle counts
    actual_k = min(k, N)

    # pytorch3d KNN: returns (dists, idx, _)
    # positions need shape (B, N, 3), so add batch dim
    pos_batch = positions.unsqueeze(0)  # (1, N, 3)

    with torch.no_grad():
        dists, idx, _ = knn_points(
            pos_batch, pos_batch,
            K=actual_k,
            return_nn=False,
            return_sorted=True
        )

    dists = dists.squeeze(0)  # (N, K)
    # dists are squared distances from pytorch3d knn_points
    dists = torch.sqrt(dists + 1e-10)  # (N, K) — Euclidean distances

    # Compute Poly6 kernel weights
    weights = poly6_kernel(dists, h)  # (N, K)

    # Density: sum of kernel weights over neighbors
    densities = weights.sum(dim=-1)  # (N,)

    return densities


# ---------------------------------------------------------------------------
# Density-guided adaptive control helpers
# ---------------------------------------------------------------------------

def compute_density_gradient_mask(densities, low_percentile=0.05, high_percentile=0.95):
    """
    Identify particles with extreme densities for adaptive control.
    Low density → may need splitting (under-resolved region)
    High density → may need pruning (over-resolved region)

    Args:
        densities: (N,) tensor
        low_percentile: threshold for low density
        high_percentile: threshold for high density
    Returns:
        low_mask, high_mask: boolean tensors
    """
    sorted_dens, _ = torch.sort(densities)
    N = densities.shape[0]
    low_thresh = sorted_dens[int(N * low_percentile)]
    high_thresh = sorted_dens[int(N * high_percentile)]

    low_mask = densities < low_thresh
    high_mask = densities > high_thresh

    return low_mask, high_mask
