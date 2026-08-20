"""Receiver-relevant geometry for frozen finite-support diagnostics."""

from __future__ import annotations

import math

import torch

from .channel import COMPLEX_DTYPE
from .receiver import mismatched_lmmse_equalizer
from .tail_learning import stream_cvar


def apply_mask(effective: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply one fixed or channel-wise Boolean support mask."""
    selected = mask.to(torch.bool)
    if selected.ndim == 2:
        selected = selected.expand_as(effective)
    if selected.shape != effective.shape:
        raise ValueError("support mask does not match effective operators")
    return torch.where(selected, effective, torch.zeros((), dtype=effective.dtype))


def exact_error_geometry(
    effective: torch.Tensor,
    retained: torch.Tensor,
    sigma2: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact geometry of the project's mismatched finite LMMSE receiver.

    The equalizer is designed from ``retained`` but acts on ``effective``.
    Consequently the error covariance includes the actual mismatch term
    ``(F A-I)(F A-I)^H`` rather than an idealized posterior covariance.
    """
    n = effective.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma = torch.as_tensor(sigma2, dtype=torch.float64)
    equalizer = mismatched_lmmse_equalizer(retained, sigma)
    response = equalizer @ effective
    residual = response - identity
    covariance = residual @ residual.mH + sigma[..., None, None] * (equalizer @ equalizer.mH)
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).real
    desired = torch.abs(torch.diagonal(response, dim1=-2, dim2=-1)).square()
    interference = torch.sum(torch.abs(response).square(), dim=-1) - desired
    output_noise = sigma[..., None] * torch.sum(torch.abs(equalizer).square(), dim=-1)
    gamma = desired / torch.clamp(interference + output_noise, min=1e-300)

    singular = torch.linalg.svdvals(retained)
    gram_eigenvalues = singular.square()
    weights = gram_eigenvalues / torch.clamp(gram_eigenvalues.sum(dim=-1, keepdim=True), min=1e-300)
    effective_rank = torch.exp(-torch.sum(weights * torch.log(torch.clamp(weights, min=1e-300)), dim=-1))
    diagonal_scale = torch.sqrt(torch.clamp(diagonal[..., :, None] * diagonal[..., None, :], min=1e-300))
    correlation = covariance / diagonal_scale
    offdiagonal = correlation - torch.diag_embed(torch.diagonal(correlation, dim1=-2, dim2=-1))
    retained_energy = torch.sum(torch.abs(retained).square(), dim=(-2, -1)) / torch.sum(
        torch.abs(effective).square(), dim=(-2, -1)
    )
    return {
        "equalizer": equalizer,
        "response": response,
        "error_covariance": covariance,
        "stream_mse": diagonal,
        "gamma": gamma,
        "singular_values": singular,
        "gram_eigenvalues": gram_eigenvalues,
        "lambda_min": gram_eigenvalues[..., -1],
        "lambda_max": gram_eigenvalues[..., 0],
        "condition_number": singular[..., 0] / torch.clamp(singular[..., -1], min=1e-300),
        "effective_rank": effective_rank,
        "equalizer_noise_enhancement": torch.sum(torch.abs(equalizer).square(), dim=(-2, -1)) / n,
        "retained_energy": retained_energy,
        "mean_stream_mse": diagonal.mean(dim=-1),
        "worst_stream_mse": diagonal.max(dim=-1).values,
        "stream_mse_cvar95": stream_cvar(diagonal, 0.95),
        "median_gamma": gamma.median(dim=-1).values,
        "worst_gamma": gamma.min(dim=-1).values,
        "gamma_lower_cvar05": torch.topk(gamma, max(1, math.ceil(0.05 * n)), dim=-1, largest=False).values.mean(dim=-1),
        "error_anisotropy": diagonal.max(dim=-1).values / torch.clamp(diagonal.min(dim=-1).values, min=1e-300),
        "offdiagonal_error_correlation_rms": torch.sqrt(torch.sum(torch.abs(offdiagonal).square(), dim=(-2, -1)) / (n * (n - 1))),
    }


def distribution_summary(values: torch.Tensor, *, high_is_bad: bool = True) -> dict[str, float]:
    """Finite distribution summary with an explicit 5% adverse-tail mean."""
    flat = torch.as_tensor(values).detach().reshape(-1).to(torch.float64)
    flat = flat[torch.isfinite(flat)]
    if len(flat) == 0:
        return {key: float("nan") for key in ("mean", "median", "p05", "p95", "adverse_cvar05")}
    count = max(1, math.ceil(0.05 * len(flat)))
    adverse = torch.topk(flat, count, largest=high_is_bad).values
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p05": float(torch.quantile(flat, 0.05)),
        "p95": float(torch.quantile(flat, 0.95)),
        "adverse_cvar05": float(adverse.mean()),
    }


def support_geometry(mask: torch.Tensor, labels: torch.Tensor, order: torch.Tensor) -> dict[str, float]:
    """Compare a row support against Stage-2B communities and centroid order."""
    selected = mask.to(torch.bool)
    same = labels[:, None] == labels[None, :]
    within = selected & same
    positions = torch.empty_like(order)
    positions[order] = torch.arange(len(order), dtype=order.dtype)
    distance = torch.abs(positions[:, None] - positions[None, :])
    distance = torch.minimum(distance, len(order) - distance).to(torch.float64)
    count = torch.clamp(selected.sum().to(torch.float64), min=1.0)
    mutual = selected & selected.T
    adjacency = selected | selected.T
    visited: set[int] = set()
    component_sizes: list[int] = []
    for start in range(len(selected)):
        if start in visited:
            continue
        frontier = [start]; visited.add(start); size = 0
        while frontier:
            node = frontier.pop(); size += 1
            neighbors = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor); frontier.append(neighbor)
        component_sizes.append(size)
    return {
        "within_stage2b_community_fraction": float(within.sum() / count),
        "mean_centroid_order_distance": float(distance[selected].mean()),
        "p95_centroid_order_distance": float(torch.quantile(distance[selected], 0.95)),
        "mutual_edge_fraction": float(mutual.sum() / count),
        "undirected_component_count": float(len(component_sizes)),
        "largest_component_fraction": float(max(component_sizes) / len(selected)),
    }
