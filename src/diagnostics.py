"""Frozen receiver-output diagnostics for Stage 1A.1."""

from __future__ import annotations

import torch

from .channel import COMPLEX_DTYPE
from .receiver import rowwise_topk


def mse_decomposition(
    effective: torch.Tensor,
    equalizer: torch.Tensor,
    sigma2: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    n = effective.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    b = equalizer @ effective
    diagonal = torch.diagonal(b, dim1=-2, dim2=-1)
    diagonal_distortion = torch.sum(torch.abs(diagonal - 1) ** 2, dim=-1) / n
    offdiagonal = b - torch.diag_embed(diagonal)
    offdiagonal_distortion = torch.sum(torch.abs(offdiagonal) ** 2, dim=(-2, -1)) / n
    noise = torch.as_tensor(sigma2, dtype=torch.float64) * torch.sum(
        torch.abs(equalizer) ** 2, dim=(-2, -1)
    ) / n
    total = diagonal_distortion + offdiagonal_distortion + noise
    return {
        "b": b,
        "total": total,
        "diagonal": diagonal_distortion,
        "offdiagonal": offdiagonal_distortion,
        "noise": noise,
    }


def per_stream_diagnostics(
    b: torch.Tensor,
    equalizer: torch.Tensor,
    sigma2: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    diagonal = torch.diagonal(b, dim1=-2, dim2=-1)
    desired = torch.abs(diagonal) ** 2
    interference = torch.sum(torch.abs(b) ** 2, dim=-1) - desired
    output_noise = torch.as_tensor(sigma2, dtype=torch.float64) * torch.sum(
        torch.abs(equalizer) ** 2, dim=-1
    )
    gamma = desired / (interference + output_noise)
    return {
        "gamma": gamma,
        "desired_magnitude": torch.abs(diagonal),
        "desired_phase": torch.angle(diagonal),
        "interference": interference,
        "output_noise": output_noise,
    }


def representation_diagnostics(effective: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    n = effective.shape[-1]
    truncated = rowwise_topk(effective, k)
    retained = torch.sum(torch.abs(truncated) ** 2, dim=(-2, -1)) / torch.sum(
        torch.abs(effective) ** 2, dim=(-2, -1)
    )
    singular = torch.linalg.svdvals(truncated)
    condition = singular[..., 0] / singular[..., -1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    limiting = torch.sum(
        torch.abs(torch.linalg.pinv(truncated, rtol=1e-12) @ effective - identity) ** 2,
        dim=(-2, -1),
    ) / n
    return {
        "retained_energy": retained,
        "condition_number": condition,
        "limiting_distortion": limiting,
    }


def distribution_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().reshape(-1).to(torch.float64)
    return {
        "mean": float(torch.mean(flat)),
        "median": float(torch.median(flat)),
        "p10": float(torch.quantile(flat, 0.10)),
        "p05": float(torch.quantile(flat, 0.05)),
        "p01": float(torch.quantile(flat, 0.01)),
    }
