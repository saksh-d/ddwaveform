"""Effective-channel truncation and mismatched finite-budget LMMSE receiver."""

import torch

from .channel import COMPLEX_DTYPE


def effective_operator(channel: torch.Tensor, waveform: torch.Tensor) -> torch.Tensor:
    return waveform.mH @ channel @ waveform


def rowwise_topk(matrix: torch.Tensor, k: int) -> torch.Tensor:
    n = matrix.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    if k == n:
        return matrix.clone()
    order = torch.argsort(torch.abs(matrix), dim=-1, descending=True, stable=True)
    keep_indices = order[..., :k]
    mask = torch.zeros_like(matrix, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, True)
    return torch.where(mask, matrix, torch.zeros((), dtype=matrix.dtype))


def mismatched_lmmse_equalizer(a_tilde: torch.Tensor, sigma2: float | torch.Tensor) -> torch.Tensor:
    n = a_tilde.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    gram = a_tilde.mH @ a_tilde + torch.as_tensor(sigma2, dtype=torch.float64)[..., None, None] * identity
    return torch.linalg.solve(gram, a_tilde.mH)


def analytic_communication_mse(a: torch.Tensor, equalizer: torch.Tensor, sigma2: float | torch.Tensor) -> torch.Tensor:
    n = a.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    distortion = torch.sum(torch.abs(equalizer @ a - identity) ** 2, dim=(-2, -1))
    noise = torch.as_tensor(sigma2, dtype=torch.float64) * torch.sum(
        torch.abs(equalizer) ** 2, dim=(-2, -1)
    )
    return (distortion + noise) / n


def finite_budget_receiver(a: torch.Tensor, sigma2: float | torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_tilde = rowwise_topk(a, k)
    equalizer = mismatched_lmmse_equalizer(a_tilde, sigma2)
    mse = analytic_communication_mse(a, equalizer, sigma2)
    return a_tilde, equalizer, mse


def full_information_mse(channel: torch.Tensor, sigma2: float | torch.Tensor) -> torch.Tensor:
    """Independent trace expression tr[(I + rho H^H H)^-1]/N."""
    n = channel.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma2_tensor = torch.as_tensor(sigma2, dtype=torch.float64)
    rho = 1.0 / sigma2_tensor
    system = identity + rho[..., None, None] * (channel.mH @ channel)
    covariance = torch.linalg.solve(system, identity.expand(system.shape))
    return torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1).real / n
