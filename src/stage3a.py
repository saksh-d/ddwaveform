"""Formal coupling-law and Q-concentration utilities for frozen Stage 3A analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from . import config
from .channel import COMPLEX_DTYPE, REAL_DTYPE, assemble_dd_channel, make_generator


@dataclass(frozen=True)
class RawChannelBatch:
    """Unnormalized DD channels and their exact project normalization."""

    raw: torch.Tensor
    normalized: torch.Tensor
    delays: torch.Tensor
    dopplers: torch.Tensor
    gains: torch.Tensor
    alpha: torch.Tensor


def sample_raw_dd(
    count: int,
    seed: int,
    path_variances: tuple[float, ...],
    n: int = config.N,
) -> RawChannelBatch:
    """Sample iid-location raw channels with explicit independent gain variances."""
    if count < 1 or not path_variances or any(value <= 0 for value in path_variances):
        raise ValueError("count and every path variance must be positive")
    generator = make_generator(seed)
    p = len(path_variances)
    delays = config.DELAY_MIN + (config.DELAY_MAX - config.DELAY_MIN) * torch.rand(
        (count, p), dtype=REAL_DTYPE, generator=generator
    )
    dopplers = config.DOPPLER_MIN + (config.DOPPLER_MAX - config.DOPPLER_MIN) * torch.rand(
        (count, p), dtype=REAL_DTYPE, generator=generator
    )
    standard = (
        torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
    ).to(COMPLEX_DTYPE) / math.sqrt(2.0)
    scales = torch.sqrt(torch.as_tensor(path_variances, dtype=REAL_DTYPE))
    gains = standard * scales
    raw = assemble_dd_channel(delays, dopplers, gains, n=n)
    alpha = math.sqrt(n) / torch.linalg.matrix_norm(raw, dim=(-2, -1))
    return RawChannelBatch(raw, alpha[:, None, None] * raw, delays, dopplers, gains, alpha)


def q_concentration(q: np.ndarray, k: int) -> dict[str, object]:
    """Row and aggregate top-k Q mass plus concise concentration summaries."""
    matrix = np.asarray(q, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError("Q must be a nonnegative square matrix")
    n = matrix.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    row_total = matrix.sum(axis=1)
    if np.any(row_total <= 0):
        raise ValueError("every Q row must have positive mass")
    row_fraction = ordered[:, :k].sum(axis=1) / row_total
    probabilities = matrix / row_total[:, None]
    effective_support = 1.0 / np.sum(probabilities**2, axis=1)
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    normalized_entropy = entropy_terms.sum(axis=1) / math.log(n)
    return {
        "k": k,
        "eta_q": float(ordered[:, :k].sum() / matrix.sum()),
        "row_fraction": row_fraction,
        "row_mean": float(np.mean(row_fraction)),
        "row_median": float(np.median(row_fraction)),
        "row_p05": float(np.quantile(row_fraction, 0.05)),
        "row_worst": float(np.min(row_fraction)),
        "effective_support": effective_support,
        "effective_support_mean": float(np.mean(effective_support)),
        "effective_support_median": float(np.median(effective_support)),
        "normalized_entropy": normalized_entropy,
        "normalized_entropy_mean": float(np.mean(normalized_entropy)),
    }


def concentration_curve(q: np.ndarray) -> np.ndarray:
    """Aggregate eta_K^Q for K=1,...,N."""
    matrix = np.asarray(q, dtype=float)
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    return np.sum(np.cumsum(ordered, axis=1), axis=0) / np.sum(matrix)


def concentration_thresholds(
    q: np.ndarray, thresholds: tuple[float, ...] = (0.90, 0.95, 0.99, 0.995)
) -> dict[float, int]:
    curve = concentration_curve(q)
    return {threshold: int(np.searchsorted(curve, threshold, side="left") + 1) for threshold in thresholds}


def fixed_mask_expected_retained(c_normalized: np.ndarray, mask: np.ndarray) -> float:
    """Exact expected retained fraction when normalized ||H||_F^2=N."""
    coupling = np.asarray(c_normalized, dtype=float)
    support = np.asarray(mask, dtype=bool)
    if coupling.shape != support.shape:
        raise ValueError("coupling and mask must have matching shapes")
    return float(np.sum(coupling[support]) / np.sum(coupling))


def relative_frobenius_error(observed: np.ndarray, expected: np.ndarray) -> float:
    observed_array = np.asarray(observed, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    return float(np.linalg.norm(observed_array - expected_array) / np.linalg.norm(expected_array))


def batch_mean_relative_standard_error(batch_matrices: np.ndarray) -> float:
    """Frobenius norm of elementwise mean SE relative to the sample mean."""
    values = np.asarray(batch_matrices, dtype=float)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError("at least two matrix-valued batch means are required")
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
    return float(np.linalg.norm(standard_error) / np.linalg.norm(np.mean(values, axis=0)))
