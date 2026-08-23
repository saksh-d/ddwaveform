"""Population comparisons for Stage 4B learned-solution reproducibility."""

from __future__ import annotations

import math

import numpy as np
import torch

from .stage2d import matrix_agreement
from .stage4a import basis_fidelity, phase_permutation_align


def matrix_pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    """Phase/permutation-aligned matrix metrics with the full overlap distribution."""
    aligned, _ = phase_permutation_align(left, right)
    overlaps = torch.abs(torch.sum(left.conj() * aligned, dim=0)).detach().cpu().numpy()
    return {
        **basis_fidelity(left, right),
        "column_overlap_p25": float(np.quantile(overlaps, 0.25)),
        "column_overlap_median": float(np.median(overlaps)),
        "column_overlap_p75": float(np.quantile(overlaps, 0.75)),
        "column_overlaps": overlaps.tolist(),
    }


def aligned_operator_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    operators: torch.Tensor,
    q_left: np.ndarray,
    q_right: np.ndarray,
    *,
    k: int = 24,
    chunk_size: int = 32,
) -> dict[str, float]:
    """Compare complex atomic responses after legitimate basis alignment.

    ``d_b`` is the atom-wise mean relative squared Frobenius discrepancy.
    Coherences are global normalized inner products across every atom/coefficient.
    """
    aligned, _ = phase_permutation_align(left, right)
    discrepancy = 0.0
    complex_inner = 0.0j
    magnitude_inner = 0.0
    left_energy = 0.0
    right_energy = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(operators), chunk_size):
            current = operators[start : start + chunk_size]
            left_b = left.mH @ current @ left
            right_b = aligned.mH @ current @ aligned
            left_norm = torch.sum(torch.abs(left_b).square(), dim=(-2, -1))
            difference = torch.sum(torch.abs(left_b - right_b).square(), dim=(-2, -1))
            discrepancy += float(torch.sum(difference / torch.clamp(left_norm, min=1e-300)))
            complex_inner += complex(torch.sum(left_b.conj() * right_b))
            magnitude_inner += float(torch.sum(torch.abs(left_b) * torch.abs(right_b)))
            left_energy += float(torch.sum(torch.abs(left_b).square()))
            right_energy += float(torch.sum(torch.abs(right_b).square()))
            count += len(current)
    denominator = math.sqrt(max(left_energy * right_energy, 1e-300))
    agreement = matrix_agreement(q_left, q_right, k)
    return {
        "d_b": discrepancy / count,
        "complex_coefficient_coherence": abs(complex_inner) / denominator,
        "magnitude_coefficient_coherence": magnitude_inner / denominator,
        "q_relative_frobenius_distance": float(np.linalg.norm(q_left - q_right) / np.linalg.norm(q_left)),
        "q_correlation": float(np.corrcoef(q_left.reshape(-1), q_right.reshape(-1))[0, 1]),
        "top24_q_support_overlap": float(agreement["topk_overlap_mean"]),
    }


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    """Channel-paired percentile interval for right-minus-left mean."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape or left_values.ndim != 1 or len(left_values) < 2:
        raise ValueError("paired samples must be matching one-dimensional arrays")
    delta = right_values - left_values
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "paired_count": int(len(delta)),
        "resamples": int(resamples),
    }


def population_variation(values: np.ndarray) -> dict[str, float]:
    """Report ensemble variation without defining success by one tolerance."""
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "coefficient_of_variation": float(np.std(array, ddof=1) / abs(np.mean(array)))
        if len(array) > 1 and np.mean(array) != 0 else 0.0,
    }
