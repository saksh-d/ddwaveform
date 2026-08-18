"""Stage 2B frozen waveform and effective-operator forensic utilities."""

from __future__ import annotations

import math

import numpy as np
import torch

from . import config
from .audit import expected_afdm_column_offset
from .channel import COMPLEX_DTYPE, assemble_dd_channel
from .receiver import effective_operator, finite_budget_receiver, rowwise_topk

PERCENTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
PATH_COLORS = np.asarray(
    ((0.13, 0.47, 0.71), (0.84, 0.15, 0.16), (0.17, 0.63, 0.17), (0.58, 0.40, 0.74))
)


def positive_shift_coordinates(n: int = config.N, shift: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero display coordinates for y[r]=x[(r-shift) mod N]."""
    columns = np.arange(n)
    rows = (columns + shift) % n
    return rows, columns


def spectral_centroids(waveform: torch.Tensor, fft_size: int = 16384) -> np.ndarray:
    values = waveform.detach().resolve_conj().cpu().numpy()
    spectrum = np.fft.fft(values, n=fft_size, axis=0)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.fftfreq(fft_size)
    circular = np.sum(power * np.exp(2j * np.pi * frequencies[:, None]), axis=0)
    return np.angle(circular) / (2 * np.pi)


def representative_atom_indices(
    waveform: torch.Tensor, percentiles: tuple[float, ...] = PERCENTILES
) -> tuple[np.ndarray, np.ndarray]:
    centroids = spectral_centroids(waveform)
    order = np.argsort(centroids)
    positions = np.rint(np.asarray(percentiles) * (len(order) - 1)).astype(int)
    return order[positions], centroids


def dense_atom_spectra(
    waveform: torch.Tensor, indices: np.ndarray, fft_size: int = 16384, floor_db: float = -80.0
) -> tuple[np.ndarray, np.ndarray]:
    values = waveform[:, indices].detach().resolve_conj().cpu().numpy()
    spectrum = np.fft.fftshift(np.fft.fft(values, n=fft_size, axis=0), axes=0)
    magnitude = np.abs(spectrum)
    magnitude /= np.maximum(magnitude.max(axis=0, keepdims=True), 1e-300)
    db = np.maximum(20 * np.log10(np.maximum(magnitude, 1e-300)), floor_db)
    return np.fft.fftshift(np.fft.fftfreq(fft_size)), db


def single_path_operator(
    waveform: torch.Tensor, delay: float, doppler: float, gain: complex = 1.0 + 0j
) -> torch.Tensor:
    physical = assemble_dd_channel(
        torch.tensor([delay], dtype=torch.float64),
        torch.tensor([doppler], dtype=torch.float64),
        torch.tensor([gain], dtype=COMPLEX_DTYPE),
    )
    return effective_operator(physical, waveform)


def path_resolved_operators(
    waveform: torch.Tensor, delays: torch.Tensor, dopplers: torch.Tensor, gains: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [single_path_operator(waveform, float(tau), float(nu), complex(gain)) for tau, nu, gain in zip(delays, dopplers, gains)]
    )


def path_rgb(operators: torch.Tensor, maximum: float | None = None) -> np.ndarray:
    magnitude = torch.abs(operators).detach().cpu().numpy()
    scale = float(magnitude.max()) if maximum is None else maximum
    intensity = np.clip(magnitude / max(scale, 1e-300), 0, 1)
    colors = PATH_COLORS[: len(operators)]
    rgb = np.einsum("pij,pc->ijc", intensity, colors)
    return np.clip(rgb, 0, 1)


def measured_afdm_offsets(operator: torch.Tensor) -> tuple[int, ...]:
    dominant_columns = torch.argmax(torch.abs(operator), dim=-1)
    rows = torch.arange(operator.shape[-1])
    offsets = (dominant_columns - rows) % operator.shape[-1]
    return tuple(int(value) for value in torch.unique(offsets))


def operator_offset_metrics(operator: torch.Tensor) -> dict[str, object]:
    """Summarize energy over circular column-minus-row offsets."""
    n = operator.shape[-1]
    energy = torch.abs(operator) ** 2
    profile = torch.zeros(n, dtype=torch.float64)
    rows = torch.arange(n)
    for offset in range(n):
        profile[offset] = torch.sum(energy[rows, (rows + offset) % n])
    profile /= torch.sum(profile)
    order = torch.argsort(profile, descending=True)
    cumulative = torch.cumsum(profile[order], dim=0)
    width95 = int(torch.nonzero(cumulative >= 0.95, as_tuple=False)[0]) + 1
    dominant = int(torch.argmax(profile))
    circular = torch.sum(profile * torch.exp(2j * math.pi * torch.arange(n) / n))
    circular_center = float((torch.angle(circular) * n / (2 * math.pi)) % n)
    return {
        "dominant_offset": dominant,
        "circular_energy_center": circular_center,
        "dominant_offset_energy_fraction": float(profile[dominant]),
        "offset_width_95pct": width95,
    }


def afdm_path_audit(
    waveform: torch.Tensor,
    paths: tuple[tuple[int, int], ...],
    c1: float,
) -> list[dict[str, object]]:
    rows = []
    for delay, doppler in paths:
        operator = single_path_operator(waveform, delay, doppler)
        expected = expected_afdm_column_offset(delay, doppler, config.N, c1)
        measured = measured_afdm_offsets(operator)
        rows.append(
            {
                "delay": delay,
                "doppler": doppler,
                "expected_offset": expected,
                "measured_offsets": ";".join(map(str, measured)),
                "matches": measured == (expected,),
                "minimum_row_peak": float(torch.min(torch.max(torch.abs(operator), dim=-1).values)),
            }
        )
    return rows


def ensemble_operator_statistics(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    k: int = 24,
    chunk: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    energy = {name: torch.zeros((config.N, config.N), dtype=torch.float64) for name in waveforms}
    support = {name: torch.zeros((config.N, config.N), dtype=torch.float64) for name in waveforms}
    with torch.no_grad():
        for start in range(0, len(channels), chunk):
            current = channels[start : start + chunk]
            for name, waveform in waveforms.items():
                effective = effective_operator(current, waveform)
                energy[name] += torch.sum(torch.abs(effective) ** 2, dim=0)
                support[name] += torch.sum(torch.abs(rowwise_topk(effective, k)) > 0, dim=0)
    return (
        {name: (value / len(channels)).numpy() for name, value in energy.items()},
        {name: (value / len(channels)).numpy() for name, value in support.items()},
    )


def difficulty_quantile_indices(
    channels: torch.Tensor, quantiles: tuple[float, ...] = (0.10, 0.50, 0.90), snr_db: int = 18, k: int = 24
) -> tuple[list[int], np.ndarray]:
    sigma2 = 10.0 ** (-snr_db / 10.0)
    _, _, difficulty = finite_budget_receiver(channels, sigma2, k)
    values = difficulty.detach().cpu().numpy()
    indices = [int(np.argmin(np.abs(values - np.quantile(values, quantile)))) for quantile in quantiles]
    return indices, values


def spectral_overlap_matrix(waveform: torch.Tensor, fft_size: int = 4096) -> np.ndarray:
    spectrum = np.fft.fft(waveform.detach().resolve_conj().cpu().numpy(), n=fft_size, axis=0)
    power = np.abs(spectrum) ** 2
    power /= np.maximum(power.sum(axis=0, keepdims=True), 1e-300)
    root = np.sqrt(power)
    return root.T @ root


def coupling_matrix(energy: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (energy + energy.T)
    scale = np.sqrt(np.maximum(np.outer(np.diag(symmetric), np.diag(symmetric)), 1e-300))
    result = symmetric / scale
    np.fill_diagonal(result, 0.0)
    return np.maximum(result, 0.0)


def _deterministic_kmeans(values: np.ndarray, count: int, iterations: int = 100) -> np.ndarray:
    centers = [int(np.argmax(np.linalg.norm(values - values.mean(axis=0), axis=1)))]
    while len(centers) < count:
        distance = np.min(
            np.stack([np.sum((values - values[index]) ** 2, axis=1) for index in centers]), axis=0
        )
        centers.append(int(np.argmax(distance)))
    centroids = values[centers].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        distances = np.stack(
            [np.sum((values - centroid) ** 2, axis=1) for centroid in centroids], axis=1
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for label in range(count):
            if np.any(labels == label):
                centroids[label] = values[labels == label].mean(axis=0)
    return labels


def spectral_communities(similarity: np.ndarray, maximum_groups: int = 8) -> tuple[np.ndarray, dict[str, object]]:
    graph = 0.5 * (similarity + similarity.T)
    graph = graph.copy()
    np.fill_diagonal(graph, 0.0)
    degrees = graph.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degrees, 1e-15))
    laplacian = np.eye(len(graph)) - inverse[:, None] * graph * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    candidates = range(2, min(maximum_groups, len(graph) - 1) + 1)
    gaps = {count: float(eigenvalues[count] - eigenvalues[count - 1]) for count in candidates}
    count = max(candidates, key=lambda value: (gaps[value], -value))
    embedding = eigenvectors[:, :count]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-15)
    labels = _deterministic_kmeans(embedding, count)
    return labels, {
        "group_count": count,
        "eigengaps": {str(key): value for key, value in gaps.items()},
        "first_eigenvalues": eigenvalues[: maximum_groups + 1].tolist(),
    }


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        return result
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float:
    from math import comb

    total = len(left)
    left_values = np.unique(left)
    right_values = np.unique(right)
    contingency = np.asarray([[np.sum((left == a) & (right == b)) for b in right_values] for a in left_values])
    pairs = sum(comb(int(value), 2) for value in contingency.flat if value >= 2)
    left_pairs = sum(comb(int(value), 2) for value in contingency.sum(axis=1) if value >= 2)
    right_pairs = sum(comb(int(value), 2) for value in contingency.sum(axis=0) if value >= 2)
    expected = left_pairs * right_pairs / comb(total, 2)
    maximum = 0.5 * (left_pairs + right_pairs)
    return float((pairs - expected) / (maximum - expected)) if maximum != expected else 1.0


def community_summary(
    energy: np.ndarray,
    overlap: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    coupling = coupling_matrix(energy)
    labels, audit = spectral_communities(coupling)
    spectral_labels, spectral_audit = spectral_communities(overlap)
    upper = np.triu_indices(len(energy), 1)
    same = labels[:, None] == labels[None, :]
    within = float(energy[same].sum())
    cross = float(energy[~same].sum())
    order = np.argsort(centroids)
    ordered_labels = labels[order]
    transitions = int(np.sum(ordered_labels[1:] != ordered_labels[:-1]))
    groups = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        groups.append(
            {
                "label": int(label),
                "size": int(len(members)),
                "members": members.tolist(),
                "centroid_min": float(centroids[members].min()),
                "centroid_max": float(centroids[members].max()),
                "centroid_mean": float(centroids[members].mean()),
            }
        )
    summary = {
        **audit,
        "spectral_group_count": spectral_audit["group_count"],
        "group_sizes": [group["size"] for group in groups],
        "groups": groups,
        "within_energy_fraction": within / (within + cross),
        "within_to_cross_energy_ratio": within / max(cross, 1e-300),
        "spectral_operator_rank_correlation": rank_correlation(overlap[upper], coupling[upper]),
        "operator_spectral_adjusted_rand": adjusted_rand_index(labels, spectral_labels),
        "centroid_sorted_label_transitions": transitions,
        "raw_centroid_index_pearson": float(np.corrcoef(np.arange(len(centroids)), centroids)[0, 1]),
        "circular_unwrapped_centroid_index_pearson": float(
            np.corrcoef(
                np.arange(len(centroids)),
                np.unwrap(2 * np.pi * centroids) / (2 * np.pi),
            )[0, 1]
        ),
        "centroid_order": order.tolist(),
    }
    return labels, summary
