"""Frozen Phase-1 evaluation utilities.

No optimizer lives in this module.  Every finite receiver uses its own
realization-wise row-top-K effective operator.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src import config
from src.audit import normalize_operators
from src.channel import COMPLEX_DTYPE, REAL_DTYPE, ChannelBatch, assemble_dd_channel, make_generator
from src.receiver import effective_operator, full_information_mse, rowwise_topk
from src.stage2b import spectral_communities


EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
FRACTIONAL_AXES = ("delay", "doppler", "joint")
K_VALUES = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
PRIMARY_K = 24
PRIMARY_RHO = PRIMARY_K / config.N


def median_full_information_channel(
    channels: torch.Tensor, snr_db: float
) -> tuple[int, np.ndarray]:
    """Select a physical median channel without any waveform or finite-K input."""
    values = full_information_mse(channels, 10.0 ** (-snr_db / 10.0)).detach().cpu().numpy()
    median = float(np.median(values))
    index = int(np.argmin(np.abs(values - median)))
    return index, values


def balanced_symbol_indices(
    true_indices: torch.Tensor,
    *,
    symbol_count: int,
    per_symbol: int,
) -> np.ndarray:
    """Return a deterministic equal-size display subset for every symbol class."""
    flat = true_indices.detach().cpu().numpy().reshape(-1)
    selected = []
    for symbol in range(symbol_count):
        candidates = np.flatnonzero(flat == symbol)
        if len(candidates) < per_symbol:
            raise ValueError(f"symbol {symbol} has only {len(candidates)} samples; need {per_symbol}")
        selected.append(candidates[:per_symbol])
    return np.concatenate(selected)


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int,
) -> dict[str, float]:
    """Paired channel bootstrap for mean(left-right)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("paired inputs must be equal-length one-dimensional arrays with at least two samples")
    indices = np.random.default_rng(seed).integers(0, len(left), size=(resamples, len(left)))
    values = np.mean(left[indices] - right[indices], axis=1)
    return {
        "mean_difference": float(np.mean(left - right)),
        "ci95_low": float(np.quantile(values, .025)),
        "ci95_high": float(np.quantile(values, .975)),
        "bootstrap_resamples": resamples,
    }


def _inward_sign(anchor: torch.Tensor, lower: float, upper: float, generator: torch.Generator) -> torch.Tensor:
    sign = torch.where(
        torch.rand(anchor.shape, dtype=REAL_DTYPE, generator=generator) < 0.5,
        -torch.ones_like(anchor),
        torch.ones_like(anchor),
    )
    return torch.where(anchor <= lower, torch.ones_like(sign), torch.where(anchor >= upper, -torch.ones_like(sign), sign))


def controlled_fractionality_channels(
    count: int,
    seed: int,
    epsilon: float,
    axis: str,
) -> ChannelBatch:
    """Paired channels at an exact normalized offset from integer DD anchors."""
    if axis not in FRACTIONAL_AXES:
        raise ValueError(f"axis must be one of {FRACTIONAL_AXES}, got {axis}")
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("epsilon must lie in [0, 0.5]")
    generator = make_generator(seed)
    shape = (count, config.P)
    anchor_tau = torch.randint(0, 5, shape, generator=generator).to(REAL_DTYPE)
    anchor_nu = torch.randint(-1, 2, shape, generator=generator).to(REAL_DTYPE)
    tau_sign = _inward_sign(anchor_tau, 0.0, 4.0, generator)
    nu_sign = _inward_sign(anchor_nu, -1.0, 1.0, generator)
    tau = anchor_tau + (epsilon * tau_sign if axis in ("delay", "joint") else 0.0)
    nu = anchor_nu + (epsilon * nu_sign if axis in ("doppler", "joint") else 0.0)
    scale = math.sqrt(1.0 / (2 * config.P))
    gains = scale * (
        torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    matrices, scales = normalize_operators(assemble_dd_channel(tau, nu, gains), config.N)
    return ChannelBatch(matrices, tau, nu, gains, scales)


def retained_energy_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    ks: tuple[int, ...] = K_VALUES,
) -> list[dict[str, object]]:
    """Per-channel energy retained by each waveform's own oracle row-top-K."""
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name, waveform in waveforms.items():
            operator = effective_operator(channels, waveform)
            total = torch.sum(torch.abs(operator) ** 2, dim=(-2, -1))
            for k in ks:
                kept = rowwise_topk(operator, k)
                fractions = torch.sum(torch.abs(kept) ** 2, dim=(-2, -1)) / total
                for index, value in enumerate(fractions):
                    rows.append({
                        "channel_index": index,
                        "waveform": name,
                        "k": k,
                        "rho": k / operator.shape[-1],
                        "retained_energy": float(value),
                    })
    return rows


def thresholded_graph(full_energy: np.ndarray, k: int) -> np.ndarray:
    """Symmetric graph induced by row-top-K of one unchanged energy statistic."""
    retained = rowwise_topk(torch.as_tensor(full_energy, dtype=torch.float64), k).numpy()
    graph = 0.5 * (retained + retained.T)
    np.fill_diagonal(graph, 0.0)
    return graph


def connected_components(adjacency: np.ndarray) -> np.ndarray:
    labels = np.full(len(adjacency), -1, dtype=int)
    label = 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        labels[start] = label
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in np.flatnonzero(adjacency[current] > 0):
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(int(neighbor))
        label += 1
    return labels


def weighted_modularity(adjacency: np.ndarray, labels: np.ndarray) -> float:
    total_twice = float(adjacency.sum())
    if total_twice <= 0:
        return 0.0
    degree = adjacency.sum(axis=1)
    expected = np.outer(degree, degree) / total_twice
    same = labels[:, None] == labels[None, :]
    return float(np.sum((adjacency - expected)[same]) / total_twice)


def graph_diagnostics(full_energy: np.ndarray, ks: tuple[int, ...] = K_VALUES) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Threshold one fixed map; report topology without asserting a group law."""
    graphs: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    previous_support: np.ndarray | None = None
    for k in ks:
        graph = thresholded_graph(full_energy, k)
        support = graph > 0
        labels, spectral = spectral_communities(graph)
        components = connected_components(graph)
        same = labels[:, None] == labels[None, :]
        within = float(full_energy[same].sum())
        between = float(full_energy[~same].sum())
        overlap = float("nan")
        if previous_support is not None:
            overlap = float(np.logical_and(previous_support, support).sum() / np.logical_or(previous_support, support).sum())
        rows.append({
            "k": k,
            "rho": k / len(graph),
            "support_overlap_previous": overlap,
            "connected_components": int(len(np.unique(components))),
            "community_count": int(spectral["group_count"]),
            "community_sizes": ";".join(str(int(np.sum(labels == value))) for value in np.unique(labels)),
            "within_energy_fraction": within / (within + between),
            "within_to_between_energy": within / max(between, 1e-300),
            "weighted_modularity": weighted_modularity(graph, labels),
        })
        graphs[f"k{k}"] = graph
        previous_support = support
    return graphs, rows
