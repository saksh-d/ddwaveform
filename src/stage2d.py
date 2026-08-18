"""Frozen DD-coupling and fixed-support diagnostics for Stage 2D."""

from __future__ import annotations

import math

import numpy as np
import torch

from . import config
from .channel import COMPLEX_DTYPE, assemble_dd_channel, complex_awgn, make_generator, noise_variance
from .metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from .receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from .stage2b import spectral_centroids, spectral_overlap_matrix
from .tail_learning import stream_cvar

PRIOR_SEED = 11060
STRUCTURE_SEED = 11061
SOBOL_SEED = 11062
TEST_SEED = 11063
SYMBOL_SEED = 11064
NOISE_SEED = 11065


def score_support_mask(scores: np.ndarray | torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic row-wise top-k mask for fixed predictor scores."""
    values = torch.as_tensor(scores, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("support scores must be a square matrix")
    if not 1 <= k <= values.shape[-1]:
        raise ValueError(f"k must lie in [1, {values.shape[-1]}], got {k}")
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, order[:, :k], True)
    return mask


def centroid_neighborhood_mask(waveform: torch.Tensor, k: int) -> torch.Tensor:
    """Choose the k nearest circular positions in spectral-centroid order."""
    n = waveform.shape[-1]
    order = np.argsort(spectral_centroids(waveform), kind="mergesort")
    positions = np.empty(n, dtype=int)
    positions[order] = np.arange(n)
    distance = np.abs(positions[:, None] - positions[None, :])
    circular = np.minimum(distance, n - distance)
    # Tiny index term makes ties deterministic without changing unequal distances.
    scores = -circular.astype(float) - 1e-12 * np.arange(n)[None, :]
    return score_support_mask(scores, k)


def empirical_coupling(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor], chunk: int = 32
) -> dict[str, np.ndarray]:
    """C_ij = E_H |[W^H H W]_ij|^2 on one common held-out bank."""
    n = channels.shape[-1]
    totals = {name: torch.zeros((n, n), dtype=torch.float64) for name in waveforms}
    with torch.no_grad():
        for start in range(0, len(channels), chunk):
            current = channels[start : start + chunk]
            for name, waveform in waveforms.items():
                totals[name] += torch.sum(torch.abs(effective_operator(current, waveform)) ** 2, dim=0)
    return {name: (value / len(channels)).numpy() for name, value in totals.items()}


def empirical_support_prior(
    channels: torch.Tensor, waveform: torch.Tensor, k: int, chunk: int = 32
) -> np.ndarray:
    """P_K(i,j), estimated only from a dedicated construction bank."""
    n = waveform.shape[-1]
    total = torch.zeros((n, n), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(channels), chunk):
            effective = effective_operator(channels[start : start + chunk], waveform)
            total += torch.sum(torch.abs(rowwise_topk(effective, k)) > 0, dim=0)
    return (total / len(channels)).numpy()


def sobol_dd_samples(count: int, seed: int = SOBOL_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic scrambled Sobol samples over the exact DD-v1 path support."""
    if count < 2:
        raise ValueError("at least two ambiguity samples are required")
    unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(count).to(torch.float64)
    delays = config.DELAY_MIN + (config.DELAY_MAX - config.DELAY_MIN) * unit[:, 0]
    dopplers = config.DOPPLER_MIN + (config.DOPPLER_MAX - config.DOPPLER_MIN) * unit[:, 1]
    return delays, dopplers


def ambiguity_coupling(
    waveform: torch.Tensor,
    delays: torch.Tensor,
    dopplers: torch.Tensor,
    chunk: int = 32,
) -> np.ndarray:
    """Q_ij = E_{tau,nu}|w_i^H Delta(nu) Pi(tau) w_j|^2."""
    if delays.shape != dopplers.shape:
        raise ValueError("delay and Doppler samples must have matching shapes")
    n = waveform.shape[-1]
    total = torch.zeros((n, n), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(delays), chunk):
            tau = delays[start : start + chunk]
            nu = dopplers[start : start + chunk]
            paths = assemble_dd_channel(
                tau[:, None], nu[:, None],
                torch.ones((len(tau), 1), dtype=COMPLEX_DTYPE), n=n,
            )
            transformed = effective_operator(paths, waveform)
            total += torch.sum(torch.abs(transformed) ** 2, dim=0)
    return (total / len(delays)).numpy()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_rank(left: np.ndarray, right: np.ndarray) -> float:
    """Tie-aware Spearman correlation without a SciPy dependency."""
    a = _average_ranks(np.asarray(left, dtype=float).reshape(-1))
    b = _average_ranks(np.asarray(right, dtype=float).reshape(-1))
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scaled_frobenius_discrepancy(predictor: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return min_{alpha>=0} ||alpha X-Y||_F/||Y||_F and alpha."""
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(target, dtype=float)
    scale = max(float(np.sum(x * y) / max(np.sum(x * x), 1e-300)), 0.0)
    return float(np.linalg.norm(scale * x - y) / max(np.linalg.norm(y), 1e-300)), scale


def matrix_agreement(left: np.ndarray, right: np.ndarray, k: int) -> dict[str, float]:
    """Off-diagonal global/row ranks, scale discrepancy, and row top-k overlap."""
    n = len(left)
    off = ~np.eye(n, dtype=bool)
    row_ranks = [spearman_rank(left[row, np.arange(n) != row], right[row, np.arange(n) != row]) for row in range(n)]
    left_mask = score_support_mask(left, k).numpy()
    right_mask = score_support_mask(right, k).numpy()
    overlaps = np.sum(left_mask & right_mask, axis=1) / k
    discrepancy, scale = scaled_frobenius_discrepancy(left[off], right[off])
    return {
        "spearman_offdiagonal": spearman_rank(left[off], right[off]),
        "row_spearman_mean": float(np.nanmean(row_ranks)),
        "row_spearman_median": float(np.nanmedian(row_ranks)),
        "topk_overlap_mean": float(np.mean(overlaps)),
        "topk_overlap_p10": float(np.quantile(overlaps, 0.10)),
        "scaled_frobenius_discrepancy": discrepancy,
        "positive_scale": scale,
    }


def circular_spectral_separation(waveform: torch.Tensor) -> np.ndarray:
    centroids = spectral_centroids(waveform)
    raw = np.abs(centroids[:, None] - centroids[None, :])
    return np.minimum(raw, 1.0 - raw)


def locality_curve_rows(
    separation: np.ndarray,
    overlap: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    bins: int = 10,
) -> list[dict[str, object]]:
    """Continuous off-diagonal Q/C locality curves versus distance and overlap."""
    off = ~np.eye(len(c), dtype=bool)
    rows: list[dict[str, object]] = []
    for variable, values, edges in (
        ("spectral_separation", separation[off], np.linspace(0.0, 0.5, bins + 1)),
        ("spectral_overlap", overlap[off], np.quantile(overlap[off], np.linspace(0, 1, bins + 1))),
    ):
        edges = np.maximum.accumulate(edges)
        for index in range(bins):
            selected = (values >= edges[index]) & (values <= edges[index + 1] if index == bins - 1 else values < edges[index + 1])
            if not np.any(selected):
                continue
            for object_name, matrix in (("Q", q), ("C", c)):
                data = matrix[off][selected]
                rows.append({
                    "variable": variable,
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "center": float(np.mean(values[selected])),
                    "object": object_name,
                    "mean_coupling": float(np.mean(data)),
                    "median_coupling": float(np.median(data)),
                    "count": int(np.sum(selected)),
                })
    return rows


def apply_support(effective: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != effective.shape[-2:] and mask.shape != effective.shape:
        raise ValueError("support mask shape does not match effective operator")
    return torch.where(mask.to(torch.bool), effective, torch.zeros((), dtype=effective.dtype))


def support_metrics(
    effective: torch.Tensor,
    predicted_mask: torch.Tensor,
    k: int,
    sigma2: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-channel structural and mismatched-receiver metrics for one support rule."""
    oracle = torch.abs(rowwise_topk(effective, k)) > 0
    mask = predicted_mask.to(torch.bool)
    if mask.ndim == 2:
        mask = mask.expand_as(oracle)
    elif mask.shape != oracle.shape:
        raise ValueError("predicted support must be fixed 2-D or channel-wise")
    intersection = torch.sum(oracle & mask, dim=-1)
    union = torch.sum(oracle | mask, dim=-1)
    energy = torch.abs(effective) ** 2
    row_total = torch.sum(energy, dim=-1)
    row_retained = torch.sum(torch.where(mask, energy, 0.0), dim=-1) / row_total
    truncated = apply_support(effective, predicted_mask)
    equalizer = mismatched_lmmse_equalizer(truncated, sigma2)
    mse = analytic_communication_mse(effective, equalizer, sigma2)
    b = equalizer @ effective
    streams = torch.abs(torch.diagonal(b, dim1=-2, dim2=-1) - 1) ** 2
    streams += torch.sum(torch.abs(b - torch.diag_embed(torch.diagonal(b, dim1=-2, dim2=-1))) ** 2, dim=-1)
    streams += torch.as_tensor(sigma2, dtype=torch.float64) * torch.sum(torch.abs(equalizer) ** 2, dim=-1)
    singular = torch.linalg.svdvals(truncated)
    n = effective.shape[-1]
    limiting = torch.sum(torch.abs(torch.linalg.pinv(truncated, rtol=1e-12) @ effective - torch.eye(n, dtype=COMPLEX_DTYPE)) ** 2, dim=(-2, -1)) / n
    return {
        "support_recall": intersection.to(torch.float64).mean(dim=-1) / k,
        "jaccard": (intersection / union).mean(dim=-1),
        "retained_energy": torch.sum(torch.where(mask, energy, 0.0), dim=(-2, -1)) / torch.sum(energy, dim=(-2, -1)),
        "row_retained_energy": row_retained,
        "analytic_mse": mse,
        "limiting_distortion": limiting,
        "condition_number": singular[..., 0] / singular[..., -1],
        "stream_cvar_095": stream_cvar(streams, 0.95),
        "equalizer": equalizer,
    }


def support_communication_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, dict[str, torch.Tensor | None]],
    k_values: dict[str, int],
    constellations: tuple[Constellation, ...],
    snrs_db: tuple[int, ...],
    target_bits: int = 1_000_000,
    symbol_seed: int = SYMBOL_SEED,
    noise_seed: int = NOISE_SEED,
) -> list[dict[str, object]]:
    """Paired physical-noise BER for oracle and fixed support masks."""
    rows: list[dict[str, object]] = []
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    with torch.no_grad():
        for modulation_index, constellation in enumerate(constellations):
            blocks = math.ceil(target_bits / (len(channels) * config.N * constellation.bits_per_symbol))
            bits_per_channel = blocks * config.N * constellation.bits_per_symbol
            symbols_per_channel = blocks * config.N
            for snr_index, snr_db in enumerate(snrs_db):
                bits = random_bits((len(channels), blocks, config.N, constellation.bits_per_symbol), make_generator(symbol_seed + 100 * modulation_index + snr_index))
                symbols, true_indices = constellation_encode(bits, constellation)
                sigma2 = noise_variance(snr_db)
                noise = complex_awgn((len(channels), blocks, config.N), sigma2, make_generator(noise_seed + 100 * modulation_index + snr_index))
                for name, waveform in waveforms.items():
                    received = torch.einsum("ij,cbj->cbi", waveform.mH, torch.einsum("cij,cbj->cbi", channels, torch.einsum("ij,cbj->cbi", waveform, symbols)) + noise)
                    for method, fixed_mask in masks[name].items():
                        truncated = rowwise_topk(effective[name], k_values[name]) if fixed_mask is None else apply_support(effective[name], fixed_mask)
                        equalizer = mismatched_lmmse_equalizer(truncated, sigma2)
                        estimates = torch.einsum("cij,cbj->cbi", equalizer, received)
                        decoded_bits, decoded_indices = nearest_neighbor_decode(estimates, constellation)
                        bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                        symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                        for channel_index in range(len(channels)):
                            be = int(bit_errors[channel_index])
                            se = int(symbol_errors[channel_index])
                            rows.append({
                                "channel_index": channel_index,
                                "modulation": constellation.name,
                                "snr_db": snr_db,
                                "waveform": name,
                                "k": k_values[name],
                                "support_rule": method,
                                "bit_errors": be,
                                "total_bits": bits_per_channel,
                                "ber": be / bits_per_channel,
                                "symbol_errors": se,
                                "total_symbols": symbols_per_channel,
                                "ser": se / symbols_per_channel,
                            })
    return rows
