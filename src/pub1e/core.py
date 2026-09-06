"""Scientific core for Pub-1E.

No optimizer or waveform update lives here. Receiver restrictions never alter
the full physical/effective operator used to form observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from src import config
from src.audit import normalize_operators
from src.channel import COMPLEX_DTYPE, REAL_DTYPE, ChannelBatch, assemble_dd_channel, complex_awgn, make_generator, noise_variance
from src.metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk


@dataclass(frozen=True)
class ResidualStatistics:
    mean: torch.Tensor
    receive_covariance: torch.Tensor
    samples: int


def _inward_sign(anchor: torch.Tensor, lower: float, upper: float, generator: torch.Generator) -> torch.Tensor:
    signs = torch.where(
        torch.rand(anchor.shape, dtype=REAL_DTYPE, generator=generator) < 0.5,
        -torch.ones_like(anchor),
        torch.ones_like(anchor),
    )
    return torch.where(anchor <= lower, torch.ones_like(signs), torch.where(anchor >= upper, -torch.ones_like(signs), signs))


def controlled_fractionality_2d(count: int, seed: int, epsilon_tau: float, epsilon_nu: float) -> ChannelBatch:
    """Exact offsets from shared integer anchors with inward boundary signs."""
    if not 0.0 <= epsilon_tau <= 0.5 or not 0.0 <= epsilon_nu <= 0.5:
        raise ValueError("fractional offsets must lie in [0, 0.5]")
    generator = make_generator(seed)
    shape = (count, config.P)
    anchor_tau = torch.randint(0, 5, shape, generator=generator).to(REAL_DTYPE)
    anchor_nu = torch.randint(-1, 2, shape, generator=generator).to(REAL_DTYPE)
    tau = anchor_tau + epsilon_tau * _inward_sign(anchor_tau, 0.0, 4.0, generator)
    nu = anchor_nu + epsilon_nu * _inward_sign(anchor_nu, -1.0, 1.0, generator)
    gain_scale = math.sqrt(1.0 / (2 * config.P))
    gains = gain_scale * (
        torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    matrices, scales = normalize_operators(assemble_dd_channel(tau, nu, gains), config.N)
    return ChannelBatch(matrices, tau, nu, gains, scales)


def shifted_prior_channels(count: int, seed: int, family: str, severity: float) -> ChannelBatch:
    """One-factor prior shifts; severity 1/0 is the nominal member as documented."""
    if family not in {"delay", "doppler", "power"}:
        raise ValueError("family must be delay, doppler, or power")
    generator = make_generator(seed)
    shape = (count, config.P)
    delay_factor = severity if family == "delay" else 1.0
    doppler_factor = severity if family == "doppler" else 1.0
    delays = 4.0 * delay_factor * torch.rand(shape, dtype=REAL_DTYPE, generator=generator)
    dopplers = 0.75 * doppler_factor * (2 * torch.rand(shape, dtype=REAL_DTYPE, generator=generator) - 1)
    exponent = severity if family == "power" else 0.0
    weights = torch.exp(-exponent * delays / max(4.0 * delay_factor, 1e-12))
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    gains = torch.sqrt(weights / 2) * (
        torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    matrices, scales = normalize_operators(assemble_dd_channel(delays, dopplers, gains), config.N)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def fixed_energy_support(channels: torch.Tensor, waveform: torch.Tensor, k: int) -> torch.Tensor:
    """Row-top-K support of independently estimated ensemble effective energy."""
    effective = effective_operator(channels, waveform)
    scores = torch.mean(torch.abs(effective) ** 2, dim=0)
    return torch.abs(rowwise_topk(scores, k)) > 0


def apply_support(matrix: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return torch.where(support, matrix, torch.zeros((), dtype=matrix.dtype))


def residual_statistics(channels: torch.Tensor, waveform: torch.Tensor, k: int) -> ResidualStatistics:
    actual = effective_operator(channels, waveform)
    omitted = actual - rowwise_topk(actual, k)
    mean = torch.mean(omitted, dim=0)
    centered = omitted - mean
    covariance = torch.mean(centered @ centered.mH, dim=0)
    covariance = 0.5 * (covariance + covariance.mH)
    return ResidualStatistics(mean, covariance, len(channels))


def historical_equalizer(restricted: torch.Tensor, sigma2: float | torch.Tensor, regularization_scale: float = 1.0) -> torch.Tensor:
    return mismatched_lmmse_equalizer(restricted, regularization_scale * torch.as_tensor(sigma2, dtype=REAL_DTYPE))


def residual_aware_equalizer(restricted: torch.Tensor, sigma2: float | torch.Tensor, stats: ResidualStatistics) -> torch.Tensor:
    """Moment-matched approximation; not conditional-optimal for realized support."""
    b = restricted + stats.mean
    n = b.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    system = b @ b.mH + stats.receive_covariance + torch.as_tensor(sigma2, dtype=REAL_DTYPE)[..., None, None] * identity
    return torch.linalg.solve(system, b).mH


def receiver_equalizer(
    actual: torch.Tensor,
    k: int,
    sigma2: float | torch.Tensor,
    family: str,
    *,
    regularization_scale: float = 1.0,
    stats: ResidualStatistics | None = None,
    fixed_support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if family == "full":
        restricted = actual
        return restricted, historical_equalizer(restricted, sigma2)
    restricted = rowwise_topk(actual, k) if fixed_support is None else apply_support(actual, fixed_support)
    if family == "residual_aware":
        if stats is None:
            raise ValueError("residual-aware receiver requires independent statistics")
        return restricted, residual_aware_equalizer(restricted, sigma2, stats)
    if family in {"historical", "tuned"}:
        return restricted, historical_equalizer(restricted, sigma2, regularization_scale)
    raise ValueError(f"unknown receiver family: {family}")


def select_regularization(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    k: int,
    snrs_db: tuple[int, ...],
    candidates: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    actual = effective_operator(channels, waveform)
    restricted = rowwise_topk(actual, k)
    rows = []
    for scale in candidates:
        values = []
        for snr in snrs_db:
            equalizer = historical_equalizer(restricted, noise_variance(snr), scale)
            values.append(float(torch.mean(analytic_communication_mse(actual, equalizer, noise_variance(snr)))))
        rows.append({"regularization_scale": scale, "mean_analytic_mse": float(np.mean(values))})
    selected = min(rows, key=lambda row: (row["mean_analytic_mse"], row["regularization_scale"]))
    return float(selected["regularization_scale"]), rows


def support_metrics(actual: torch.Tensor, support: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    oracle = torch.abs(rowwise_topk(actual, k)) > 0
    expanded = support.expand_as(oracle)
    energy = torch.abs(actual) ** 2
    retained = torch.sum(torch.where(expanded, energy, 0.0), dim=(-2, -1)) / torch.sum(energy, dim=(-2, -1))
    recall = torch.sum(expanded & oracle, dim=-1).to(REAL_DTYPE).mean(dim=-1) / k
    return retained, recall


def communication_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    constellations: tuple[Constellation, ...],
    snrs_db: tuple[int, ...],
    k: int,
    receiver_family: str,
    bits_per_channel: int,
    symbol_seed: int,
    noise_seed: int,
    *,
    regularization: dict[str, float] | None = None,
    statistics: dict[str, ResidualStatistics] | None = None,
    supports: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, object]]:
    """Exact physical observation with channel-level counts and paired inputs."""
    rows: list[dict[str, object]] = []
    for modulation_index, constellation in enumerate(constellations):
        blocks = math.ceil(bits_per_channel / (config.N * constellation.bits_per_symbol))
        tested_bits = blocks * config.N * constellation.bits_per_symbol
        tested_symbols = blocks * config.N
        for snr_index, snr in enumerate(snrs_db):
            bits = random_bits((len(channels), blocks, config.N, constellation.bits_per_symbol), make_generator(symbol_seed + 100 * modulation_index + snr_index))
            symbols, true_indices = constellation_encode(bits, constellation)
            physical_noise = complex_awgn((len(channels), blocks, config.N), noise_variance(snr), make_generator(noise_seed + 100 * modulation_index + snr_index))
            for name, waveform in waveforms.items():
                x = torch.einsum("ij,cbj->cbi", waveform, symbols)
                y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
                observed = torch.einsum("ij,cbj->cbi", waveform.mH, y)
                actual = effective_operator(channels, waveform)
                restricted, equalizer = receiver_equalizer(
                    actual, k, noise_variance(snr), receiver_family,
                    regularization_scale=(regularization or {}).get(name, 1.0) if receiver_family == "tuned" else 1.0,
                    stats=(statistics or {}).get(name), fixed_support=(supports or {}).get(name),
                )
                estimate = torch.einsum("cij,cbj->cbi", equalizer, observed)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                retained_energy = torch.sum(torch.abs(restricted) ** 2, dim=(-2, -1)) / torch.sum(torch.abs(actual) ** 2, dim=(-2, -1))
                for channel_index in range(len(channels)):
                    be = int(bit_errors[channel_index])
                    se = int(symbol_errors[channel_index])
                    rows.append({
                        "channel_index": channel_index, "modulation": constellation.name, "snr_db": snr,
                        "waveform": name, "receiver_family": receiver_family, "k": k,
                        "bit_errors": be, "total_bits": tested_bits, "ber": be / tested_bits,
                        "symbol_errors": se, "total_symbols": tested_symbols, "ser": se / tested_symbols,
                        "retained_energy": float(retained_energy[channel_index]),
                    })
    return rows


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for values, selected in groups.items():
        errors = sum(int(row["bit_errors"]) for row in selected)
        bits = sum(int(row["total_bits"]) for row in selected)
        output.append({**dict(zip(keys, values)), "bit_errors": errors, "total_bits": bits, "ber": errors / bits, "n_channels": len(selected), "mean_retained_energy": float(np.mean([float(row["retained_energy"]) for row in selected]))})
    return output


def paired_channel_interval(left: list[dict[str, object]], right: list[dict[str, object]], seed: int, resamples: int = 10_000) -> dict[str, float]:
    left = sorted(left, key=lambda row: int(row["channel_index"]))
    right = sorted(right, key=lambda row: int(row["channel_index"]))
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired rows must have equal channel counts")
    differences = np.asarray([float(a["ber"]) - float(b["ber"]) for a, b in zip(left, right)])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(resamples, len(differences)))
    sampled = np.mean(differences[indices], axis=1)
    return {"mean_difference": float(np.mean(differences)), "ci95_low": float(np.quantile(sampled, .025)), "ci95_high": float(np.quantile(sampled, .975)), "resamples": resamples}


def safe_log10_ber_ratio(left_errors: int, left_bits: int, right_errors: int, right_bits: int) -> tuple[float, bool]:
    left = (left_errors + 0.5) / (left_bits + 1.0)
    right = (right_errors + 0.5) / (right_bits + 1.0)
    return float(np.log10(left / right)), left_errors < 5 or right_errors < 5
