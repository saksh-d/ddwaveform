"""Stage 2E frozen-waveform communication and physical-forensic utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from . import config
from .audit import normalize_operators
from .channel import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    ChannelBatch,
    assemble_dd_channel,
    complex_awgn,
    make_generator,
    noise_variance,
)
from .metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from .receiver import (
    analytic_communication_mse,
    effective_operator,
    mismatched_lmmse_equalizer,
    rowwise_topk,
)
from .stage2b import operator_offset_metrics, spectral_centroids
from .stage2d import ambiguity_coupling, apply_support, score_support_mask
from .waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform


AFDM_SEARCH_MULTIPLIERS = tuple(range(1, 12))
AFDM_INTEGER_CANONICAL_MULTIPLIER = 3
AFDM_GUARDED_CANONICAL_MULTIPLIER = 5


@dataclass(frozen=True)
class Regime:
    name: str
    delay_kind: str
    delay_max: float
    doppler_kind: str
    doppler_max: float
    paths: int = config.P


REGIMES = (
    Regime("GRID-DD", "integer", 4.0, "integer", 1.0),
    Regime("fractional Doppler", "integer", 4.0, "continuous", 0.75),
    Regime("fractional delay", "continuous", 4.0, "continuous", 0.20),
    Regime("DD-v1", "continuous", 4.0, "continuous", 0.75),
    Regime("wider Doppler", "continuous", 4.0, "continuous", 1.50),
    Regime("wider delay", "continuous", 8.0, "continuous", 0.75),
    Regime("more paths", "continuous", 4.0, "continuous", 0.75, paths=5),
)


def afdm_doppler_components(nu: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-integer AFDM decomposition nu=alpha+a with a in [-1/2,1/2)."""
    values = np.asarray(nu, dtype=float)
    alpha = np.floor(values + 0.5).astype(int)
    fractional = values - alpha
    return alpha, fractional


def afdm_design_record(
    nu_min: float = config.DOPPLER_MIN,
    nu_max: float = config.DOPPLER_MAX,
    leakage_guard: int = config.AFDM_DDV1_LEAKAGE_GUARD,
    n: int = config.N,
) -> dict[str, object]:
    """Derive integer and guarded published AFDM support rules from channel support."""
    probes = np.asarray([nu_min, nu_max, -0.5, 0.5 - np.finfo(float).eps])
    integer, fractional = afdm_doppler_components(probes)
    alpha_max = int(np.max(np.abs(integer)))
    integer_multiplier = 2 * alpha_max + 1
    guarded_multiplier = 2 * (alpha_max + leakage_guard) + 1
    return {
        "normalized_doppler_interval": [float(nu_min), float(nu_max)],
        "decomposition": "nu=alpha+a, alpha=floor(nu+1/2), a in [-1/2,1/2)",
        "alpha_max": alpha_max,
        "fractional_remainders_at_endpoints": fractional[:2].tolist(),
        "xi_nu": leakage_guard,
        "integer_multiplier_2Nc1": integer_multiplier,
        "integer_c1": integer_multiplier / (2 * n),
        "guarded_multiplier_2Nc1": guarded_multiplier,
        "guarded_c1": guarded_multiplier / (2 * n),
        "feasibility_left_side": 2 * (alpha_max + leakage_guard) * (int(config.DELAY_MAX) + 1) + int(config.DELAY_MAX),
        "feasibility_n": n,
    }


def stage2e_waveforms(learned: torch.Tensor, afdm_multiplier: int) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=afdm_multiplier / (2 * config.N)),
        "Learned": learned,
    }


def _gains(count: int, paths: int, generator: torch.Generator) -> torch.Tensor:
    scale = math.sqrt(1.0 / (2 * paths))
    return (scale * (
        torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator)
    )).to(COMPLEX_DTYPE)


def sample_regime(regime: Regime, count: int, seed: int, n: int = config.N) -> ChannelBatch:
    """Sample a documented stress regime with per-realization Frobenius normalization."""
    generator = make_generator(seed)
    p = regime.paths
    if regime.delay_kind == "integer":
        delays = torch.randint(0, int(regime.delay_max) + 1, (count, p), generator=generator).to(REAL_DTYPE)
    else:
        delays = regime.delay_max * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator)
    if regime.doppler_kind == "integer":
        bound = int(regime.doppler_max)
        dopplers = torch.randint(-bound, bound + 1, (count, p), generator=generator).to(REAL_DTYPE)
    else:
        dopplers = (2 * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator) - 1) * regime.doppler_max
    gains = _gains(count, p, generator)
    raw = assemble_dd_channel(delays, dopplers, gains, n=n)
    matrices, scales = normalize_operators(raw, n)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def regime_q_samples(regime: Regime, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Path-law samples for Q; independent of multipath gains and communication banks."""
    unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(count).to(REAL_DTYPE)
    if regime.delay_kind == "integer":
        delays = torch.floor(unit[:, 0] * (int(regime.delay_max) + 1)).clamp(max=int(regime.delay_max))
    else:
        delays = regime.delay_max * unit[:, 0]
    if regime.doppler_kind == "integer":
        values = 2 * int(regime.doppler_max) + 1
        dopplers = torch.floor(unit[:, 1] * values).clamp(max=values - 1) - int(regime.doppler_max)
    else:
        dopplers = (2 * unit[:, 1] - 1) * regime.doppler_max
    return delays, dopplers


def q_masks_for_regime(
    waveforms: dict[str, torch.Tensor], regime: Regime, k: int, samples: int, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    delays, dopplers = regime_q_samples(regime, samples, seed)
    q = {name: ambiguity_coupling(waveform, delays, dopplers) for name, waveform in waveforms.items()}
    return {name: score_support_mask(value, k) for name, value in q.items()}, q


def afdm_validation_rows(
    channels: torch.Tensor,
    snrs_db: tuple[int, ...],
    multipliers: tuple[int, ...] = AFDM_SEARCH_MULTIPLIERS,
    k: int = 24,
    c2: float = config.AFDM_C2,
) -> list[dict[str, object]]:
    """Validation-only exact analytic finite-receiver sweep."""
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for multiplier in multipliers:
            waveform = afdm_waveform(c1=multiplier / (2 * config.N), c2=c2)
            effective = effective_operator(channels, waveform)
            retained = rowwise_topk(effective, k)
            for snr_db in snrs_db:
                sigma2 = noise_variance(snr_db)
                equalizer = mismatched_lmmse_equalizer(retained, sigma2)
                mse = analytic_communication_mse(effective, equalizer, sigma2)
                for index, value in enumerate(mse):
                    rows.append({
                        "channel_index": index,
                        "multiplier_2Nc1": multiplier,
                        "c1": multiplier / (2 * config.N),
                        "c2": c2,
                        "k": k,
                        "snr_db": snr_db,
                        "analytic_mse": float(value),
                    })
    return rows


def select_afdm_multiplier(rows: list[dict[str, object]]) -> tuple[int, list[dict[str, object]]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["multiplier_2Nc1"]), []).append(float(row["analytic_mse"]))
    summary = [{"multiplier_2Nc1": key, "c1": key / (2 * config.N), "mean_analytic_mse": float(np.mean(value))}
               for key, value in sorted(grouped.items())]
    selected = min(summary, key=lambda row: (row["mean_analytic_mse"], row["multiplier_2Nc1"]))
    return int(selected["multiplier_2Nc1"]), summary


def fixed_support_metrics(
    channels: torch.Tensor, waveform: torch.Tensor, mask: torch.Tensor, k: int, snr_db: float
) -> dict[str, torch.Tensor]:
    effective = effective_operator(channels, waveform)
    oracle_mask = torch.abs(rowwise_topk(effective, k)) > 0
    expanded = mask.expand_as(oracle_mask)
    energy = torch.abs(effective) ** 2
    retained = torch.sum(torch.where(expanded, energy, 0.0), dim=(-2, -1)) / torch.sum(energy, dim=(-2, -1))
    recall = torch.sum(expanded & oracle_mask, dim=-1).to(torch.float64).mean(dim=-1) / k
    truncated = apply_support(effective, mask)
    equalizer = mismatched_lmmse_equalizer(truncated, noise_variance(snr_db))
    mse = analytic_communication_mse(effective, equalizer, noise_variance(snr_db))
    # The operational zero-noise limit is evaluated with a tiny ridge. Exact
    # batched SVD can fail to converge for deliberately singular fixed masks.
    limiting_equalizer = mismatched_lmmse_equalizer(truncated, 1e-10)
    limiting = torch.sum(torch.abs(limiting_equalizer @ effective - torch.eye(config.N, dtype=COMPLEX_DTYPE)) ** 2, dim=(-2, -1)) / config.N
    return {"retained_energy": retained, "support_recall": recall, "analytic_mse": mse, "limiting_distortion": limiting}


def communication_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    receiver_masks: dict[str, dict[str, torch.Tensor | None]],
    constellations: tuple[Constellation, ...],
    snrs_db: tuple[int, ...],
    target_bits: int,
    symbol_seed: int,
    noise_seed: int,
) -> list[dict[str, object]]:
    """Paired BER for named fixed/oracle/full receiver masks."""
    rows: list[dict[str, object]] = []
    n = channels.shape[-1]
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    with torch.no_grad():
        for modulation_index, constellation in enumerate(constellations):
            blocks = math.ceil(target_bits / (len(channels) * n * constellation.bits_per_symbol))
            bits_per_channel = blocks * n * constellation.bits_per_symbol
            for snr_index, snr_db in enumerate(snrs_db):
                bits = random_bits((len(channels), blocks, n, constellation.bits_per_symbol), make_generator(symbol_seed + 100 * modulation_index + snr_index))
                symbols, indices = constellation_encode(bits, constellation)
                noise = complex_awgn((len(channels), blocks, n), noise_variance(snr_db), make_generator(noise_seed + 100 * modulation_index + snr_index))
                for name, waveform in waveforms.items():
                    transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
                    physical = torch.einsum("cij,cbj->cbi", channels, transmitted) + noise
                    received = torch.einsum("ij,cbj->cbi", waveform.mH, physical)
                    for mode, mask in receiver_masks[name].items():
                        if mode == "full":
                            truncated = effective[name]
                            k = n
                        elif mode.startswith("oracle"):
                            k = int(mode.removeprefix("oracle"))
                            truncated = rowwise_topk(effective[name], k)
                        else:
                            k = int(mode.split("K")[-1])
                            if mask is None:
                                raise ValueError(f"missing fixed mask for {name}/{mode}")
                            truncated = apply_support(effective[name], mask)
                        equalizer = mismatched_lmmse_equalizer(truncated, noise_variance(snr_db))
                        estimate = torch.einsum("cij,cbj->cbi", equalizer, received)
                        decoded, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                        bit_errors = torch.sum(decoded != bits, dim=(1, 2, 3))
                        symbol_errors = torch.sum(decoded_indices != indices, dim=(1, 2))
                        for channel_index in range(len(channels)):
                            rows.append({
                                "channel_index": channel_index, "modulation": constellation.name,
                                "snr_db": snr_db, "waveform": name, "receiver_mode": mode, "k": k,
                                "bit_errors": int(bit_errors[channel_index]), "total_bits": bits_per_channel,
                                "symbol_errors": int(symbol_errors[channel_index]), "total_symbols": blocks * n,
                            })
    return rows


def path_separation_record(waveform: torch.Tensor, delay: float, doppler: float, predicted: int | None) -> dict[str, object]:
    operator = effective_operator(assemble_dd_channel(
        torch.tensor([delay]), torch.tensor([doppler]), torch.tensor([1 + 0j], dtype=COMPLEX_DTYPE)
    ), waveform)
    summary = operator_offset_metrics(operator)
    for k in (3, 16, 24, 32):
        top = rowwise_topk(operator, k)
        summary[f"retained_energy_k{k}"] = float(torch.sum(torch.abs(top) ** 2) / torch.sum(torch.abs(operator) ** 2))
    summary.update({"delay": delay, "doppler": doppler, "predicted_offset": predicted,
                    "observed_offset": summary["dominant_offset"]})
    return summary


def atom_spectral_time_metrics(waveform: torch.Tensor, fft_size: int = 16384) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    values = waveform.detach().resolve_conj().cpu().numpy()
    spectrum = np.fft.fftshift(np.fft.fft(values, n=fft_size, axis=0), axes=0)
    power = np.abs(spectrum) ** 2
    power /= np.maximum(power.sum(axis=0, keepdims=True), 1e-300)
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size))
    centroids = spectral_centroids(waveform, fft_size)
    delta = ((frequencies[:, None] - centroids[None, :] + 0.5) % 1.0) - 0.5
    bandwidth = np.sqrt(np.sum(power * delta ** 2, axis=0))
    entropy = -np.sum(power * np.log(np.maximum(power, 1e-300)), axis=0) / np.log(fft_size)
    sorted_power = np.sort(power, axis=0)[::-1]
    magnitude = np.abs(values)
    time_power = magnitude ** 2
    time = np.arange(values.shape[0])[:, None]
    time_centroid = np.sum(time * time_power, axis=0)
    time_spread = np.sqrt(np.sum((time - time_centroid) ** 2 * time_power, axis=0))
    metrics = {
        "centroid": centroids, "rms_bandwidth": bandwidth, "spectral_entropy": entropy,
        "top4_spectral_energy": sorted_power[:4].sum(axis=0),
        "top8_spectral_energy": sorted_power[:8].sum(axis=0),
        "top16_spectral_energy": sorted_power[:16].sum(axis=0),
        "magnitude_cv": magnitude.std(axis=0) / np.maximum(magnitude.mean(axis=0), 1e-300),
        "peak_magnitude": magnitude.max(axis=0), "time_centroid": time_centroid, "time_spread": time_spread,
    }
    db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-8))
    return metrics, frequencies, db


def stft_power(atom: np.ndarray, window_length: int = 16, hop: int = 2, fft_size: int = 128) -> np.ndarray:
    window = np.hanning(window_length)
    padded = np.pad(atom, (window_length // 2, window_length // 2), mode="wrap")
    starts = range(0, len(padded) - window_length + 1, hop)
    result = np.stack([np.fft.fftshift(np.fft.fft(padded[start:start + window_length] * window, fft_size)) for start in starts])
    power = np.abs(result) ** 2
    return power.T / max(float(power.max()), 1e-300)


def ambiguity_surface(
    left: torch.Tensor, right: torch.Tensor, delays: np.ndarray, dopplers: np.ndarray
) -> np.ndarray:
    """Squared cross ambiguity |left^H Delta(nu) Pi(tau) right|^2."""
    n = len(left)
    output = np.empty((len(dopplers), len(delays)))
    with torch.no_grad():
        for column, delay in enumerate(delays):
            delayed = assemble_dd_channel(
                torch.tensor([float(delay)]), torch.tensor([0.0]), torch.tensor([1 + 0j], dtype=COMPLEX_DTYPE), n=n
            ) @ right
            for row, doppler in enumerate(dopplers):
                samples = torch.arange(n, dtype=REAL_DTYPE)
                shifted = torch.exp(-2j * math.pi * float(doppler) * samples / n) * delayed
                output[row, column] = float(torch.abs(torch.vdot(left, shifted)) ** 2)
    return output


def dd_concentration_maps(
    waveform: torch.Tensor, delays: np.ndarray, dopplers: np.ndarray
) -> dict[str, np.ndarray]:
    shape = (len(dopplers), len(delays))
    maps = {name: np.empty(shape) for name in ("top3", "top8", "top16", "top24", "participation", "row_entropy")}
    for row, doppler in enumerate(dopplers):
        for column, delay in enumerate(delays):
            operator = effective_operator(assemble_dd_channel(
                torch.tensor([float(delay)]), torch.tensor([float(doppler)]), torch.tensor([1 + 0j], dtype=COMPLEX_DTYPE)
            ), waveform)
            energy = torch.abs(operator) ** 2
            total = torch.sum(energy)
            for k in (3, 8, 16, 24):
                maps[f"top{k}"][row, column] = float(torch.sum(torch.abs(rowwise_topk(operator, k)) ** 2) / total)
            probability = energy / torch.sum(energy, dim=-1, keepdim=True)
            maps["participation"][row, column] = float(torch.mean(1 / torch.sum(probability ** 2, dim=-1)))
            maps["row_entropy"][row, column] = float(torch.mean(-torch.sum(probability * torch.log(torch.clamp(probability, min=1e-300)), dim=-1) / math.log(config.N)))
    return maps


def papr_samples(waveform: torch.Tensor, symbols: torch.Tensor, oversample: int = 1) -> np.ndarray:
    samples = torch.einsum("ij,bj->bi", waveform, symbols).detach().cpu().numpy()
    if oversample > 1:
        spectrum = np.fft.fft(samples, axis=-1)
        shifted = np.fft.fftshift(spectrum, axes=-1)
        pad = (samples.shape[-1] * (oversample - 1)) // 2
        shifted = np.pad(shifted, ((0, 0), (pad, pad)))
        samples = np.fft.ifft(np.fft.ifftshift(shifted, axes=-1), axis=-1) * oversample
    power = np.abs(samples) ** 2
    return 10 * np.log10(np.max(power, axis=-1) / np.mean(power, axis=-1))


def simple_pairwise_explanation(q: np.ndarray, features: dict[str, np.ndarray]) -> dict[str, object]:
    """Rank correlations plus standardized linear regression; no black-box model."""
    off = np.triu_indices(len(q), 1)
    target = np.log10(np.maximum(q[off], 1e-15))
    names = list(features)
    x = np.column_stack([np.asarray(features[name])[off] for name in names])
    ranks = lambda values: np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float)
    correlations = {name: float(np.corrcoef(ranks(x[:, index]), ranks(target))[0, 1]) for index, name in enumerate(names)}
    standardized = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-15)
    y = (target - target.mean()) / max(target.std(), 1e-15)
    design = np.column_stack([np.ones(len(y)), standardized])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coefficients
    r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"spearman": correlations, "standardized_coefficients": dict(zip(names, coefficients[1:].tolist())), "linear_r2": float(r2)}
