"""Frozen-waveform qualification utilities for Stage 2A."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from . import config
from .channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance
from .diagnostics import representation_diagnostics
from .metrics import (
    Constellation,
    constellation_encode,
    nearest_neighbor_decode,
    qam16_constellation,
    qpsk_constellation,
    random_bits,
)
from .receiver import effective_operator, finite_budget_receiver, rowwise_topk
from .tail_learning import per_stream_exact_mse, stream_cvar
from .waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform

FROZEN_CHECKPOINT_SHA256 = "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc"
WAVEFORM_ORDER = ("IDENTITY", "OFDM", "OTFS", "AFDM", "Learned")
FIGURE_WAVEFORMS = ("Learned", "OFDM", "OTFS", "AFDM")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_waveform(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    """Load and validate the immutable validation-selected Stage-1A.2 tensor."""
    digest = file_sha256(path)
    if digest != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(f"unexpected frozen checkpoint SHA-256: {digest}")
    waveform = torch.load(path, weights_only=True)
    if waveform.shape != (config.N, config.N) or waveform.dtype != COMPLEX_DTYPE:
        raise RuntimeError(f"unexpected frozen tensor: shape={waveform.shape}, dtype={waveform.dtype}")
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    residual = float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity))
    if residual >= 1e-12:
        raise RuntimeError(f"frozen waveform is not unitary: residual={residual:.3e}")
    return waveform, {"path": str(path), "sha256": digest, "unitarity_residual": residual}


def qualification_waveforms(learned: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * config.N)),
        "Learned": learned,
    }


def analytic_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    snrs_db: tuple[int, ...] = config.STAGE1A_SNRS_DB,
    k: int = config.STAGE1A_K,
    chunk: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name in WAVEFORM_ORDER:
            waveform = waveforms[name]
            for start in range(0, len(channels), chunk):
                current = channels[start : start + chunk]
                effective = effective_operator(current, waveform)
                diagnostic = representation_diagnostics(effective, k)
                for snr_db in snrs_db:
                    streams = per_stream_exact_mse(waveform, current, noise_variance(snr_db), k)
                    mean = streams.mean(dim=-1)
                    tail = stream_cvar(streams, 0.95)
                    for offset in range(len(current)):
                        rows.append(
                            {
                                "channel_index": start + offset,
                                "snr_db": snr_db,
                                "waveform": name,
                                "k": k,
                                "analytic_mse": float(mean[offset]),
                                "mean_per_stream_mse": float(streams[offset].mean()),
                                "stream_cvar_095": float(tail[offset]),
                                "retained_energy": float(diagnostic["retained_energy"][offset]),
                                "limiting_distortion": float(diagnostic["limiting_distortion"][offset]),
                                "condition_number": float(diagnostic["condition_number"][offset]),
                            }
                        )
    return rows


def communication_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    constellations: tuple[Constellation, ...] = (qpsk_constellation(), qam16_constellation()),
    snrs_db: tuple[int, ...] = config.STAGE1A_SNRS_DB,
    k_values: tuple[int, ...] = (config.STAGE1A_K,),
    target_bits: int = 1_000_000,
    symbol_seed: int = config.STAGE1A2_SYMBOL_SEED,
    noise_seed: int = config.STAGE1A2_NOISE_SEED,
) -> list[dict[str, object]]:
    """Paired BER/SER evaluation with common channels, symbols, and physical noise."""
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        effective = {
            name: effective_operator(channels, waveform) for name, waveform in waveforms.items()
        }
        for modulation_index, constellation in enumerate(constellations):
            blocks = math.ceil(
                target_bits / (len(channels) * config.N * constellation.bits_per_symbol)
            )
            bits_per_channel = blocks * config.N * constellation.bits_per_symbol
            symbols_per_channel = blocks * config.N
            for snr_index, snr_db in enumerate(snrs_db):
                bits = random_bits(
                    (len(channels), blocks, config.N, constellation.bits_per_symbol),
                    make_generator(symbol_seed + 100 * modulation_index + snr_index),
                )
                symbols, true_indices = constellation_encode(bits, constellation)
                sigma2 = noise_variance(snr_db)
                noise = complex_awgn(
                    (len(channels), blocks, config.N),
                    sigma2,
                    make_generator(noise_seed + 100 * modulation_index + snr_index),
                )
                for name in WAVEFORM_ORDER:
                    waveform = waveforms[name]
                    x = torch.einsum("ij,cbj->cbi", waveform, symbols)
                    y = torch.einsum("cij,cbj->cbi", channels, x) + noise
                    received = torch.einsum("ij,cbj->cbi", waveform.mH, y)
                    for k in k_values:
                        _, equalizer, _ = finite_budget_receiver(effective[name], sigma2, k)
                        estimate = torch.einsum("cij,cbj->cbi", equalizer, received)
                        decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                        bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                        symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                        for channel_index in range(len(channels)):
                            be = int(bit_errors[channel_index])
                            se = int(symbol_errors[channel_index])
                            rows.append(
                                {
                                    "channel_index": channel_index,
                                    "modulation": constellation.name,
                                    "snr_db": snr_db,
                                    "waveform": name,
                                    "k": k,
                                    "bit_errors": be,
                                    "total_bits": bits_per_channel,
                                    "ber": be / bits_per_channel,
                                    "symbol_errors": se,
                                    "total_symbols": symbols_per_channel,
                                    "ser": se / symbols_per_channel,
                                }
                            )
    return rows


def aggregate_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for values, selected in groups.items():
        result = dict(zip(keys, values))
        if "bit_errors" in selected[0]:
            bit_errors = sum(int(row["bit_errors"]) for row in selected)
            bits = sum(int(row["total_bits"]) for row in selected)
            symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
            symbols = sum(int(row["total_symbols"]) for row in selected)
            result.update(
                bit_errors=bit_errors,
                total_bits=bits,
                ber=bit_errors / bits,
                symbol_errors=symbol_errors,
                total_symbols=symbols,
                ser=symbol_errors / symbols,
                zero_ber_95pct_upper=3 / bits if bit_errors == 0 else "",
            )
        else:
            for metric in (
                "analytic_mse",
                "mean_per_stream_mse",
                "stream_cvar_095",
                "retained_energy",
                "limiting_distortion",
                "condition_number",
            ):
                data = np.asarray([float(row[metric]) for row in selected])
                result[f"{metric}_mean"] = float(np.mean(data))
                result[f"{metric}_median"] = float(np.median(data))
                result[f"{metric}_p90"] = float(np.quantile(data, 0.90))
                result[f"{metric}_p95"] = float(np.quantile(data, 0.95))
        result["n_channels"] = len(selected)
        output.append(result)
    return output


def paired_ber_bootstrap(
    rows: list[dict[str, object]],
    comparator_names: tuple[str, ...] = ("OFDM", "OTFS", "AFDM", "IDENTITY"),
    resamples: int = config.STAGE1A2_BOOTSTRAP_RESAMPLES,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rng = np.random.default_rng(config.STAGE2A_BOOTSTRAP_SEED)
    settings = sorted({(str(r["modulation"]), int(r["snr_db"]), int(r["k"])) for r in rows})
    for modulation, snr_db, k in settings:
        learned = sorted(
            (r for r in rows if r["modulation"] == modulation and r["snr_db"] == snr_db and r["k"] == k and r["waveform"] == "Learned"),
            key=lambda row: int(row["channel_index"]),
        )
        for comparator in comparator_names:
            baseline = sorted(
                (r for r in rows if r["modulation"] == modulation and r["snr_db"] == snr_db and r["k"] == k and r["waveform"] == comparator),
                key=lambda row: int(row["channel_index"]),
            )
            delta = np.asarray([float(a["ber"]) - float(b["ber"]) for a, b in zip(learned, baseline)])
            indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
            means = delta[indices].mean(axis=1)
            output.append(
                {
                    "modulation": modulation,
                    "snr_db": snr_db,
                    "k": k,
                    "comparison": f"Learned minus {comparator}",
                    "mean_ber_difference": float(delta.mean()),
                    "ci95_lower": float(np.quantile(means, 0.025)),
                    "ci95_upper": float(np.quantile(means, 0.975)),
                    "resamples": resamples,
                }
            )
    return output


def k_sweep_analytic_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    k_values: tuple[int, ...] = config.STAGE2A_K_VALUES,
    snr_db: int = 18,
    chunk: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name in WAVEFORM_ORDER:
            for start in range(0, len(channels), chunk):
                current = channels[start : start + chunk]
                effective = effective_operator(current, waveforms[name])
                for k in k_values:
                    diagnostic = representation_diagnostics(effective, k)
                    _, _, mse = finite_budget_receiver(effective, noise_variance(snr_db), k)
                    for offset in range(len(current)):
                        rows.append(
                            {
                                "channel_index": start + offset,
                                "snr_db": snr_db,
                                "waveform": name,
                                "k": k,
                                "analytic_mse": float(mse[offset]),
                                "mean_per_stream_mse": float(mse[offset]),
                                "stream_cvar_095": float("nan"),
                                "retained_energy": float(diagnostic["retained_energy"][offset]),
                                "limiting_distortion": float(diagnostic["limiting_distortion"][offset]),
                                "condition_number": float(diagnostic["condition_number"][offset]),
                            }
                        )
    return rows


def representative_channel_index(channels: torch.Tensor, snr_db: int = 18, k: int = 24) -> tuple[int, np.ndarray]:
    """Choose nearest-to-median IDENTITY MSE before inspecting principal waveforms."""
    _, _, difficulty = finite_budget_receiver(channels, noise_variance(snr_db), k)
    values = difficulty.detach().cpu().numpy()
    median = np.median(values)
    return int(np.argmin(np.abs(values - median))), values


def _pairwise_transform_similarity(waveform: torch.Tensor, modulation: bool) -> np.ndarray:
    n = waveform.shape[0]
    samples = torch.arange(n, dtype=torch.float64)
    similarities = []
    for left in range(n):
        reference = waveform[:, left]
        for right in range(left + 1, n):
            candidate = waveform[:, right]
            transformed = torch.stack(
                [
                    candidate * torch.exp(2j * math.pi * shift * samples / n)
                    if modulation
                    else torch.roll(candidate, shift)
                    for shift in range(n)
                ]
            )
            similarities.append(float(torch.max(torch.abs(transformed.conj() @ reference))))
    return np.asarray(similarities)


def learned_structure(waveform: torch.Tensor) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    values = waveform.detach().cpu().numpy()
    n = waveform.shape[0]
    magnitude = np.abs(values)
    phase = np.unwrap(np.angle(values), axis=0)
    phase_derivative = np.diff(phase, axis=0)
    spectrum_complex = np.fft.fftshift(np.fft.fft(values, axis=0, norm="ortho"), axes=0)
    spectrum = np.abs(spectrum_complex)
    power = spectrum**2
    frequencies = np.fft.fftshift(np.fft.fftfreq(n))
    rows: list[dict[str, object]] = []
    for column in range(n):
        weights = magnitude[:, column] ** 2
        coefficients = {}
        residuals = {}
        sample = np.arange(n, dtype=float)
        for degree in (1, 2):
            fit = np.polyfit(sample, phase[:, column], degree, w=np.sqrt(weights))
            residual = phase[:, column] - np.polyval(fit, sample)
            coefficients[degree] = fit
            residuals[degree] = float(np.sqrt(np.average(residual**2, weights=weights)))
        spectral_power = power[:, column] / power[:, column].sum()
        circular = np.sum(spectral_power * np.exp(2j * np.pi * frequencies))
        centroid = float(np.angle(circular) / (2 * np.pi))
        distance = ((frequencies - centroid + 0.5) % 1.0) - 0.5
        entropy = float(-np.sum(spectral_power * np.log(spectral_power + 1e-300)) / np.log(n))
        rows.append(
            {
                "column": column,
                "magnitude_mean": float(magnitude[:, column].mean()),
                "magnitude_std": float(magnitude[:, column].std()),
                "magnitude_cv": float(magnitude[:, column].std() / magnitude[:, column].mean()),
                "magnitude_peak_to_mean": float(magnitude[:, column].max() / magnitude[:, column].mean()),
                "phase_linear_slope": float(coefficients[1][0]),
                "phase_linear_rmse": residuals[1],
                "phase_quadratic_coefficient": float(coefficients[2][0]),
                "phase_quadratic_rmse": residuals[2],
                "spectral_centroid_cycles_per_sample": centroid,
                "spectral_rms_bandwidth": float(np.sqrt(np.sum(spectral_power * distance**2))),
                "spectral_entropy_normalized": entropy,
                "spectral_top1_energy": float(np.sort(spectral_power)[-1]),
                "spectral_top4_energy": float(np.sort(spectral_power)[-4:].sum()),
                "spectral_top8_energy": float(np.sort(spectral_power)[-8:].sum()),
            }
        )
    shift = _pairwise_transform_similarity(waveform, modulation=False)
    modulation = _pairwise_transform_similarity(waveform, modulation=True)
    sorted_magnitudes = np.sort(magnitude, axis=0)
    permutation_residuals = []
    for left in range(n):
        for right in range(left + 1, n):
            permutation_residuals.append(
                np.linalg.norm(sorted_magnitudes[:, left] - sorted_magnitudes[:, right])
                / np.linalg.norm(sorted_magnitudes[:, left])
            )
    summary = {
        "column_magnitude_cv_median": float(np.median([r["magnitude_cv"] for r in rows])),
        "column_magnitude_cv_max": float(np.max([r["magnitude_cv"] for r in rows])),
        "phase_linear_rmse_median": float(np.median([r["phase_linear_rmse"] for r in rows])),
        "phase_quadratic_rmse_median": float(np.median([r["phase_quadratic_rmse"] for r in rows])),
        "phase_quadratic_rmse_below_0p5_fraction": float(np.mean([r["phase_quadratic_rmse"] < 0.5 for r in rows])),
        "spectral_entropy_median": float(np.median([r["spectral_entropy_normalized"] for r in rows])),
        "spectral_top4_energy_median": float(np.median([r["spectral_top4_energy"] for r in rows])),
        "spectral_top8_energy_median": float(np.median([r["spectral_top8_energy"] for r in rows])),
        "spectral_rms_bandwidth_median": float(np.median([r["spectral_rms_bandwidth"] for r in rows])),
        "cyclic_shift_similarity_median": float(np.median(shift)),
        "cyclic_shift_similarity_p95": float(np.quantile(shift, 0.95)),
        "modulation_similarity_median": float(np.median(modulation)),
        "modulation_similarity_p95": float(np.quantile(modulation, 0.95)),
        "permuted_magnitude_residual_median": float(np.median(permutation_residuals)),
        "permuted_magnitude_residual_p05": float(np.quantile(permutation_residuals, 0.05)),
        "centroid_adjacent_circular_step_median": float(
            np.median(np.abs(((np.diff([r["spectral_centroid_cycles_per_sample"] for r in rows]) + 0.5) % 1.0) - 0.5))
        ),
        "centroid_unwrapped_index_correlation": float(
            np.corrcoef(
                np.arange(n),
                np.unwrap(2 * np.pi * np.asarray([r["spectral_centroid_cycles_per_sample"] for r in rows])) / (2 * np.pi),
            )[0, 1]
        ),
    }
    arrays = {
        "magnitude": magnitude,
        "wrapped_phase": np.angle(values),
        "unwrapped_phase": phase,
        "phase_derivative": phase_derivative,
        "fft_magnitude": spectrum,
        "frequencies": frequencies,
        "cyclic_shift_similarities": shift,
        "modulation_similarities": modulation,
    }
    return rows, summary, arrays


def effective_operator_arrays(
    channel: torch.Tensor, waveforms: dict[str, torch.Tensor], k: int = 24
) -> dict[str, np.ndarray]:
    output = {}
    for name in FIGURE_WAVEFORMS[1:] + ("Learned",):
        effective = effective_operator(channel, waveforms[name])
        output[f"{name}_A"] = effective.detach().cpu().numpy()
        output[f"{name}_T{k}"] = rowwise_topk(effective, k).detach().cpu().numpy()
    return output
