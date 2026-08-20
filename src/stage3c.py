"""Frozen saturated-Q communication-geometry diagnostics for Stage 3C."""

from __future__ import annotations

import math

import numpy as np
import torch

from .channel import complex_awgn, make_generator, noise_variance
from .metrics import Constellation, constellation_encode, nearest_neighbor_decode, nearest_neighbor_margin, random_bits
from .receiver import effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from .receiver_geometry import apply_mask


def retained_operator(
    effective: torch.Tensor,
    support_rule: str,
    k: int,
    fixed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct the exact operator used to design each receiver mode."""
    if support_rule == "Full":
        return effective
    if support_rule == "Oracle":
        return rowwise_topk(effective, k)
    if support_rule == "Q":
        if fixed_mask is None:
            raise ValueError("fixed Q reception requires a mask")
        return apply_mask(effective, fixed_mask)
    raise ValueError("support rule must be Full, Oracle, or Q")


def residual_anisotropy(residual: torch.Tensor) -> torch.Tensor:
    """Per-channel real/imag residual covariance eigenvalue ratio."""
    real = residual.real.reshape(residual.shape[0], -1)
    imag = residual.imag.reshape(residual.shape[0], -1)
    real = real - real.mean(dim=-1, keepdim=True)
    imag = imag - imag.mean(dim=-1, keepdim=True)
    c00 = torch.mean(real.square(), dim=-1)
    c11 = torch.mean(imag.square(), dim=-1)
    c01 = torch.mean(real * imag, dim=-1)
    trace = c00 + c11
    radius = torch.sqrt(torch.clamp((c00 - c11).square() + 4 * c01.square(), min=0.0))
    maximum = 0.5 * (trace + radius)
    minimum = 0.5 * (trace - radius)
    return maximum / torch.clamp(minimum, min=1e-300)


def normalized_minimum_distance(constellation: Constellation) -> float:
    distances = torch.abs(constellation.points[:, None] - constellation.points[None, :])
    distances.fill_diagonal_(torch.inf)
    return float(torch.min(distances))


def communication_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    constellations: tuple[Constellation, ...],
    snrs_db: tuple[int, ...],
    *,
    k: int,
    target_bits: int,
    symbol_seed: int,
    noise_seed: int,
) -> list[dict[str, object]]:
    """Common-bank BER and constellation diagnostics for all receiver modes."""
    rows: list[dict[str, object]] = []
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    with torch.no_grad():
        for modulation_index, constellation in enumerate(constellations):
            dmin = normalized_minimum_distance(constellation)
            blocks = math.ceil(target_bits / (len(channels) * channels.shape[-1] * constellation.bits_per_symbol))
            bits_per_channel = blocks * channels.shape[-1] * constellation.bits_per_symbol
            symbols_per_channel = blocks * channels.shape[-1]
            for snr_index, snr_db in enumerate(snrs_db):
                bits = random_bits(
                    (len(channels), blocks, channels.shape[-1], constellation.bits_per_symbol),
                    make_generator(symbol_seed + 100 * modulation_index + snr_index),
                )
                symbols, true_indices = constellation_encode(bits, constellation)
                sigma2 = noise_variance(snr_db)
                physical_noise = complex_awgn(
                    (len(channels), blocks, channels.shape[-1]), sigma2,
                    make_generator(noise_seed + 100 * modulation_index + snr_index),
                )
                for name, waveform in waveforms.items():
                    transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
                    received = torch.einsum(
                        "ij,cbj->cbi", waveform.mH,
                        torch.einsum("cij,cbj->cbi", channels, transmitted) + physical_noise,
                    )
                    for support_rule in ("Full", "Oracle", "Q"):
                        retained = retained_operator(effective[name], support_rule, k, masks[name])
                        equalizer = mismatched_lmmse_equalizer(retained, sigma2)
                        estimate = torch.einsum("cij,cbj->cbi", equalizer, received)
                        decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                        margin = nearest_neighbor_margin(estimate, true_indices, constellation) / dmin
                        residual = estimate - symbols
                        bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                        symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                        empirical_mse = torch.mean(torch.abs(residual).square(), dim=(1, 2))
                        anisotropy = residual_anisotropy(residual)
                        margin_flat = margin.reshape(len(channels), -1)
                        for channel_index in range(len(channels)):
                            channel_margin = margin_flat[channel_index]
                            errors = int(bit_errors[channel_index])
                            symbol_error_count = int(symbol_errors[channel_index])
                            rows.append({
                                "channel_index": channel_index,
                                "modulation": constellation.name,
                                "snr_db": snr_db,
                                "waveform": name,
                                "support_rule": support_rule,
                                "k": channels.shape[-1] if support_rule == "Full" else k,
                                "bit_errors": errors,
                                "total_bits": bits_per_channel,
                                "ber": errors / bits_per_channel,
                                "symbol_errors": symbol_error_count,
                                "total_symbols": symbols_per_channel,
                                "ser": symbol_error_count / symbols_per_channel,
                                "empirical_mse": float(empirical_mse[channel_index]),
                                "constellation_scaled_variance": float(empirical_mse[channel_index]) / (dmin * dmin),
                                "normalized_margin_mean": float(channel_margin.mean()),
                                "normalized_margin_p05": float(torch.quantile(channel_margin, 0.05)),
                                "negative_margin_fraction": float((channel_margin < 0).to(torch.float64).mean()),
                                "residual_anisotropy": float(anisotropy[channel_index]),
                            })
    return rows


def paired_delta_summary(
    learned: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(learned, dtype=float)
    return {
        "mean_candidate_minus_learned": float(np.mean(delta)),
        "median_candidate_minus_learned": float(np.median(delta)),
        "p05_candidate_minus_learned": float(np.quantile(delta, 0.05)),
        "p95_candidate_minus_learned": float(np.quantile(delta, 0.95)),
        "fraction_candidate_lower": float(np.mean(delta < 0)),
    }


def support_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Mean row-wise intersection fraction for equal-size masks."""
    if left.shape != right.shape:
        raise ValueError("support masks must match")
    denominator = torch.clamp(left.to(torch.float64).sum(dim=-1), min=1.0)
    return float(((left & right).sum(dim=-1) / denominator).mean())
