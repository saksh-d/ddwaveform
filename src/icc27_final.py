"""Scientific kernels for the evaluation-only ICC27 N=64 confirmation."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from src.audit import normalize_operators
from src.channel import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    ChannelBatch,
    assemble_dd_channel,
    complex_awgn,
    make_generator,
    noise_variance,
)
from src.metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from src.pub1f.continuation import apply_support, fixed_energy_support
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk


NAMESPACE = "icc27-final-v1"


def semantic_seed(role: str, **fields: object) -> int:
    encoded = "|".join([NAMESPACE, role] + [f"{key}={fields[key]}" for key in sorted(fields)])
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def sample_channels(condition: str, count: int, seed: int, n: int = 64) -> ChannelBatch:
    definitions = {
        "nominal": (0.0, 4.0, -0.75, 0.75, 3),
        "wider_delay": (0.0, 8.0, -0.75, 0.75, 3),
        "wider_doppler": (0.0, 4.0, -1.5, 1.5, 3),
        "more_paths": (0.0, 4.0, -0.75, 0.75, 5),
    }
    generator = make_generator(seed)
    if condition == "integer_grid":
        paths = 3
        delays = torch.randint(0, 5, (count, paths), generator=generator).to(REAL_DTYPE)
        dopplers = torch.randint(-1, 2, (count, paths), generator=generator).to(REAL_DTYPE)
    elif condition in definitions:
        delay_min, delay_max, doppler_min, doppler_max, paths = definitions[condition]
        delays = delay_min + (delay_max - delay_min) * torch.rand(
            (count, paths), dtype=REAL_DTYPE, generator=generator
        )
        dopplers = doppler_min + (doppler_max - doppler_min) * torch.rand(
            (count, paths), dtype=REAL_DTYPE, generator=generator
        )
    else:
        raise ValueError(f"unknown ICC condition: {condition}")
    gain_scale = math.sqrt(1.0 / (2 * paths))
    gains = gain_scale * (
        torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    matrices, scales = normalize_operators(assemble_dd_channel(delays, dopplers, gains, n), n)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def receiver_state(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    mode: str,
    k: int,
    snr_db: int,
    fixed_support: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    actual = effective_operator(channels, waveform)
    if mode == "fixed_statistical_k24":
        if fixed_support is None:
            raise ValueError("fixed support is required")
        retained = apply_support(actual, fixed_support)
    elif mode == "realized_topk_k24":
        retained = rowwise_topk(actual, k)
    elif mode == "full_k64":
        retained = actual
    else:
        raise ValueError(f"unknown receiver mode: {mode}")
    sigma2 = float(noise_variance(snr_db))
    equalizer = mismatched_lmmse_equalizer(retained, sigma2)
    retained_energy = torch.sum(torch.abs(retained) ** 2, dim=(-2, -1)) / torch.sum(
        torch.abs(actual) ** 2, dim=(-2, -1)
    )
    return {
        "actual": actual,
        "equalizer": equalizer,
        "retained_energy": retained_energy,
        "analytic_mse": analytic_communication_mse(actual, equalizer, sigma2),
    }


def communication_rows(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    constellation: Constellation,
    state: dict[str, torch.Tensor],
    *,
    condition: str,
    bank_role: str,
    snr_db: int,
    bits_per_channel: int,
) -> list[dict[str, object]]:
    n = waveform.shape[-1]
    blocks = math.ceil(bits_per_channel / (n * constellation.bits_per_symbol))
    seed_fields = {
        "bank_role": bank_role,
        "condition": condition,
        "modulation": constellation.name,
        "snr_db": snr_db,
    }
    bits = random_bits(
        (len(channels), blocks, n, constellation.bits_per_symbol),
        make_generator(semantic_seed("symbols", **seed_fields)),
    )
    symbols, _ = constellation_encode(bits, constellation)
    noise = complex_awgn(
        (len(channels), blocks, n),
        noise_variance(snr_db),
        make_generator(semantic_seed("noise", **seed_fields)),
    )
    physical = torch.einsum("ij,cbj->cbi", waveform, symbols)
    received = torch.einsum("cij,cbj->cbi", channels, physical) + noise
    observed = torch.einsum("ij,cbj->cbi", waveform.mH, received)
    estimates = torch.einsum("cij,cbj->cbi", state["equalizer"], observed)
    decoded_bits, _ = nearest_neighbor_decode(estimates, constellation)
    errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
    tested_bits = blocks * n * constellation.bits_per_symbol
    return [
        {
            "channel_id": index,
            "bit_errors": int(errors[index]),
            "bits": tested_bits,
            "retained_energy": float(state["retained_energy"][index]),
            "analytic_mse": float(state["analytic_mse"][index]),
        }
        for index in range(len(channels))
    ]


def summarize(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summaries = []
    for key, values in groups.items():
        errors = sum(int(row["bit_errors"]) for row in values)
        bits = sum(int(row["bits"]) for row in values)
        summaries.append(
            {
                **dict(zip(group_keys, key)),
                "bit_errors": errors,
                "bits": bits,
                "ber": errors / bits,
                "channels": len(values),
                "mean_retained_energy": float(np.mean([float(row["retained_energy"]) for row in values])),
                "mean_analytic_mse": float(np.mean([float(row["analytic_mse"]) for row in values])),
            }
        )
    return summaries


__all__ = [
    "NAMESPACE",
    "communication_rows",
    "fixed_energy_support",
    "receiver_state",
    "sample_channels",
    "semantic_seed",
    "summarize",
]
