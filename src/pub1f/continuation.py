"""Continuation-only scientific kernels for Pub-1F."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from src.channel import COMPLEX_DTYPE, REAL_DTYPE, complex_awgn, make_generator, noise_variance
from src.metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from src.pub1e.core import ResidualStatistics
from src.pub1f.core import semantic_seed
from src.receiver import effective_operator, mismatched_lmmse_equalizer, rowwise_topk


CONTINUATION_NAMESPACE = "pub1f-cont-v1"


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def apply_support(actual: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return torch.where(support, actual, torch.zeros((), dtype=actual.dtype))


def fixed_energy_support(channels: torch.Tensor, waveform: torch.Tensor, k: int) -> torch.Tensor:
    effective = effective_operator(channels, waveform)
    scores = torch.mean(torch.abs(effective) ** 2, dim=0)
    return torch.abs(rowwise_topk(scores, k)) > 0


def matching_residual_statistics(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    k: int,
    support_policy: str,
    fixed_support: torch.Tensor | None = None,
) -> ResidualStatistics:
    actual = effective_operator(channels, waveform)
    if support_policy == "oracle_row_topk":
        retained = rowwise_topk(actual, k)
    elif support_policy == "fixed_statistical":
        if fixed_support is None:
            raise ValueError("fixed support required")
        retained = apply_support(actual, fixed_support)
    else:
        raise ValueError(f"unknown support policy: {support_policy}")
    omitted = actual - retained
    mean = omitted.mean(dim=0)
    centered = omitted - mean
    covariance = torch.mean(centered @ centered.mH, dim=0)
    return ResidualStatistics(mean, 0.5 * (covariance + covariance.mH), len(channels))


def direct_equalizer(
    actual: torch.Tensor,
    k: int,
    sigma2: float,
    receiver: str,
    *,
    regularization_scale: float = 1.0,
    support_policy: str = "oracle_row_topk",
    fixed_support: torch.Tensor | None = None,
    statistics: ResidualStatistics | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if receiver == "full":
        return mismatched_lmmse_equalizer(actual, sigma2), actual
    retained = rowwise_topk(actual, k) if support_policy == "oracle_row_topk" else apply_support(actual, fixed_support)
    if receiver == "residual_aware":
        if statistics is None:
            raise ValueError("residual statistics required")
        b = retained + statistics.mean
        identity = torch.eye(actual.shape[-1], dtype=COMPLEX_DTYPE)
        system = b @ b.mH + statistics.receive_covariance + sigma2 * identity
        return torch.linalg.solve(system, b).mH, retained
    scale = regularization_scale if receiver == "tuned" else 1.0
    return mismatched_lmmse_equalizer(retained, scale * sigma2), retained


def generic_communication_counts(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    constellation: Constellation,
    snr_db: int,
    k: int,
    receiver: str,
    bits_per_channel: int,
    *,
    condition: str,
    bank_role: str,
    waveform_key: str,
    regularization_scale: float = 1.0,
    support_policy: str = "oracle_row_topk",
    fixed_support: torch.Tensor | None = None,
    statistics: ResidualStatistics | None = None,
) -> list[dict[str, object]]:
    n = waveform.shape[-1]
    blocks = math.ceil(bits_per_channel / (n * constellation.bits_per_symbol))
    fields = {"bank_role": bank_role, "condition": condition, "modulation": constellation.name, "snr_db": snr_db}
    bits = random_bits((len(channels), blocks, n, constellation.bits_per_symbol), make_generator(semantic_seed("symbols", namespace=CONTINUATION_NAMESPACE, **fields)))
    symbols, true_indices = constellation_encode(bits, constellation)
    noise = complex_awgn((len(channels), blocks, n), noise_variance(snr_db), make_generator(semantic_seed("noise", namespace=CONTINUATION_NAMESPACE, **fields)))
    physical = torch.einsum("ij,cbj->cbi", waveform, symbols)
    received = torch.einsum("cij,cbj->cbi", channels, physical) + noise
    observed = torch.einsum("ij,cbj->cbi", waveform.mH, received)
    actual = effective_operator(channels, waveform)
    equalizer, retained = direct_equalizer(actual, k, float(noise_variance(snr_db)), receiver, regularization_scale=regularization_scale, support_policy=support_policy, fixed_support=fixed_support, statistics=statistics)
    estimate = torch.einsum("cij,cbj->cbi", equalizer, observed)
    decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
    bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
    symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
    tested_bits = blocks * n * constellation.bits_per_symbol
    tested_symbols = blocks * n
    retained_energy = torch.sum(torch.abs(retained) ** 2, dim=(-2, -1)) / torch.sum(torch.abs(actual) ** 2, dim=(-2, -1))
    return [{
        "channel_index": index, "waveform_key": waveform_key,
        "bit_errors": int(bit_errors[index]), "total_bits": tested_bits, "ber": int(bit_errors[index]) / tested_bits,
        "symbol_errors": int(symbol_errors[index]), "total_symbols": tested_symbols, "ser": int(symbol_errors[index]) / tested_symbols,
        "retained_energy": float(retained_energy[index]),
    } for index in range(len(channels))]


@dataclass
class DenseCGSnapshot:
    estimate: torch.Tensor
    relative_residual: torch.Tensor
    logical_iterations: torch.Tensor
    physical_iterations: int


def dense_cg_snapshots(
    retained: torch.Tensor,
    observations: torch.Tensor,
    sigma2: float,
    budgets: tuple[int, ...],
    tolerance: float = 1e-10,
) -> dict[int, DenseCGSnapshot]:
    """Dense matrix-free batched CG control; masked members still execute matvecs."""
    if retained.ndim != 3 or observations.ndim != 3 or retained.shape[0] != observations.shape[0]:
        raise ValueError("retained [C,N,N] and observations [C,B,N] required")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be strictly increasing")
    rhs = torch.einsum("cij,cbi->cbj", retained.conj(), observations)
    estimate = torch.zeros_like(rhs); residual = rhs.clone(); direction = rhs.clone()
    residual_sq = torch.sum(residual.conj() * residual, dim=-1).real
    rhs_norm = torch.sqrt(residual_sq).clamp_min(torch.finfo(REAL_DTYPE).tiny)
    relative = torch.sqrt(residual_sq) / rhs_norm
    converged = relative <= tolerance
    logical = torch.zeros_like(relative, dtype=torch.int64)
    snapshots = {}
    for iteration in range(1, budgets[-1] + 1):
        forward = torch.einsum("cij,cbj->cbi", retained, direction)
        product = torch.einsum("cij,cbi->cbj", retained.conj(), forward) + sigma2 * direction
        denominator = torch.sum(direction.conj() * product, dim=-1).real
        active = ~converged
        valid = active & torch.isfinite(denominator) & (denominator > 0)
        alpha = torch.where(valid, residual_sq / denominator.clamp_min(torch.finfo(REAL_DTYPE).tiny), 0.0)
        estimate = estimate + alpha[..., None] * direction
        residual = residual - alpha[..., None] * product
        next_sq = torch.sum(residual.conj() * residual, dim=-1).real
        logical[valid] = iteration
        relative = torch.sqrt(next_sq) / rhs_norm
        converged |= relative <= tolerance
        beta = torch.where(valid & ~converged, next_sq / residual_sq.clamp_min(torch.finfo(REAL_DTYPE).tiny), 0.0)
        direction = residual + beta[..., None] * direction
        residual_sq = next_sq
        if iteration in budgets:
            snapshots[iteration] = DenseCGSnapshot(estimate.clone(), relative.clone(), logical.clone(), iteration)
    return snapshots


def generic_iterative_counts(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    constellation: Constellation,
    snr_db: int,
    k: int,
    bits_per_channel: int,
    budgets: tuple[int, ...],
    *,
    condition: str,
    bank_role: str,
    waveform_key: str,
    support_policy: str = "oracle_row_topk",
    fixed_support: torch.Tensor | None = None,
    tolerance: float = 1e-10,
    regularization_scale: float = 1.0,
) -> list[dict[str, object]]:
    n = waveform.shape[-1]; blocks = math.ceil(bits_per_channel / (n * constellation.bits_per_symbol))
    fields = {"bank_role": bank_role, "condition": condition, "modulation": constellation.name, "snr_db": snr_db}
    bits = random_bits((len(channels), blocks, n, constellation.bits_per_symbol), make_generator(semantic_seed("symbols", namespace=CONTINUATION_NAMESPACE, **fields)))
    symbols, true_indices = constellation_encode(bits, constellation)
    noise = complex_awgn((len(channels), blocks, n), noise_variance(snr_db), make_generator(semantic_seed("noise", namespace=CONTINUATION_NAMESPACE, **fields)))
    received = torch.einsum("cij,cbj->cbi", channels, torch.einsum("ij,cbj->cbi", waveform, symbols)) + noise
    observed = torch.einsum("ij,cbj->cbi", waveform.mH, received)
    actual = effective_operator(channels, waveform)
    retained = rowwise_topk(actual, k) if support_policy == "oracle_row_topk" else apply_support(actual, fixed_support)
    snapshots = dense_cg_snapshots(retained, observed, regularization_scale * float(noise_variance(snr_db)), budgets, tolerance)
    output=[]; tested_bits=blocks*n*constellation.bits_per_symbol
    for budget, snapshot in snapshots.items():
        decoded_bits, _ = nearest_neighbor_decode(snapshot.estimate, constellation)
        errors = torch.sum(decoded_bits != bits, dim=(1,2,3))
        for index in range(len(channels)):
            output.append({"channel_index":index,"waveform_key":waveform_key,"iteration_budget":budget,
                "bit_errors":int(errors[index]),"total_bits":tested_bits,"ber":int(errors[index])/tested_bits,
                "mean_relative_residual":float(snapshot.relative_residual[index].mean()),
                "mean_logical_iterations":float(snapshot.logical_iterations[index].to(REAL_DTYPE).mean()),
                "physical_iterations":snapshot.physical_iterations})
    return output


def own_full_degradation(restricted_rows: list[dict[str, object]], full_rows: list[dict[str, object]]) -> dict[str, float]:
    restricted_errors=sum(int(row["bit_errors"]) for row in restricted_rows); restricted_bits=sum(int(row["total_bits"]) for row in restricted_rows)
    full_errors=sum(int(row["bit_errors"]) for row in full_rows); full_bits=sum(int(row["total_bits"]) for row in full_rows)
    restricted_ber=restricted_errors/restricted_bits; full_ber=full_errors/full_bits
    return {"restricted_ber":restricted_ber,"own_full_ber":full_ber,"absolute_degradation":restricted_ber-full_ber,"ratio_to_own_full":restricted_ber/full_ber if full_ber else float("inf")}
