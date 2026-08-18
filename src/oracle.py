"""Channel-specific projected-unitary optimization for the Stage 0F diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from . import config
from .channel import COMPLEX_DTYPE, noise_variance
from .receiver import rowwise_topk
from .waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform, random_unitary


@dataclass
class OracleBatchResult:
    initialization_names: tuple[str, ...]
    initialization_objectives: torch.Tensor
    best_objectives_by_restart: torch.Tensor
    final_objectives: torch.Tensor
    best_steps_by_restart: torch.Tensor
    best_matrices_by_restart: torch.Tensor
    best_objectives: torch.Tensor
    best_matrices: torch.Tensor
    winning_restart_indices: torch.Tensor
    nonfinite_seen: torch.Tensor
    max_unitarity_residual: float
    traces: dict[str, list[float]]


def polar_project(matrix: torch.Tensor) -> torch.Tensor:
    """Nearest unitary matrix in Frobenius norm via the polar SVD factor."""
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def finite_budget_objective(
    waveform: torch.Tensor,
    channel: torch.Tensor,
    k_values: Iterable[int],
    snrs_db: Iterable[float],
) -> torch.Tensor:
    """Mean exact hard-top-K communication MSE over supplied K/SNR points."""
    a = waveform.mH @ channel @ waveform
    n = a.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma2 = noise_variance(torch.as_tensor(tuple(snrs_db), dtype=torch.float64))
    leading_dims = a.ndim - 2
    sigma_shape = (1,) * leading_dims + (sigma2.numel(), 1, 1)
    values = []
    for k in k_values:
        a_tilde = rowwise_topk(a, int(k))
        gram = a_tilde.mH @ a_tilde
        system = gram.unsqueeze(-3) + sigma2.reshape(sigma_shape) * identity
        rhs = a_tilde.mH.unsqueeze(-3).expand_as(system)
        equalizer = torch.linalg.solve(system, rhs)
        residual = equalizer @ a.unsqueeze(-3) - identity
        distortion = torch.sum(torch.abs(residual) ** 2, dim=(-2, -1))
        noise = sigma2.reshape((1,) * leading_dims + (sigma2.numel(),)) * torch.sum(
            torch.abs(equalizer) ** 2, dim=(-2, -1)
        )
        values.append(torch.mean((distortion + noise) / n, dim=-1))
    return torch.mean(torch.stack(values, dim=-1), dim=-1)


def learning_rate_for_step(step: int) -> float:
    if step <= 150:
        return 3e-2
    if step <= 250:
        return 1e-2
    return 3e-3


def oracle_initializations(
    channel_count: int,
    afdm_c1: float,
    random_seed: int,
    n: int = config.N,
) -> tuple[tuple[str, ...], torch.Tensor]:
    names = ("IDENTITY", "OFDM", "OTFS", "AFDM", "RANDOM_HAAR_1", "RANDOM_HAAR_2")
    if n != config.N:
        raise ValueError("Stage 0F initializations are defined for N=64")
    fixed = (identity_waveform(n), ofdm_waveform(n), otfs_waveform(), afdm_waveform(n=n, c1=afdm_c1))
    per_channel = []
    for channel_index in range(channel_count):
        random_one = random_unitary(n, random_seed + 2 * channel_index)
        random_two = random_unitary(n, random_seed + 2 * channel_index + 1)
        per_channel.append(torch.stack((*fixed, random_one, random_two)))
    return names, torch.stack(per_channel)


def optimize_oracle_batch(
    channels: torch.Tensor,
    afdm_c1: float,
    k_values: tuple[int, ...],
    snrs_db: tuple[int, ...] = config.ORACLE_SNRS_DB,
    steps: int = config.ORACLE_STEPS,
    random_seed: int = config.ORACLE_RANDOM_SEED,
    trace_channel_index: int = 0,
) -> OracleBatchResult:
    """Independently optimize six starts per channel, vectorized for execution."""
    channel_count = channels.shape[0]
    names, starts = oracle_initializations(channel_count, afdm_c1, random_seed, channels.shape[-1])
    expanded_channels = channels[:, None]
    with torch.no_grad():
        initial = finite_budget_objective(starts, expanded_channels, k_values, snrs_db)
    parameter = torch.nn.Parameter(starts.clone())
    optimizer = torch.optim.Adam([parameter], lr=learning_rate_for_step(1))
    best_values = initial.clone()
    best_matrices = starts.clone()
    best_steps = torch.zeros_like(initial, dtype=torch.int64)
    nonfinite_seen = ~torch.isfinite(initial)
    traces = {name: [float(initial[trace_channel_index, index])] for index, name in enumerate(names)}

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = learning_rate_for_step(step)
        optimizer.zero_grad(set_to_none=True)
        objectives = finite_budget_objective(parameter, expanded_channels, k_values, snrs_db)
        finite = torch.isfinite(objectives)
        nonfinite_seen |= ~finite
        safe_sum = torch.where(finite, objectives, torch.zeros_like(objectives)).sum()
        safe_sum.backward()
        optimizer.step()
        with torch.no_grad():
            matrix_finite = torch.isfinite(parameter.real).all(dim=(-2, -1)) & torch.isfinite(parameter.imag).all(
                dim=(-2, -1)
            )
            nonfinite_seen |= ~matrix_finite
            parameter[~matrix_finite] = best_matrices[~matrix_finite]
            parameter.copy_(polar_project(parameter))
            projected_objectives = finite_budget_objective(parameter, expanded_channels, k_values, snrs_db)
            finite_projected = torch.isfinite(projected_objectives)
            nonfinite_seen |= ~finite_projected
            improved = finite_projected & (projected_objectives < best_values)
            best_values[improved] = projected_objectives[improved]
            best_matrices[improved] = parameter.detach()[improved]
            best_steps[improved] = step
            for index, name in enumerate(names):
                traces[name].append(float(projected_objectives[trace_channel_index, index]))

    with torch.no_grad():
        final = finite_budget_objective(parameter, expanded_channels, k_values, snrs_db)
        winner = torch.argmin(best_values, dim=1)
        channel_indices = torch.arange(channel_count)
        selected_values = best_values[channel_indices, winner]
        selected_matrices = best_matrices[channel_indices, winner]
        identity = torch.eye(channels.shape[-1], dtype=COMPLEX_DTYPE)
        residuals = torch.linalg.matrix_norm(best_matrices.mH @ best_matrices - identity, dim=(-2, -1))
    return OracleBatchResult(
        names,
        initial,
        best_values,
        final,
        best_steps,
        best_matrices,
        selected_values,
        selected_matrices,
        winner,
        nonfinite_seen,
        float(torch.max(residuals)),
        traces,
    )


def tune_afdm_for_budgets(
    validation_channels: torch.Tensor,
    k_values: tuple[int, ...] = config.ORACLE_K_VALUES,
    snrs_db: tuple[int, ...] = config.ORACLE_SNRS_DB,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Validation-only AFDM selection for each K and the equal-weight multi-K objective."""
    objective_sets = {f"K{k}": (k,) for k in k_values}
    objective_sets["MULTIK"] = k_values
    rows: list[dict[str, object]] = []
    selected: dict[str, int] = {}
    for objective_name, objective_ks in objective_sets.items():
        best_multiplier = -1
        best_score = float("inf")
        for multiplier in config.AFDM_C1_CANDIDATE_MULTIPLIERS:
            c1 = multiplier / (2 * config.N)
            waveform = afdm_waveform(c1=c1)
            scores = finite_budget_objective(waveform, validation_channels, objective_ks, snrs_db)
            score = float(torch.mean(scores))
            rows.append(
                {
                    "objective": objective_name,
                    "k_values": ";".join(map(str, objective_ks)),
                    "c1_multiplier_m": multiplier,
                    "c1": c1,
                    "validation_objective": score,
                }
            )
            if score < best_score:
                best_score = score
                best_multiplier = multiplier
        selected[objective_name] = best_multiplier
    return rows, selected
