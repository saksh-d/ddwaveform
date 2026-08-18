"""Modulation-independent stream-tail refinement for Stage 1A.2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from . import config
from .channel import COMPLEX_DTYPE, make_generator, noise_variance, sample_dd_v1
from .oracle import polar_project
from .receiver import rowwise_topk


@dataclass(frozen=True)
class TailObjectiveSpec:
    name: str
    beta: float
    tail_lambda: float


@dataclass
class TailTrainingResult:
    specs: tuple[TailObjectiveSpec, ...]
    initial_mean_mse: torch.Tensor
    initial_cvar90: torch.Tensor
    initial_cvar95: torch.Tensor
    best_mean_mse: torch.Tensor
    best_cvar90: torch.Tensor
    best_cvar95: torch.Tensor
    best_steps: torch.Tensor
    best_matrices: torch.Tensor
    final_mean_mse: torch.Tensor
    final_cvar90: torch.Tensor
    final_cvar95: torch.Tensor
    final_matrices: torch.Tensor
    max_unitarity_residuals: torch.Tensor
    nonfinite_seen: torch.Tensor
    runtime_seconds: float
    trace_steps: list[int]
    training_traces: dict[str, list[float]]
    validation_mean_traces: dict[str, list[float]]
    validation_cvar95_traces: dict[str, list[float]]


def stream_tail_count(n: int, beta: float) -> int:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    return max(1, math.ceil((1.0 - beta) * n))


def stream_cvar(values: torch.Tensor, beta: float) -> torch.Tensor:
    """Mean of the deterministic largest ceil((1-beta)N) stream values."""
    count = stream_tail_count(values.shape[-1], beta)
    return torch.topk(values, count, dim=-1, largest=True, sorted=True).values.mean(dim=-1)


def per_stream_exact_mse(
    waveforms: torch.Tensor,
    channels: torch.Tensor,
    sigma2: torch.Tensor | float,
    k: int,
) -> torch.Tensor:
    """Exact expected MSE for each symbol stream.

    Waveforms are ``[R,N,N]`` or ``[N,N]``; channels are ``[B,N,N]``.
    The result is ``[R,B,N]`` or ``[B,N]``.
    """
    single = waveforms.ndim == 2
    matrices = waveforms.unsqueeze(0) if single else waveforms
    effective = matrices[:, None].mH @ channels[None] @ matrices[:, None]
    truncated = rowwise_topk(effective, k)
    n = channels.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma = torch.as_tensor(sigma2, dtype=torch.float64)
    if sigma.ndim == 0:
        sigma = sigma.expand(channels.shape[0])
    system = truncated.mH @ truncated + sigma[None, :, None, None] * identity
    equalizer = torch.linalg.solve(system, truncated.mH)
    residual = equalizer @ effective - identity
    distortion = torch.sum(torch.abs(residual) ** 2, dim=-1)
    noise = sigma[None, :, None] * torch.sum(torch.abs(equalizer) ** 2, dim=-1)
    result = distortion + noise
    return result[0] if single else result


def tail_objective(stream_mse: torch.Tensor, beta: float, tail_lambda: float) -> torch.Tensor:
    mean_mse = stream_mse.mean(dim=-1)
    tail_excess = stream_cvar(stream_mse, beta) - mean_mse
    return mean_mse + tail_lambda * tail_excess


def validation_tail_metrics(
    waveforms: torch.Tensor,
    validation_channels: torch.Tensor,
    *,
    k: int = config.STAGE1A_K,
    snrs_db: tuple[int, ...] = config.STAGE1A_SNRS_DB,
    channel_chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Frozen-bank mean and within-example stream-tail metrics."""
    single = waveforms.ndim == 2
    matrices = waveforms.unsqueeze(0) if single else waveforms
    totals = {
        "mean_mse": torch.zeros(matrices.shape[0], dtype=torch.float64),
        "cvar90": torch.zeros(matrices.shape[0], dtype=torch.float64),
        "cvar95": torch.zeros(matrices.shape[0], dtype=torch.float64),
    }
    count = 0
    with torch.no_grad():
        for start in range(0, validation_channels.shape[0], channel_chunk_size):
            channels = validation_channels[start : start + channel_chunk_size]
            for snr_db in snrs_db:
                streams = per_stream_exact_mse(matrices, channels, noise_variance(snr_db), k)
                totals["mean_mse"] += streams.mean(dim=-1).sum(dim=1)
                totals["cvar90"] += stream_cvar(streams, 0.90).sum(dim=1)
                totals["cvar95"] += stream_cvar(streams, 0.95).sum(dim=1)
                count += channels.shape[0]
    result = {name: value / count for name, value in totals.items()}
    return {name: value[0] for name, value in result.items()} if single else result


def select_feasible_checkpoint(
    mean_mse: torch.Tensor,
    cvar95: torch.Tensor,
    reference_mean_mse: float | torch.Tensor,
    gate: float = config.STAGE1A2_MEAN_MSE_GATE,
) -> int | None:
    """Select feasible lowest-CVaR checkpoint, exactly tie-broken by mean MSE."""
    threshold = float(torch.as_tensor(reference_mean_mse)) * gate
    feasible = torch.nonzero(mean_mse <= threshold, as_tuple=False).flatten().tolist()
    if not feasible:
        return None
    return min(feasible, key=lambda index: (float(cvar95[index]), float(mean_mse[index]), index))


def stage1a2_learning_rate(step: int, calibration: bool = False) -> float:
    if calibration:
        return 3e-3
    if step <= 500:
        return 3e-3
    if step <= 800:
        return 1e-3
    return 3e-4


def optimize_tail_waveforms(
    initial_matrices: torch.Tensor,
    specs: tuple[TailObjectiveSpec, ...],
    validation_channels: torch.Tensor,
    reference_mean_mse: float | torch.Tensor,
    *,
    steps: int,
    batch_size: int = config.STAGE1A2_TRAIN_BATCH_SIZE,
    validation_cadence: int = config.STAGE1A2_VALIDATION_CADENCE,
    k: int = config.STAGE1A_K,
    train_channel_seed: int,
    train_snr_seed: int,
    calibration: bool = False,
) -> TailTrainingResult:
    """Projected Adam with paired online data and feasibility-aware checkpoints."""
    if initial_matrices.shape[0] != len(specs):
        raise ValueError("one objective specification is required per matrix")
    n = initial_matrices.shape[-1]
    starts = polar_project(initial_matrices.to(COMPLEX_DTYPE))
    initial = validation_tail_metrics(starts, validation_channels, k=k)
    parameter = torch.nn.Parameter(starts.clone())
    optimizer = torch.optim.Adam([parameter], lr=stage1a2_learning_rate(1, calibration))
    threshold = float(torch.as_tensor(reference_mean_mse)) * config.STAGE1A2_MEAN_MSE_GATE
    feasible = initial["mean_mse"] <= threshold
    best_mean = initial["mean_mse"].clone()
    best_cvar90 = initial["cvar90"].clone()
    best_cvar95 = initial["cvar95"].clone()
    best_steps = torch.zeros(len(specs), dtype=torch.int64)
    best_matrices = starts.clone()
    # The mandated starts are feasible. Keep defensive state if a custom caller is not.
    best_cvar95[~feasible] = torch.inf
    nonfinite = ~torch.isfinite(initial["mean_mse"]) | ~torch.isfinite(initial["cvar95"])
    max_residuals = torch.zeros(len(specs), dtype=torch.float64)
    trace_steps = [0]
    training_traces = {spec.name: [float("nan")] for spec in specs}
    validation_mean_traces = {
        spec.name: [float(initial["mean_mse"][index])] for index, spec in enumerate(specs)
    }
    validation_cvar95_traces = {
        spec.name: [float(initial["cvar95"][index])] for index, spec in enumerate(specs)
    }
    last_training = torch.full((len(specs),), torch.nan, dtype=torch.float64)
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = stage1a2_learning_rate(step, calibration)
        channels = sample_dd_v1(batch_size, train_channel_seed + step, n=n).matrices
        snr_generator = make_generator(train_snr_seed + step)
        snrs_db = 12.0 + 8.0 * torch.rand(batch_size, dtype=torch.float64, generator=snr_generator)
        streams = per_stream_exact_mse(parameter, channels, noise_variance(snrs_db), k)
        objectives = torch.stack(
            [
                tail_objective(streams[index], spec.beta, spec.tail_lambda).mean()
                for index, spec in enumerate(specs)
            ]
        )
        optimizer.zero_grad(set_to_none=True)
        finite_objectives = torch.isfinite(objectives)
        nonfinite |= ~finite_objectives
        torch.where(finite_objectives, objectives, torch.zeros_like(objectives)).sum().backward()
        gradients_finite = torch.isfinite(parameter.grad.real).all(dim=(-2, -1)) & torch.isfinite(
            parameter.grad.imag
        ).all(dim=(-2, -1))
        nonfinite |= ~gradients_finite
        parameter.grad[~gradients_finite] = 0
        optimizer.step()
        with torch.no_grad():
            matrices_finite = torch.isfinite(parameter.real).all(dim=(-2, -1)) & torch.isfinite(
                parameter.imag
            ).all(dim=(-2, -1))
            nonfinite |= ~matrices_finite
            parameter[~matrices_finite] = best_matrices[~matrices_finite]
            parameter.copy_(polar_project(parameter))
            residuals = torch.linalg.matrix_norm(
                parameter.mH @ parameter - torch.eye(n, dtype=COMPLEX_DTYPE)
            )
            max_residuals = torch.maximum(max_residuals, residuals)
            last_training = objectives.detach()

        if step % validation_cadence == 0 or step == steps:
            validation = validation_tail_metrics(parameter.detach(), validation_channels, k=k)
            validation_finite = torch.isfinite(validation["mean_mse"]) & torch.isfinite(
                validation["cvar95"]
            )
            nonfinite |= ~validation_finite
            validation_feasible = validation["mean_mse"] <= threshold
            improved = validation_finite & validation_feasible & (
                (validation["cvar95"] < best_cvar95)
                | (
                    (validation["cvar95"] == best_cvar95)
                    & (validation["mean_mse"] < best_mean)
                )
            )
            best_mean[improved] = validation["mean_mse"][improved]
            best_cvar90[improved] = validation["cvar90"][improved]
            best_cvar95[improved] = validation["cvar95"][improved]
            best_steps[improved] = step
            best_matrices[improved] = parameter.detach()[improved]
            trace_steps.append(step)
            for index, spec in enumerate(specs):
                training_traces[spec.name].append(float(last_training[index]))
                validation_mean_traces[spec.name].append(float(validation["mean_mse"][index]))
                validation_cvar95_traces[spec.name].append(float(validation["cvar95"][index]))

    final_matrices = parameter.detach().clone()
    final = validation_tail_metrics(final_matrices, validation_channels, k=k)
    return TailTrainingResult(
        specs=specs,
        initial_mean_mse=initial["mean_mse"],
        initial_cvar90=initial["cvar90"],
        initial_cvar95=initial["cvar95"],
        best_mean_mse=best_mean,
        best_cvar90=best_cvar90,
        best_cvar95=best_cvar95,
        best_steps=best_steps,
        best_matrices=best_matrices,
        final_mean_mse=final["mean_mse"],
        final_cvar90=final["cvar90"],
        final_cvar95=final["cvar95"],
        final_matrices=final_matrices,
        max_unitarity_residuals=max_residuals,
        nonfinite_seen=nonfinite,
        runtime_seconds=time.perf_counter() - started,
        trace_steps=trace_steps,
        training_traces=training_traces,
        validation_mean_traces=validation_mean_traces,
        validation_cvar95_traces=validation_cvar95_traces,
    )
