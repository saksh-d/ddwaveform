"""Universal dense-unitary learning for the Stage 1A existence test."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from . import config
from .channel import COMPLEX_DTYPE, make_generator, noise_variance, sample_dd_v1
from .oracle import polar_project
from .receiver import rowwise_topk
from .waveforms import (
    afdm_waveform,
    identity_waveform,
    ofdm_waveform,
    otfs_waveform,
    random_unitary,
)


@dataclass
class UniversalTrainingResult:
    initialization_names: tuple[str, ...]
    initial_validation: torch.Tensor
    best_validation: torch.Tensor
    best_steps: torch.Tensor
    final_validation: torch.Tensor
    best_matrices: torch.Tensor
    final_matrices: torch.Tensor
    max_unitarity_residuals: torch.Tensor
    nonfinite_seen: torch.Tensor
    runtime_seconds: float
    trace_steps: list[int]
    training_traces: dict[str, list[float]]
    validation_traces: dict[str, list[float]]


def stage1a_initializations(n: int = config.N) -> tuple[tuple[str, ...], torch.Tensor]:
    if n != config.N:
        raise ValueError("Stage 1A named initializations are defined for N=64")
    names = ("IDENTITY", "OFDM", "OTFS", "AFDM_K24_TUNED", "HAAR_1", "HAAR_2", "HAAR_3")
    matrices = (
        identity_waveform(n),
        ofdm_waveform(n),
        otfs_waveform(),
        afdm_waveform(n=n, c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * n)),
        random_unitary(n, config.STAGE1A_HAAR_SEED),
        random_unitary(n, config.STAGE1A_HAAR_SEED + 1),
        random_unitary(n, config.STAGE1A_HAAR_SEED + 2),
    )
    return names, torch.stack(matrices)


def stage1a_learning_rate(step: int) -> float:
    if step <= 1000:
        return 1e-2
    if step <= 1600:
        return 3e-3
    return 1e-3


def hard_topk_communication_mse(
    waveforms: torch.Tensor,
    channels: torch.Tensor,
    sigma2: torch.Tensor | float,
    k: int,
) -> torch.Tensor:
    """Per-waveform/per-channel exact communication MSE with paired sigma2.

    `waveforms` is `[R,N,N]` (or `[N,N]`), `channels` is `[B,N,N]`, and
    `sigma2` is scalar or `[B]`. The result is `[R,B]` (or `[B]`).
    """
    single = waveforms.ndim == 2
    if single:
        waveforms = waveforms.unsqueeze(0)
    a = waveforms[:, None].mH @ channels[None] @ waveforms[:, None]
    a_tilde = rowwise_topk(a, k)
    n = channels.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma = torch.as_tensor(sigma2, dtype=torch.float64)
    if sigma.ndim == 0:
        sigma = sigma.expand(channels.shape[0])
    system = a_tilde.mH @ a_tilde + sigma[None, :, None, None] * identity
    equalizer = torch.linalg.solve(system, a_tilde.mH)
    residual = equalizer @ a - identity
    distortion = torch.sum(torch.abs(residual) ** 2, dim=(-2, -1))
    noise = sigma[None] * torch.sum(torch.abs(equalizer) ** 2, dim=(-2, -1))
    values = (distortion + noise) / n
    return values[0] if single else values


def validation_objective(
    waveforms: torch.Tensor,
    validation_channels: torch.Tensor,
    k: int = config.STAGE1A_K,
    snrs_db: tuple[int, ...] = config.STAGE1A_SNRS_DB,
    channel_chunk_size: int = 32,
) -> torch.Tensor:
    """Frozen-bank mean over channels and equally weighted validation SNRs."""
    single = waveforms.ndim == 2
    matrices = waveforms.unsqueeze(0) if single else waveforms
    totals = torch.zeros(matrices.shape[0], dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, validation_channels.shape[0], channel_chunk_size):
            channels = validation_channels[start : start + channel_chunk_size]
            for snr_db in snrs_db:
                totals += hard_topk_communication_mse(
                    matrices, channels, noise_variance(snr_db), k
                ).sum(dim=1)
                count += channels.shape[0]
    result = totals / count
    return result[0] if single else result


def optimize_universal_waveforms(
    initial_matrices: torch.Tensor,
    initialization_names: tuple[str, ...],
    validation_channels: torch.Tensor,
    *,
    steps: int = config.STAGE1A_STEPS,
    batch_size: int = config.STAGE1A_TRAIN_BATCH_SIZE,
    validation_cadence: int = config.STAGE1A_VALIDATION_CADENCE,
    k: int = config.STAGE1A_K,
    train_channel_seed: int = config.STAGE1A_TRAIN_CHANNEL_SEED,
    train_snr_seed: int = config.STAGE1A_TRAIN_SNR_SEED,
) -> UniversalTrainingResult:
    """Projected Adam with fresh DD-v1 channels and exact hard top-K."""
    if initial_matrices.shape[0] != len(initialization_names):
        raise ValueError("one initialization name is required per matrix")
    n = initial_matrices.shape[-1]
    initial_matrices = polar_project(initial_matrices.to(COMPLEX_DTYPE))
    initial_validation = validation_objective(initial_matrices, validation_channels, k)
    parameter = torch.nn.Parameter(initial_matrices.clone())
    optimizer = torch.optim.Adam([parameter], lr=stage1a_learning_rate(1))
    best_validation = initial_validation.clone()
    best_steps = torch.zeros(len(initialization_names), dtype=torch.int64)
    best_matrices = initial_matrices.clone()
    nonfinite_seen = ~torch.isfinite(initial_validation)
    max_residuals = torch.zeros(len(initialization_names), dtype=torch.float64)
    trace_steps = [0]
    training_traces = {name: [float("nan")] for name in initialization_names}
    validation_traces = {
        name: [float(initial_validation[index])] for index, name in enumerate(initialization_names)
    }
    last_training = torch.full((len(initialization_names),), torch.nan, dtype=torch.float64)
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = stage1a_learning_rate(step)
        channels = sample_dd_v1(batch_size, train_channel_seed + step, n=n).matrices
        snr_generator = make_generator(train_snr_seed + step)
        snrs_db = 12.0 + 8.0 * torch.rand(batch_size, dtype=torch.float64, generator=snr_generator)
        sigma2 = noise_variance(snrs_db)
        optimizer.zero_grad(set_to_none=True)
        objectives = hard_topk_communication_mse(parameter, channels, sigma2, k).mean(dim=1)
        finite = torch.isfinite(objectives)
        nonfinite_seen |= ~finite
        torch.where(finite, objectives, torch.zeros_like(objectives)).sum().backward()
        gradients_finite = torch.isfinite(parameter.grad.real).all(dim=(-2, -1)) & torch.isfinite(
            parameter.grad.imag
        ).all(dim=(-2, -1))
        nonfinite_seen |= ~gradients_finite
        parameter.grad[~gradients_finite] = 0
        optimizer.step()
        with torch.no_grad():
            matrices_finite = torch.isfinite(parameter.real).all(dim=(-2, -1)) & torch.isfinite(
                parameter.imag
            ).all(dim=(-2, -1))
            nonfinite_seen |= ~matrices_finite
            parameter[~matrices_finite] = best_matrices[~matrices_finite]
            parameter.copy_(polar_project(parameter))
            residuals = torch.linalg.matrix_norm(parameter.mH @ parameter - torch.eye(n, dtype=COMPLEX_DTYPE))
            max_residuals = torch.maximum(max_residuals, residuals)
            last_training = objectives.detach()

        if step % validation_cadence == 0 or step == steps:
            validation = validation_objective(parameter.detach(), validation_channels, k)
            finite_validation = torch.isfinite(validation)
            nonfinite_seen |= ~finite_validation
            improved = finite_validation & (validation < best_validation)
            best_validation[improved] = validation[improved]
            best_steps[improved] = step
            best_matrices[improved] = parameter.detach()[improved]
            trace_steps.append(step)
            for index, name in enumerate(initialization_names):
                training_traces[name].append(float(last_training[index]))
                validation_traces[name].append(float(validation[index]))

    final_matrices = parameter.detach().clone()
    final_validation = validation_objective(final_matrices, validation_channels, k)
    return UniversalTrainingResult(
        initialization_names=initialization_names,
        initial_validation=initial_validation,
        best_validation=best_validation,
        best_steps=best_steps,
        final_validation=final_validation,
        best_matrices=best_matrices,
        final_matrices=final_matrices,
        max_unitarity_residuals=max_residuals,
        nonfinite_seen=nonfinite_seen,
        runtime_seconds=time.perf_counter() - started,
        trace_steps=trace_steps,
        training_traces=training_traces,
        validation_traces=validation_traces,
    )
