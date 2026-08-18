"""Receiver-budget causality utilities for Stage 2C."""

from __future__ import annotations

import time

import numpy as np
import torch

from . import config
from .channel import COMPLEX_DTYPE, make_generator, noise_variance, sample_dd_v1
from .diagnostics import representation_diagnostics
from .oracle import polar_project
from .stage2b import (
    community_summary,
    ensemble_operator_statistics,
    spectral_centroids,
    spectral_overlap_matrix,
)
from .tail_learning import (
    TailObjectiveSpec,
    TailTrainingResult,
    per_stream_exact_mse,
    stage1a2_learning_rate,
    stream_cvar,
    tail_objective,
    validation_tail_metrics,
)
from .waveforms import ofdm_waveform, random_unitary

TRAIN_K_VALUES = (16, 24, 32)
EVAL_K_VALUES = config.STAGE2A_K_VALUES
INITIALIZATION_NAMES = ("OFDM", "HAAR_1", "HAAR_2")
HAAR_SEEDS = (config.STAGE1A_HAAR_SEED, config.STAGE1A_HAAR_SEED + 1)
TAIL_BETA = 0.95
TAIL_LAMBDA = 0.40
TRAIN_CHANNEL_SEED = config.STAGE1A2_PRODUCTION_CHANNEL_SEED
TRAIN_SNR_SEED = config.STAGE1A2_PRODUCTION_SNR_SEED
BER_SYMBOL_SEED = 11050
BER_NOISE_SEED = 11051


def stage2c_initializations() -> tuple[tuple[str, ...], torch.Tensor]:
    """Return fixed raw starts; no K=24-trained tensor is used."""
    matrices = (
        ofdm_waveform(),
        random_unitary(config.N, HAAR_SEEDS[0]),
        random_unitary(config.N, HAAR_SEEDS[1]),
    )
    return INITIALIZATION_NAMES, torch.stack(matrices)


def optimize_budget_waveforms(
    initial_matrices: torch.Tensor,
    validation_channels: torch.Tensor,
    *,
    k: int,
    steps: int = config.STAGE1A2_PRODUCTION_STEPS,
    batch_size: int = config.STAGE1A2_TRAIN_BATCH_SIZE,
    validation_cadence: int = config.STAGE1A2_VALIDATION_CADENCE,
    train_channel_seed: int = TRAIN_CHANNEL_SEED,
    train_snr_seed: int = TRAIN_SNR_SEED,
) -> TailTrainingResult:
    """Exact Stage-1A.2 optimization with a per-independent-start 2% gate."""
    names = INITIALIZATION_NAMES[: len(initial_matrices)]
    specs = tuple(TailObjectiveSpec(name, TAIL_BETA, TAIL_LAMBDA) for name in names)
    matrices = polar_project(initial_matrices.to(COMPLEX_DTYPE))
    initial = validation_tail_metrics(matrices, validation_channels, k=k)
    thresholds = initial["mean_mse"] * config.STAGE1A2_MEAN_MSE_GATE
    parameter = torch.nn.Parameter(matrices.clone())
    optimizer = torch.optim.Adam([parameter], lr=stage1a2_learning_rate(1))
    best_mean = initial["mean_mse"].clone()
    best_cvar90 = initial["cvar90"].clone()
    best_cvar95 = initial["cvar95"].clone()
    best_steps = torch.zeros(len(specs), dtype=torch.int64)
    best_matrices = matrices.clone()
    max_residuals = torch.zeros(len(specs), dtype=torch.float64)
    nonfinite = ~torch.isfinite(initial["mean_mse"]) | ~torch.isfinite(initial["cvar95"])
    trace_steps = [0]
    training_traces = {name: [float("nan")] for name in names}
    validation_mean_traces = {
        name: [float(initial["mean_mse"][index])] for index, name in enumerate(names)
    }
    validation_cvar95_traces = {
        name: [float(initial["cvar95"][index])] for index, name in enumerate(names)
    }
    last_training = torch.full((len(specs),), torch.nan, dtype=torch.float64)
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = stage1a2_learning_rate(step)
        channels = sample_dd_v1(batch_size, train_channel_seed + step).matrices
        snr_generator = make_generator(train_snr_seed + step)
        snrs_db = 12.0 + 8.0 * torch.rand(batch_size, dtype=torch.float64, generator=snr_generator)
        streams = per_stream_exact_mse(parameter, channels, noise_variance(snrs_db), k)
        objectives = torch.stack(
            [tail_objective(streams[index], TAIL_BETA, TAIL_LAMBDA).mean() for index in range(len(specs))]
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
                parameter.mH @ parameter - torch.eye(config.N, dtype=COMPLEX_DTYPE)
            )
            max_residuals = torch.maximum(max_residuals, residuals)
            last_training = objectives.detach()

        if step % validation_cadence == 0 or step == steps:
            validation = validation_tail_metrics(parameter.detach(), validation_channels, k=k)
            finite = torch.isfinite(validation["mean_mse"]) & torch.isfinite(validation["cvar95"])
            nonfinite |= ~finite
            improved = finite & (validation["mean_mse"] <= thresholds) & (
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
            for index, name in enumerate(names):
                training_traces[name].append(float(last_training[index]))
                validation_mean_traces[name].append(float(validation["mean_mse"][index]))
                validation_cvar95_traces[name].append(float(validation["cvar95"][index]))

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


def select_validation_run(result: TailTrainingResult) -> int:
    """Validation-only selection; no BER or structural metric is accepted."""
    eligible = [index for index in range(len(result.specs)) if not bool(result.nonfinite_seen[index])]
    if not eligible:
        raise RuntimeError("all Stage 2C starts encountered non-finite values")
    return min(
        eligible,
        key=lambda index: (
            float(result.best_cvar95[index]),
            float(result.best_mean_mse[index]),
            index,
        ),
    )


def structural_diagnostics(
    waveform: torch.Tensor, channels: torch.Tensor, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    energy_map, support_map = ensemble_operator_statistics(channels, {"Learned": waveform}, k=k)
    energy = energy_map["Learned"]
    support = support_map["Learned"]
    centroids = spectral_centroids(waveform)
    overlap = spectral_overlap_matrix(waveform)
    labels, summary = community_summary(energy, overlap, centroids)
    same = labels[:, None] == labels[None, :]
    sizes = np.asarray(summary["group_sizes"])
    summary.update(
        predicted_group_count=int(np.ceil(config.N / k)),
        maximum_group_size=int(sizes.max()),
        median_group_size=float(np.median(sizes)),
        within_topk_support_fraction=float(support[same].sum() / support.sum()),
        mean_support_probability_within=float(support[same].mean()),
        mean_support_probability_across=float(support[~same].mean()),
    )
    return energy, support, labels, summary


def cross_budget_analytic_rows(
    channels: torch.Tensor,
    selected_waveforms: dict[int, torch.Tensor],
    snr_db: int = 18,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_k, waveform in selected_waveforms.items():
        for start in range(0, len(channels), 32):
            current = channels[start : start + 32]
            for eval_k in EVAL_K_VALUES:
                streams = per_stream_exact_mse(waveform, current, noise_variance(snr_db), eval_k)
                diagnostics = representation_diagnostics(waveform.mH @ current @ waveform, eval_k)
                for offset in range(len(current)):
                    rows.append(
                        {
                            "channel_index": start + offset,
                            "training_k": train_k,
                            "evaluation_k": eval_k,
                            "snr_db": snr_db,
                            "analytic_mse": float(streams[offset].mean()),
                            "retained_energy": float(diagnostics["retained_energy"][offset]),
                            "limiting_distortion": float(diagnostics["limiting_distortion"][offset]),
                        }
                    )
    return rows
