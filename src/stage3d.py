"""Constrained finite-receiver geometry intervention for Stage 3D."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .channel import COMPLEX_DTYPE, noise_variance
from .oracle import polar_project
from .receiver import effective_operator
from .receiver_geometry import apply_mask, exact_error_geometry
from .stage3b import differentiable_q, hard_row_concentration
from .tail_learning import stream_cvar

K = 24
GEOMETRY_SNR_DB = 18
TAIL_BETA = 0.95
Q_EPSILON = 2.5e-4
Q_PENALTY = 25.0
TAIL_LAMBDAS = (0.25, 1.0)

Q_TRAIN_SEED = 11100
Q_VALIDATION_SEED = 11101
GEOMETRY_TRAIN_SEED = 11102
GEOMETRY_VALIDATION_SEED = 11103
EVALUATION_Q_SEED = 11104
EVALUATION_CHANNEL_SEED = 11105
SYMBOL_SEED = 11106
NOISE_SEED = 11107

Q_TRAIN_SAMPLES = 128
Q_VALIDATION_SAMPLES = 256
GEOMETRY_TRAIN_CHANNELS = 32
GEOMETRY_VALIDATION_CHANNELS = 64
EVALUATION_Q_SAMPLES = 2048
EVALUATION_CHANNELS = 512
TRAINING_STEPS = 1000
VALIDATION_CADENCE = 25


@dataclass(frozen=True)
class GeometryObjectiveSpec:
    arm: str
    tail_lambda: float

    @property
    def name(self) -> str:
        return "MEAN" if self.arm == "mean" else f"TAIL_{self.tail_lambda:g}"


@dataclass
class GeometryTrainingResult:
    specs: tuple[GeometryObjectiveSpec, ...]
    start_names: tuple[str, ...]
    initial_metrics: dict[str, torch.Tensor]
    q_floors: torch.Tensor
    reference_mean: torch.Tensor
    reference_tail: torch.Tensor
    candidates: dict[tuple[int, int], list[dict[str, object]]]
    final_matrices: torch.Tensor
    max_unitarity_residuals: torch.Tensor
    nonfinite_seen: torch.Tensor
    trace_rows: list[dict[str, object]]
    runtime_seconds: float


def stage3d_learning_rate(step: int) -> float:
    if step <= 500:
        return 3e-3
    if step <= 800:
        return 1e-3
    return 3e-4


def adverse_channel_stream_tail(stream_mse: torch.Tensor, beta: float = TAIL_BETA) -> torch.Tensor:
    """Stage-3C two-level upper stream/channel error tail."""
    if stream_mse.ndim < 2:
        raise ValueError("stream MSE must have channel and stream dimensions")
    per_channel = stream_cvar(stream_mse, beta)
    return stream_cvar(per_channel, beta)


def fixed_q_geometry_metrics(
    waveforms: torch.Tensor,
    channels: torch.Tensor,
    q_operators: torch.Tensor,
    *,
    snr_db: float = GEOMETRY_SNR_DB,
    k: int = K,
    chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Exact differentiable fixed-Q receiver statistics for one frozen bank."""
    single = waveforms.ndim == 2
    matrices = waveforms.unsqueeze(0) if single else waveforms
    q = differentiable_q(matrices, q_operators, chunk_size=chunk_size)
    eta, rows, masks = hard_row_concentration(q, k)
    effective = effective_operator(channels[None], matrices[:, None])
    retained = apply_mask(effective, masks[:, None].expand_as(effective))
    geometry = exact_error_geometry(effective, retained, noise_variance(snr_db))
    stream_mse = geometry["stream_mse"]
    result = {
        "mean": stream_mse.mean(dim=(-2, -1)),
        "tail": adverse_channel_stream_tail(stream_mse),
        "eta": eta,
        "row_p05": torch.quantile(rows, 0.05, dim=-1),
        "mask": masks,
        "retained_energy": geometry["retained_energy"].mean(dim=-1),
        "condition_median": geometry["condition_number"].median(dim=-1).values,
        "condition_p95": torch.quantile(geometry["condition_number"], 0.95, dim=-1),
    }
    return {key: value[0] for key, value in result.items()} if single else result


def normalized_geometry_loss(
    mean: torch.Tensor,
    tail: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_tail: torch.Tensor,
    tail_lambda: torch.Tensor,
) -> torch.Tensor:
    return mean / reference_mean + tail_lambda * tail / reference_tail


def pareto_nondominated(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Stable unique Pareto set for minimization of validation mean and tail."""
    ordered = sorted(rows, key=lambda row: (float(row["validation_mean"]), float(row["validation_tail"]), str(row["spec"])))
    kept: list[dict[str, object]] = []
    for row in ordered:
        mean = float(row["validation_mean"]); tail = float(row["validation_tail"])
        dominated = any(
            float(other["validation_mean"]) <= mean
            and float(other["validation_tail"]) <= tail
            and (float(other["validation_mean"]) < mean or float(other["validation_tail"]) < tail)
            for other in ordered
        )
        duplicate = any(float(other["validation_mean"]) == mean and float(other["validation_tail"]) == tail for other in kept)
        if not dominated and not duplicate:
            kept.append(row)
    return kept


def optimize_geometry_waveforms(
    starting_matrices: torch.Tensor,
    start_names: tuple[str, ...],
    specs: tuple[GeometryObjectiveSpec, ...],
    q_training_operators: torch.Tensor,
    q_validation_operators: torch.Tensor,
    geometry_training_channels: torch.Tensor,
    geometry_validation_channels: torch.Tensor,
    *,
    steps: int = TRAINING_STEPS,
    validation_cadence: int = VALIDATION_CADENCE,
) -> GeometryTrainingResult:
    """Projected Adam with validation-only Q feasibility and analytic selection."""
    if len(starting_matrices) != len(start_names):
        raise ValueError("one name is required per starting matrix")
    if not specs:
        raise ValueError("at least one objective specification is required")
    starts = polar_project(starting_matrices.to(COMPLEX_DTYPE))
    expanded = starts[None].expand(len(specs), -1, -1, -1).clone()
    parameters = torch.nn.Parameter(expanded.reshape(-1, *starts.shape[-2:]))
    spec_indices = torch.arange(len(specs)).repeat_interleave(len(starts))
    start_indices = torch.arange(len(starts)).repeat(len(specs))
    lambdas = torch.tensor([spec.tail_lambda for spec in specs], dtype=torch.float64).repeat_interleave(len(starts))

    with torch.no_grad():
        initial = fixed_q_geometry_metrics(starts, geometry_validation_channels, q_validation_operators)
        training_initial = fixed_q_geometry_metrics(starts, geometry_training_channels, q_training_operators)
    floors = initial["eta"] - Q_EPSILON
    training_floors = training_initial["eta"] - Q_EPSILON
    reference_mean = initial["mean"].clone()
    reference_tail = initial["tail"].clone()
    optimizer = torch.optim.Adam([parameters], lr=stage3d_learning_rate(1))
    candidates: dict[tuple[int, int], list[dict[str, object]]] = {
        (spec_index, start_index): []
        for spec_index in range(len(specs)) for start_index in range(len(starts))
    }
    max_residual = torch.zeros(len(parameters), dtype=torch.float64)
    nonfinite = torch.zeros(len(parameters), dtype=torch.bool)
    trace_rows: list[dict[str, object]] = []
    previous_masks: torch.Tensor | None = None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = stage3d_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        training = fixed_q_geometry_metrics(parameters, geometry_training_channels, q_training_operators)
        base = normalized_geometry_loss(
            training["mean"], training["tail"],
            reference_mean[start_indices], reference_tail[start_indices], lambdas,
        )
        violation = torch.relu(training_floors[start_indices] - training["eta"])
        objective = base + Q_PENALTY * (violation / Q_EPSILON).square()
        finite = torch.isfinite(objective)
        nonfinite |= ~finite
        torch.where(finite, objective, torch.zeros_like(objective)).sum().backward()
        gradient_norm = torch.linalg.vector_norm(parameters.grad.reshape(len(parameters), -1), dim=-1)
        gradient_finite = torch.isfinite(gradient_norm)
        nonfinite |= ~gradient_finite
        parameters.grad[~gradient_finite] = 0
        masks = training["mask"].detach()
        support_flip = torch.zeros(len(parameters), dtype=torch.float64) if previous_masks is None else (
            (masks != previous_masks).to(torch.float64).mean(dim=(-2, -1))
        )
        previous_masks = masks
        optimizer.step()
        with torch.no_grad():
            matrix_finite = torch.isfinite(parameters.real).all(dim=(-2, -1)) & torch.isfinite(parameters.imag).all(dim=(-2, -1))
            nonfinite |= ~matrix_finite
            parameters[~matrix_finite] = expanded.reshape_as(parameters)[~matrix_finite]
            parameters.copy_(polar_project(parameters))
            residual = torch.linalg.matrix_norm(
                parameters.mH @ parameters - torch.eye(parameters.shape[-1], dtype=COMPLEX_DTYPE)
            )
            max_residual = torch.maximum(max_residual, residual)

        if step % validation_cadence == 0 or step == steps:
            with torch.no_grad():
                validation = fixed_q_geometry_metrics(parameters, geometry_validation_channels, q_validation_operators)
            validation_loss = normalized_geometry_loss(
                validation["mean"], validation["tail"],
                reference_mean[start_indices], reference_tail[start_indices], lambdas,
            )
            validation_feasible = validation["eta"] >= floors[start_indices]
            nonfinite |= ~(torch.isfinite(validation_loss) & torch.isfinite(validation["eta"]))
            for run_index in range(len(parameters)):
                spec_index = int(spec_indices[run_index]); start_index = int(start_indices[run_index])
                row = {
                    "spec": specs[spec_index].name,
                    "arm": specs[spec_index].arm,
                    "tail_lambda": specs[spec_index].tail_lambda,
                    "start": start_names[start_index],
                    "step": step,
                    "training_mean": float(training["mean"][run_index]),
                    "training_tail": float(training["tail"][run_index]),
                    "training_eta": float(training["eta"][run_index]),
                    "training_row_p05": float(training["row_p05"][run_index]),
                    "validation_mean": float(validation["mean"][run_index]),
                    "validation_tail": float(validation["tail"][run_index]),
                    "validation_eta": float(validation["eta"][run_index]),
                    "validation_row_p05": float(validation["row_p05"][run_index]),
                    "validation_retained_energy": float(validation["retained_energy"][run_index]),
                    "validation_condition_median": float(validation["condition_median"][run_index]),
                    "validation_condition_p95": float(validation["condition_p95"][run_index]),
                    "validation_loss": float(validation_loss[run_index]),
                    "q_floor": float(floors[start_index]),
                    "q_floor_violation": float(torch.relu(floors[start_index] - validation["eta"][run_index])),
                    "feasible": bool(validation_feasible[run_index]),
                    "gradient_norm": float(gradient_norm[run_index]),
                    "support_flip_rate": float(support_flip[run_index]),
                    "unitarity_residual": float(residual[run_index]),
                    "finite": bool(torch.isfinite(validation_loss[run_index])),
                }
                trace_rows.append(row)
                if row["feasible"] and row["finite"]:
                    candidates[(spec_index, start_index)].append({**row, "matrix": parameters[run_index].detach().clone()})

    return GeometryTrainingResult(
        specs=specs, start_names=start_names, initial_metrics=initial, q_floors=floors,
        reference_mean=reference_mean, reference_tail=reference_tail, candidates=candidates,
        final_matrices=parameters.detach().clone(), max_unitarity_residuals=max_residual,
        nonfinite_seen=nonfinite, trace_rows=trace_rows,
        runtime_seconds=time.perf_counter() - started,
    )


def select_analytic_checkpoints(result: GeometryTrainingResult) -> dict[str, dict[str, object]]:
    """Select Arm-A minima and Arm-B Pareto points without communication data."""
    selected: dict[str, dict[str, object]] = {}
    for start_index, start_name in enumerate(result.start_names):
        mean_candidates = result.candidates[(0, start_index)]
        if mean_candidates:
            best = min(mean_candidates, key=lambda row: (float(row["validation_mean"]), float(row["validation_tail"]), int(row["step"])))
            selected[f"GEOM_MEAN_FROM_{start_name}"] = best
        tail_best: list[dict[str, object]] = []
        for spec_index, spec in enumerate(result.specs[1:], start=1):
            candidates = result.candidates[(spec_index, start_index)]
            if candidates:
                tail_best.append(min(candidates, key=lambda row: (float(row["validation_loss"]), float(row["validation_mean"]), int(row["step"]))))
        for frontier_index, row in enumerate(pareto_nondominated(tail_best), start=1):
            selected[f"GEOM_PARETO_{frontier_index}_FROM_{start_name}"] = row
    return selected
