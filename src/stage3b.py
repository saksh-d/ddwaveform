"""Direct fixed-bank Q-concentration optimization for Stage 3B."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from . import config
from .channel import COMPLEX_DTYPE, REAL_DTYPE, assemble_dd_channel
from .oracle import polar_project
from .waveforms import ofdm_waveform, random_unitary

TRAIN_K = 24
INITIALIZATION_NAMES = ("OFDM", "HAAR_1", "HAAR_2")
TRAIN_SEED = 11080
VALIDATION_SEED = 11081
EVALUATION_Q_SEED = 11082
EVALUATION_CHANNEL_SEED = 11083
BER_SYMBOL_SEED = 11084
BER_NOISE_SEED = 11085
TRAIN_SAMPLES = 128
VALIDATION_SAMPLES = 256
EVALUATION_Q_SAMPLES = 2048
EVALUATION_CHANNELS = 512
ROBUST_LAMBDA = 0.40
LOWER_TAIL_FRACTION = 0.10


@dataclass
class QTrainingResult:
    arm: str
    initialization_names: tuple[str, ...]
    best_matrices: torch.Tensor
    best_steps: torch.Tensor
    initial_validation_eta: torch.Tensor
    initial_validation_lower_tail: torch.Tensor
    best_validation_eta: torch.Tensor
    best_validation_lower_tail: torch.Tensor
    final_matrices: torch.Tensor
    final_validation_eta: torch.Tensor
    final_validation_lower_tail: torch.Tensor
    max_unitarity_residuals: torch.Tensor
    nonfinite_seen: torch.Tensor
    trace_rows: list[dict[str, object]]
    runtime_seconds: float


def fixed_initializations(n: int = config.N) -> tuple[tuple[str, ...], torch.Tensor]:
    """The exact OFDM and Stage-1A Haar starts required by the intervention."""
    if n != config.N:
        raise ValueError("named Stage 3B initializations are fixed at project N")
    matrices = (
        ofdm_waveform(n),
        random_unitary(n, config.STAGE1A_HAAR_SEED),
        random_unitary(n, config.STAGE1A_HAAR_SEED + 1),
    )
    return INITIALIZATION_NAMES, torch.stack(matrices)


def atomic_operators(delays: torch.Tensor, dopplers: torch.Tensor, n: int = config.N) -> torch.Tensor:
    """Build the frozen Delta(nu) Pi(tau) bank with unit path gains."""
    delays = torch.as_tensor(delays, dtype=REAL_DTYPE)
    dopplers = torch.as_tensor(dopplers, dtype=REAL_DTYPE)
    if delays.ndim != 1 or delays.shape != dopplers.shape or len(delays) < 1:
        raise ValueError("delays and Dopplers must be matching nonempty vectors")
    gains = torch.ones((len(delays), 1), dtype=COMPLEX_DTYPE)
    return assemble_dd_channel(delays[:, None], dopplers[:, None], gains, n=n)


def differentiable_q(
    waveforms: torch.Tensor,
    operators: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Estimate Q=mean |W^H Delta Pi W|^2 without breaking autograd."""
    single = waveforms.ndim == 2
    matrices = waveforms.unsqueeze(0) if single else waveforms
    if matrices.ndim != 3 or operators.ndim != 3:
        raise ValueError("waveforms and operators must contain square matrices")
    if matrices.shape[-2:] != operators.shape[-2:]:
        raise ValueError("waveform and operator dimensions do not match")
    total = torch.zeros(
        (len(matrices), matrices.shape[-1], matrices.shape[-1]), dtype=REAL_DTYPE
    )
    for start in range(0, len(operators), chunk_size):
        current = operators[start : start + chunk_size]
        transformed = matrices[:, None].mH @ current[None] @ matrices[:, None]
        total = total + torch.sum(torch.abs(transformed) ** 2, dim=1)
    result = total / len(operators)
    return result[0] if single else result


def hard_row_concentration(q: torch.Tensor, k: int = TRAIN_K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aggregate eta, row fractions, and detached hard top-k masks."""
    single = q.ndim == 2
    matrices = q.unsqueeze(0) if single else q
    n = matrices.shape[-1]
    if matrices.ndim != 3 or matrices.shape[-2] != n or not 1 <= k <= n:
        raise ValueError("Q must be square and k must be valid")
    indices = torch.argsort(matrices.detach(), dim=-1, descending=True, stable=True)[..., :k]
    mask = torch.zeros_like(matrices, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    selected = torch.where(mask, matrices, torch.zeros((), dtype=matrices.dtype))
    row_total = torch.sum(matrices, dim=-1)
    rows = torch.sum(selected, dim=-1) / row_total
    eta = torch.sum(selected, dim=(-2, -1)) / torch.sum(matrices, dim=(-2, -1))
    if single:
        return eta[0], rows[0], mask[0]
    return eta, rows, mask


def lower_row_tail(rows: torch.Tensor, fraction: float = LOWER_TAIL_FRACTION) -> torch.Tensor:
    """Mean of the weakest fixed fraction of rows; selection is detached."""
    if not 0 < fraction <= 1:
        raise ValueError("tail fraction must lie in (0,1]")
    single = rows.ndim == 1
    values = rows.unsqueeze(0) if single else rows
    count = max(1, math.ceil(values.shape[-1] * fraction))
    indices = torch.argsort(values.detach(), dim=-1, stable=True)[..., :count]
    tail = torch.gather(values, -1, indices).mean(dim=-1)
    return tail[0] if single else tail


def q_objective(
    q: torch.Tensor,
    arm: str,
    *,
    k: int = TRAIN_K,
    robust_lambda: float = ROBUST_LAMBDA,
    tail_fraction: float = LOWER_TAIL_FRACTION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The two preregistered Q-only Stage-3B objective arms."""
    eta, rows, mask = hard_row_concentration(q, k)
    tail = lower_row_tail(rows, tail_fraction)
    if arm == "mean":
        objective = 1.0 - eta
    elif arm == "robust":
        objective = (1.0 - eta) + robust_lambda * (1.0 - tail)
    else:
        raise ValueError("arm must be 'mean' or 'robust'")
    return objective, {"eta": eta, "lower_tail": tail, "row_fraction": rows, "mask": mask}


def stage3b_learning_rate(step: int) -> float:
    if step <= 500:
        return 1e-2
    if step <= 800:
        return 3e-3
    return 1e-3


def _validation_metrics(
    matrices: torch.Tensor, operators: torch.Tensor, arm: str, chunk_size: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        return q_objective(differentiable_q(matrices, operators, chunk_size=chunk_size), arm)


def optimize_q_waveforms(
    initial_matrices: torch.Tensor,
    initialization_names: tuple[str, ...],
    training_operators: torch.Tensor,
    validation_operators: torch.Tensor,
    *,
    arm: str,
    steps: int = 1000,
    validation_cadence: int = 25,
    chunk_size: int = 32,
) -> QTrainingResult:
    """Projected Adam on one fixed Q bank with independent Q-only validation."""
    if len(initial_matrices) != len(initialization_names):
        raise ValueError("one name is required per initialization")
    matrices = polar_project(initial_matrices.to(COMPLEX_DTYPE))
    initial_objective, initial = _validation_metrics(matrices, validation_operators, arm, chunk_size)
    parameter = torch.nn.Parameter(matrices.clone())
    optimizer = torch.optim.Adam([parameter], lr=stage3b_learning_rate(1))
    best_objective = initial_objective.clone()
    best_eta = initial["eta"].clone()
    best_tail = initial["lower_tail"].clone()
    best_steps = torch.zeros(len(matrices), dtype=torch.int64)
    best_matrices = matrices.clone()
    max_residuals = torch.zeros(len(matrices), dtype=REAL_DTYPE)
    nonfinite = ~torch.isfinite(initial_objective)
    trace_rows: list[dict[str, object]] = []
    previous_masks: torch.Tensor | None = None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = stage3b_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        training_q = differentiable_q(parameter, training_operators, chunk_size=chunk_size)
        objectives, training = q_objective(training_q, arm)
        finite = torch.isfinite(objectives)
        nonfinite |= ~finite
        torch.where(finite, objectives, torch.zeros_like(objectives)).sum().backward()
        gradient_norms = torch.linalg.vector_norm(parameter.grad.reshape(len(parameter), -1), dim=-1)
        gradient_finite = torch.isfinite(gradient_norms)
        nonfinite |= ~gradient_finite
        parameter.grad[~gradient_finite] = 0
        current_masks = training["mask"].detach()
        churn = torch.zeros(len(parameter), dtype=REAL_DTYPE) if previous_masks is None else torch.mean(
            (current_masks != previous_masks).to(REAL_DTYPE), dim=(-2, -1)
        )
        previous_masks = current_masks
        optimizer.step()
        with torch.no_grad():
            matrix_finite = torch.isfinite(parameter.real).all(dim=(-2, -1)) & torch.isfinite(parameter.imag).all(dim=(-2, -1))
            nonfinite |= ~matrix_finite
            parameter[~matrix_finite] = best_matrices[~matrix_finite]
            parameter.copy_(polar_project(parameter))
            residuals = torch.linalg.matrix_norm(
                parameter.mH @ parameter - torch.eye(parameter.shape[-1], dtype=COMPLEX_DTYPE)
            )
            max_residuals = torch.maximum(max_residuals, residuals)

        if step % validation_cadence == 0 or step == steps:
            validation_objective, validation = _validation_metrics(parameter.detach(), validation_operators, arm, chunk_size)
            finite_validation = torch.isfinite(validation_objective)
            nonfinite |= ~finite_validation
            improved = finite_validation & (
                (validation_objective < best_objective)
                | ((validation_objective == best_objective) & (validation["lower_tail"] > best_tail))
            )
            best_objective[improved] = validation_objective[improved]
            best_eta[improved] = validation["eta"][improved]
            best_tail[improved] = validation["lower_tail"][improved]
            best_steps[improved] = step
            best_matrices[improved] = parameter.detach()[improved]
            for index, name in enumerate(initialization_names):
                trace_rows.append({
                    "arm": arm,
                    "initialization": name,
                    "step": step,
                    "training_objective": float(objectives[index]),
                    "training_eta": float(training["eta"][index]),
                    "training_lower_tail": float(training["lower_tail"][index]),
                    "validation_objective": float(validation_objective[index]),
                    "validation_eta": float(validation["eta"][index]),
                    "validation_lower_tail": float(validation["lower_tail"][index]),
                    "gradient_norm": float(gradient_norms[index]),
                    "support_churn": float(churn[index]),
                    "unitarity_residual": float(residuals[index]),
                    "selected_so_far": bool(improved[index]),
                })

    final_objective, final = _validation_metrics(parameter.detach(), validation_operators, arm, chunk_size)
    nonfinite |= ~torch.isfinite(final_objective)
    return QTrainingResult(
        arm=arm,
        initialization_names=initialization_names,
        best_matrices=best_matrices,
        best_steps=best_steps,
        initial_validation_eta=initial["eta"],
        initial_validation_lower_tail=initial["lower_tail"],
        best_validation_eta=best_eta,
        best_validation_lower_tail=best_tail,
        final_matrices=parameter.detach().clone(),
        final_validation_eta=final["eta"],
        final_validation_lower_tail=final["lower_tail"],
        max_unitarity_residuals=max_residuals,
        nonfinite_seen=nonfinite,
        trace_rows=trace_rows,
        runtime_seconds=time.perf_counter() - started,
    )


def _maximum_assignment(scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact square assignment by the O(N^3) Hungarian primal-dual algorithm."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("assignment scores must be square")
    cost = -values
    n = len(cost)
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]
            delta = np.inf; j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur; way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment, float(values[np.arange(n), assignment].sum())


def aligned_basis_similarity(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    """Exact column-phase/permutation-invariant unitary basis similarity."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("both bases must be matching square matrices")
    overlaps = torch.abs(left.mH @ right).square().detach().cpu().numpy()
    permutation, total = _maximum_assignment(overlaps)
    mean_overlap = total / len(overlaps)
    return {
        "mean_matched_squared_overlap": mean_overlap,
        "aligned_chordal_distance": float(math.sqrt(max(0.0, 1.0 - mean_overlap))),
        "minimum_matched_squared_overlap": float(np.min(overlaps[np.arange(len(overlaps)), permutation])),
        "permutation": permutation.tolist(),
    }
