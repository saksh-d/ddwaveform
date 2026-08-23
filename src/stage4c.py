"""Depth-scaled cubic-phase cascade fitting for frozen Stage 4C targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE, dft_matrix
from .stage4a import phase_permutation_align


PHASE_DEGREE = 3


@dataclass
class CascadeFitResult:
    parameters: torch.Tensor
    validation_loss: torch.Tensor
    selected_steps: torch.Tensor
    history: list[dict[str, object]]
    nonfinite_seen: torch.Tensor


def cascade_parameter_count(depth: int) -> int:
    """Identifiable real parameters after removing every constant/global phase."""
    if depth < 1:
        raise ValueError("depth must be positive")
    return PHASE_DEGREE * (depth + 1)


def chirp_cascade(parameters: torch.Tensor, depth: int, n: int) -> torch.Tensor:
    """Return D0 F^H D1 ... F^H DL with cubic, constant-free phases."""
    expected = cascade_parameter_count(depth)
    values = torch.as_tensor(parameters, dtype=REAL_DTYPE)
    if values.shape[-1] != expected:
        raise ValueError(f"depth {depth} requires {expected} parameters")
    shaped = values.reshape(*values.shape[:-1], depth + 1, PHASE_DEGREE)
    x = torch.linspace(-1.0, 1.0, n, dtype=REAL_DTYPE)
    powers = torch.stack(tuple(x**power for power in range(1, PHASE_DEGREE + 1)))
    phases = torch.exp(1j * torch.einsum("...lp,pn->...ln", shaped, powers)).to(COMPLEX_DTYPE)
    inverse_fft = dft_matrix(n).mH
    waveform = phases[..., 0, :, None] * inverse_fft * phases[..., 1, None, :]
    for layer in range(2, depth + 1):
        waveform = (waveform @ inverse_fft) * phases[..., layer, None, :]
    return waveform


def canonicalize_target(
    target: torch.Tensor, reference: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix one phase/permutation convention against F^H before any fitting."""
    n = target.shape[-1]
    reference_matrix = dft_matrix(n).mH if reference is None else reference
    aligned, match = phase_permutation_align(reference_matrix, target)
    permutation = torch.as_tensor(match["permutation"], dtype=torch.int64)
    selected = target[:, permutation]
    inner = torch.sum(reference_matrix.conj() * selected, dim=0)
    phases = torch.where(
        torch.abs(inner) > 0,
        torch.exp(-1j * torch.angle(inner)),
        torch.ones_like(inner),
    )
    if not torch.allclose(aligned, selected * phases[None, :]):
        raise RuntimeError("canonical alignment reconstruction failed")
    return aligned, permutation, phases


def restore_native_order(
    canonical: torch.Tensor, permutation: torch.Tensor, phases: torch.Tensor
) -> torch.Tensor:
    """Undo the fixed target convention; no fit-dependent alignment is allowed."""
    if canonical.shape[-1] != len(permutation) or phases.shape != permutation.shape:
        raise ValueError("alignment metadata does not match waveform")
    output = torch.empty_like(canonical)
    output[..., :, permutation] = canonical * phases.conj()[..., None, :]
    return output


def matrix_objective(candidates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stage-4A aligned-atom loss after the permutation convention is frozen."""
    single = candidates.ndim == 2
    left = candidates.unsqueeze(0) if single else candidates
    right = targets.unsqueeze(0) if targets.ndim == 2 else targets
    if left.shape != right.shape:
        raise ValueError("candidate and target batches must match")
    overlaps = torch.sum(right.conj() * left, dim=-2)
    losses = 1.0 - torch.mean(torch.abs(overlaps).square(), dim=-1)
    return losses[0] if single else losses


def atomic_responses(waveforms: torch.Tensor, operators: torch.Tensor) -> torch.Tensor:
    matrices = waveforms.unsqueeze(0) if waveforms.ndim == 2 else waveforms
    if matrices.ndim != 3 or operators.ndim != 3:
        raise ValueError("waveforms and operators must be matrix batches")
    return matrices[:, None].mH @ operators[None] @ matrices[:, None]


def operator_objective(
    candidates: torch.Tensor,
    target_responses: torch.Tensor,
    operators: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Mean atom-wise normalized complex B discrepancy."""
    matrices = candidates.unsqueeze(0) if candidates.ndim == 2 else candidates
    if target_responses.shape[:2] != (len(matrices), len(operators)):
        raise ValueError("target responses do not match candidate/operator batches")
    total = torch.zeros(len(matrices), dtype=REAL_DTYPE)
    for start in range(0, len(operators), chunk_size):
        stop = min(start + chunk_size, len(operators))
        response = atomic_responses(matrices, operators[start:stop])
        target = target_responses[:, start:stop]
        numerator = torch.sum(torch.abs(response - target).square(), dim=(-2, -1))
        denominator = torch.sum(torch.abs(target).square(), dim=(-2, -1))
        total = total + torch.sum(numerator / torch.clamp(denominator, min=1e-300), dim=-1)
    return total / len(operators)


def fitting_learning_rate(step: int, steps: int) -> float:
    fraction = step / max(steps, 1)
    if fraction <= 0.50:
        return 0.04
    if fraction <= 0.80:
        return 0.015
    return 0.005


def fit_cascade_batch(
    targets: torch.Tensor,
    *,
    depth: int,
    arm: str,
    restart: int,
    steps: int,
    seed: int,
    training_operators: torch.Tensor | None = None,
    validation_operators: torch.Tensor | None = None,
    validation_cadence: int = 20,
    chunk_size: int = 16,
) -> CascadeFitResult:
    """Fit all targets together; each target retains its own validation winner."""
    if targets.ndim != 3 or targets.shape[-2] != targets.shape[-1]:
        raise ValueError("targets must be a batch of square matrices")
    if arm not in {"M", "B"}:
        raise ValueError("arm must be M or B")
    if steps < 1 or validation_cadence < 1:
        raise ValueError("steps and validation cadence must be positive")
    if arm == "B" and (training_operators is None or validation_operators is None):
        raise ValueError("Arm B requires independent training and validation banks")
    count = cascade_parameter_count(depth)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    initial = torch.zeros((len(targets), count), dtype=REAL_DTYPE)
    if restart > 0:
        initial = 0.10 * torch.randn(initial.shape, dtype=REAL_DTYPE, generator=generator)
    parameters = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameters], lr=fitting_learning_rate(1, steps))
    target_train = atomic_responses(targets, training_operators).detach() if arm == "B" else None
    target_validation = atomic_responses(targets, validation_operators).detach() if arm == "B" else None
    best_loss = torch.full((len(targets),), torch.inf, dtype=REAL_DTYPE)
    best_parameters = parameters.detach().clone()
    selected_steps = torch.zeros(len(targets), dtype=torch.int64)
    nonfinite = torch.zeros(len(targets), dtype=torch.bool)
    history: list[dict[str, object]] = []

    def losses(values: torch.Tensor, validation: bool) -> torch.Tensor:
        candidates = chirp_cascade(values, depth, targets.shape[-1])
        if arm == "M":
            return matrix_objective(candidates, targets)
        operators = validation_operators if validation else training_operators
        responses = target_validation if validation else target_train
        assert operators is not None and responses is not None
        return operator_objective(candidates, responses, operators, chunk_size=chunk_size)

    for step in range(1, steps + 1):
        optimizer.param_groups[0]["lr"] = fitting_learning_rate(step, steps)
        optimizer.zero_grad(set_to_none=True)
        training_loss = losses(parameters, False)
        finite = torch.isfinite(training_loss)
        nonfinite |= ~finite
        torch.where(finite, training_loss, torch.zeros_like(training_loss)).sum().backward()
        gradient_norm = torch.linalg.vector_norm(parameters.grad, dim=-1)
        gradient_finite = torch.isfinite(gradient_norm)
        nonfinite |= ~gradient_finite
        parameters.grad[~gradient_finite] = 0
        optimizer.step()
        if step % validation_cadence == 0 or step == steps:
            with torch.no_grad():
                validation_loss = losses(parameters, True)
                finite_validation = torch.isfinite(validation_loss)
                nonfinite |= ~finite_validation
                improved = finite_validation & (validation_loss < best_loss)
                best_loss[improved] = validation_loss[improved]
                best_parameters[improved] = parameters.detach()[improved]
                selected_steps[improved] = step
                for index in range(len(targets)):
                    history.append({
                        "target_index": index,
                        "depth": depth,
                        "arm": arm,
                        "restart": restart,
                        "step": step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "training_loss": float(training_loss[index]),
                        "validation_loss": float(validation_loss[index]),
                        "gradient_norm": float(gradient_norm[index]),
                        "selected_so_far": bool(improved[index]),
                    })
    return CascadeFitResult(best_parameters, best_loss, selected_steps, history, nonfinite)


def operator_fidelity(
    target: torch.Tensor,
    candidate: torch.Tensor,
    operators: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[dict[str, float], np.ndarray]:
    """Held-out complex, magnitude, distribution, and adverse-tail fidelity."""
    discrepancies: list[np.ndarray] = []
    complex_inner = 0.0j
    magnitude_inner = 0.0
    target_energy = 0.0
    candidate_energy = 0.0
    with torch.no_grad():
        for start in range(0, len(operators), chunk_size):
            current = operators[start : start + chunk_size]
            target_b = target.mH @ current @ target
            candidate_b = candidate.mH @ current @ candidate
            denominator = torch.sum(torch.abs(target_b).square(), dim=(-2, -1))
            discrepancy = torch.sum(torch.abs(target_b - candidate_b).square(), dim=(-2, -1)) / denominator
            discrepancies.append(discrepancy.cpu().numpy())
            complex_inner += complex(torch.sum(target_b.conj() * candidate_b))
            magnitude_inner += float(torch.sum(torch.abs(target_b) * torch.abs(candidate_b)))
            target_energy += float(torch.sum(torch.abs(target_b).square()))
            candidate_energy += float(torch.sum(torch.abs(candidate_b).square()))
    values = np.concatenate(discrepancies)
    denominator = math.sqrt(max(target_energy * candidate_energy, 1e-300))
    tail_count = max(1, math.ceil(0.05 * len(values)))
    return {
        "d_b": float(np.mean(values)),
        "complex_coefficient_coherence": abs(complex_inner) / denominator,
        "magnitude_coefficient_coherence": magnitude_inner / denominator,
        "per_atom_p05": float(np.quantile(values, 0.05)),
        "per_atom_median": float(np.median(values)),
        "per_atom_p95": float(np.quantile(values, 0.95)),
        "per_atom_maximum": float(np.max(values)),
        "adverse_tail_cvar95": float(np.mean(np.sort(values)[-tail_count:])),
    }, values


def parameter_population_metrics(parameter_vectors: np.ndarray) -> dict[str, object]:
    """Population summaries after exact phase/order symmetries are fixed."""
    values = np.asarray(parameter_vectors, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("parameter vectors must contain a population")
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    coefficient = np.divide(
        standard_deviation,
        np.abs(mean),
        out=np.full_like(standard_deviation, np.nan),
        where=np.abs(mean) > 1e-12,
    )
    distances = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distances.append(float(np.linalg.norm(values[left] - values[right]) / math.sqrt(values.shape[1])))
    return {
        "mean_parameter_vector": mean.tolist(),
        "standard_deviation_vector": standard_deviation.tolist(),
        "coefficient_of_variation_vector": coefficient.tolist(),
        "pairwise_rms_distance_mean": float(np.mean(distances)),
        "pairwise_rms_distance_minimum": float(np.min(distances)),
        "pairwise_rms_distance_maximum": float(np.max(distances)),
        "symmetry_adjusted_distance": "identical to raw: global phases removed; layer order and FFT signs fixed",
    }
