"""Relative-unitary geometry and generator decompositions for frozen Stage 4D."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg
import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE
from .stage4a import cyclic_shift, integer_weyl_coefficients, reconstruct_integer_weyl


@dataclass(frozen=True)
class LogarithmResult:
    generator: torch.Tensor
    eigenphases: np.ndarray
    branch_margin: float
    skew_residual: float
    reconstruction_residual: float
    schur_offdiagonal_residual: float
    gate_passed: bool


def skew_project(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection onto the skew-Hermitian tangent space."""
    return 0.5 * (matrix - matrix.mH)


def polar_unitary(matrix: torch.Tensor) -> torch.Tensor:
    """Nearest unitary polar factor in Frobenius norm."""
    left, _, right_h = torch.linalg.svd(matrix.to(COMPLEX_DTYPE))
    return left @ right_h


def unitary_exponential(generator: torch.Tensor) -> torch.Tensor:
    """Dense matrix exponential returned in the project's complex dtype."""
    value = linalg.expm(generator.detach().cpu().numpy())
    return torch.from_numpy(value).to(COMPLEX_DTYPE)


def unitary_logarithm(
    unitary: torch.Tensor, *, branch_tolerance: float = 0.05
) -> LogarithmResult:
    """Principal logarithm of a unitary with an explicit minus-one gate."""
    n = unitary.shape[-1]
    if unitary.shape != (n, n):
        raise ValueError("unitary must be square")
    array = unitary.detach().cpu().numpy()
    triangular, vectors = linalg.schur(array, output="complex")
    phases = np.angle(np.diag(triangular))
    generator_np = (vectors * (1j * phases)[None, :]) @ vectors.conj().T
    generator = skew_project(torch.from_numpy(generator_np).to(COMPLEX_DTYPE))
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    denominator = max(float(torch.linalg.matrix_norm(unitary)), 1e-300)
    branch_margin = float(np.min(np.pi - np.abs(phases)))
    return LogarithmResult(
        generator=generator,
        eigenphases=np.sort(phases),
        branch_margin=branch_margin,
        skew_residual=float(torch.linalg.matrix_norm(generator + generator.mH)),
        reconstruction_residual=float(
            torch.linalg.matrix_norm(unitary_exponential(generator) - unitary) / denominator
        ),
        schur_offdiagonal_residual=float(
            np.linalg.norm(triangular - np.diag(np.diag(triangular)))
            / max(np.linalg.norm(triangular), 1e-300)
        ),
        gate_passed=bool(
            branch_margin >= branch_tolerance
            and torch.linalg.matrix_norm(unitary.mH @ unitary - identity) < 1e-10
        ),
    )


def unitary_logarithm_branch(
    unitary: torch.Tensor,
    *,
    branch: str,
    cut_window: float = 0.05,
) -> LogarithmResult:
    """Exact one-sided branch alternatives for eigenphases near minus one."""
    if branch not in {"principal", "positive_cut", "negative_cut"}:
        raise ValueError("branch must be principal, positive_cut, or negative_cut")
    if branch == "principal":
        return unitary_logarithm(unitary)
    array = unitary.detach().cpu().numpy()
    triangular, vectors = linalg.schur(array, output="complex")
    phases = np.angle(np.diag(triangular))
    near = np.pi - np.abs(phases) < cut_window
    adjusted = phases.copy()
    if branch == "positive_cut":
        adjusted[near & (adjusted < 0)] += 2 * np.pi
    else:
        adjusted[near & (adjusted > 0)] -= 2 * np.pi
    generator_np = (vectors * (1j * adjusted)[None, :]) @ vectors.conj().T
    generator = skew_project(torch.from_numpy(generator_np).to(COMPLEX_DTYPE))
    denominator = max(float(torch.linalg.matrix_norm(unitary)), 1e-300)
    margin = float(np.min(np.pi - np.abs(phases)))
    return LogarithmResult(
        generator=generator,
        eigenphases=np.sort(adjusted),
        branch_margin=margin,
        skew_residual=float(torch.linalg.matrix_norm(generator + generator.mH)),
        reconstruction_residual=float(torch.linalg.matrix_norm(unitary_exponential(generator) - unitary) / denominator),
        schur_offdiagonal_residual=float(
            np.linalg.norm(triangular - np.diag(np.diag(triangular)))
            / max(np.linalg.norm(triangular), 1e-300)
        ),
        gate_passed=False,
    )


def cayley_generator(unitary: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Cayley tangent A=2(R-I)(R+I)^-1 with conditioning diagnostics."""
    n = unitary.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    plus = unitary + identity
    condition = float(torch.linalg.cond(plus))
    raw = 2 * torch.linalg.solve(plus.mT, (unitary - identity).mT).mT
    generator = skew_project(raw)
    reconstructed = torch.linalg.solve(
        (identity - 0.5 * generator).mT,
        (identity + 0.5 * generator).mT,
    ).mT
    return generator, {
        "condition_r_plus_i": condition,
        "skew_residual": float(torch.linalg.matrix_norm(generator + generator.mH)),
        "projection_change": float(
            torch.linalg.matrix_norm(generator - raw) / torch.clamp(torch.linalg.matrix_norm(raw), min=1e-300)
        ),
        "reconstruction_residual": float(
            torch.linalg.matrix_norm(reconstructed - unitary) / torch.linalg.matrix_norm(unitary)
        ),
    }


def align_columns(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fix one deterministic column phase/permutation gauge to a reference."""
    from .stage4a import phase_permutation_align

    aligned, match = phase_permutation_align(reference, candidate)
    permutation = torch.as_tensor(match["permutation"], dtype=torch.int64)
    selected = candidate[:, permutation]
    inner = torch.sum(reference.conj() * selected, dim=0)
    phases = torch.where(
        torch.abs(inner) > 0,
        torch.exp(-1j * torch.angle(inner)),
        torch.ones_like(inner),
    )
    return aligned, {
        "permutation": permutation.tolist(),
        "phases_real": phases.real.tolist(),
        "phases_imag": phases.imag.tolist(),
        "mean_matched_squared_overlap": float(match["mean_matched_squared_overlap"]),
    }


def unitary_distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Dimension-normalized geodesic and chordal distances."""
    n = left.shape[-1]
    relative = left.mH @ right
    logarithm = unitary_logarithm(relative, branch_tolerance=0.0)
    return {
        "geodesic_distance": float(torch.linalg.matrix_norm(logarithm.generator) / math.sqrt(n)),
        "chordal_distance": float(torch.linalg.matrix_norm(left - right) / math.sqrt(n)),
        "branch_margin": logarithm.branch_margin,
    }


def karcher_mean(
    waveforms: torch.Tensor,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 64,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Intrinsic unitary mean initialized by the polar Euclidean centroid."""
    if waveforms.ndim != 3 or waveforms.shape[-2] != waveforms.shape[-1]:
        raise ValueError("waveforms must be a nonempty batch of square matrices")
    mean = polar_unitary(torch.mean(waveforms, dim=0))
    trace: list[float] = []
    minimum_margin = math.inf
    for iteration in range(max_iterations):
        logs = [unitary_logarithm(mean.mH @ waveform, branch_tolerance=0.0) for waveform in waveforms]
        minimum_margin = min(minimum_margin, *(item.branch_margin for item in logs))
        average = skew_project(torch.mean(torch.stack([item.generator for item in logs]), dim=0))
        norm = float(torch.linalg.matrix_norm(average))
        trace.append(norm)
        if norm < tolerance:
            break
        mean = polar_unitary(mean @ unitary_exponential(average))
    identity = torch.eye(mean.shape[-1], dtype=COMPLEX_DTYPE)
    return mean, {
        "method": "intrinsic Karcher iteration from polar-projected Euclidean centroid",
        "iterations": len(trace),
        "converged": bool(trace[-1] < tolerance),
        "final_mean_tangent_norm": trace[-1],
        "minimum_iteration_branch_margin": minimum_margin,
        "unitarity_residual": float(torch.linalg.matrix_norm(mean.mH @ mean - identity)),
        "tangent_norm_trace": trace,
    }


def _entropy_support(energy: np.ndarray) -> tuple[float, float]:
    probability = energy.reshape(-1) / max(float(np.sum(energy)), 1e-300)
    positive = probability > 0
    entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))
    return entropy / math.log(len(probability)), 1.0 / max(float(np.sum(probability**2)), 1e-300)


def circulant_projection(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[-1]
    rows = torch.arange(n)
    output = torch.zeros_like(matrix)
    for offset in range(n):
        columns = (rows - offset) % n
        output[rows, columns] = torch.mean(matrix[rows, columns])
    return output


def toeplitz_projection(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[-1]
    output = torch.zeros_like(matrix)
    rows, columns = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    offsets = rows - columns
    for offset in range(-(n - 1), n):
        selected = offsets == offset
        output[selected] = torch.mean(matrix[selected])
    return output


def kron_rank_one_projection(matrix: torch.Tensor, left_size: int = 8) -> torch.Tensor:
    """Nearest unconstrained A kron B projection for N=left_size*right_size."""
    n = matrix.shape[-1]
    if matrix.shape != (n, n) or n % left_size:
        raise ValueError("matrix dimension must factor by left_size")
    right_size = n // left_size
    reshaped = matrix.reshape(left_size, right_size, left_size, right_size)
    rearranged = reshaped.permute(0, 2, 1, 3).reshape(left_size**2, right_size**2)
    left, singular, right_h = torch.linalg.svd(rearranged)
    factor_left = (torch.sqrt(singular[0]) * left[:, 0]).reshape(left_size, left_size)
    factor_right = (torch.sqrt(singular[0]) * right_h[0]).reshape(right_size, right_size)
    return torch.kron(factor_left.contiguous(), factor_right.contiguous())


def relative_unitary_metrics(relative: torch.Tensor) -> dict[str, object]:
    """Quantitative matrix structure of one OFDM-relative unitary."""
    n = relative.shape[-1]
    energy = torch.abs(relative).square().detach().cpu().numpy()
    total = float(np.sum(energy))
    order = np.argsort(energy.reshape(-1), kind="stable")[::-1]
    entropy, support = _entropy_support(energy)
    rows = np.arange(n)
    offsets = (rows[:, None] - rows[None, :]) % n
    distance = np.minimum(offsets, n - offsets)
    circulant = circulant_projection(relative)
    toeplitz = toeplitz_projection(relative)
    kron = kron_rank_one_projection(relative)
    logarithm = unitary_logarithm(relative)
    displacement_shift = relative - torch.roll(torch.roll(relative, 1, 0), 1, 1)
    displacement_singular = torch.linalg.svdvals(displacement_shift).detach().cpu().numpy()
    displacement_energy = displacement_singular**2
    displacement_probability = displacement_energy / max(float(np.sum(displacement_energy)), 1e-300)
    block = energy.reshape(8, 8, 8, 8).sum(axis=(1, 3))
    return {
        "unitarity_residual": float(
            torch.linalg.matrix_norm(relative.mH @ relative - torch.eye(n, dtype=COMPLEX_DTYPE))
        ),
        "diagonal_energy": float(np.trace(energy) / total),
        "band1_energy": float(np.sum(energy[distance <= 1]) / total),
        "band3_energy": float(np.sum(energy[distance <= 3]) / total),
        "largest_entry_energy": float(energy.reshape(-1)[order[0]] / total),
        "top16_entry_energy": float(np.sum(energy.reshape(-1)[order[:16]]) / total),
        "top64_entry_energy": float(np.sum(energy.reshape(-1)[order[:64]]) / total),
        "entry_effective_support": support,
        "entry_normalized_entropy": entropy,
        "circulant_relative_residual": float(torch.linalg.matrix_norm(relative - circulant) / torch.linalg.matrix_norm(relative)),
        "toeplitz_relative_residual": float(torch.linalg.matrix_norm(relative - toeplitz) / torch.linalg.matrix_norm(relative)),
        "kron8x8_relative_residual": float(torch.linalg.matrix_norm(relative - kron) / torch.linalg.matrix_norm(relative)),
        "largest_block_energy": float(np.max(block) / total),
        "displacement_effective_rank": float(1.0 / max(np.sum(displacement_probability**2), 1e-300)),
        "displacement_rank_1e10": int(np.sum(displacement_singular > displacement_singular[0] * 1e-10)) if displacement_singular[0] > 0 else 0,
        "eigenphase_minimum": float(np.min(logarithm.eigenphases)),
        "eigenphase_maximum": float(np.max(logarithm.eigenphases)),
        "eigenphase_standard_deviation": float(np.std(logarithm.eigenphases)),
        "branch_margin": logarithm.branch_margin,
        "log_gate_passed": logarithm.gate_passed,
        "log_skew_residual": logarithm.skew_residual,
        "log_reconstruction_residual": logarithm.reconstruction_residual,
        "schur_offdiagonal_residual": logarithm.schur_offdiagonal_residual,
    }


def generator_spectrum(generator: torch.Tensor) -> dict[str, object]:
    """Skew-Hermitian eigenspectrum and energy ranks."""
    hermitian = (-1j * skew_project(generator)).to(COMPLEX_DTYPE)
    eigenvalues = torch.linalg.eigvalsh(hermitian).real
    magnitudes = torch.sort(torch.abs(eigenvalues), descending=True).values
    energy = magnitudes.square()
    cumulative = torch.cumsum(energy, 0) / torch.sum(energy)
    result: dict[str, object] = {
        "eigenvalue_real_maximum": float(torch.max(torch.abs(torch.linalg.eigvals(generator).real))),
        "frobenius_norm": float(torch.linalg.matrix_norm(generator)),
        "eigenvalue_magnitudes": magnitudes.tolist(),
        "cumulative_energy": cumulative.tolist(),
    }
    for threshold, label in ((0.90, "90"), (0.95, "95"), (0.99, "99"), (0.995, "99_5")):
        result[f"rank_{label}"] = int(
            torch.searchsorted(cumulative, torch.tensor(threshold, dtype=REAL_DTYPE)).item() + 1
        )
    return result


def _approximation_row(
    representation: str,
    complexity: int,
    target: torch.Tensor,
    approximation: torch.Tensor,
    **extra: object,
) -> dict[str, object]:
    target_energy = torch.sum(torch.abs(target).square())
    residual = torch.sum(torch.abs(target - approximation).square()) / torch.clamp(target_energy, min=1e-300)
    identity = torch.eye(target.shape[-1], dtype=COMPLEX_DTYPE)
    return {
        "representation": representation,
        "complexity": complexity,
        "relative_squared_residual": float(residual),
        "skew_residual": float(torch.linalg.matrix_norm(approximation + approximation.mH)),
        "exponential_unitarity_residual": float(
            torch.linalg.matrix_norm(
                unitary_exponential(approximation).mH @ unitary_exponential(approximation) - identity
            )
        ),
        **extra,
    }


def spectral_truncations(
    generator: torch.Tensor, ranks: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]]]:
    """Exact skew-Hermitian eigenmode truncations ordered by energy."""
    hermitian = (-1j * skew_project(generator)).to(COMPLEX_DTYPE)
    eigenvalues, vectors = torch.linalg.eigh(hermitian)
    order = torch.argsort(torch.abs(eigenvalues), descending=True, stable=True)
    approximations: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for rank in ranks:
        selected = order[: min(rank, len(order))]
        approximation = 1j * (vectors[:, selected] * eigenvalues[selected][None, :]) @ vectors[:, selected].mH
        approximation = skew_project(approximation)
        approximations[rank] = approximation
        rows.append(
            _approximation_row(
                "spectral",
                rank,
                generator,
                approximation,
                real_coefficients=rank * (2 * generator.shape[-1] + 1),
                storage_note="dense complex eigenvector plus one real eigenvalue per retained mode",
            )
        )
    return approximations, rows


def weyl_truncations(
    generator: torch.Tensor, ranks: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]], dict[str, object]]:
    """Nested coefficient truncation followed by explicit skew projection."""
    coefficients = integer_weyl_coefficients(generator)
    energy = torch.abs(coefficients).square()
    order = torch.argsort(energy.reshape(-1), descending=True, stable=True)
    cumulative = torch.cumsum(energy.reshape(-1)[order], 0) / torch.sum(energy)
    entropy, support = _entropy_support(energy.detach().cpu().numpy())
    approximations: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for rank in ranks:
        retained = torch.zeros_like(coefficients)
        retained.reshape(-1)[order[: min(rank, len(order))]] = coefficients.reshape(-1)[order[: min(rank, len(order))]]
        raw = reconstruct_integer_weyl(retained)
        approximation = skew_project(raw)
        approximations[rank] = approximation
        rows.append(
            _approximation_row(
                "weyl",
                rank,
                generator,
                approximation,
                raw_preprojection_residual=float(
                    torch.sum(torch.abs(generator - raw).square()) / torch.sum(torch.abs(generator).square())
                ),
                skew_projection_change=float(
                    torch.linalg.matrix_norm(approximation - raw)
                    / torch.clamp(torch.linalg.matrix_norm(raw), min=1e-300)
                ),
                selected_complex_coefficients=min(rank, coefficients.numel()),
                storage_real_scalars=2 * min(rank, coefficients.numel()),
            )
        )
    top = order[:64].detach().cpu().numpy()
    tau, nu = np.unravel_index(top, coefficients.shape)
    metadata = {
        "normalized_entropy": entropy,
        "effective_support": support,
        "coefficient_magnitudes": torch.sqrt(energy.reshape(-1)[order]).tolist(),
        "cumulative_energy": cumulative.tolist(),
        "top64_tau": tau.tolist(),
        "top64_nu": nu.tolist(),
        "top64_phase": torch.angle(coefficients.reshape(-1)[order[:64]]).tolist(),
    }
    return approximations, rows, metadata


def entry_truncations(
    generator: torch.Tensor,
    ranks: tuple[int, ...],
    *,
    basis: torch.Tensor | None = None,
    name: str = "time_entry",
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]]]:
    """Nested matrix-entry truncation in one fixed orthonormal coordinate basis."""
    coordinates = generator if basis is None else basis.mH @ generator @ basis
    order = torch.argsort(torch.abs(coordinates).square().reshape(-1), descending=True, stable=True)
    approximations: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for rank in ranks:
        retained = torch.zeros_like(coordinates)
        retained.reshape(-1)[order[: min(rank, len(order))]] = coordinates.reshape(-1)[order[: min(rank, len(order))]]
        raw = retained if basis is None else basis @ retained @ basis.mH
        approximation = skew_project(raw)
        approximations[rank] = approximation
        rows.append(_approximation_row(name, rank, generator, approximation, storage_real_scalars=2 * rank))
    return approximations, rows


def banded_truncations(
    generator: torch.Tensor, half_widths: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]]]:
    """Nested circular-band projections in the native time basis."""
    n = generator.shape[-1]
    indices = torch.arange(n)
    offsets = (indices[:, None] - indices[None, :]) % n
    distance = torch.minimum(offsets, n - offsets)
    approximations: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for half_width in half_widths:
        raw = torch.where(distance <= half_width, generator, torch.zeros_like(generator))
        approximation = skew_project(raw)
        count = int(torch.sum(distance <= half_width))
        approximations[half_width] = approximation
        rows.append(_approximation_row("circular_band", count, generator, approximation, half_width=half_width, storage_real_scalars=2 * count))
    return approximations, rows


def fixed_structure_projections(generator: torch.Tensor) -> list[dict[str, object]]:
    """Single fixed Toeplitz, circulant, block-diagonal, and 8x8 Kron projections."""
    n = generator.shape[-1]
    block_mask = torch.zeros((n, n), dtype=torch.bool)
    for start in range(0, n, 8):
        block_mask[start : start + 8, start : start + 8] = True
    projections = {
        "toeplitz": (skew_project(toeplitz_projection(generator)), 2 * (2 * n - 1)),
        "circulant": (skew_project(circulant_projection(generator)), 2 * n),
        "block_diagonal_8": (skew_project(torch.where(block_mask, generator, torch.zeros_like(generator))), 2 * 8 * 8 * 8),
        "kron_rank1_8x8": (skew_project(kron_rank_one_projection(generator)), 4 * 8 * 8),
    }
    return [
        _approximation_row(name, complexity, generator, approximation, storage_real_scalars=complexity)
        for name, (approximation, complexity) in projections.items()
    ]


def physical_generator_dictionary(n: int) -> tuple[tuple[str, ...], torch.Tensor]:
    """Small fixed delay, Doppler, and chirp generator dictionary."""
    shift = cyclic_shift(n)
    samples = torch.arange(n, dtype=REAL_DTYPE)
    modulation = torch.diag(torch.exp(2j * math.pi * samples / n).to(COMPLEX_DTYPE))
    names: list[str] = []
    values: list[torch.Tensor] = []

    def add_pair(prefix: str, unitary: torch.Tensor) -> None:
        names.extend((f"{prefix}_sine", f"{prefix}_cosine"))
        values.extend((0.5 * (unitary - unitary.mH), 0.5j * (unitary + unitary.mH)))

    for step in (1, 2, 4, 8):
        add_pair(f"delay_{step}", torch.linalg.matrix_power(shift, step))
    for step in (1, 2, 4, 8):
        add_pair(f"doppler_{step}", torch.linalg.matrix_power(modulation, step))
    x = torch.linspace(-1.0, 1.0, n, dtype=REAL_DTYPE)
    for degree in (1, 2, 3):
        names.append(f"chirp_diagonal_degree_{degree}")
        values.append(1j * torch.diag(x**degree).to(COMPLEX_DTYPE))
    normalized = []
    for value in values:
        current = skew_project(value)
        normalized.append(current / torch.clamp(torch.linalg.matrix_norm(current), min=1e-300))
    return tuple(names), torch.stack(normalized)


def fit_physical_dictionary(generator: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    """Real least-squares fit in the fixed physical skew-generator dictionary."""
    names, dictionary = physical_generator_dictionary(generator.shape[-1])
    design = torch.cat((dictionary.real.reshape(len(dictionary), -1), dictionary.imag.reshape(len(dictionary), -1)), dim=1).T
    target_vector = torch.cat((generator.real.reshape(-1), generator.imag.reshape(-1)))
    coefficients = torch.linalg.lstsq(design, target_vector).solution
    approximation = skew_project(torch.einsum("a,aij->ij", coefficients.to(COMPLEX_DTYPE), dictionary))
    contribution_order = torch.argsort(torch.abs(coefficients), descending=True, stable=True)
    commutators = []
    dominant = contribution_order[: min(6, len(contribution_order))]
    for left_index in range(len(dominant)):
        for right_index in range(left_index + 1, len(dominant)):
            left = dictionary[dominant[left_index]]
            right = dictionary[dominant[right_index]]
            denominator = torch.linalg.matrix_norm(left) * torch.linalg.matrix_norm(right)
            commutators.append(float(torch.linalg.matrix_norm(left @ right - right @ left) / denominator))
    row = _approximation_row(
        "fixed_physical_dictionary",
        len(names),
        generator,
        approximation,
        real_coefficients=len(names),
        names=list(names),
        coefficients=coefficients.tolist(),
        dominant_names=[names[index] for index in dominant.tolist()],
        dominant_pair_commutator_mean=float(np.mean(commutators)),
        dominant_pair_commutator_maximum=float(np.max(commutators)),
    )
    return approximation, row


def tangent_analysis(tangents: torch.Tensor) -> dict[str, object]:
    """Descriptive five-sample tangent geometry; no intrinsic-dimension claim."""
    if tangents.ndim != 3:
        raise ValueError("tangents must be a matrix batch")
    vectors = torch.cat((tangents.real.reshape(len(tangents), -1), tangents.imag.reshape(len(tangents), -1)), dim=1)
    centered = vectors - torch.mean(vectors, dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    cumulative = torch.cumsum(energy, 0) / torch.clamp(torch.sum(energy), min=1e-300)
    similarities = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            denominator = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            similarities.append(float(torch.dot(vectors[left], vectors[right]) / torch.clamp(denominator, min=1e-300)))
    return {
        "tangent_norms": torch.linalg.vector_norm(vectors, dim=1).tolist(),
        "pairwise_cosine_similarities": similarities,
        "centered_singular_values": singular.tolist(),
        "centered_cumulative_energy": cumulative.tolist(),
        "centered_rank_upper_bound": min(len(tangents) - 1, int(torch.linalg.matrix_rank(centered))),
        "interpretation": "descriptive variation among five samples; not an intrinsic manifold dimension estimate",
    }
