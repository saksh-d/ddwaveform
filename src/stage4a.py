"""Frozen structural-generator diagnostics for Stage 4A."""

from __future__ import annotations

import math

import numpy as np
import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE, assemble_dd_channel
from .stage3b import aligned_basis_similarity


def cyclic_shift(n: int, step: int = 1) -> torch.Tensor:
    """S e_k = e_(k+step), matching the project's circular-delay orientation."""
    output = torch.zeros((n, n), dtype=COMPLEX_DTYPE)
    columns = torch.arange(n)
    output[(columns + step) % n, columns] = 1
    return output


def canonical_generators(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return W S W^H and W D W^H in physical coordinates."""
    n = waveform.shape[-1]
    if waveform.shape != (n, n):
        raise ValueError("waveform must be square")
    samples = torch.arange(n, dtype=REAL_DTYPE)
    modulation = torch.diag(torch.exp(2j * math.pi * samples / n).to(COMPLEX_DTYPE))
    return waveform @ cyclic_shift(n) @ waveform.mH, waveform @ modulation @ waveform.mH


def integer_weyl_coefficients(matrix: torch.Tensor) -> torch.Tensor:
    """c[tau,nu] for U=Delta(nu)Pi(tau), using the exact project sign."""
    n = matrix.shape[-1]
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square")
    rows = torch.arange(n)
    diagonals = torch.stack([matrix[rows, (rows - tau) % n] for tau in range(n)])
    return torch.fft.ifft(diagonals, dim=-1)


def reconstruct_integer_weyl(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse of integer_weyl_coefficients."""
    n = coefficients.shape[-1]
    if coefficients.shape != (n, n):
        raise ValueError("coefficients must be square")
    diagonals = torch.fft.fft(coefficients, dim=-1)
    matrix = torch.zeros((n, n), dtype=coefficients.dtype)
    rows = torch.arange(n)
    for tau in range(n):
        matrix[rows, (rows - tau) % n] = diagonals[tau]
    return matrix


def _entropy_and_support(energy: np.ndarray) -> tuple[float, float]:
    probabilities = energy.reshape(-1) / max(float(np.sum(energy)), 1e-300)
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return entropy / math.log(len(probabilities)), 1.0 / float(np.sum(probabilities**2))


def dd_expansion_metrics(coefficients: torch.Tensor, k: int = 24) -> dict[str, object]:
    """Concentration and geometry of one generator's orthogonal Weyl expansion."""
    energy = torch.abs(coefficients).square().detach().cpu().numpy()
    n = energy.shape[0]
    flat_order = np.argsort(energy.reshape(-1), kind="stable")[::-1]
    total = float(np.sum(energy))
    entropy, support = _entropy_and_support(energy)
    top_count = min(k, n * n)
    top = flat_order[:top_count]
    tau, nu = np.unravel_index(top, energy.shape)
    phases = np.angle(coefficients.detach().cpu().numpy().reshape(-1)[top])
    phase_unwrapped = np.unwrap(phases)
    phase_design = np.column_stack((np.ones(top_count), np.arange(top_count)))
    phase_fit = phase_design @ np.linalg.lstsq(phase_design, phase_unwrapped, rcond=None)[0]
    phase_residual = float(np.sqrt(np.mean((phase_unwrapped - phase_fit) ** 2)))
    if len(np.unique(tau)) >= 2:
        design = np.column_stack((np.ones(top_count), tau))
        trajectory = design @ np.linalg.lstsq(design, nu, rcond=None)[0]
        trajectory_residual = float(np.sqrt(np.mean((nu - trajectory) ** 2)) / n)
    else:
        trajectory_residual = 0.0
    return {
        "largest_coefficient_energy": float(energy.reshape(-1)[flat_order[0]] / total),
        "top4_energy": float(np.sum(energy.reshape(-1)[flat_order[:4]]) / total),
        "top8_energy": float(np.sum(energy.reshape(-1)[flat_order[:8]]) / total),
        "top24_energy": float(np.sum(energy.reshape(-1)[flat_order[:24]]) / total),
        "effective_support": support,
        "normalized_entropy": entropy,
        "dominant_delays": tau.tolist(),
        "dominant_dopplers": nu.tolist(),
        "dominant_delay_count": int(len(np.unique(tau))),
        "dominant_doppler_count": int(len(np.unique(nu))),
        "trajectory_residual_normalized": trajectory_residual,
        "dominant_phase_linear_residual": phase_residual,
    }


def generator_matrix_metrics(generator: torch.Tensor) -> dict[str, float]:
    """Coordinate-locality, bandedness, circulantness, and action concentration."""
    n = generator.shape[-1]
    energy = torch.abs(generator).square().detach().cpu().numpy()
    total = float(np.sum(energy))
    rows = np.arange(n)
    offsets = (rows[:, None] - rows[None, :]) % n
    circular_distance = np.minimum(offsets, n - offsets)
    circulant = np.zeros_like(generator.detach().cpu().numpy())
    raw = generator.detach().cpu().numpy()
    for offset in range(n):
        selected = offsets == offset
        circulant[selected] = np.mean(raw[selected])
    probabilities = energy / np.maximum(np.sum(energy, axis=0, keepdims=True), 1e-300)
    action_effective_support = np.mean(1.0 / np.sum(probabilities**2, axis=0))
    eigenvalues = torch.linalg.eigvals(generator).detach().cpu().numpy()
    eigenphases = np.sort(np.mod(np.angle(eigenvalues), 2 * math.pi))
    eigenphase_gaps = np.diff(np.r_[eigenphases, eigenphases[0] + 2 * math.pi])
    entropy, support = _entropy_and_support(energy)
    return {
        "largest_entry_energy": float(np.max(energy) / total),
        "top24_entry_energy": float(np.sum(np.sort(energy.reshape(-1))[::-1][:24]) / total),
        "entry_effective_support": support,
        "entry_normalized_entropy": entropy,
        "diagonal_energy": float(np.trace(energy) / total),
        "band1_energy": float(np.sum(energy[circular_distance <= 1]) / total),
        "band3_energy": float(np.sum(energy[circular_distance <= 3]) / total),
        "mean_circular_displacement": float(np.sum(energy * circular_distance) / total),
        "circulant_relative_error": float(np.linalg.norm(raw - circulant) / np.linalg.norm(raw)),
        "action_effective_support_mean": float(action_effective_support),
        "action_e0_effective_support": float(1.0 / np.sum(probabilities[:, 0] ** 2)),
        "action_eN3_effective_support": float(1.0 / np.sum(probabilities[:, n // 3] ** 2)),
        "eigenvalue_modulus_max_error": float(np.max(np.abs(np.abs(eigenvalues) - 1.0))),
        "eigenphase_gap_std": float(np.std(eigenphase_gaps)),
    }


def phase_permutation_align(target: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    """Align candidate columns to target using only exact communication equivalences."""
    result = aligned_basis_similarity(target, candidate)
    permutation = torch.as_tensor(result["permutation"], dtype=torch.int64)
    aligned = candidate[:, permutation]
    inner = torch.sum(target.conj() * aligned, dim=0)
    phases = torch.where(torch.abs(inner) > 0, torch.exp(-1j * torch.angle(inner)), torch.ones_like(inner))
    return aligned * phases[None, :], result


def basis_fidelity(target: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    """Raw and communication-equivalence-aligned atom/matrix fidelity."""
    aligned, match = phase_permutation_align(target, candidate)
    raw_nmse = float(torch.sum(torch.abs(candidate - target) ** 2) / torch.sum(torch.abs(target) ** 2))
    aligned_nmse = float(torch.sum(torch.abs(aligned - target) ** 2) / torch.sum(torch.abs(target) ** 2))
    overlaps = torch.abs(torch.sum(target.conj() * aligned, dim=0)).detach().cpu().numpy()
    target_spectrum = torch.fft.fft(target, dim=0)
    candidate_spectrum = torch.fft.fft(aligned, dim=0)
    magnitude_error = torch.linalg.vector_norm(torch.abs(candidate) - torch.abs(target)) / torch.linalg.vector_norm(torch.abs(target))
    spectral_magnitude_error = torch.linalg.vector_norm(torch.abs(candidate_spectrum) - torch.abs(target_spectrum)) / torch.linalg.vector_norm(torch.abs(target_spectrum))
    time_phase_error = torch.angle(aligned * target.conj())
    frequency_phase_error = torch.angle(candidate_spectrum * target_spectrum.conj())
    return {
        "raw_matrix_nmse": raw_nmse,
        "aligned_matrix_nmse": aligned_nmse,
        "mean_column_overlap": float(np.mean(overlaps)),
        "column_overlap_p05": float(np.quantile(overlaps, 0.05)),
        "column_overlap_min": float(np.min(overlaps)),
        "mean_matched_squared_overlap": match["mean_matched_squared_overlap"],
        "time_magnitude_relative_error": float(magnitude_error),
        "frequency_magnitude_relative_error": float(spectral_magnitude_error),
        "time_phase_circular_rms": float(torch.sqrt(torch.mean(time_phase_error.square()))),
        "frequency_phase_circular_rms": float(torch.sqrt(torch.mean(frequency_phase_error.square()))),
        "permutation": match["permutation"],
    }


def _integer_dd_atom(tau: int, nu: int, n: int) -> torch.Tensor:
    return assemble_dd_channel(
        torch.tensor([float(tau)]), torch.tensor([float(nu)]),
        torch.ones(1, dtype=COMPLEX_DTYPE), n=n,
    )


def compressed_generators(generator: torch.Tensor, sparse_terms: tuple[int, ...] = (4, 8, 24)) -> dict[str, torch.Tensor]:
    """Predeclared DD-sparse and polynomial-phase cyclic-shift approximations."""
    n = generator.shape[-1]
    coefficients = integer_weyl_coefficients(generator)
    flat_order = torch.argsort(torch.abs(coefficients).square().reshape(-1), descending=True, stable=True)
    output: dict[str, torch.Tensor] = {}
    best = int(flat_order[0])
    tau, nu = divmod(best, n)
    phase = torch.angle(coefficients[tau, nu])
    output["single_dd_atom"] = torch.exp(1j * phase) * _integer_dd_atom(tau, nu, n)
    for terms in sparse_terms:
        selected = torch.zeros_like(coefficients)
        selected.reshape(-1)[flat_order[:terms]] = coefficients.reshape(-1)[flat_order[:terms]]
        output[f"dd_sparse_{terms}"] = reconstruct_integer_weyl(selected)

    energy_by_delay = torch.sum(torch.abs(coefficients).square(), dim=-1)
    shift = int(torch.argmax(energy_by_delay))
    rows = torch.arange(n)
    diagonal = generator[rows, (rows - shift) % n]
    phase_samples = np.unwrap(torch.angle(diagonal).detach().cpu().numpy())
    x = np.linspace(-1.0, 1.0, n)
    weights = np.abs(diagonal.detach().cpu().numpy()) + 1e-12
    for degree, name in ((1, "delay_doppler_phase"), (2, "chirped_shift"), (3, "cubic_phase_shift")):
        design = np.column_stack([x**power for power in range(degree + 1)])
        weighted = design * np.sqrt(weights)[:, None]
        coefficients_fit = np.linalg.lstsq(weighted, phase_samples * np.sqrt(weights), rcond=None)[0]
        fitted_phase = torch.from_numpy(design @ coefficients_fit).to(REAL_DTYPE)
        approximation = torch.zeros_like(generator)
        approximation[rows, (rows - shift) % n] = torch.exp(1j * fitted_phase)
        output[name] = approximation
    return output


def orbit_metrics(waveform: torch.Tensor, generator: torch.Tensor) -> dict[str, object]:
    """Accumulated native-order orbit error; no dense refitting or reordering."""
    n = waveform.shape[-1]
    current = waveform[:, 0]
    estimates = []
    for _ in range(n):
        estimates.append(current)
        current = generator @ current
    estimate = torch.stack(estimates, dim=1)
    inner = torch.sum(waveform.conj() * estimate, dim=0)
    norms = torch.linalg.vector_norm(estimate, dim=0)
    overlaps = torch.abs(inner) / torch.clamp(norms, min=1e-300)
    phase_aligned = estimate * torch.exp(-1j * torch.angle(inner))[None, :]
    errors = torch.sum(torch.abs(phase_aligned - waveform) ** 2, dim=0)
    return {
        "raw_orbit_nmse": float(torch.sum(torch.abs(estimate - waveform) ** 2) / n),
        "phase_aligned_orbit_nmse": float(torch.mean(errors)),
        "mean_column_overlap": float(torch.mean(overlaps)),
        "column_overlap_p05": float(torch.quantile(overlaps, 0.05)),
        "final_column_overlap": float(overlaps[-1]),
        "orbit_norm_min": float(torch.min(norms)),
        "orbit_norm_max": float(torch.max(norms)),
    }


def native_atom_diagnostics(waveform: torch.Tensor, representative: tuple[int, ...] = (0, 16, 32, 48)) -> list[dict[str, object]]:
    """Compact phase, spectrum, and polynomial-phase diagnostics for native atoms."""
    rows: list[dict[str, object]] = []
    n = waveform.shape[-1]
    x = np.linspace(-1.0, 1.0, n)
    for column in representative:
        atom = waveform[:, column]
        phase = np.unwrap(torch.angle(atom).detach().cpu().numpy())
        magnitude = torch.abs(atom).detach().cpu().numpy()
        spectral = torch.abs(torch.fft.fft(atom)).detach().cpu().numpy()
        row: dict[str, object] = {
            "column": column,
            "time_magnitude_cv": float(np.std(magnitude) / max(np.mean(magnitude), 1e-300)),
            "spectral_magnitude_cv": float(np.std(spectral) / max(np.mean(spectral), 1e-300)),
            "phase_increment_std": float(np.std(np.diff(phase))),
            "phase_increment_mean": float(np.mean(np.diff(phase))),
            "spectral_centroid_bin": float(np.sum(np.arange(n) * spectral**2) / max(np.sum(spectral**2), 1e-300)),
            "spectral_peak_bin": int(np.argmax(spectral)),
        }
        for degree in (1, 2, 3):
            design = np.column_stack([x**power for power in range(degree + 1)])
            coefficients = np.linalg.lstsq(design, phase, rcond=None)[0]
            fit = design @ coefficients
            row[f"phase_degree{degree}_rms"] = float(np.sqrt(np.mean((phase - fit) ** 2)))
            row[f"phase_degree{degree}_highest_coefficient"] = float(coefficients[-1])
        rows.append(row)
    return rows
