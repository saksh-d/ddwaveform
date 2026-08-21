"""Low-parameter unitary transform families for frozen Stage 4A fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE, dft_matrix
from .oracle import polar_project
from .stage3b import aligned_basis_similarity
from .stage4a import basis_fidelity, cyclic_shift


@dataclass(frozen=True)
class StructuredFit:
    family: str
    waveform: torch.Tensor
    parameters: dict[str, object]
    free_real_parameters: int
    storage_complexity: str
    application_complexity: str
    factor_depth: int
    fidelity: dict[str, object]


def polynomial_phase(coefficients: torch.Tensor, n: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, n, dtype=REAL_DTYPE)
    phase = sum(coefficients[power] * x**power for power in range(len(coefficients)))
    return torch.exp(1j * phase).to(COMPLEX_DTYPE)


def polynomial_fourier(parameters: torch.Tensor, degree: int, n: int) -> torch.Tensor:
    width = degree + 1
    left = polynomial_phase(parameters[:width], n)
    right = polynomial_phase(parameters[width:], n)
    return left[:, None] * dft_matrix(n).mH * right[None, :]


def chirp_fourier(parameters: torch.Tensor, n: int) -> torch.Tensor:
    samples = torch.arange(n, dtype=REAL_DTYPE)
    left = torch.exp(2j * math.pi * parameters[0] * samples.square()).to(COMPLEX_DTYPE)
    right = torch.exp(2j * math.pi * parameters[1] * samples.square()).to(COMPLEX_DTYPE)
    return left[:, None] * dft_matrix(n).mH * right[None, :]


def shallow_fft_product(parameters: torch.Tensor, degree: int, n: int) -> torch.Tensor:
    width = degree + 1
    phases = [polynomial_phase(parameters[index * width : (index + 1) * width], n) for index in range(3)]
    inverse_fft = dft_matrix(n).mH
    return phases[0][:, None] * inverse_fft @ (phases[1][:, None] * inverse_fft) * phases[2][None, :]


def _fit_factor_family(
    target: torch.Tensor,
    family: str,
    builder,
    parameter_count: int,
    *,
    steps: int,
    seed: int,
    learning_rate: float = 0.04,
    initial_parameters: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    initial = 0.05 * torch.randn(parameter_count, dtype=REAL_DTYPE, generator=generator)
    if initial_parameters is not None:
        initial = initial_parameters.detach().to(REAL_DTYPE).clone()
    parameters = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    permutation = torch.arange(target.shape[-1])
    best_loss = float("inf")
    best_parameters = parameters.detach().clone()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidate = builder(parameters)
        if step % 20 == 0:
            permutation = torch.as_tensor(
                aligned_basis_similarity(target, candidate.detach())["permutation"], dtype=torch.int64
            )
        matched = candidate[:, permutation]
        overlap = torch.sum(target.conj() * matched, dim=0)
        loss = 1.0 - torch.mean(torch.abs(overlap).square())
        current_parameters = parameters.detach().clone()
        loss.backward(); optimizer.step()
        if float(loss) < best_loss:
            best_loss = float(loss); best_parameters = current_parameters
    return builder(best_parameters).detach(), best_parameters


def fit_polynomial_fourier(target: torch.Tensor, degree: int, steps: int = 400, seed: int = 11100) -> StructuredFit:
    n = target.shape[-1]
    count = 2 * (degree + 1)
    builder = lambda values: polynomial_fourier(values, degree, n)
    candidates = [
        _fit_factor_family(target, f"FOURIER_POLY_D{degree}", builder, count, steps=steps, seed=seed + degree),
        (builder(torch.zeros(count, dtype=REAL_DTYPE)), torch.zeros(count, dtype=REAL_DTYPE)),
    ]
    waveform, parameters = min(candidates, key=lambda item: basis_fidelity(target, item[0])["aligned_matrix_nmse"])
    return StructuredFit(
        f"FOURIER_POLY_D{degree}", waveform, {"degree": degree, "phase_coefficients": parameters.tolist()},
        count - 1, "O(log N) coefficients plus permutation", "two diagonal phases and one IFFT: O(N log N)",
        3, basis_fidelity(target, waveform),
    )


def fit_chirp_fourier(target: torch.Tensor, steps: int = 400, seed: int = 11110) -> StructuredFit:
    n = target.shape[-1]
    builder = lambda values: chirp_fourier(values, n)
    starts = (
        torch.zeros(2, dtype=REAL_DTYPE),
        torch.tensor([1 / (2 * n), math.sqrt(2) / (100 * n)], dtype=REAL_DTYPE),
    )
    candidates = [
        _fit_factor_family(target, "CHIRP_FOURIER", builder, 2, steps=steps, seed=seed,
                           learning_rate=0.0005, initial_parameters=start)
        for start in starts
    ]
    waveform, parameters = min(candidates, key=lambda item: basis_fidelity(target, item[0])["aligned_matrix_nmse"])
    return StructuredFit(
        "CHIRP_FOURIER", waveform, {"c1": float(parameters[0]), "c2": float(parameters[1])},
        2, "2 real chirp coefficients plus permutation", "two chirp diagonals and one IFFT: O(N log N)",
        3, basis_fidelity(target, waveform),
    )


def fit_shallow_fft_product(target: torch.Tensor, degree: int = 2, steps: int = 500, seed: int = 11120) -> StructuredFit:
    n = target.shape[-1]
    count = 3 * (degree + 1)
    waveform, parameters = _fit_factor_family(
        target, "SHALLOW_2FFT", lambda values: shallow_fft_product(values, degree, n), count,
        steps=steps, seed=seed,
    )
    return StructuredFit(
        "SHALLOW_2FFT", waveform,
        {"degree": degree, "phase_coefficients": parameters.tolist()}, count - 2,
        "O(log N) coefficients plus permutation", "three diagonal phases and two IFFTs: O(N log N)",
        5, basis_fidelity(target, waveform),
    )


def _modulation(n: int, step: int, sign: int = -1) -> torch.Tensor:
    samples = torch.arange(n, dtype=REAL_DTYPE)
    return torch.diag(torch.exp(sign * 2j * math.pi * step * samples / n).to(COMPLEX_DTYPE))


def _gabor_orbit(prototype: torch.Tensor, delay_step: int, modulation_sign: int = -1) -> torch.Tensor:
    n = len(prototype)
    delay_count = n // delay_step
    doppler_count = delay_step
    delay = cyclic_shift(n, delay_step)
    doppler = _modulation(n, delay_count, modulation_sign)
    columns = []
    for doppler_index in range(doppler_count):
        modulated = torch.linalg.matrix_power(doppler, doppler_index)
        for delay_index in range(delay_count):
            columns.append(modulated @ torch.linalg.matrix_power(delay, delay_index) @ prototype)
    return torch.stack(columns, dim=1)


def fit_gabor_prototype(target: torch.Tensor) -> StructuredFit:
    """Fit a critical-lattice prototype in native order, then take its exact tight orbit."""
    n = target.shape[-1]
    candidates: list[tuple[float, int, int, torch.Tensor, torch.Tensor]] = []
    delay_steps = tuple(step for step in range(1, n + 1) if n % step == 0)
    for delay_step in delay_steps:
        delay_count = n // delay_step
        for modulation_sign in (-1, 1):
            doppler = _modulation(n, delay_count, modulation_sign)
            delay = cyclic_shift(n, delay_step)
            inverse_atoms = []
            column = 0
            for doppler_index in range(delay_step):
                modulated = torch.linalg.matrix_power(doppler, doppler_index)
                for delay_index in range(delay_count):
                    operator = modulated @ torch.linalg.matrix_power(delay, delay_index)
                    inverse_atoms.append(operator.mH @ target[:, column]); column += 1
            prototype = torch.mean(torch.stack(inverse_atoms), dim=0)
            raw = _gabor_orbit(prototype, delay_step, modulation_sign)
            tight = polar_project(raw)
            fidelity = basis_fidelity(target, tight)
            candidates.append((float(fidelity["aligned_matrix_nmse"]), delay_step, modulation_sign, tight, tight[:, 0]))
    _, delay_step, modulation_sign, waveform, prototype = min(candidates, key=lambda row: row[0])
    residual = float(torch.linalg.matrix_norm(_gabor_orbit(prototype, delay_step, modulation_sign) - waveform))
    return StructuredFit(
        "GABOR_TIGHT_PROTOTYPE", waveform,
        {"delay_step": delay_step, "doppler_step": n // delay_step, "modulation_sign": modulation_sign,
         "prototype_real": prototype.real.tolist(),
         "prototype_imag": prototype.imag.tolist(), "orbit_reconstruction_residual": residual},
        2 * n - 2, "O(N) prototype samples plus two integer lattice steps",
        "prototype time-frequency shifts; O(N^2) direct, structured implementations possible", 2,
        basis_fidelity(target, waveform),
    )


def fit_family_hierarchy(target: torch.Tensor, steps: int = 400) -> list[StructuredFit]:
    """Predeclared A/B/C/D hierarchy; selection uses structural fidelity only."""
    return [
        fit_polynomial_fourier(target, degree, steps=steps) for degree in (1, 2, 3)
    ] + [
        fit_chirp_fourier(target, steps=steps),
        fit_gabor_prototype(target),
        fit_shallow_fft_product(target, steps=max(steps, 500)),
    ]
