import math

import torch

from src.channel import COMPLEX_DTYPE, assemble_dd_channel
from src.stage4a import (
    basis_fidelity,
    canonical_generators,
    cyclic_shift,
    dd_expansion_metrics,
    integer_weyl_coefficients,
    orbit_metrics,
    reconstruct_integer_weyl,
)
from src.structural_fitting import chirp_fourier, fit_chirp_fourier, fit_gabor_prototype, fit_polynomial_fourier
from src.waveforms import afdm_waveform, ofdm_waveform, random_unitary
from scripts.stage4a import communication_table_lines


def test_integer_weyl_expansion_round_trip_and_project_convention():
    matrix = random_unitary(n=8, seed=19)
    coefficients = integer_weyl_coefficients(matrix)
    assert torch.allclose(reconstruct_integer_weyl(coefficients), matrix, atol=1e-12, rtol=1e-12)
    atom = assemble_dd_channel(
        torch.tensor([3.0]), torch.tensor([2.0]), torch.ones(1, dtype=COMPLEX_DTYPE), n=8
    )
    atom_coefficients = integer_weyl_coefficients(atom)
    assert torch.abs(atom_coefficients[3, 2] - 1) < 1e-12
    assert torch.sum(torch.abs(atom_coefficients).square()) - 1 < 1e-12


def test_exact_canonical_shift_generates_native_columns():
    waveform = random_unitary(n=8, seed=20)
    shift_generator, _ = canonical_generators(waveform)
    result = orbit_metrics(waveform, shift_generator)
    assert result["phase_aligned_orbit_nmse"] < 1e-25
    assert result["final_column_overlap"] > 1 - 1e-12


def test_generator_diagnostic_separates_ofdm_from_haar_control():
    ofdm_shift, _ = canonical_generators(ofdm_waveform(n=8))
    haar_shift, _ = canonical_generators(random_unitary(n=8, seed=22))
    ofdm = dd_expansion_metrics(integer_weyl_coefficients(ofdm_shift), k=8)
    haar = dd_expansion_metrics(integer_weyl_coefficients(haar_shift), k=8)
    assert ofdm["largest_coefficient_energy"] > 1 - 1e-12
    assert haar["largest_coefficient_energy"] < 0.25


def test_alignment_removes_only_column_phase_and_permutation():
    waveform = random_unitary(n=8, seed=21)
    permutation = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    phases = torch.exp(1j * torch.arange(8, dtype=torch.float64))
    equivalent = waveform[:, permutation] * phases[None, :]
    fidelity = basis_fidelity(waveform, equivalent)
    assert fidelity["aligned_matrix_nmse"] < 1e-25
    assert fidelity["raw_matrix_nmse"] > 0.1


def test_chirp_family_contains_project_afdm_convention():
    parameters = torch.tensor([1 / 16, math.sqrt(2) / 800], dtype=torch.float64)
    assert torch.allclose(chirp_fourier(parameters, 8), afdm_waveform(n=8, c1=parameters[0], c2=parameters[1]))
    assert fit_chirp_fourier(afdm_waveform(n=8, c1=parameters[0], c2=parameters[1]), steps=1).fidelity[
        "aligned_matrix_nmse"
    ] < 1e-20


def test_polynomial_fourier_fit_recovers_ofdm_control():
    assert fit_polynomial_fourier(ofdm_waveform(n=8), degree=2, steps=1).fidelity[
        "aligned_matrix_nmse"
    ] < 1e-20


def test_gabor_fit_recovers_a_unitary_classical_orbit():
    target = ofdm_waveform(n=8)
    fit = fit_gabor_prototype(target)
    identity = torch.eye(8, dtype=COMPLEX_DTYPE)
    assert torch.linalg.matrix_norm(fit.waveform.mH @ fit.waveform - identity) < 1e-12
    assert fit.fidelity["aligned_matrix_nmse"] < 1e-20
    assert fit.parameters["orbit_reconstruction_residual"] < 1e-10


def test_communication_report_table_uses_local_summary_lookup():
    rows = [
        {"waveform": "W", "modulation": modulation, "snr_db": 18, "support_rule": support, "ber": value}
        for modulation, support, value in (
            ("QPSK", "Fixed-Q", 0.1), ("16QAM", "Fixed-Q", 0.2),
            ("QPSK", "Oracle", 0.3), ("16QAM", "Oracle", 0.4),
        )
    ]
    assert communication_table_lines(rows, ("W",)) == ["| W | 0.1 | 0.2 | 0.3 | 0.4 |"]
