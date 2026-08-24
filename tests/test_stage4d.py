from pathlib import Path
import re

import numpy as np
import torch

from scripts.stage4d import (
    BAND_WIDTHS,
    BRANCHES,
    BRANCH_TOLERANCE,
    COMMUNICATION_LEVELS,
    ENTRY_RANKS,
    SPECTRAL_RANKS,
    WEYL_RANKS,
    branch_comparison,
)
from src.channel import COMPLEX_DTYPE
from src.stage4d import (
    align_columns,
    banded_truncations,
    cayley_generator,
    entry_truncations,
    fit_physical_dictionary,
    karcher_mean,
    kron_rank_one_projection,
    polar_unitary,
    spectral_truncations,
    tangent_analysis,
    unitary_exponential,
    unitary_logarithm,
    unitary_logarithm_branch,
    weyl_truncations,
)
from src.waveforms import random_unitary


def _small_generator(n: int = 8) -> torch.Tensor:
    generator = torch.Generator(device="cpu"); generator.manual_seed(19001)
    raw = torch.randn((n, n), dtype=torch.float64, generator=generator) + 1j * torch.randn(
        (n, n), dtype=torch.float64, generator=generator
    )
    return 0.08 * (raw - raw.mH).to(COMPLEX_DTYPE)


def test_principal_unitary_logarithm_gate_and_round_trip():
    generator = _small_generator()
    unitary = unitary_exponential(generator)
    result = unitary_logarithm(unitary)
    assert result.gate_passed
    assert result.branch_margin > 1.0
    assert result.skew_residual < 1e-13
    assert result.reconstruction_residual < 1e-12
    assert torch.linalg.matrix_norm(result.generator - generator) < 1e-11


def test_branch_gate_rejects_minus_one_eigenvalue():
    phases = torch.tensor([np.pi - 1e-3, 0.2, -0.4], dtype=torch.float64)
    unitary = torch.diag(torch.exp(1j * phases)).to(COMPLEX_DTYPE)
    result = unitary_logarithm(unitary, branch_tolerance=0.05)
    assert not result.gate_passed
    assert result.branch_margin < 0.002
    assert result.reconstruction_residual < 1e-12


def test_valid_branch_and_cayley_alternatives_reconstruct_same_unitary():
    phases = torch.tensor([np.pi - 1e-3, -np.pi + 2e-3, 0.2], dtype=torch.float64)
    unitary = torch.diag(torch.exp(1j * phases)).to(COMPLEX_DTYPE)
    positive = unitary_logarithm_branch(unitary, branch="positive_cut")
    negative = unitary_logarithm_branch(unitary, branch="negative_cut")
    cayley, metrics = cayley_generator(unitary)
    assert positive.reconstruction_residual < 1e-12
    assert negative.reconstruction_residual < 1e-12
    assert torch.linalg.matrix_norm(positive.generator - negative.generator) > 1.0
    assert metrics["reconstruction_residual"] < 1e-10
    assert torch.linalg.matrix_norm(cayley + cayley.mH) < 1e-10


def test_alignment_and_karcher_center_recover_common_gauge():
    base = random_unitary(8, 19002)
    permutation = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    phases = torch.exp(1j * torch.arange(8, dtype=torch.float64))
    equivalent = base[:, permutation] * phases[None, :]
    aligned, _ = align_columns(base, equivalent)
    center, diagnostics = karcher_mean(torch.stack((base, aligned)))
    assert torch.linalg.matrix_norm(aligned - base) < 1e-12
    assert diagnostics["converged"]
    assert torch.linalg.matrix_norm(center - base) < 1e-11


def test_nested_generator_truncations_are_skew_and_exact_at_full_complexity():
    generator = _small_generator()
    spectral, spectral_rows = spectral_truncations(generator, (1, 4, 8))
    weyl, weyl_rows, _ = weyl_truncations(generator, (1, 8, 64))
    entries, entry_rows = entry_truncations(generator, (1, 8, 64))
    bands, band_rows = banded_truncations(generator, (0, 1, 4))
    for approximation in (*spectral.values(), *weyl.values(), *entries.values(), *bands.values()):
        assert torch.linalg.matrix_norm(approximation + approximation.mH) < 1e-12
    assert spectral_rows[-1]["relative_squared_residual"] < 1e-24
    assert weyl_rows[-1]["relative_squared_residual"] < 1e-24
    assert entry_rows[-1]["relative_squared_residual"] < 1e-24
    assert band_rows[-1]["relative_squared_residual"] < 1e-24


def test_fixed_physical_dictionary_is_small_real_skew_fit():
    generator = _small_generator()
    approximation, metrics = fit_physical_dictionary(generator)
    assert metrics["real_coefficients"] == 19
    assert len(metrics["names"]) == 19
    assert torch.linalg.matrix_norm(approximation + approximation.mH) < 1e-12
    assert 0 <= metrics["relative_squared_residual"] <= 1.01


def test_kron_projection_recovers_exact_8_by_8_factorization():
    generator = torch.Generator(device="cpu"); generator.manual_seed(19004)
    left = torch.randn((8, 8), dtype=COMPLEX_DTYPE, generator=generator)
    right = torch.randn((8, 8), dtype=COMPLEX_DTYPE, generator=generator)
    target = torch.kron(left, right)
    approximation = kron_rank_one_projection(target)
    assert torch.linalg.matrix_norm(target - approximation) / torch.linalg.matrix_norm(target) < 1e-12


def test_tangent_analysis_respects_five_sample_rank_bound():
    tangents = torch.stack([factor * _small_generator() for factor in (-2, -1, 0, 1, 2)])
    metrics = tangent_analysis(tangents)
    assert metrics["centered_rank_upper_bound"] <= 4
    assert "not an intrinsic" in metrics["interpretation"]


def test_stage4d_structural_layer_contains_no_communication_selection():
    source = Path("src/stage4d.py").read_text()
    assert re.search(r"(^|[^a-z])ber([^a-z]|$)", source.lower()) is None
    assert "optimiz" not in source.lower()


def test_reconstruction_levels_are_predeclared_nested_subsets():
    assert BRANCHES == ("principal", "positive_cut", "negative_cut")
    assert set(COMMUNICATION_LEVELS["spectral"]) <= set(SPECTRAL_RANKS)
    assert set(COMMUNICATION_LEVELS["weyl"]) <= set(WEYL_RANKS)
    assert set(COMMUNICATION_LEVELS["time_entry"]) <= set(ENTRY_RANKS)
    assert set(COMMUNICATION_LEVELS["circular_band"]) <= set(BAND_WIDTHS)
    assert tuple(sorted(WEYL_RANKS)) == WEYL_RANKS


def test_cayley_gate_is_derived_from_log_branch_tolerance():
    rows = [
        {"branch": branch, "representation": "weyl", "complexity": 64, "relative_squared_residual": 0.8}
        for branch in BRANCHES
    ]
    metadata = {
        **{branch: {"spectrum": {"rank_95": 40}} for branch in BRANCHES},
        "cayley": {"condition_r_plus_i": 1.1 / BRANCH_TOLERANCE},
    }
    result = branch_comparison(rows, metadata)
    assert result["cayley_condition_limit"] == 1 / BRANCH_TOLERANCE
    assert result["cayley_ill_conditioned"]
