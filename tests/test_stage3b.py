"""Exact differentiability and protocol checks for Stage 3B."""

import math

import numpy as np
import torch

from src import config
from src.channel import make_generator
from src.stage2d import sobol_dd_samples
from src.stage3b import (
    INITIALIZATION_NAMES,
    LOWER_TAIL_FRACTION,
    ROBUST_LAMBDA,
    TRAIN_K,
    aligned_basis_similarity,
    atomic_operators,
    differentiable_q,
    fixed_initializations,
    hard_row_concentration,
    lower_row_tail,
    optimize_q_waveforms,
    q_objective,
)
from src.waveforms import random_unitary


def test_fixed_banks_produce_exact_q_mass_identities_for_unitary_waveform():
    n = 8
    waveform = random_unitary(n, 24001)
    delays, dopplers = sobol_dd_samples(32, seed=24002)
    q = differentiable_q(waveform, atomic_operators(delays, dopplers, n=n), chunk_size=8)
    assert torch.allclose(q.sum(dim=0), torch.ones(n, dtype=torch.float64), atol=2e-13, rtol=0)
    assert torch.allclose(q.sum(dim=1), torch.ones(n, dtype=torch.float64), atol=2e-13, rtol=0)
    assert abs(float(q.sum()) - n) < 2e-13


def test_q_objectives_have_finite_nonzero_waveform_gradients():
    n = 8
    generator = make_generator(24003)
    parameter = torch.nn.Parameter(
        torch.randn((2, n, n), dtype=torch.float64, generator=generator).to(torch.complex128)
        + 1j * torch.randn((2, n, n), dtype=torch.float64, generator=generator).to(torch.complex128)
    )
    delays, dopplers = sobol_dd_samples(24, seed=24004)
    q = differentiable_q(parameter, atomic_operators(delays, dopplers, n=n), chunk_size=6)
    mean, _ = q_objective(q, "mean", k=3)
    robust, _ = q_objective(q, "robust", k=3)
    (mean.sum() + robust.sum()).backward()
    norms = torch.linalg.vector_norm(parameter.grad.reshape(2, -1), dim=-1)
    assert torch.isfinite(norms).all()
    assert torch.all(norms > 0)


def test_detached_hard_support_updates_and_preserves_selected_gradients():
    q = torch.tensor([
        [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0],
        [4.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0],
    ], dtype=torch.float64, requires_grad=True)
    _, _, first = hard_row_concentration(q, 2)
    changed = q.detach().clone(); changed[0, 0] = 0.0; changed[0, 2] = 8.0
    _, rows, second = hard_row_concentration(changed.requires_grad_(), 2)
    assert not torch.equal(first[0], second[0])
    rows.sum().backward()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()
    assert torch.count_nonzero(changed.grad) > 0


def test_lower_tail_and_robust_arm_are_exactly_preregistered_q_only_quantities():
    rows = torch.arange(1, 11, dtype=torch.float64) / 10
    assert LOWER_TAIL_FRACTION == 0.10
    assert ROBUST_LAMBDA == 0.40
    assert lower_row_tail(rows) == rows[0]
    q = torch.diag(torch.arange(1, 9, dtype=torch.float64))
    mean, metrics = q_objective(q, "mean", k=1)
    robust, _ = q_objective(q, "robust", k=1)
    assert mean == 0
    assert metrics["lower_tail"] == 1
    assert robust == 0


def test_alignment_is_invariant_to_column_phase_and_permutation():
    basis = random_unitary(8, 24005)
    permutation = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
    phases = torch.exp(1j * torch.arange(8, dtype=torch.float64))
    equivalent = basis[:, permutation] * phases
    result = aligned_basis_similarity(basis, equivalent)
    assert math.isclose(result["mean_matched_squared_overlap"], 1.0, abs_tol=2e-15)
    assert result["aligned_chordal_distance"] < 5e-8
    assert result["minimum_matched_squared_overlap"] > 1 - 2e-15


def test_one_step_training_is_finite_unitary_and_validation_selected():
    names, starts = fixed_initializations()
    assert names == INITIALIZATION_NAMES == ("OFDM", "HAAR_1", "HAAR_2")
    train_delays, train_dopplers = sobol_dd_samples(2, seed=24006)
    val_delays, val_dopplers = sobol_dd_samples(2, seed=24007)
    result = optimize_q_waveforms(
        starts[:1], names[:1],
        atomic_operators(train_delays, train_dopplers),
        atomic_operators(val_delays, val_dopplers),
        arm="mean", steps=1, validation_cadence=1, chunk_size=2,
    )
    assert not bool(result.nonfinite_seen[0])
    identity = torch.eye(config.N, dtype=torch.complex128)
    residual = torch.linalg.matrix_norm(result.best_matrices[0].mH @ result.best_matrices[0] - identity)
    assert float(residual) < 1e-12
    assert len(result.trace_rows) == 1
