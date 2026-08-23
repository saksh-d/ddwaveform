import numpy as np
import pytest
import torch

from src.channel import COMPLEX_DTYPE, dft_matrix
from src.stage2d import sobol_dd_samples
from src.stage3b import atomic_operators, fixed_initializations
from src.stage4c import (
    canonicalize_target,
    cascade_parameter_count,
    chirp_cascade,
    fit_cascade_batch,
    matrix_objective,
    operator_fidelity,
    parameter_population_metrics,
    restore_native_order,
)
from src.structural_fitting import polynomial_fourier


@pytest.mark.parametrize("depth,count", [(1, 6), (2, 9), (3, 12), (4, 15)])
def test_cascade_parameter_count_and_unitarity(depth: int, count: int):
    parameters = torch.linspace(-0.2, 0.3, count, dtype=torch.float64)
    waveform = chirp_cascade(parameters, depth, 16)
    identity = torch.eye(16, dtype=COMPLEX_DTYPE)
    assert cascade_parameter_count(depth) == count
    assert torch.linalg.matrix_norm(waveform.mH @ waveform - identity) < 1e-12


def test_depth_one_equals_stage4a_cubic_family_after_constant_removal():
    identifiable = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64)
    stage4a_parameters = torch.tensor(
        [1.7, *identifiable[:3], -0.8, *identifiable[3:]], dtype=torch.float64
    )
    expected = polynomial_fourier(stage4a_parameters, 3, 16)
    observed = chirp_cascade(identifiable, 1, 16)
    phase = torch.exp(torch.tensor(0.9j, dtype=COMPLEX_DTYPE))
    assert torch.linalg.matrix_norm(expected - phase * observed) < 1e-12


def test_fixed_alignment_round_trip_and_matrix_objective():
    _, starts = fixed_initializations()
    target = starts[1]
    canonical, permutation, phases = canonicalize_target(target, dft_matrix(64).mH)
    restored = restore_native_order(canonical, permutation, phases)
    assert torch.linalg.matrix_norm(restored - target) < 1e-12
    assert matrix_objective(canonical, canonical) < 1e-14


def test_arm_b_identity_has_zero_heldout_discrepancy():
    delays, dopplers = sobol_dd_samples(8, 11220)
    operators = atomic_operators(delays, dopplers, n=16)
    target = chirp_cascade(torch.zeros(6), 1, 16)
    metrics, per_atom = operator_fidelity(target, target, operators)
    assert metrics["d_b"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["complex_coefficient_coherence"] == pytest.approx(1.0, abs=1e-14)
    assert np.max(per_atom) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("arm", ["M", "B"])
def test_fitting_is_validation_only_and_finite(arm: str):
    target = chirp_cascade(torch.linspace(-0.1, 0.1, 6), 1, 16).unsqueeze(0)
    delays_train, dopplers_train = sobol_dd_samples(4, 11221)
    delays_val, dopplers_val = sobol_dd_samples(4, 11222)
    result = fit_cascade_batch(
        target,
        depth=1,
        arm=arm,
        restart=0,
        steps=3,
        seed=11223,
        training_operators=atomic_operators(delays_train, dopplers_train, n=16),
        validation_operators=atomic_operators(delays_val, dopplers_val, n=16),
        validation_cadence=1,
        chunk_size=2,
    )
    assert torch.isfinite(result.validation_loss).all()
    assert not result.nonfinite_seen.any()
    assert len(result.history) == 3
    assert all(row["validation_loss"] >= 0 for row in result.history)


def test_parameter_population_summary_removes_no_hidden_symmetry():
    values = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
    metrics = parameter_population_metrics(values)
    assert metrics["mean_parameter_vector"] == pytest.approx([1.0, 3.0])
    assert metrics["pairwise_rms_distance_maximum"] > metrics["pairwise_rms_distance_minimum"]
