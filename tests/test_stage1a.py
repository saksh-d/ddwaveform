"""Focused deterministic checks for universal Stage 1A learning."""

import torch

from src import config
from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.learning import (
    hard_topk_communication_mse,
    optimize_universal_waveforms,
    stage1a_initializations,
    validation_objective,
)
from src.oracle import polar_project
from src.receiver import full_information_mse
from src.waveforms import identity_waveform, random_unitary


def _small_run(seed: int = 12000):
    n = 8
    starts = torch.stack((torch.eye(n, dtype=COMPLEX_DTYPE), random_unitary(n, seed)))
    validation = sample_dd_v1(4, seed + 1, n=n).matrices
    return optimize_universal_waveforms(
        starts,
        ("IDENTITY", "HAAR"),
        validation,
        steps=2,
        batch_size=2,
        validation_cadence=1,
        k=4,
        train_channel_seed=seed + 2,
        train_snr_seed=seed + 3,
    )


def test_named_initializations_begin_unitary():
    _, matrices = stage1a_initializations()
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    assert torch.max(torch.linalg.matrix_norm(matrices.mH @ matrices - identity)) < 1e-12


def test_polar_projection_restores_unitarity():
    generator = torch.Generator().manual_seed(12001)
    raw = torch.randn((3, 8, 8), dtype=torch.float64, generator=generator) + 1j * torch.randn(
        (3, 8, 8), dtype=torch.float64, generator=generator
    )
    projected = polar_project(raw.to(COMPLEX_DTYPE))
    identity = torch.eye(8, dtype=COMPLEX_DTYPE)
    assert torch.max(torch.linalg.matrix_norm(projected.mH @ projected - identity)) < 1e-12


def test_short_training_is_finite_unitary_and_reproducible():
    first = _small_run()
    second = _small_run()
    assert not torch.any(first.nonfinite_seen)
    assert torch.all(torch.isfinite(first.best_validation))
    assert torch.max(first.max_unitarity_residuals) < 1e-12
    assert torch.equal(first.best_steps, second.best_steps)
    assert torch.allclose(first.best_validation, second.best_validation, atol=0, rtol=0)
    assert torch.allclose(first.best_matrices, second.best_matrices, atol=0, rtol=0)


def test_training_loss_and_gradient_are_finite():
    waveform = torch.nn.Parameter(identity_waveform(8))
    channels = sample_dd_v1(2, 12010, n=8).matrices
    loss = hard_topk_communication_mse(waveform, channels, noise_variance(15), 4).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(waveform.grad.real))
    assert torch.all(torch.isfinite(waveform.grad.imag))


def test_best_validation_checkpoint_is_retained():
    result = _small_run(12020)
    for index, name in enumerate(result.initialization_names):
        trace = torch.tensor(result.validation_traces[name], dtype=torch.float64)
        best_index = int(torch.argmin(trace))
        assert result.best_steps[index] == result.trace_steps[best_index]
        assert result.best_validation[index] == trace[best_index]


def test_evaluation_does_not_mutate_waveform():
    waveform = random_unitary(8, 12030)
    before = waveform.clone()
    channels = sample_dd_v1(3, 12031, n=8).matrices
    _ = validation_objective(waveform, channels, k=4, snrs_db=(12, 20), channel_chunk_size=2)
    assert torch.equal(waveform, before)


def test_learned_full_information_aggregate_invariant():
    result = _small_run(12040)
    channels = sample_dd_v1(4, 12050, n=8).matrices
    sigma2 = noise_variance(18)
    learned = hard_topk_communication_mse(result.best_matrices[0], channels, sigma2, 8)
    trace = full_information_mse(channels, sigma2)
    assert torch.max(torch.abs(learned - trace)) < 1e-12


def test_stage1a_evaluation_seed_is_independent():
    forbidden = {config.VALIDATION_SEED, config.TEST_SEED, config.STAGE0_EVAL_SEED}
    assert config.STAGE1A_EVAL_SEED not in forbidden
