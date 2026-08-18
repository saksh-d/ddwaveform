"""Focused deterministic checks for Stage 1A.2 stream-tail refinement."""

from pathlib import Path
import math

import torch

from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.learning import hard_topk_communication_mse
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import full_information_mse
from src.tail_learning import (
    TailObjectiveSpec,
    optimize_tail_waveforms,
    per_stream_exact_mse,
    select_feasible_checkpoint,
    stream_cvar,
    tail_objective,
)
from src.waveforms import identity_waveform, random_unitary


def test_per_stream_exact_mse_averages_to_communication_loss():
    channels = sample_dd_v1(3, 14000, n=8).matrices
    waveforms = torch.stack((identity_waveform(8), random_unitary(8, 14001)))
    sigma2 = torch.tensor([0.03, 0.05, 0.08], dtype=torch.float64)
    streams = per_stream_exact_mse(waveforms, channels, sigma2, 4)
    aggregate = hard_topk_communication_mse(waveforms, channels, sigma2, 4)
    assert torch.max(torch.abs(streams.mean(dim=-1) - aggregate)) < 1e-14


def test_stream_cvar_known_vector_and_tail_count():
    values = torch.arange(1, 65, dtype=torch.float64)
    assert stream_cvar(values, 0.90) == torch.arange(58, 65, dtype=torch.float64).mean()
    assert stream_cvar(values, 0.95) == torch.arange(61, 65, dtype=torch.float64).mean()


def test_upper_tail_cvar_exceeds_mean_and_excess_is_nonnegative():
    generator = torch.Generator().manual_seed(14002)
    values = torch.rand((5, 64), dtype=torch.float64, generator=generator)
    for beta in (0.90, 0.95):
        excess = stream_cvar(values, beta) - values.mean(dim=-1)
        assert torch.all(excess >= -1e-15)


def test_selected_worst_stream_gradients_are_finite():
    waveform = torch.nn.Parameter(identity_waveform(8))
    channels = sample_dd_v1(2, 14003, n=8).matrices
    streams = per_stream_exact_mse(waveform, channels, noise_variance(15), 4)
    loss = tail_objective(streams, 0.90, 0.2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(waveform.grad.real))
    assert torch.all(torch.isfinite(waveform.grad.imag))


def test_lambda_zero_is_original_mean_mse_objective():
    channels = sample_dd_v1(2, 14004, n=8).matrices
    waveform = random_unitary(8, 14005)
    streams = per_stream_exact_mse(waveform, channels, noise_variance(18), 4)
    original = hard_topk_communication_mse(waveform, channels, noise_variance(18), 4)
    assert torch.max(torch.abs(tail_objective(streams, 0.95, 0.0) - original)) < 1e-14


def test_checkpoint_selection_obeys_mean_gate_then_cvar():
    means = torch.tensor([1.019, 1.021, 1.010, 1.015], dtype=torch.float64)
    tails = torch.tensor([0.8, 0.1, 0.7, 0.7], dtype=torch.float64)
    assert select_feasible_checkpoint(means, tails, 1.0) == 2
    assert select_feasible_checkpoint(torch.tensor([1.03]), torch.tensor([0.1]), 1.0) is None


def test_short_projected_optimization_is_unitary_and_finite():
    n = 8
    starts = torch.stack((identity_waveform(n), random_unitary(n, 14006)))
    specs = (
        TailObjectiveSpec("MEAN", 0.95, 0.0),
        TailObjectiveSpec("TAIL", 0.95, 0.2),
    )
    validation = sample_dd_v1(4, 14007, n=n).matrices
    reference = per_stream_exact_mse(starts[0], validation, noise_variance(15), 4).mean()
    result = optimize_tail_waveforms(
        starts,
        specs,
        validation,
        reference,
        steps=2,
        batch_size=2,
        validation_cadence=1,
        k=4,
        train_channel_seed=14008,
        train_snr_seed=14009,
        calibration=True,
    )
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    assert not torch.any(result.nonfinite_seen)
    assert torch.max(torch.linalg.matrix_norm(result.final_matrices.mH @ result.final_matrices - identity)) < 1e-12


def test_training_module_has_no_constellation_dependency():
    source = Path("src/tail_learning.py").read_text().lower()
    assert "qpsk" not in source
    assert "qam" not in source
    assert "constellation" not in source


def test_evaluation_constellations_remain_unit_energy():
    for constellation in (qpsk_constellation(), qam16_constellation()):
        energy = float(torch.mean(torch.abs(constellation.points) ** 2))
        assert math.isclose(energy, 1.0, rel_tol=0.0, abs_tol=1e-15)


def test_tail_refined_waveform_preserves_full_information_invariant():
    n = 8
    channels = sample_dd_v1(4, 14010, n=n).matrices
    waveform = random_unitary(n, 14011)
    values = per_stream_exact_mse(waveform, channels, noise_variance(20), n).mean(dim=-1)
    trace = full_information_mse(channels, noise_variance(20))
    assert torch.max(torch.abs(values - trace)) < 1e-12
