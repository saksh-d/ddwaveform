"""Deterministic checks for the frozen Stage 1A.1 forensic tools."""

import math

import torch

from src import config
from src.channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.diagnostics import mse_decomposition, per_stream_diagnostics
from src.metrics import (
    constellation_encode,
    nearest_neighbor_decode,
    nearest_neighbor_margin,
    qam16_constellation,
    qpsk_constellation,
    qpsk_decode,
    qpsk_encode,
    random_bits,
)
from src.receiver import effective_operator, finite_budget_receiver
from src.waveforms import identity_waveform


def test_exact_mse_decomposition_and_finite_stream_quantities():
    channel = sample_dd_v1(3, 13000, n=8).matrices
    waveform = identity_waveform(8)
    effective = effective_operator(channel, waveform)
    _, equalizer, total = finite_budget_receiver(effective, noise_variance(15), 4)
    parts = mse_decomposition(effective, equalizer, noise_variance(15))
    assert torch.max(torch.abs(total - parts["total"])) < 1e-14
    assert torch.max(torch.abs(parts["total"] - parts["diagonal"] - parts["offdiagonal"] - parts["noise"])) < 1e-14
    stream = per_stream_diagnostics(parts["b"], equalizer, noise_variance(15))
    assert all(torch.all(torch.isfinite(value)) for value in stream.values())


def test_constellations_are_unit_power_and_mapping_round_trips():
    for constellation in (qpsk_constellation(), qam16_constellation()):
        assert math.isclose(float(torch.mean(torch.abs(constellation.points) ** 2)), 1.0, abs_tol=1e-15)
        symbols, indices = constellation_encode(constellation.bit_labels, constellation)
        decoded, decoded_indices = nearest_neighbor_decode(symbols, constellation)
        assert torch.equal(decoded, constellation.bit_labels)
        assert torch.equal(decoded_indices, indices)


def test_generic_qpsk_detector_agrees_with_existing_detector():
    generator = make_generator(13001)
    estimates = torch.randn(100, dtype=torch.float64, generator=generator) + 1j * torch.randn(
        100, dtype=torch.float64, generator=generator
    )
    generic, _ = nearest_neighbor_decode(estimates.to(COMPLEX_DTYPE), qpsk_constellation())
    assert torch.equal(generic, qpsk_decode(estimates))


def test_nearest_neighbor_margin_sign_tracks_boundary_crossing():
    constellation = qpsk_constellation()
    true_bits = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    true_symbols, indices = constellation_encode(true_bits, constellation)
    estimates = torch.stack((true_symbols[0], torch.tensor(complex(-0.2, 0.7), dtype=COMPLEX_DTYPE)))
    margins = nearest_neighbor_margin(estimates, indices, constellation)
    assert margins[0] > 0
    assert margins[1] < 0


def test_analytic_empirical_mse_agree_in_deterministic_toy():
    count, blocks, n = 4, 4096, 8
    channels = sample_dd_v1(count, 13002, n=n).matrices
    waveform = identity_waveform(n)
    effective = effective_operator(channels, waveform)
    sigma2 = noise_variance(15)
    _, equalizer, analytic = finite_budget_receiver(effective, sigma2, 4)
    generator = make_generator(13003)
    bits = random_bits((count, blocks, n, 2), generator)
    symbols = qpsk_encode(bits)
    noise = complex_awgn((count, blocks, n), sigma2, generator)
    output = torch.einsum("cij,cbj->cbi", effective, symbols) + noise
    estimate = torch.einsum("cij,cbj->cbi", equalizer, output)
    empirical = torch.mean(torch.abs(estimate - symbols) ** 2, dim=(1, 2))
    assert torch.max(torch.abs(empirical - analytic) / analytic) < 0.04


def test_frozen_checkpoint_not_mutated_and_matches_training_payload():
    path = "outputs/stage1a/checkpoints/haar_2_best.pt"
    frozen = torch.load(path, weights_only=True)
    before = frozen.clone()
    payload = torch.load("outputs/stage1a/checkpoints/training.pt", weights_only=False)
    result = payload["result"]
    index = result.initialization_names.index("HAAR_2")
    assert torch.equal(frozen, result.best_matrices[index])
    channels = sample_dd_v1(2, 13004).matrices
    _ = finite_budget_receiver(effective_operator(channels, frozen), noise_variance(12), 24)
    assert torch.equal(frozen, before)


def test_forensic_seeds_are_independent_from_prior_and_final_seeds():
    forensic = {
        config.STAGE1A1_CHANNEL_SEED,
        config.STAGE1A1_SYMBOL_SEED,
        config.STAGE1A1_NOISE_SEED,
        config.STAGE1A1_BOOTSTRAP_SEED,
    }
    prior = {
        config.VALIDATION_SEED,
        config.STAGE0_EVAL_SEED,
        config.TEST_SEED,
        config.STAGE1A_EVAL_SEED,
        config.STAGE1A_TRAIN_CHANNEL_SEED,
        config.STAGE1A_TRAIN_SNR_SEED,
        config.STAGE1A_BOOTSTRAP_SEED,
        config.STAGE1A_BER_SEED,
    }
    assert len(forensic) == 4
    assert forensic.isdisjoint(prior)
