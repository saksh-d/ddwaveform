"""Focused exact and controlled statistical checks for Stage 3A."""

import numpy as np
import torch

from src.receiver import effective_operator
from src.stage2d import ambiguity_coupling, score_support_mask, sobol_dd_samples
from src.stage3a import (
    concentration_curve,
    concentration_thresholds,
    fixed_mask_expected_retained,
    q_concentration,
    relative_frobenius_error,
    sample_raw_dd,
)
from src.waveforms import identity_waveform, random_unitary


def test_raw_sampler_applies_exact_project_normalization():
    batch = sample_raw_dd(12, 23001, (1 / 3, 1 / 3, 1 / 3), n=8)
    reconstructed = batch.alpha[:, None, None] * batch.raw
    assert torch.allclose(batch.normalized, reconstructed, atol=0, rtol=0)
    assert torch.allclose(
        torch.sum(torch.abs(batch.normalized) ** 2, dim=(-2, -1)) / 8,
        torch.ones(12, dtype=torch.float64),
        atol=2e-13,
    )


def test_q_concentration_definitions_and_thresholds_are_exact():
    q = np.diag([1.0, 2.0, 3.0, 4.0])
    metrics = q_concentration(q, 1)
    assert metrics["eta_q"] == 1.0
    assert np.all(metrics["row_fraction"] == 1.0)
    assert np.all(metrics["effective_support"] == 1.0)
    assert np.allclose(concentration_curve(q), 1.0)
    assert concentration_thresholds(q) == {0.90: 1, 0.95: 1, 0.99: 1, 0.995: 1}


def test_normalized_fixed_support_expectation_equals_masked_c_mass():
    waveform = random_unitary(8, 23002)
    channels = sample_raw_dd(128, 23003, (1 / 3, 1 / 3, 1 / 3), n=8).normalized
    effective = effective_operator(channels, waveform)
    c = torch.mean(torch.abs(effective) ** 2, dim=0).numpy()
    mask = score_support_mask(c, 3).numpy()
    realized = torch.sum(torch.abs(effective[:, mask]) ** 2, dim=1) / 8
    assert abs(float(torch.mean(realized)) - fixed_mask_expected_retained(c, mask)) < 2e-15
    assert abs(float(np.sum(c)) - 8.0) < 2e-13


def test_unnormalized_equal_power_q_law_holds_statistically():
    n = 8
    waveform = random_unitary(n, 23004)
    delays, dopplers = sobol_dd_samples(2048, seed=23005)
    q = ambiguity_coupling(waveform, delays, dopplers, chunk=64)
    batch = sample_raw_dd(4096, 23006, (1 / 3, 1 / 3, 1 / 3), n=n)
    c_raw = torch.mean(torch.abs(effective_operator(batch.raw, waveform)) ** 2, dim=0).numpy()
    assert relative_frobenius_error(c_raw, q) < 0.07


def test_unequal_path_power_law_has_sum_variance_scaling():
    n = 8
    waveform = identity_waveform(n)
    delays, dopplers = sobol_dd_samples(2048, seed=23007)
    q = ambiguity_coupling(waveform, delays, dopplers, chunk=64)
    powers = (0.9, 0.4, 0.2)
    batch = sample_raw_dd(4096, 23008, powers, n=n)
    c_raw = torch.mean(torch.abs(effective_operator(batch.raw, waveform)) ** 2, dim=0).numpy()
    assert relative_frobenius_error(c_raw, sum(powers) * q) < 0.07
