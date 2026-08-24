"""Focused Stage 2E audit and frozen-forensic checks."""

import numpy as np
import torch

from src import config
from src.stage2e import (
    AFDM_SEARCH_MULTIPLIERS,
    REGIMES,
    afdm_design_record,
    afdm_doppler_components,
    afdm_validation_rows,
    ambiguity_surface,
    atom_spectral_time_metrics,
    fixed_support_metrics,
    papr_samples,
    q_masks_for_regime,
    sample_regime,
    select_afdm_multiplier,
    simple_pairwise_explanation,
    stage2e_waveforms,
)
from src.waveforms import identity_waveform, ofdm_waveform


def test_afdm_doppler_decomposition_and_support_rule_are_derived():
    alpha, fractional = afdm_doppler_components(np.asarray([-0.75, -0.5, 0.49, 0.75]))
    assert alpha.tolist() == [-1, 0, 0, 1]
    assert np.all(fractional >= -0.5) and np.all(fractional < 0.5)
    record = afdm_design_record()
    assert record["alpha_max"] == 1
    assert record["integer_multiplier_2Nc1"] == 3
    assert record["guarded_multiplier_2Nc1"] == 5
    assert record["feasibility_left_side"] < config.N


def test_regime_samplers_are_reproducible_normalized_and_distinct():
    for regime in REGIMES:
        first = sample_regime(regime, 3, 12100, n=16)
        second = sample_regime(regime, 3, 12100, n=16)
        assert torch.equal(first.matrices, second.matrices)
        assert first.matrices.shape == (3, 16, 16)
        assert torch.allclose(
            torch.sum(torch.abs(first.matrices) ** 2, dim=(-2, -1)) / 16,
            torch.ones(3, dtype=torch.float64),
        )
        assert first.delays.shape[-1] == regime.paths


def test_validation_sweep_selection_is_validation_objective_only():
    channels = sample_regime(REGIMES[3], 4, 12101, n=config.N).matrices
    rows = afdm_validation_rows(channels, (15,), multipliers=(1, 3, 5), k=24)
    selected, summary = select_afdm_multiplier(rows)
    assert selected in (1, 3, 5)
    assert len(summary) == 3
    assert all("mean_analytic_mse" in row for row in summary)
    assert tuple(range(1, 12)) == AFDM_SEARCH_MULTIPLIERS


def test_q_masks_have_exact_budget_and_are_waveform_specific():
    waveforms = {"I": identity_waveform(), "F": ofdm_waveform()}
    masks, q = q_masks_for_regime(waveforms, REGIMES[3], 12, 32, 12102)
    assert set(masks) == set(q) == {"I", "F"}
    assert torch.all(masks["I"].sum(dim=-1) == 12)
    assert not np.allclose(q["I"], q["F"])


def test_spectral_metrics_ambiguity_and_papr_are_well_formed():
    waveform = ofdm_waveform(n=16)
    metrics, frequencies, spectra = atom_spectral_time_metrics(waveform, fft_size=256)
    assert len(frequencies) == 256 and spectra.shape == (256, 16)
    assert np.all((metrics["spectral_entropy"] >= 0) & (metrics["spectral_entropy"] <= 1))
    delays = np.asarray([0.0, 1.0])
    dopplers = np.asarray([-1.0, 0.0, 1.0])
    surface = ambiguity_surface(waveform[:, 0], waveform[:, 0], delays, dopplers)
    assert surface.shape == (3, 2) and np.isclose(surface[1, 0], 1.0)
    symbols = torch.ones((4, 16), dtype=torch.complex128)
    papr = papr_samples(waveform, symbols, oversample=4)
    assert papr.shape == (4,) and np.all(np.isfinite(papr))


def test_pairwise_explanation_is_deterministic_and_bounded():
    q = np.asarray([[0.0, 1.0, 0.2], [1.0, 0.0, 0.4], [0.2, 0.4, 0.0]])
    feature = np.asarray([[0.0, 1.0, 0.1], [1.0, 0.0, 0.5], [0.1, 0.5, 0.0]])
    result = simple_pairwise_explanation(q, {"feature": feature})
    assert set(result) == {"spearman", "standardized_coefficients", "linear_r2"}
    assert result["linear_r2"] <= 1.0 + 1e-12


def test_stage2e_waveform_set_uses_requested_afdm_and_frozen_learned_tensor():
    learned = identity_waveform()
    waveforms = stage2e_waveforms(learned, 5)
    assert tuple(waveforms) == ("IDENTITY", "OFDM", "OTFS", "AFDM", "Learned")
    assert waveforms["Learned"].data_ptr() == learned.data_ptr()
    assert all(value.shape == (config.N, config.N) for value in waveforms.values())


def test_limiting_distortion_is_finite_for_rank_deficient_fixed_support():
    channels = torch.eye(config.N, dtype=torch.complex128)[None]
    mask = torch.zeros((config.N, config.N), dtype=torch.bool)
    mask[:, :24] = True
    result = fixed_support_metrics(channels, identity_waveform(), mask, 24, 18)
    assert torch.all(torch.isfinite(result["limiting_distortion"]))
