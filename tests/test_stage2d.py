"""Focused checks for Stage 2D fixed-support diagnostics."""

import numpy as np
import torch

from src.channel import sample_dd_v1
from src.stage2d import (
    ambiguity_coupling,
    apply_support,
    centroid_neighborhood_mask,
    empirical_coupling,
    matrix_agreement,
    score_support_mask,
    sobol_dd_samples,
    support_metrics,
)
from src.waveforms import identity_waveform, random_unitary


def test_fixed_support_masks_are_deterministic_and_have_exact_row_budget():
    scores = np.tile(np.arange(8, dtype=float), (8, 1))
    mask = score_support_mask(scores, 3)
    assert torch.all(mask.sum(dim=-1) == 3)
    assert torch.equal(mask, score_support_mask(scores, 3))
    neighborhood = centroid_neighborhood_mask(random_unitary(8, 22000), 4)
    assert torch.all(neighborhood.sum(dim=-1) == 4)


def test_identity_single_path_ambiguity_matches_empirical_path_coupling():
    delays, dopplers = sobol_dd_samples(16, seed=22001)
    waveform = identity_waveform(8)
    q = ambiguity_coupling(waveform, delays, dopplers, chunk=8)
    from src.channel import assemble_dd_channel
    channels = assemble_dd_channel(delays[:, None], dopplers[:, None], torch.ones((16, 1), dtype=torch.complex128), n=8)
    c = empirical_coupling(channels, {"I": waveform}, chunk=8)["I"]
    assert np.allclose(q, c, atol=1e-13)


def test_matrix_agreement_is_exact_for_positive_scaling():
    rng = np.random.default_rng(22002)
    matrix = rng.random((8, 8))
    metrics = matrix_agreement(matrix, 3.0 * matrix, 3)
    assert metrics["spearman_offdiagonal"] == 1.0
    assert metrics["topk_overlap_mean"] == 1.0
    assert metrics["scaled_frobenius_discrepancy"] < 1e-14


def test_predicted_support_uses_actual_coefficients_and_receiver_metrics_are_finite():
    channel = sample_dd_v1(3, 22003, n=8).matrices
    mask = score_support_mask(np.eye(8), 4)
    truncated = apply_support(channel, mask)
    assert torch.equal(truncated[mask.expand_as(channel)], channel[mask.expand_as(channel)])
    assert torch.count_nonzero(truncated[~mask.expand_as(channel)]) == 0
    metrics = support_metrics(channel, mask, 4, 0.1)
    assert torch.isfinite(metrics["analytic_mse"]).all()
    assert torch.isfinite(metrics["stream_cvar_095"]).all()
    assert torch.all((metrics["support_recall"] >= 0) & (metrics["support_recall"] <= 1))
