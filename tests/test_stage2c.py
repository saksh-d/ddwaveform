"""Focused deterministic Stage 2C checks."""

import math
from pathlib import Path

import numpy as np
import torch

from src import config
from src.channel import sample_dd_v1
from src.stage2a import FROZEN_CHECKPOINT_SHA256, file_sha256
from src.stage2c import (
    HAAR_SEEDS,
    INITIALIZATION_NAMES,
    TAIL_BETA,
    TAIL_LAMBDA,
    optimize_budget_waveforms,
    select_validation_run,
    stage2c_initializations,
    structural_diagnostics,
)


def test_stage2c_raw_starts_are_fixed_unitary_and_do_not_use_checkpoint():
    names, first = stage2c_initializations()
    _, second = stage2c_initializations()
    assert names == INITIALIZATION_NAMES == ("OFDM", "HAAR_1", "HAAR_2")
    assert HAAR_SEEDS == (config.STAGE1A_HAAR_SEED, config.STAGE1A_HAAR_SEED + 1)
    assert torch.equal(first, second)
    identity = torch.eye(config.N, dtype=torch.complex128)
    assert float(torch.linalg.matrix_norm(first.mH @ first - identity).max()) < 1e-12


def test_stage2c_loss_and_predictions_are_preregistered_only():
    assert TAIL_BETA == 0.95
    assert TAIL_LAMBDA == 0.40
    assert {k: math.ceil(config.N / k) for k in (16, 24, 32)} == {16: 4, 24: 3, 32: 2}


def test_stage2c_one_step_smoke_is_finite_unitary_and_validation_selected():
    _, starts = stage2c_initializations()
    validation = sample_dd_v1(4, 16001).matrices
    result = optimize_budget_waveforms(
        starts[:1], validation, k=16, steps=1, batch_size=2, validation_cadence=1,
        train_channel_seed=16002, train_snr_seed=16003,
    )
    selected = select_validation_run(result)
    assert selected == 0
    assert not bool(result.nonfinite_seen[0])
    identity = torch.eye(config.N, dtype=torch.complex128)
    assert float(torch.linalg.matrix_norm(result.best_matrices[0].mH @ result.best_matrices[0] - identity)) < 1e-12


def test_structural_metrics_include_unimposed_group_prediction_and_support_partition():
    waveform = stage2c_initializations()[1][0]
    channels = sample_dd_v1(8, 16004).matrices
    energy, support, labels, summary = structural_diagnostics(waveform, channels, 16)
    assert energy.shape == support.shape == (config.N, config.N)
    assert labels.shape == (config.N,)
    assert summary["predicted_group_count"] == 4
    assert 2 <= summary["group_count"] <= 8
    assert np.isclose(support.sum(), config.N * 16)
    assert 0 <= summary["within_topk_support_fraction"] <= 1


def test_authoritative_stage1a2_checkpoint_hash_is_unchanged():
    path = Path("outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt")
    assert file_sha256(path) == FROZEN_CHECKPOINT_SHA256
