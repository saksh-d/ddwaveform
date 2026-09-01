from pathlib import Path
import inspect

import numpy as np
import torch

from src import config
from src.pub1a.core import CHECKPOINT_SHA256, REGIME_BY_KEY, waveform_set
from src.pub1c.core import (
    EPSILONS,
    K_VALUES,
    balanced_symbol_indices,
    connected_components,
    controlled_fractionality_channels,
    graph_diagnostics,
    median_full_information_channel,
    paired_bootstrap_difference,
    retained_energy_rows,
)
from scripts.stage_pub1c import SCALING_RHO_VALUES, scaling_ks
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2e import sample_regime


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
TRAINED_HASHES = {
    16: "a91d4fb2066d35a7b8e13ffbf869801544e63e8328a4318b04f5dab7ad49a115",
    24: "73a401b6034c04863628684ae20be9a90c3ee866e1204445c205ab70027a8fa6",
    32: "d8a04b5ce658e488021ef3c2f007556cea094a92320c31b4b7d05d0e23cd4e90",
}


def test_frozen_inputs_have_exact_hashes():
    assert file_sha256(FROZEN) == CHECKPOINT_SHA256
    for k, digest in TRAINED_HASHES.items():
        assert file_sha256(ROOT / f"outputs/stage2c/checkpoints/k{k}_ofdm_best.pt") == digest


def test_controlled_fractionality_is_exact_paired_and_reproducible():
    for axis in ("delay", "doppler", "joint"):
        zero = controlled_fractionality_channels(4, 14000, 0.0, axis)
        half = controlled_fractionality_channels(4, 14000, 0.5, axis)
        repeat = controlled_fractionality_channels(4, 14000, 0.5, axis)
        assert torch.equal(half.matrices, repeat.matrices)
        assert torch.equal(zero.pre_normalization_gains, half.pre_normalization_gains)
        delay_offset = torch.abs(half.delays - zero.delays)
        doppler_offset = torch.abs(half.dopplers - zero.dopplers)
        assert torch.allclose(delay_offset, torch.full_like(delay_offset, .5 if axis in ("delay", "joint") else 0.0))
        assert torch.allclose(doppler_offset, torch.full_like(doppler_offset, .5 if axis in ("doppler", "joint") else 0.0))
        assert float(half.delays.min()) >= 0 and float(half.delays.max()) <= 4
        assert float(half.dopplers.min()) >= -1 and float(half.dopplers.max()) <= 1


def test_oracle_retained_energy_is_monotone_and_full_at_rho_one():
    learned, _ = load_frozen_waveform(FROZEN)
    channels = controlled_fractionality_channels(3, 14001, EPSILONS[-1], "joint").matrices
    rows = retained_energy_rows(channels, {k: v for k, v in waveform_set(learned, 3).items() if k != "OFDM"})
    for waveform in ("Learned", "OTFS", "AFDM"):
        for channel in range(3):
            values = [row["retained_energy"] for row in rows if row["waveform"] == waveform and row["channel_index"] == channel]
            assert np.all(np.diff(values) >= -1e-14)
            assert abs(values[-1] - 1.0) < 1e-14


def test_graph_thresholds_use_one_map_and_nested_supports():
    generator = np.random.default_rng(14002)
    energy = generator.random((config.N, config.N))
    graphs, rows = graph_diagnostics(energy)
    assert [row["k"] for row in rows] == list(K_VALUES)
    previous = None
    for k in K_VALUES:
        current = graphs[f"k{k}"] > 0
        if previous is not None:
            assert np.all(previous <= current)
        previous = current
    assert len(np.unique(connected_components(np.eye(4)))) == 4


def test_p4_rho_grid_scales_to_exact_integer_row_budgets():
    assert scaling_ks(64) == (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
    assert scaling_ks(128) == (16, 24, 32, 40, 48, 56, 64, 80, 96, 128)
    assert scaling_ks(256) == (32, 48, 64, 80, 96, 112, 128, 160, 192, 256)
    assert tuple(k / 128 for k in scaling_ks(128)) == SCALING_RHO_VALUES
    assert tuple(k / 256 for k in scaling_ks(256)) == SCALING_RHO_VALUES


def test_pub1c_core_has_no_training_entrypoint():
    source = (ROOT / "src/pub1c/core.py").read_text()
    assert "torch.optim" not in source
    assert "backward(" not in source


def test_paired_bootstrap_difference_is_deterministic_and_directional():
    left = np.asarray([0.1, 0.2, 0.3, 0.4])
    right = left + 0.5
    first = paired_bootstrap_difference(left, right, resamples=1000, seed=14003)
    second = paired_bootstrap_difference(left, right, resamples=1000, seed=14003)
    assert first == second
    assert first["ci95_high"] < 0


def test_median_selector_is_waveform_independent_and_deterministic():
    channels = controlled_fractionality_channels(17, 14004, .5, "joint").matrices
    first_index, first_values = median_full_information_channel(channels, 15)
    second_index, second_values = median_full_information_channel(channels, 15)
    assert first_index == second_index
    assert np.array_equal(first_values, second_values)
    assert first_index == int(np.argmin(np.abs(first_values - np.median(first_values))))


def test_balanced_symbol_indices_are_exact_and_stable():
    labels = torch.arange(16).repeat_interleave(205)
    first = balanced_symbol_indices(labels, symbol_count=16, per_symbol=200)
    second = balanced_symbol_indices(labels, symbol_count=16, per_symbol=200)
    assert np.array_equal(first, second)
    assert len(first) == 3200
    assert np.array_equal(np.bincount(labels.numpy()[first], minlength=16), np.full(16, 200))


def test_named_constellation_regimes_keep_legacy_path_laws():
    grid = sample_regime(REGIME_BY_KEY["grid_dd"].stage2e(), 8, 14005)
    fractional = sample_regime(REGIME_BY_KEY["fractional_doppler"].stage2e(), 8, 14006)
    offgrid = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 8, 14007)
    assert torch.equal(grid.delays, torch.round(grid.delays))
    assert torch.equal(grid.dopplers, torch.round(grid.dopplers))
    assert torch.equal(fractional.delays, torch.round(fractional.delays))
    assert torch.any(fractional.dopplers != torch.round(fractional.dopplers))
    assert torch.any(offgrid.delays != torch.round(offgrid.delays))
    assert torch.any(offgrid.dopplers != torch.round(offgrid.dopplers))


def test_15db_phase_constructs_one_shared_symbol_and_noise_stream_without_alignment():
    from scripts.stage_pub1c import phase_constellations_15db
    source = inspect.getsource(phase_constellations_15db)
    assert source.count("bits = random_bits") == 1
    assert source.count("noise = complex_awgn") == 1
    assert "torch.angle" not in source
    assert "estimate /" not in source
    assert "estimate *" not in source


def test_contrast_phase_uses_physical_selection_and_paired_integer_control():
    from scripts.stage_pub1c import phase_constellations_contrast
    source = inspect.getsource(phase_constellations_contrast)
    assert "median_full_information_channel(offgrid.matrices, 18)" in source
    assert '("integer_dd", integer.matrices[selected_index])' in source
    assert '("offgrid_dd", offgrid.matrices[selected_index])' in source
    assert source.count("bits = random_bits") == 1
    assert source.count("noise = complex_awgn") == 1
    assert '"selection_uses_ber_or_evm": False' in source
    assert "torch.angle" not in source
