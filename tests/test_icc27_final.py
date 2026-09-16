from __future__ import annotations

import json
from pathlib import Path

import torch

from src.icc27_final import (
    fixed_energy_support,
    receiver_state,
    sample_channels,
    semantic_seed,
    summarize,
)
from src.waveforms import ofdm_waveform


ROOT = Path(__file__).resolve().parents[1]


def test_semantic_seeds_are_stable_and_role_separated():
    assert semantic_seed("channels", condition="nominal") == semantic_seed("channels", condition="nominal")
    assert semantic_seed("channels", condition="nominal") != semantic_seed("support", condition="nominal")
    assert semantic_seed("channels", condition="nominal") != semantic_seed("channels", condition="wider_delay")


def test_channel_definitions_and_normalization():
    cases = {
        "nominal": (4.0, 0.75, 3),
        "wider_delay": (8.0, 0.75, 3),
        "wider_doppler": (4.0, 1.5, 3),
        "more_paths": (4.0, 0.75, 5),
    }
    for index, (condition, (delay_max, doppler_max, paths)) in enumerate(cases.items()):
        batch = sample_channels(condition, 7, 9000 + index)
        assert batch.delays.shape == (7, paths)
        assert batch.dopplers.shape == (7, paths)
        assert torch.all((batch.delays >= 0) & (batch.delays <= delay_max))
        assert torch.all((batch.dopplers >= -doppler_max) & (batch.dopplers <= doppler_max))
        energy = torch.sum(torch.abs(batch.matrices) ** 2, dim=(-2, -1)) / 64
        assert torch.allclose(energy, torch.ones_like(energy), atol=1e-12, rtol=0)


def test_integer_grid_is_exact_and_nontrivial():
    batch = sample_channels("integer_grid", 32, 9010)
    assert torch.all(batch.delays == torch.round(batch.delays))
    assert torch.all(batch.dopplers == torch.round(batch.dopplers))
    assert set(batch.dopplers.flatten().tolist()) == {-1.0, 0.0, 1.0}


def test_fixed_support_has_exact_row_budget_and_full_receiver_is_exact():
    waveform = ofdm_waveform()
    support_bank = sample_channels("nominal", 8, 9020).matrices
    test_bank = sample_channels("nominal", 3, 9021).matrices
    support = fixed_energy_support(support_bank, waveform, 24)
    assert torch.all(support.sum(dim=-1) == 24)
    state = receiver_state(test_bank, waveform, "full_k64", 24, 18, support)
    assert torch.allclose(state["retained_energy"], torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.all(torch.isfinite(state["analytic_mse"]))


def test_summary_preserves_raw_counts():
    rows = [
        {"waveform": "A", "bit_errors": 1, "bits": 10, "retained_energy": 0.5, "analytic_mse": 0.2},
        {"waveform": "A", "bit_errors": 2, "bits": 10, "retained_energy": 0.7, "analytic_mse": 0.4},
    ]
    result = summarize(rows, ("waveform",))[0]
    assert result["bit_errors"] == 3
    assert result["bits"] == 20
    assert result["ber"] == 0.15
    assert result["mean_retained_energy"] == 0.6
    assert result["mean_analytic_mse"] == 0.30000000000000004


def test_frozen_config_has_requested_scope():
    config = json.loads((ROOT / "configs/icc27_final_n64_confirm.json").read_text())
    assert config["namespace"] == "icc27-final-v1"
    assert (config["n"], config["k"], config["snr_db"]) == (64, 24, 18)
    assert config["banks"]["nominal_channels"] == 512
    assert config["banks"]["stress_channels_per_condition"] == 256
    assert config["receiver"]["support_recalibration"] is False
    assert config["receiver"]["test_ber_tuning"] is False
