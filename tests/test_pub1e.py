import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import config
from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.metrics import qpsk_constellation
from src.pub1e.core import (
    ResidualStatistics,
    controlled_fractionality_2d,
    fixed_energy_support,
    historical_equalizer,
    receiver_equalizer,
    residual_aware_equalizer,
    residual_statistics,
    safe_log10_ber_ratio,
    select_regularization,
    shifted_prior_channels,
)
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform
from src.waveforms import ofdm_waveform, otfs_waveform
from scripts.stage_pub1e import DATA, _select_series, read_csv


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
EXPECTED_HASH = "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc"


def test_protocol_is_frozen_and_base_bank_seed_labels_are_unique():
    protocol = json.loads((ROOT / "configs/stage_pub1e.json").read_text())
    assert protocol["status"] == "frozen_before_production"
    seeds = [record["seed"] for record in protocol["banks"].values() if "seed" in record]
    assert len(seeds) == len(set(seeds))
    assert protocol["selection"]["test_ber_checkpoint_selection"] is False


def test_authoritative_checkpoint_hash_and_unitarity():
    assert file_sha256(FROZEN) == EXPECTED_HASH
    waveform, record = load_frozen_waveform(FROZEN)
    assert record["unitarity_residual"] < 1e-12
    assert waveform.shape == (64, 64)


def test_controlled_2d_pairing_and_exact_offsets():
    integer = controlled_fractionality_2d(4, 18100, 0.0, 0.0)
    joint = controlled_fractionality_2d(4, 18100, 0.3, 0.4)
    assert torch.equal(integer.pre_normalization_gains, joint.pre_normalization_gains)
    assert torch.allclose(torch.abs(joint.delays - integer.delays), torch.full((4, 3), 0.3, dtype=torch.float64))
    assert torch.allclose(torch.abs(joint.dopplers - integer.dopplers), torch.full((4, 3), 0.4, dtype=torch.float64))
    assert torch.allclose(torch.sum(torch.abs(joint.matrices) ** 2, dim=(-2, -1)) / 64, torch.ones(4, dtype=torch.float64))


def test_fixed_support_is_waveform_specific_and_exact_k():
    bank = sample_dd_v1(8, 18101).matrices
    supports = [fixed_energy_support(bank, waveform, 7) for waveform in (ofdm_waveform(), otfs_waveform())]
    assert all(torch.all(torch.sum(mask, dim=-1) == 7) for mask in supports)
    assert not torch.equal(*supports)


def test_residual_covariance_is_hermitian_psd():
    bank = sample_dd_v1(12, 18102).matrices
    stats = residual_statistics(bank, ofdm_waveform(), 8)
    assert torch.allclose(stats.receive_covariance, stats.receive_covariance.mH, atol=1e-12)
    assert float(torch.min(torch.linalg.eigvalsh(stats.receive_covariance))) > -1e-12


def test_receiver_equations_match_direct_reference():
    generator = torch.Generator().manual_seed(18103)
    a = torch.randn((2, 5, 5), dtype=torch.float64, generator=generator).to(COMPLEX_DTYPE)
    restricted = rowwise_topk(a, 3)
    sigma2 = noise_variance(18)
    got = historical_equalizer(restricted, sigma2, 2.0)
    identity = torch.eye(5, dtype=COMPLEX_DTYPE)
    expected = torch.linalg.solve(restricted.mH @ restricted + 2.0 * sigma2 * identity, restricted.mH)
    assert torch.allclose(got, expected, atol=1e-12)
    mean = torch.full((5, 5), 0.01 + 0.02j, dtype=COMPLEX_DTYPE)
    covariance = torch.diag(torch.linspace(.01, .05, 5)).to(COMPLEX_DTYPE)
    stats = ResidualStatistics(mean, covariance, 100)
    b = restricted + mean
    expected_residual = torch.linalg.solve(b @ b.mH + covariance + sigma2 * identity, b).mH
    assert torch.allclose(residual_aware_equalizer(restricted, sigma2, stats), expected_residual, atol=1e-12)


def test_k_equals_n_receiver_consistency():
    bank = sample_dd_v1(4, 18104).matrices
    waveform = ofdm_waveform()
    actual = effective_operator(bank, waveform)
    stats = residual_statistics(bank, waveform, config.N)
    _, historical = receiver_equalizer(actual, config.N, noise_variance(18), "historical")
    _, residual = receiver_equalizer(actual, config.N, noise_variance(18), "residual_aware", stats=stats)
    assert torch.max(torch.abs(stats.mean)) < 1e-15
    assert torch.max(torch.abs(stats.receive_covariance)) < 1e-25
    assert torch.allclose(historical, residual, atol=1e-11)


def test_regularization_selection_and_shifted_banks_are_deterministic():
    bank = sample_dd_v1(6, 18105).matrices
    scale, rows = select_regularization(bank, ofdm_waveform(), 8, (15, 18), (.5, 1.0, 2.0))
    assert scale in {.5, 1.0, 2.0} and len(rows) == 3
    first = shifted_prior_channels(4, 18106, "power", .75)
    second = shifted_prior_channels(4, 18106, "power", .75)
    assert torch.equal(first.matrices, second.matrices)


def test_zero_error_ratio_is_finite_and_marked():
    value, insufficient = safe_log10_ber_ratio(0, 1000, 4, 1000)
    assert np.isfinite(value)
    assert insufficient


def test_historical_receiver_ignores_tuned_regularization_map():
    from src.pub1e.core import communication_rows
    channels = sample_dd_v1(2, 18107).matrices
    waveform = {"OFDM": ofdm_waveform()}
    baseline = communication_rows(channels, waveform, (qpsk_constellation(),), (18,), 8, "historical", 128, 18108, 18109)
    supplied = communication_rows(channels, waveform, (qpsk_constellation(),), (18,), 8, "historical", 128, 18108, 18109, regularization={"OFDM": 4.0})
    assert baseline == supplied


def test_plot_selector_rejects_missing_and_duplicate_keys():
    rows = [{"x": 1, "waveform": "A"}, {"x": 1, "waveform": "A"}]
    with pytest.raises(RuntimeError, match="got 2"):
        _select_series(rows, "x", (1,), waveform="A")
    with pytest.raises(RuntimeError, match="got 0"):
        _select_series(rows, "x", (2,), waveform="A")


def test_e1_budget_plot_series_are_waveform_specific():
    rows = read_csv(DATA / "e1_aggregate.csv"); ks = (8, 16, 24, 32, 48, 64)
    series = {
        waveform: _select_series(rows, "k", ks, condition="joint_0.5", modulation="16QAM", snr_db=18, receiver_family="historical", waveform=waveform)
        for waveform in ("Learned", "OTFS", "AFDM", "OFDM")
    }
    assert all([row["waveform"] for row in selected] == [waveform] * 6 for waveform, selected in series.items())
    assert len({tuple(float(row["ber"]) for row in selected) for selected in series.values()}) == 4


def test_saved_e2_e3_e4_plot_series_have_unique_complete_keys():
    e2 = read_csv(DATA / "e2_aggregate.csv")
    for modulation in ("16QAM", "QPSK"):
        for waveform in ("Learned", "OTFS", "AFDM", "OFDM"):
            assert len(_select_series(e2, "snr_db", (18,), epsilon_tau=.2, epsilon_nu=.2, modulation=modulation, waveform=waveform, receiver_family="historical", k=24)) == 1
    e3 = read_csv(DATA / "e3_aggregate.csv")
    assert len(_select_series(e3, "severity", (1., 1.5, 2.), shift_family="delay", modulation="16QAM", waveform="Learned", support_policy="oracle_row_topk", receiver_family="historical", k=24, snr_db=18)) == 3
    e4 = read_csv(DATA / "e4_aggregate.csv")
    assert len(_select_series(e4, "snr_db", (12, 15, 18, 20, 25), condition="favorable_joint_0.5", modulation="16QAM", waveform="AFDM historical", receiver_family="historical", k=24)) == 5


def test_production_raw_tables_remain_byte_exact():
    expected = {
        "e1_per_channel.csv": "fb48bd0b709e958d38d087b06b90d945a7d7cbfa4a52b920b953ee0332d4a1fb",
        "e2_per_channel.csv": "239991e8eeee5eb9c2305920a25685fb626818a321102994f8558b19923e2da6",
        "e3_per_channel.csv": "31ccd56860984fe8afdb2a4155c8375ba2948864493b91f62f11e7561aab7776",
        "e4_per_channel.csv": "ab1a2334250dc72f4ddd5d9e1b6968cf3b2ca17c1c437867f0a8b26f5998aeea",
    }
    assert {name: file_sha256(DATA / name) for name in expected} == expected
