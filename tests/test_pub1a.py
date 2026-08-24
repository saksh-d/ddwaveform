"""Focused Pub-1A frozen qualification gates."""

from pathlib import Path

import numpy as np
import torch

from src import config
from src.channel import COMPLEX_DTYPE, assemble_dd_channel
from src.metrics import qam16_constellation, qam64_constellation, qpsk_constellation
from src.pub1a.core import (
    AFDM_ADMISSIBLE_MULTIPLIERS, CHECKPOINT_SHA256, REGIME_BY_KEY,
    afdm_structural_audit, fixed_statistical_masks, practicality_data, waveform_set,
)
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2d import apply_support
from src.stage2e import communication_rows, sample_regime
from src.waveforms import afdm_waveform, otfs_waveform
from scripts.stage_pub1a import waveform_peak_normalize_db


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"


def test_frozen_checkpoint_hash_and_unitarity():
    assert file_sha256(FROZEN) == CHECKPOINT_SHA256
    learned, record = load_frozen_waveform(FROZEN)
    assert record["unitarity_residual"] < 1e-12
    assert torch.allclose(learned.mH @ learned, torch.eye(config.N, dtype=COMPLEX_DTYPE), atol=1e-12)


def test_otfs_and_admissible_afdm_are_unitary_clean_transforms():
    generator = torch.Generator().manual_seed(13000)
    symbols = torch.randn(config.N, dtype=COMPLEX_DTYPE, generator=generator)
    for waveform in (otfs_waveform(), *(afdm_waveform(c1=m / (2 * config.N), c2=config.AFDM_C2) for m in AFDM_ADMISSIBLE_MULTIPLIERS)):
        assert torch.allclose(waveform.mH @ waveform, torch.eye(config.N, dtype=COMPLEX_DTYPE), atol=1e-12)
        assert torch.allclose(waveform.mH @ (waveform @ symbols), symbols, atol=1e-12)


def test_afdm_integer_offsets_multipath_distinct_and_fractional_invariants():
    channels = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 4, 13001).matrices
    audit = afdm_structural_audit(channels)
    assert audit["hard_gate"]
    for candidate in audit["candidates"]:
        assert candidate["grid_distinct"] and candidate["predicted_equals_observed"]
        assert len({row["observed_offset"] for row in candidate["paths"]}) == 4
        assert candidate["cases"]["fractional_doppler"]["top24_energy_fraction"] <= 1 + 1e-12
        assert candidate["cases"]["full_offgrid"]["top24_energy_fraction"] <= 1 + 1e-12


def test_qpsk_qam16_qam64_have_unit_average_power_and_unique_gray_labels():
    for constellation in (qpsk_constellation(), qam16_constellation(), qam64_constellation()):
        assert torch.allclose(
            torch.mean(torch.abs(constellation.points) ** 2),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-12,
        )
        assert len(torch.unique(constellation.bit_labels, dim=0)) == len(constellation.points)


def test_fixed_and_oracle_supports_have_exact_row_budget():
    learned, _ = load_frozen_waveform(FROZEN); waveforms = waveform_set(learned, 3)
    masks, _ = fixed_statistical_masks(waveforms, REGIME_BY_KEY["dd_v1"], (4, 24, 64), 32, 13002)
    channel = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 2, 13003).matrices
    for name, waveform in waveforms.items():
        assert torch.all(masks[name]["fixedStatK4"].sum(-1) == 4)
        assert torch.all(masks[name]["fixedStatK24"].sum(-1) == 24)
        actual = effective_operator(channel, waveform)
        assert torch.all((torch.abs(rowwise_topk(actual, 24)) > 0).sum(-1) == 24)
        assert torch.count_nonzero(apply_support(actual, masks[name]["fixedStatK24"]), dim=-1).eq(24).all()


def test_common_random_number_pairing_identical_waveforms_gives_identical_counts():
    channel = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 2, 13004).matrices
    waveform = otfs_waveform(); waveforms = {"left": waveform, "right": waveform.clone()}
    masks, _ = fixed_statistical_masks(waveforms, REGIME_BY_KEY["dd_v1"], (24,), 32, 13005)
    modes = {name: {"fixedStatK24": masks[name]["fixedStatK24"]} for name in waveforms}
    rows = communication_rows(channel, waveforms, modes, (qpsk_constellation(),), (18,), 1024, 13006, 13007)
    left = [row["bit_errors"] for row in rows if row["waveform"] == "left"]
    right = [row["bit_errors"] for row in rows if row["waveform"] == "right"]
    assert left == right


def test_full_receiver_is_waveform_invariant_in_analytic_mse():
    learned, _ = load_frozen_waveform(FROZEN); waveforms = waveform_set(learned, 3)
    channel = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 2, 13008).matrices
    values = []
    from src.receiver import analytic_communication_mse, mismatched_lmmse_equalizer
    from src.channel import noise_variance
    for waveform in waveforms.values():
        operator = effective_operator(channel, waveform)
        equalizer = mismatched_lmmse_equalizer(operator, noise_variance(18))
        values.append(float(analytic_communication_mse(operator, equalizer, noise_variance(18)).mean()))
    assert max(values) - min(values) < 1e-12


def test_papr_psd_normalization_is_matched_across_waveforms():
    waveform = otfs_waveform(); result = practicality_data({"A": waveform, "B": waveform.clone()}, 16, 13009)
    for modulation in ("QPSK", "16QAM", "64QAM"):
        a = next(row for row in result["papr"] if row["modulation"] == modulation and row["waveform"] == "A")
        b = next(row for row in result["papr"] if row["modulation"] == modulation and row["waveform"] == "B")
        assert a["samples_db"] == b["samples_db"]
        assert np.allclose(result["psd"][f"{modulation}_A"], result["psd"][f"{modulation}_B"])


def test_stage2e_nominal_artifact_present_for_reproduction_gate():
    path = ROOT / "outputs/stage2e/nominal_communication_summary.csv"
    assert path.exists()
    import csv
    rows = list(csv.DictReader(path.open()))
    def ber(modulation: str, mode: str, waveform: str) -> float:
        return float(next(row["ber"] for row in rows if row["modulation"] == modulation and row["receiver_mode"] == mode and row["waveform"] == waveform and row["snr_db"] == "18"))
    assert np.isclose(ber("QPSK", "fixedQK24", "Learned"), 0.0012598037719726562)
    assert np.isclose(ber("16QAM", "fixedQK24", "Learned"), 0.025005340576171875)
    assert np.isclose(ber("QPSK", "full", "OTFS"), 0.0001163482666015625)
    assert np.isclose(ber("16QAM", "full", "AFDM"), 0.021986961364746094)


def test_waveform_peak_normalization_preserves_atom_peak_offsets():
    original = {
        "A": np.asarray([[1.0, -3.0, 2.0], [4.0, 0.0, -1.0]]),
        "B": np.asarray([[-7.0, 5.0], [3.0, 1.0]]),
    }
    before = {name: values.copy() for name, values in original.items()}
    normalized = waveform_peak_normalize_db(original)
    for name, values in original.items():
        assert np.isclose(np.max(normalized[name]), 0.0, atol=1e-12)
        assert np.max(normalized[name]) <= 1e-12
        assert np.allclose(
            np.max(normalized[name], axis=0),
            np.max(values, axis=0) - np.max(values),
        )
        assert np.array_equal(original[name], before[name])
