from __future__ import annotations

import json
from pathlib import Path

import torch

from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.pub1b.core import file_sha256
from src.pub1f.continuation import (
    CONTINUATION_NAMESPACE, dense_cg_snapshots, fixed_energy_support,
    matching_residual_statistics,
)
from src.pub1f.core import operation_ledger, semantic_seed
from src.receiver import mismatched_lmmse_equalizer, rowwise_topk
from src.waveforms import ofdm_waveform
from scripts.stage_pub1f_continuation import PHASES


ROOT = Path(__file__).resolve().parents[1]


def test_continuation_namespace_is_fresh_and_semantic():
    old = semantic_seed("training_channel", seed_id=0, stage="parent", step=1)
    new = semantic_seed("training_channel", namespace=CONTINUATION_NAMESPACE, seed_id=0, stage="parent", step=1)
    assert old != new
    assert new == semantic_seed("training_channel", namespace=CONTINUATION_NAMESPACE, step=1, stage="parent", seed_id=0)


def test_original_gate_remains_failed_and_authorization_is_separate():
    gate = json.loads((ROOT / "outputs/stage_pub1f/data/early_gate.json").read_text())
    continuation = json.loads((ROOT / "configs/stage_pub1f_continuation.json").read_text())
    assert gate["passed"] is False
    assert continuation["authorization"]["preserve_original_gate"] is True
    assert continuation["authorization"]["original_gate_outcome"] == "FAIL"


def test_dense_matrix_free_cg_matches_direct_small_system():
    channels = sample_dd_v1(2, 93001).matrices
    waveform = ofdm_waveform()
    actual = waveform.mH @ channels @ waveform
    retained = rowwise_topk(actual, 24)
    observations = torch.randn((2, 2, 64), dtype=COMPLEX_DTYPE)
    direct = torch.einsum("cij,cbj->cbi", mismatched_lmmse_equalizer(retained, noise_variance(18)), observations)
    result = dense_cg_snapshots(retained, observations, float(noise_variance(18)), (128,), tolerance=1e-12)[128]
    relative = torch.linalg.vector_norm(result.estimate - direct) / torch.linalg.vector_norm(direct)
    assert float(relative) < 1e-9


def test_fixed_support_residual_statistics_match_declared_policy():
    support_channels = sample_dd_v1(8, 93002).matrices
    residual_channels = sample_dd_v1(8, 93003).matrices
    waveform = ofdm_waveform()
    support = fixed_energy_support(support_channels, waveform, 8)
    stats = matching_residual_statistics(residual_channels, waveform, 8, "fixed_statistical", support)
    assert torch.all(support.sum(dim=-1) == 8)
    assert stats.mean.shape == (64, 64)
    assert float(torch.linalg.eigvalsh(stats.receive_covariance).min()) > -1e-10


def test_declared_scaling_operation_rows_are_exact():
    expected = {(64, 24): 222784, (128, 48): 841856, (256, 96): 3268864, (512, 192): 12878336, (1024, 384): 51119104}
    for (n, k), value in expected.items():
        assert operation_ledger(n, k, 64)["algorithmic_total_complex_equivalent_macs"] == value


def test_scaling_checkpoint_hashes_are_live():
    continuation = json.loads((ROOT / "configs/stage_pub1f_continuation.json").read_text())
    for entry in continuation["scaling"]["checkpoints"].values():
        assert file_sha256(ROOT / entry["path"]) == entry["sha256"]


def test_all_phase_order_respects_data_dependencies():
    names = tuple(PHASES)
    assert names.index("operating-validation") < names.index("communication") < names.index("intervals")
    assert names.index("scaling") < names.index("scaling-matched") < names.index("figures")
    assert names[-2:] == ("report", "manifest")
