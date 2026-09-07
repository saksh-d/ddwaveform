from __future__ import annotations

import torch

from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.metrics import qpsk_constellation
from src.pub1e.core import residual_statistics
from src.pub1f.core import (
    TrainingProtocol, communication_counts, condition_channels, operation_ledger,
    receiver_estimate, semantic_seed, storage_ledger, train_stage,
)
from src.receiver import effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from src.waveforms import ofdm_waveform, otfs_waveform


def test_semantic_seed_is_stable_order_independent_and_role_separated():
    assert semantic_seed("channel", condition="joint", snr=18) == semantic_seed("channel", snr=18, condition="joint")
    assert semantic_seed("channel", condition="joint", snr=18) != semantic_seed("noise", condition="joint", snr=18)
    assert semantic_seed("channel", condition="joint", snr=18) != semantic_seed("channel", condition="joint", snr=20)


def test_condition_banks_are_reproducible_and_role_separated():
    left = condition_channels("joint_0.5", 2, "validation")
    right = condition_channels("joint_0.5", 2, "validation")
    other = condition_channels("joint_0.5", 2, "test")
    assert torch.equal(left, right)
    assert not torch.equal(left, other)


def test_full_observation_receiver_restriction_and_k_n_collapse():
    channels = sample_dd_v1(2, 92001).matrices
    waveform = otfs_waveform()
    actual = effective_operator(channels, waveform)
    observations = torch.randn((2, 1, 64), dtype=COMPLEX_DTYPE)
    estimate, retained = receiver_estimate(actual, observations, 64, float(noise_variance(18)), "full")
    reference = torch.einsum("cij,cbj->cbi", mismatched_lmmse_equalizer(actual, noise_variance(18)), observations)
    assert torch.equal(retained, actual)
    assert torch.allclose(estimate, reference, atol=1e-12, rtol=1e-12)


def test_residual_statistics_are_waveform_specific_and_psd():
    channels = sample_dd_v1(8, 92002).matrices
    ofdm = residual_statistics(channels, ofdm_waveform(), 8)
    otfs = residual_statistics(channels, otfs_waveform(), 8)
    assert not torch.equal(ofdm.mean, otfs.mean)
    eigenvalues = torch.linalg.eigvalsh(ofdm.receive_covariance)
    assert float(eigenvalues.min()) > -1e-10


def test_communication_pairing_is_deterministic_and_uses_all_observations():
    channels = condition_channels("integer", 2, "test-pairing")
    kwargs = dict(condition="integer", bank_role="test", bits_per_channel=128)
    first = communication_counts(channels, ofdm_waveform(), qpsk_constellation(), 18, 8, "historical", **kwargs)
    second = communication_counts(channels, ofdm_waveform(), qpsk_constellation(), 18, 8, "historical", **kwargs)
    assert first == second
    assert all(row["total_bits"] == 128 for row in first)


def test_training_continuation_streams_match_but_parent_is_fresh(monkeypatch):
    import src.pub1f.core as core
    seen = []
    original = core.sample_dd_v1
    def capture(count, seed, n=64, p=3):
        seen.append(seed)
        return original(count, seed, n=n, p=p)
    monkeypatch.setattr(core, "sample_dd_v1", capture)
    validation = sample_dd_v1(2, 92003).matrices
    initial = ofdm_waveform()
    mean = TrainingProtocol(0, "mean_continuation", 1, 1, 1, 24, "parent")
    tail = TrainingProtocol(0, "tail_continuation", 1, 1, 1, 24, "parent")
    parent = TrainingProtocol(0, "parent", 1, 1, 1, 24, None)
    train_stage(initial, validation, mean)
    mean_seed = seen[-1]
    train_stage(initial, validation, tail, reference_parent_mean=1.0)
    tail_seed = seen[-1]
    train_stage(initial, validation, parent)
    parent_seed = seen[-1]
    assert mean_seed == tail_seed
    assert parent_seed != mean_seed


def test_operation_and_storage_ledgers_match_declared_formulas():
    counts = operation_ledger(64, 24, 16)
    assert counts["algorithmic_total_complex_equivalent_macs"] == 64 * 24 + 64 + 16 * (2 * 64 * 24 + 6 * 64)
    storage = storage_ledger(64, 24, residual_aware=False)
    assert storage["dense_coefficients_bytes"] == 16 * 64 * 64
    assert storage["sparse_total_bytes"] == 24 * 64 * 24
    residual = storage_ledger(64, 24, residual_aware=True)
    assert residual["residual_mean_covariance_bytes"] == 2 * 16 * 64 * 64
