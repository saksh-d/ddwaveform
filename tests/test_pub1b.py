"""Focused Pub-1B compute and scaling gates."""

from pathlib import Path
import math
import inspect
import json

import numpy as np
import torch

from src.channel import COMPLEX_DTYPE, assemble_dd_channel, make_generator, noise_variance
from src.metrics import qam16_constellation, qam64_constellation, qpsk_constellation
from src.pub1b.core import (
    N64_CHECKPOINT_SHA256,
    PUB1A_TREE_SHA256,
    SparseRowMatrix,
    afdm_candidate_multipliers,
    cg_mac_count,
    clean_reconstruction_error,
    directory_tree_sha256,
    file_sha256,
    fixed_scaled_masks,
    full_mse_invariance,
    otfs_grid,
    sample_scaled_dd,
    scaled_waveforms,
    scaling_definition,
    scaling_rule_record,
    sparse_cg_lmmse,
    sparse_cg_batch_snapshots,
    sparse_from_mask,
    sparse_from_topk,
)
from src.receiver import effective_operator, mismatched_lmmse_equalizer
from src.stage2d import score_support_mask
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform, random_unitary


ROOT = Path(__file__).resolve().parents[1]


def _system(n: int = 12, k: int = 4) -> tuple[torch.Tensor, SparseRowMatrix, torch.Tensor, float]:
    generator = make_generator(17000 + n + k)
    dense = torch.randn((n, n), dtype=COMPLEX_DTYPE, generator=generator)
    sparse = sparse_from_topk(dense, k)
    observation = torch.randn(n, dtype=COMPLEX_DTYPE, generator=generator)
    return dense, sparse, observation, 0.2


def test_sparse_iterative_complex128_matches_direct_lmmse():
    _, sparse, observation, sigma2 = _system()
    direct = mismatched_lmmse_equalizer(sparse.dense(), sigma2) @ observation
    iterative = sparse_cg_lmmse(sparse, observation, sigma2, max_iterations=96, tolerance=1e-12)
    relative = torch.linalg.vector_norm(iterative.estimate - direct) / torch.linalg.vector_norm(direct)
    assert iterative.estimate.dtype == COMPLEX_DTYPE
    assert iterative.converged and iterative.relative_residual < 1e-12
    assert float(relative) < 1e-10


def test_measured_sparse_path_does_not_call_dense(monkeypatch):
    _, sparse, observation, sigma2 = _system()
    monkeypatch.setattr(SparseRowMatrix, "dense", lambda self: (_ for _ in ()).throw(AssertionError("dense path")))
    result = sparse_cg_lmmse(sparse, observation, sigma2, max_iterations=4)
    assert result.iterations == 4


def test_sparse_mask_has_exact_n_times_k_entries_and_products_match_dense():
    dense, _, observation, _ = _system(n=10, k=3)
    mask = score_support_mask(torch.abs(dense).square(), 3)
    sparse = sparse_from_mask(dense, mask)
    assert sparse.nnz == 30
    assert torch.all(mask.sum(-1) == 3)
    assert torch.allclose(sparse.matvec(observation), sparse.dense() @ observation)
    assert torch.allclose(sparse.adjoint_matvec(observation), sparse.dense().mH @ observation)
    batch = torch.stack((observation, 2 * observation))
    assert torch.allclose(sparse.matvec(batch), batch @ sparse.dense().T)
    assert torch.allclose(sparse.adjoint_matvec(batch), batch @ sparse.dense().conj())


def test_batched_cg_snapshots_match_independent_direct_solves():
    _, sparse, observation, sigma2 = _system(n=12, k=5)
    observations = torch.stack((observation, 1j * observation))
    snapshots, _, _ = sparse_cg_batch_snapshots(sparse, observations, sigma2, (4, 96), tolerance=1e-12)
    direct = torch.stack([mismatched_lmmse_equalizer(sparse.dense(), sigma2) @ value for value in observations])
    relative = torch.linalg.vector_norm(snapshots[96].estimate - direct, dim=-1) / torch.linalg.vector_norm(direct, dim=-1)
    assert snapshots[96].converged.all()
    assert torch.all(relative < 1e-10)


def test_mac_counter_matches_toy_analytic_values():
    count = cg_mac_count(n=10, k=3, iterations=4)
    assert count.sparse_detector == 2 * 10 * 3 * 4
    assert count.vector_dot == 10 + 6 * 10 * 4
    assert count.preconditioner == 0
    assert count.setup == 10 * 3
    assert count.total == 520


def test_common_solver_rule_is_waveform_agnostic():
    for seed in (1, 2, 3):
        matrix = random_unitary(12, seed)
        sparse = sparse_from_topk(matrix, 4)
        observation = torch.ones(12, dtype=COMPLEX_DTYPE)
        result = sparse_cg_lmmse(sparse, observation, 0.1, max_iterations=6, tolerance=0.0)
        assert result.iterations == 6
        assert result.macs == cg_mac_count(12, 4, 6)


def test_full_k_iterative_converges_to_full_reference():
    channel = sample_scaled_dd(1, 17020, 64).matrices[0]
    waveform = ofdm_waveform(64)
    actual = effective_operator(channel, waveform)
    sparse = sparse_from_topk(actual, 64)
    observation = torch.randn(64, dtype=COMPLEX_DTYPE, generator=make_generator(17021))
    direct = mismatched_lmmse_equalizer(actual, noise_variance(18)) @ observation
    result = sparse_cg_lmmse(sparse, observation, noise_variance(18), max_iterations=256, tolerance=1e-12)
    assert result.converged
    assert float(torch.linalg.vector_norm(result.estimate - direct) / torch.linalg.vector_norm(direct)) < 1e-10


def test_scaling_law_preserves_normalized_supports_and_fixed_budgets():
    record = scaling_rule_record()
    assert record["status"].startswith("frozen")
    for n in (64, 128, 256):
        definition = scaling_definition(n)
        assert np.isclose(definition.delay_max / n, 4 / 64)
        assert np.isclose(definition.doppler_max / n, 0.75 / 64)
        assert definition.primary_k == 3 * n // 8
        assert definition.k_values[-1] == n


def test_scaled_channel_supports_match_definitions():
    for n in (128, 256):
        batch = sample_scaled_dd(8, 17030 + n, n)
        definition = scaling_definition(n)
        assert batch.matrices.shape == (8, n, n)
        assert torch.all((batch.delays >= 0) & (batch.delays <= definition.delay_max))
        assert torch.all(torch.abs(batch.dopplers) <= definition.doppler_max)
        norm = torch.sum(torch.abs(batch.matrices) ** 2, dim=(-2, -1)) / n
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-12)


def test_scaled_otfs_dimensions_unitarity_and_clean_reconstruction():
    for n in (128, 256):
        n_tau, n_nu = otfs_grid(n)
        waveform = otfs_waveform(n_tau, n_nu)
        assert n_tau * n_nu == n
        assert torch.allclose(waveform.mH @ waveform, torch.eye(n, dtype=COMPLEX_DTYPE), atol=2e-12)
        assert clean_reconstruction_error(waveform) < 2e-12


def test_scaled_learned_shapes_unitarity_and_clean_reconstruction():
    for n in (128, 256):
        waveform = random_unitary(n, 17100 + n)
        assert waveform.shape == (n, n)
        assert torch.linalg.matrix_norm(waveform.mH @ waveform - torch.eye(n, dtype=COMPLEX_DTYPE)) < 1e-12
        assert clean_reconstruction_error(waveform) < 2e-12


def test_scaled_full_lmmse_aggregate_invariance():
    n = 128
    learned = random_unitary(n, 17300)
    waveforms = scaled_waveforms(n, learned, afdm_candidate_multipliers(n)[0])
    channels = sample_scaled_dd(2, 17301, n).matrices
    assert full_mse_invariance(channels, waveforms) < 1e-12


def test_scaled_afdm_predicted_observed_offsets_and_unitarity():
    for n in (128, 256):
        definition = scaling_definition(n)
        alpha = int(torch.floor(torch.tensor(definition.doppler_max) + 0.5))
        delays = (0, 1, int(definition.delay_max // 2), int(definition.delay_max))
        dopplers = (-alpha, alpha, 0, -alpha)
        observed_sets = []
        for multiplier in afdm_candidate_multipliers(n):
            waveform = afdm_waveform(n=n, c1=multiplier / (2 * n), c2=math.sqrt(2.0) / (100 * n))
            assert torch.allclose(waveform.mH @ waveform, torch.eye(n, dtype=COMPLEX_DTYPE), atol=3e-12)
            observed = []
            for delay, doppler in zip(delays, dopplers):
                path = assemble_dd_channel(
                    torch.tensor([delay]), torch.tensor([doppler]), torch.tensor([1 + 0j], dtype=COMPLEX_DTYPE), n=n
                )
                effective = effective_operator(path, waveform)
                offsets = (torch.arange(n)[None, :] - torch.arange(n)[:, None]) % n
                energy = torch.zeros(n, dtype=torch.float64)
                energy.scatter_add_(0, offsets.reshape(-1), torch.abs(effective).square().reshape(-1))
                value = int(torch.argmax(energy))
                assert value == (multiplier * delay + doppler) % n
                observed.append(value)
            assert len(set(observed)) == len(observed)
            observed_sets.append(observed)


def test_constellation_normalization_for_pub1b_modulations():
    for constellation in (qpsk_constellation(), qam16_constellation(), qam64_constellation()):
        assert torch.allclose(
            torch.mean(torch.abs(constellation.points) ** 2),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-12,
        )


def test_authoritative_n64_checkpoint_and_pub1a_package_are_unchanged():
    checkpoint = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
    assert file_sha256(checkpoint) == N64_CHECKPOINT_SHA256
    assert directory_tree_sha256(ROOT / "outputs/stage_pub1a") == PUB1A_TREE_SHA256


def test_runtime_benchmark_consumes_prepared_inputs_only():
    from scripts.stage_pub1b import phase_complexity_runtime
    source = inspect.getsource(phase_complexity_runtime)
    assert "runtime_inputs.pt" in source
    assert "sample_scaled_dd" not in source
    assert "sample_regime" not in source
    environment = json.loads((ROOT / "outputs/stage_pub1b/provenance/runtime_environment.json").read_text())
    assert environment["torch_threads"] == 1


def test_scaled_fixed_support_has_exact_k_per_row():
    masks, _ = fixed_scaled_masks({"OFDM": ofdm_waveform(128)}, 128, (16, 48, 128), 4, 17400)
    for k in (16, 48, 128):
        assert torch.all(masks["OFDM"][k].sum(-1) == k)


def test_production_scaling_checkpoint_hashes_and_provenance():
    for n in (128, 256):
        directory = ROOT / "outputs/stage_pub1b/checkpoints" / f"N{n}"
        metadata = json.loads((directory / "metadata.json").read_text())
        checkpoint = ROOT / metadata["checkpoint"]
        waveform = torch.load(checkpoint, weights_only=True)
        assert metadata["pub1a_parent_commit"] == "737c720b89b64f9bea11b2fd825b8730a77cb14c"
        assert metadata["checkpoint_sha256"] == file_sha256(checkpoint)
        assert waveform.shape == (n, n) and waveform.dtype == COMPLEX_DTYPE
        assert torch.linalg.matrix_norm(waveform.mH @ waveform - torch.eye(n, dtype=COMPLEX_DTYPE)) < 1e-11
        gate = json.loads((ROOT / f"outputs/stage_pub1b/data/scaling/hard_gate_n{n}.json").read_text())
        assert gate["passed"] and all(gate["checks"].values())
