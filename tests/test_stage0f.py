import torch

from src import config
from src.audit import sample_grid_dd
from src.channel import COMPLEX_DTYPE, make_generator, sample_dd_v1
from src.oracle import finite_budget_objective, optimize_oracle_batch, polar_project, tune_afdm_for_budgets
from src.receiver import full_information_mse
from src.waveforms import identity_waveform


def test_polar_projection_is_unitary() -> None:
    generator = make_generator(301)
    raw = (
        torch.randn((3, 8, 8), dtype=torch.float64, generator=generator)
        + 1j * torch.randn((3, 8, 8), dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    projected = polar_project(raw)
    identity = torch.eye(8, dtype=COMPLEX_DTYPE)
    assert torch.allclose(projected.mH @ projected, identity, atol=1e-12, rtol=1e-10)


def test_projected_optimization_preserves_unitarity_and_finite_gradient() -> None:
    channel = sample_dd_v1(1, 302).matrices
    waveform = identity_waveform().clone().requires_grad_(True)
    objective = finite_budget_objective(waveform, channel[0], (16,), (12, 15))
    objective.backward()
    assert torch.isfinite(objective)
    assert waveform.grad is not None and torch.isfinite(waveform.grad.real).all() and torch.isfinite(waveform.grad.imag).all()
    result = optimize_oracle_batch(channel, 1 / 128, (16,), (12, 15), steps=3, random_seed=303)
    assert result.max_unitarity_residual < 1e-10
    assert not torch.any(result.nonfinite_seen)


def test_best_seen_is_not_worse_than_best_initialization() -> None:
    channel = sample_dd_v1(1, 304).matrices
    result = optimize_oracle_batch(channel, 1 / 128, (24,), (12, 15), steps=2, random_seed=305)
    assert result.best_objectives[0] <= torch.min(result.initialization_objectives[0]) + 1e-14


def test_small_oracle_run_is_seed_reproducible() -> None:
    channel = sample_dd_v1(1, 306).matrices
    first = optimize_oracle_batch(channel, 1 / 128, (32,), (12,), steps=2, random_seed=307)
    second = optimize_oracle_batch(channel, 1 / 128, (32,), (12,), steps=2, random_seed=307)
    assert torch.equal(first.initialization_objectives, second.initialization_objectives)
    assert torch.equal(first.best_objectives, second.best_objectives)
    assert torch.equal(first.best_matrices, second.best_matrices)


def test_grid_k3_matches_full_bound_and_oracle_cannot_beat_it() -> None:
    channel = sample_grid_dd(1, 308).matrices
    snrs = (12, 15)
    baseline = finite_budget_objective(identity_waveform(), channel[0], (3,), snrs)
    bound = torch.mean(torch.stack([full_information_mse(channel[0], 10.0 ** (-snr / 10.0)) for snr in snrs]))
    assert torch.allclose(baseline, bound, atol=1e-10, rtol=1e-9)
    result = optimize_oracle_batch(channel, 1 / 128, (3,), snrs, steps=2, random_seed=309)
    assert result.best_objectives[0] >= bound - 1e-10
    assert result.best_objectives[0] <= bound + 1e-10


def test_afdm_tuning_returns_each_budget_and_all_candidates() -> None:
    validation = sample_dd_v1(4, config.VALIDATION_SEED).matrices
    rows, selected = tune_afdm_for_budgets(validation)
    assert set(selected) == {"K16", "K24", "K32", "MULTIK"}
    assert len(rows) == 4 * len(config.AFDM_C1_CANDIDATE_MULTIPLIERS)
    assert all(multiplier in config.AFDM_C1_CANDIDATE_MULTIPLIERS for multiplier in selected.values())
