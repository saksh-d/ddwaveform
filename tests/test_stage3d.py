import torch

from src.channel import COMPLEX_DTYPE
from src.stage3d import (
    GeometryObjectiveSpec,
    adverse_channel_stream_tail,
    fixed_q_geometry_metrics,
    normalized_geometry_loss,
    pareto_nondominated,
)


def test_two_level_tail_matches_largest_stream_then_channel() -> None:
    values = torch.arange(1, 81, dtype=torch.float64).reshape(4, 20)
    assert adverse_channel_stream_tail(values, 0.90) == torch.tensor(79.5, dtype=torch.float64)


def test_normalized_geometry_loss_and_specs() -> None:
    loss = normalized_geometry_loss(
        torch.tensor([2.0]), torch.tensor([6.0]),
        torch.tensor([4.0]), torch.tensor([3.0]), torch.tensor([0.25]),
    )
    assert torch.equal(loss, torch.tensor([1.0]))
    assert GeometryObjectiveSpec("mean", 0.0).name == "MEAN"
    assert GeometryObjectiveSpec("tail", 0.25).name == "TAIL_0.25"


def test_pareto_filter_removes_dominated_points() -> None:
    rows = [
        {"spec": "a", "validation_mean": 1.0, "validation_tail": 3.0},
        {"spec": "b", "validation_mean": 2.0, "validation_tail": 2.0},
        {"spec": "c", "validation_mean": 3.0, "validation_tail": 3.0},
    ]
    assert [row["spec"] for row in pareto_nondominated(rows)] == ["a", "b"]


def test_fixed_q_geometry_is_exact_finite_and_differentiable() -> None:
    torch.manual_seed(9)
    n = 4
    waveform = torch.eye(n, dtype=COMPLEX_DTYPE, requires_grad=True)
    channels = torch.randn((3, n, n), dtype=COMPLEX_DTYPE)
    q_operators = torch.randn((5, n, n), dtype=COMPLEX_DTYPE)
    metrics = fixed_q_geometry_metrics(waveform, channels, q_operators, k=2, chunk_size=2)
    assert metrics["mask"].shape == (n, n)
    assert all(torch.isfinite(metrics[key]) for key in ("mean", "tail", "eta", "retained_energy"))
    (metrics["mean"] + metrics["tail"]).backward()
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
