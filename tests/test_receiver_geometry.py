import torch

from src.channel import COMPLEX_DTYPE
from src.receiver_geometry import apply_mask, distribution_summary, exact_error_geometry, support_geometry


def test_exact_error_geometry_matches_scalar_lmmse() -> None:
    effective = 2.0 * torch.eye(3, dtype=COMPLEX_DTYPE)[None]
    result = exact_error_geometry(effective, effective, 0.5)
    expected_mse = 0.5 / (4.0 + 0.5)
    assert torch.allclose(result["stream_mse"], torch.full((1, 3), expected_mse, dtype=torch.float64))
    assert torch.allclose(result["gamma"], torch.full((1, 3), 8.0, dtype=torch.float64))
    assert torch.allclose(result["condition_number"], torch.ones(1, dtype=torch.float64))
    assert torch.allclose(result["retained_energy"], torch.ones(1, dtype=torch.float64))


def test_exact_error_geometry_retains_mismatch() -> None:
    effective = torch.eye(2, dtype=COMPLEX_DTYPE)[None]
    retained = effective.clone(); retained[:, 0, 0] = 0
    result = exact_error_geometry(effective, retained, 0.1)
    assert result["stream_mse"][0, 0] == 1
    assert result["stream_mse"][0, 1] < 0.1


def test_mask_summary_and_support_geometry() -> None:
    effective = torch.ones((2, 2), dtype=COMPLEX_DTYPE)
    mask = torch.tensor([[True, False], [False, True]])
    assert torch.equal(apply_mask(effective, mask), torch.eye(2, dtype=COMPLEX_DTYPE))
    summary = distribution_summary(torch.arange(1, 21), high_is_bad=False)
    assert summary["adverse_cvar05"] == 1
    geometry = support_geometry(mask, torch.tensor([0, 0]), torch.tensor([0, 1]))
    assert geometry["within_stage2b_community_fraction"] == 1
    assert geometry["mutual_edge_fraction"] == 1
    assert geometry["undirected_component_count"] == 2
