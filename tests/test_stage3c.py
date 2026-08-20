import numpy as np
import pytest
import torch

from src.channel import COMPLEX_DTYPE
from src.metrics import qpsk_constellation
from src.stage3c import normalized_minimum_distance, paired_delta_summary, residual_anisotropy, retained_operator, support_overlap


def test_retained_operator_rules_are_exact() -> None:
    matrix = torch.tensor([[1, 3], [4, 2]], dtype=COMPLEX_DTYPE)[None]
    mask = torch.tensor([[True, False], [False, True]])
    assert torch.equal(retained_operator(matrix, "Full", 1, mask), matrix)
    assert torch.equal(retained_operator(matrix, "Q", 1, mask), torch.tensor([[[1, 0], [0, 2]]], dtype=COMPLEX_DTYPE))
    assert torch.equal(retained_operator(matrix, "Oracle", 1, mask), torch.tensor([[[0, 3], [4, 0]]], dtype=COMPLEX_DTYPE))
    with pytest.raises(ValueError):
        retained_operator(matrix, "Q", 1)


def test_residual_anisotropy_and_qpsk_scale() -> None:
    residual = torch.tensor([[1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j]], dtype=COMPLEX_DTYPE)
    assert torch.allclose(residual_anisotropy(residual), torch.ones(1, dtype=torch.float64))
    assert normalized_minimum_distance(qpsk_constellation()) == pytest.approx(np.sqrt(2.0))


def test_paired_summary_and_support_overlap() -> None:
    summary = paired_delta_summary(np.array([2.0, 4.0]), np.array([1.0, 7.0]))
    assert summary["mean_candidate_minus_learned"] == 1.0
    assert summary["fraction_candidate_lower"] == 0.5
    left = torch.tensor([[True, False], [False, True]])
    right = torch.tensor([[True, False], [True, False]])
    assert support_overlap(left, right) == 0.5
