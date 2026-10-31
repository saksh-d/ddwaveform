"""Algebra, positive controls and physical full-receiver continuity."""
import math

import pytest
import torch

from src.channel import dft_matrix, sample_dd_v1
from src.receiver import mismatched_lmmse_equalizer
from src.stage5a import ExactGivens, MultiscaleUnitary, fit_matrix, role_seed
from src.waveforms import random_unitary


@pytest.mark.parametrize("n", [4, 8, 64])
def test_exact_givens_arbitrary_unitary_and_transposed_input(n):
    w = random_unitary(n, 714).T  # exercise noncontiguous input
    representation = ExactGivens(w)
    assert len(representation.gates) == n * (n - 1) // 2
    assert torch.linalg.vector_norm(representation.matrix() - w) < 3e-12
    x = random_unitary(n, 715)[:3]
    assert torch.allclose(representation.apply_W(x), x @ w.T, atol=1e-12)
    assert torch.allclose(representation.apply_W_hermitian(x), x @ w.conj(), atol=1e-12)


@pytest.mark.parametrize("topology", ["ascending", "descending"])
def test_native_action_adjoint_and_identity_nesting(topology):
    model = MultiscaleUnitary(8, 1, topology, 716)
    model.phase.data.copy_(torch.linspace(-1., 1., 8))
    larger = MultiscaleUnitary(8, 4, topology, 717)
    larger.initialize(model.matrix().detach(), model)
    assert torch.allclose(model.matrix(), larger.matrix(), atol=2e-15)
    w, native = model.matrix().detach(), model.compile()
    x = random_unitary(8, 718)[:3]
    assert torch.allclose(native.apply_W(x), x @ w.T, atol=1e-14)
    assert torch.allclose(native.apply_W_hermitian(x), x @ w.conj(), atol=1e-14)
    assert torch.allclose(native.apply_W_hermitian(native.apply_W(x)), x, atol=1e-14)
    ledger = native.ledger()
    assert ledger["nominal_real_parameters"] == 8 + 2 * 12


def test_known_representable_target_recovery_and_phase_gradient():
    target = MultiscaleUnitary(8, 1, seed=719).matrix().detach()
    model = MultiscaleUnitary(8, 1, seed=720)
    model.initialize(target)
    model, trace, stats = fit_matrix(model, target, adam_steps=100, lbfgs_steps=150)
    assert stats["best_loss"] < 1e-8  # relative-error qualification: 1e-4
    assert all(row["gradient_norm"] >= 0 for row in trace)
    perturbed = MultiscaleUnitary(8, 1, seed=721)
    loss = abs(perturbed.matrix() - target).square().sum()
    loss.backward()
    assert float(perturbed.phase.grad.norm()) > 1e-4


def test_full_receiver_physical_solve_equals_effective_solve():
    h = sample_dd_v1(3, 722, n=8).matrices
    w = random_unitary(8, 723)
    sigma2 = 10**(-15/10)
    g_h = mismatched_lmmse_equalizer(h, sigma2)
    a = w.mH @ h @ w
    g_a = mismatched_lmmse_equalizer(a, sigma2)
    assert torch.allclose(g_a, w.mH @ g_h @ w, atol=2e-13)
    c_h = torch.eye(8) - g_h @ h
    c_a = torch.eye(8) - g_a @ a
    assert torch.allclose(c_h.diagonal(dim1=-2, dim2=-1).real.mean(-1),
                          c_a.diagonal(dim1=-2, dim2=-1).real.mean(-1), atol=1e-14)


def test_labeled_integer_generator_commutant_is_scalar():
    n = 4
    shift = torch.roll(torch.eye(n, dtype=torch.complex128), 1, 0)
    modulation = torch.diag(torch.exp(-2j * math.pi * torch.arange(n).double() / n))
    identity = torch.eye(n, dtype=torch.complex128)
    equations = torch.cat([torch.kron(identity, a.contiguous()) -
                           torch.kron(a.T.contiguous(), identity)
                           for a in (shift, modulation)])
    singular = torch.linalg.svdvals(equations)
    assert int((singular < 1e-12).sum()) == 1


def test_role_namespaces_are_distinct():
    seeds = [role_seed(role, i) for role in ("fit", "validation", "atoms", "communication")
             for i in range(20)]
    assert len(seeds) == len(set(seeds))
