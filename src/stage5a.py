"""Native complex unitary representations for the isolated Stage-5A study.

Actions use row batches (..., N); matrix columns are synthesized basis pulses.
No fitted permutation or column phase alignment is used during optimization.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn

from .channel import dft_matrix, make_generator


def role_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("stage5a-v1/".encode() +
                          "/".join(map(str, parts)).encode()).digest()[:7], "big")


def pair_schedule(n: int, passes: int, topology: str) -> torch.Tensor:
    if n < 2 or n & (n - 1) or passes < 1:
        raise ValueError("power-of-two N>=2 and positive passes required")
    if topology not in ("ascending", "descending"):
        raise ValueError("unknown fixed topology")
    scales = list(range(n.bit_length() - 1))
    if topology == "descending":
        scales.reverse()
    layers = []
    for _ in range(passes):
        for scale in scales:
            left = [j for j in range(n) if not j & (1 << scale)]
            layers.append([index for j in left for index in (j, j + (1 << scale))])
    return torch.tensor(layers, dtype=torch.int64)


def pair_action(x: torch.Tensor, permutation: torch.Tensor, inverse: torch.Tensor,
                c: torch.Tensor, s: torch.Tensor, adjoint: bool = False) -> torch.Tensor:
    pairs = x[..., permutation].reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    if adjoint:
        a, b = c * a - s * b, s.conj() * a + c * b
    else:
        a, b = c * a + s * b, -s.conj() * a + c * b
    return torch.stack((a, b), dim=-1).flatten(-2)[..., inverse]


class MultiscaleUnitary(nn.Module):
    """W=F^H D L_last ... L_first, independent two-angle SU(2) gates."""
    def __init__(self, n: int, passes: int, topology: str = "ascending", seed: int = 0):
        super().__init__()
        self.n, self.passes, self.topology = n, passes, topology
        order = pair_schedule(n, passes, topology)
        self.register_buffer("order", order)
        self.register_buffer("inverse", torch.argsort(order, dim=-1))
        generator = make_generator(seed)
        self.theta = nn.Parameter(.15 * torch.randn((len(order), n // 2),
                                                   dtype=torch.float64, generator=generator))
        self.phi = nn.Parameter(2 * math.pi * torch.rand(self.theta.shape,
                                                       dtype=torch.float64, generator=generator))
        self.phase = nn.Parameter(torch.zeros(n, dtype=torch.float64))

    def apply_W(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.order)):
            c = torch.cos(self.theta[i])
            s = torch.sin(self.theta[i]) * torch.exp(1j * self.phi[i])
            x = pair_action(x, self.order[i], self.inverse[i], c, s)
        return torch.fft.ifft(x * torch.exp(1j * self.phase), dim=-1, norm="ortho")

    def apply_W_hermitian(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft(x, dim=-1, norm="ortho") * torch.exp(-1j * self.phase)
        for i in reversed(range(len(self.order))):
            c = torch.cos(self.theta[i])
            s = torch.sin(self.theta[i]) * torch.exp(1j * self.phi[i])
            x = pair_action(x, self.order[i], self.inverse[i], c, s, True)
        return x

    def matrix(self) -> torch.Tensor:
        return self.apply_W(torch.eye(self.n, dtype=torch.complex128)).T.contiguous()

    def initialize(self, target: torch.Tensor, incumbent: "MultiscaleUnitary | None" = None):
        with torch.no_grad():
            if incumbent is None:
                self.phase.copy_(torch.angle(torch.diagonal(dft_matrix(self.n) @ target)))
            else:
                count = len(incumbent.theta)
                self.theta[:count].copy_(incumbent.theta)
                self.phi[:count].copy_(incumbent.phi)
                self.theta[count:].zero_()  # exact identity nesting, including step zero
                self.phase.copy_(incumbent.phase)

    def compile(self) -> "CompiledMultiscale":
        return CompiledMultiscale(self)


class CompiledMultiscale:
    """Cached native action. No dense W; buffers/indices fully accounted."""
    def __init__(self, model: MultiscaleUnitary):
        self.n = model.n
        self.order = model.order.clone()
        self.inverse = model.inverse.clone()
        self.c = torch.cos(model.theta.detach())
        self.s = torch.sin(model.theta.detach()) * torch.exp(1j * model.phi.detach())
        self.d = torch.exp(1j * model.phase.detach())

    def apply_W(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.order)):
            x = pair_action(x, self.order[i], self.inverse[i], self.c[i], self.s[i])
        return torch.fft.ifft(x * self.d, dim=-1, norm="ortho")

    def apply_W_hermitian(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft(x, dim=-1, norm="ortho") * self.d.conj()
        for i in reversed(range(len(self.order))):
            x = pair_action(x, self.order[i], self.inverse[i], self.c[i], self.s[i], True)
        return x

    def ledger(self) -> dict:
        gates = self.c.numel()
        return {"mixers": gates, "nominal_real_parameters": self.n + 2 * gates,
                "intrinsic_dimension_upper_bound": min(self.n**2, self.n + 2 * gates),
                "parameter_bytes_float64": 8 * (self.n + 2 * gates),
                "cached_coefficient_bytes": sum(t.numel() * t.element_size() for t in
                                                 (self.c, self.s, self.d)),
                "cached_index_bytes": sum(t.numel() * t.element_size() for t in
                                           (self.order, self.inverse)),
                "dense_bytes_complex128": 16 * self.n**2,
                "complex_multiplies_per_action_excluding_fft": 2 * gates + self.n,
                "real_scalings_per_action": 4 * gates,
                "complex_adds_per_action_excluding_fft": 2 * gates,
                "fft_length": self.n, "fft_count": 1,
                "workspace": "O(batch*N), plus backend FFT workspace; no dense W",
                "identifiability": "nominal upper bound; phase angles unidentifiable at zero mixers"}


class ExactGivens:
    """Complete QR elimination; quadratic reference, not a compact family."""
    def __init__(self, w: torch.Tensor):
        self.n = len(w)
        remaining = w.detach().clone()
        self.gates = []
        for column in range(self.n - 1):
            for row in range(self.n - 1, column, -1):
                j, k = row - 1, row
                a, b = remaining[j, column].clone(), remaining[k, column].clone()
                radius = torch.sqrt(a.abs()**2 + b.abs()**2)
                if radius == 0:
                    c, s = torch.tensor(1., dtype=torch.float64), torch.tensor(0j, dtype=w.dtype)
                else:
                    phase = a / a.abs() if a.abs() > 0 else torch.tensor(1+0j, dtype=w.dtype)
                    c, s = a.abs() / radius, phase * b.conj() / radius
                left, right = remaining[j].clone(), remaining[k].clone()
                remaining[j], remaining[k] = c * left + s * right, -s.conj() * left + c * right
                self.gates.append((j, k, c, s))
        diagonal = remaining.diagonal()
        self.d = diagonal / diagonal.abs()
        self.elimination_residual = float(torch.linalg.vector_norm(remaining - torch.diag(self.d)))

    def apply_W(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone() * self.d
        for j, k, c, s in reversed(self.gates):
            a, b = x[..., j].clone(), x[..., k].clone()
            x[..., j], x[..., k] = c * a - s * b, s.conj() * a + c * b
        return x

    def apply_W_hermitian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for j, k, c, s in self.gates:
            a, b = x[..., j].clone(), x[..., k].clone()
            x[..., j], x[..., k] = c * a + s * b, -s.conj() * a + c * b
        return x * self.d.conj()

    def matrix(self) -> torch.Tensor:
        return self.apply_W(torch.eye(self.n, dtype=torch.complex128)).T.contiguous()


def matrix_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (candidate - target).abs().square().sum() / len(target)


def fit_matrix(model: MultiscaleUnitary, target: torch.Tensor, *, adam_steps: int,
               lbfgs_steps: int) -> tuple[MultiscaleUnitary, list[dict], dict]:
    """Native complex loss; retain every improving iterate, including step zero."""
    start = time.perf_counter()
    best, best_loss = copy.deepcopy(model.state_dict()), float("inf")
    trace, evaluations = [], 0

    def closure(optimizer, phase):
        nonlocal best, best_loss, evaluations
        optimizer.zero_grad()
        loss = matrix_loss(model.matrix(), target)
        loss.backward()
        gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
        value = float(loss.detach())
        if not math.isfinite(value + gradient):
            raise RuntimeError("nonfinite fitting state")
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
        trace.append({"evaluation": evaluations, "phase": phase, "loss": value,
                      "gradient_norm": gradient, "best_loss": best_loss})
        evaluations += 1
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=.035)
    for _ in range(adam_steps):
        closure(optimizer, "adam")
        optimizer.step()
    model.load_state_dict(best if trace else model.state_dict())
    if lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_steps,
                                      tolerance_grad=1e-10, tolerance_change=1e-13,
                                      line_search_fn="strong_wolfe", history_size=30)
        optimizer.step(lambda: closure(optimizer, "lbfgs"))
    # Capture the final update too; an optimizer closure can precede its update.
    optimizer.zero_grad()
    closure(optimizer, "final")
    final_loss = trace[-1]["loss"]
    model.load_state_dict(best)
    return model, trace, {"initial_loss": trace[0]["loss"], "final_loss": final_loss,
                          "best_loss": best_loss, "evaluations": evaluations,
                          "seconds": time.perf_counter() - start}


def fidelity(target: torch.Tensor, candidate: torch.Tensor) -> dict:
    inner = torch.sum(candidate.conj() * target)
    aligned = candidate * (inner / inner.abs() if inner.abs() > 0 else 1.)
    columns = torch.sum((target - candidate).abs().square(), dim=0).sqrt().numpy()
    overlap = abs(target.mH @ candidate).numpy()**2
    # Descriptive assignment only, never changes transmitted symbols or fits.
    from .stage3b import _maximum_assignment
    permutation, _ = _maximum_assignment(overlap)
    selected = overlap[np.arange(len(target)), permutation]
    return {"native_relative_frobenius": float(matrix_loss(candidate, target).sqrt()),
            "global_phase_relative_frobenius": float(matrix_loss(aligned, target).sqrt()),
            "aligned_squared_overlap": float(selected.mean()),
            "unitarity_residual": float(torch.linalg.matrix_norm(candidate.mH @ candidate -
                                                                  torch.eye(len(candidate)))),
            "column_error_p50": float(np.median(columns)),
            "column_error_p95": float(np.quantile(columns, .95)),
            "column_error_max": float(columns.max()), "column_errors": columns.tolist()}


def operator_discrepancy(target, candidate, operators):
    a, b = target.mH @ operators @ target, candidate.mH @ operators @ candidate
    return ((a - b).abs().square().sum((-2, -1)) /
            a.abs().square().sum((-2, -1))).detach().numpy()
