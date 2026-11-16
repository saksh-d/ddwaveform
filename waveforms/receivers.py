"""Receiver support changes the model; observations always retain complete A."""
import math
import torch
from .core import REAL, COMPLEX


def support(scores, k=24):
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError("Invalid row support budget")
    order = torch.argsort(scores.detach(), dim=-1, descending=True, stable=True)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, order[..., :k], True)


def receiver(a, variance, kind="full", mask=None, k=24):
    if kind == "full":
        model = a
    elif kind == "realized":
        model = a * support(a.abs(), k)
    elif kind == "fixed" and mask is not None:
        model = a * mask
    else:
        raise ValueError("Unknown receiver or missing discovery-only mask")
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    v = torch.as_tensor(variance, dtype=a.real.dtype, device=a.device)
    return torch.linalg.solve(model.mH @ model + v[..., None, None] * eye, model.mH)


def stream_mse(a, g, variance):
    eye = torch.eye(a.shape[-1], dtype=a.dtype)
    return (g @ a - eye).abs().square().sum(-1) + variance * g.abs().square().sum(-1)


def constellation(mod):
    if mod not in ("QPSK", "16QAM"):
        raise ValueError("Only QPSK and 16QAM are qualified")
    b = 2 if mod == "QPSK" else 4
    labels = ((torch.arange(2**b)[:, None] >> torch.arange(b - 1, -1, -1)) & 1)
    if b == 2:
        points = ((1 - 2 * labels[:, 0]) + 1j * (1 - 2 * labels[:, 1])) / math.sqrt(2)
    else:
        levels = torch.tensor([-3., -1., 3., 1.], dtype=REAL)
        points = (levels[labels[:, 0] * 2 + labels[:, 1]] +
                  1j * levels[labels[:, 2] * 2 + labels[:, 3]]) / math.sqrt(10)
    return points.to(COMPLEX), labels


def encode(bits, mod):
    points, _ = constellation(mod)
    indices = (bits * 2 ** torch.arange(bits.shape[-1] - 1, -1, -1)).sum(-1)
    return points[indices]


def decode(x, mod):
    points, labels = constellation(mod)
    return labels[(x[..., None] - points).abs().argmin(-1)]
