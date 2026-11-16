"""DD-v1: finite circular fractional delay, negative Doppler, delay first."""
import math
from dataclasses import dataclass
import torch
from .core import REAL, COMPLEX, rng


def dft(n=64):
    k = torch.arange(n, dtype=REAL)
    return torch.exp(-2j * math.pi * torch.outer(k, k) / n) / math.sqrt(n)


def atoms(tau, nu, n=64):
    tau = torch.as_tensor(tau, dtype=REAL)
    nu = torch.as_tensor(nu, dtype=REAL)
    k = torch.arange(n, dtype=REAL)
    f = dft(n)
    pi = f.mH @ (torch.exp(-2j * math.pi * tau[..., None] * k / n)[..., :, None] * f)
    return torch.exp(-2j * math.pi * nu[..., None] * k / n)[..., :, None] * pi


def assemble(tau, nu, gains, n=64):
    gains = torch.as_tensor(gains, dtype=COMPLEX)
    return (gains[..., None, None] * atoms(tau, nu, n)).sum(-3)


@dataclass
class Channels:
    h: torch.Tensor
    raw: torch.Tensor
    tau: torch.Tensor
    nu: torch.Tensor
    gains: torch.Tensor
    scale: torch.Tensor


def sample(count, namespace=("advisor", "export"), n=64, p=3, *,
           delay_range=(0., 4.), doppler_range=(-.75, .75)):
    if count < 1 or n < 1 or p < 1:
        raise ValueError("count, n and p must be positive")
    generator = rng(*namespace)
    tau = delay_range[0] + (delay_range[1] - delay_range[0]) * torch.rand(count, p, dtype=REAL, generator=generator)
    nu = doppler_range[0] + (doppler_range[1] - doppler_range[0]) * torch.rand(count, p, dtype=REAL, generator=generator)
    gains = (torch.randn(count, p, dtype=REAL, generator=generator) +
             1j * torch.randn(count, p, dtype=REAL, generator=generator)) / math.sqrt(2 * p)
    raw = assemble(tau, nu, gains, n)
    scale = (raw.abs().square().sum((-2, -1)) / n).sqrt()
    return Channels(raw / scale[:, None, None], raw, tau, nu, gains, scale)


def classical(n=64, n_tau=8, n_nu=8):
    if n_tau * n_nu != n:
        raise ValueError("OTFS grid product must equal n")
    f = dft(n)
    k = torch.arange(n, dtype=REAL)
    c1, c2 = 1 / (2 * n), math.sqrt(2) / (100 * n)
    l1, l2 = torch.exp(2j * math.pi * c1 * k.square()), torch.exp(2j * math.pi * c2 * k.square())
    return {"OFDM": f.mH.contiguous(),
            "OTFS": torch.kron(dft(n_nu).mH.contiguous(), torch.eye(n_tau, dtype=COMPLEX)),
            "AFDM": l1[:, None] * f.mH * l2[None, :]}


def frequency_paths(tau, nu, gains, scale=None, n=64):
    """Independent finite-sum F H F^H, including complex path coefficients."""
    tau = torch.as_tensor(tau, dtype=REAL)
    nu = torch.as_tensor(nu, dtype=REAL)
    gains = torch.as_tensor(gains, dtype=COMPLEX)
    k = torch.arange(n, dtype=REAL)
    # Explicit finite sum avoids sinc singularities at integer multiples of N.
    offset = nu[..., None, None] + k[:, None] - k[None, :]
    doppler_f = torch.exp(-2j * math.pi * offset[..., None] * k / n).mean(-1)
    delay_f = torch.exp(-2j * math.pi * tau[..., None] * k / n)
    hf = (gains[..., None, None] * doppler_f * delay_f[..., None, :]).sum(-3)
    return hf if scale is None else hf / torch.as_tensor(scale, dtype=REAL)[..., None, None]


def bridge(w, h):
    f = dft(w.shape[-1])
    c, hf = f @ w, f @ h @ f.mH
    return {"C": c, "H_f": hf, "A_spectral": c.mH @ hf @ c,
            "basis_gram": w.mH @ w, "distorted_gram": w.mH @ h.mH @ h @ w}
