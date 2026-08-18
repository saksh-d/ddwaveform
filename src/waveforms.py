"""Classical unitary synthesis transforms used in Stage 0."""

import math

import torch

from . import config
from .channel import COMPLEX_DTYPE, REAL_DTYPE, dft_matrix, make_generator


def identity_waveform(n: int = config.N) -> torch.Tensor:
    """Untransformed time-sample control."""
    return torch.eye(n, dtype=COMPLEX_DTYPE)


def ofdm_waveform(n: int = config.N) -> torch.Tensor:
    return dft_matrix(n).mH


def otfs_waveform(n_tau: int = config.N_TAU, n_nu: int = config.N_NU) -> torch.Tensor:
    """W_OTFS = F_nu^H kron I_tau, with delay index varying fastest."""
    return torch.kron(dft_matrix(n_nu).mH.contiguous(), torch.eye(n_tau, dtype=COMPLEX_DTYPE))


def chirp_diagonal(c: float, n: int = config.N) -> torch.Tensor:
    samples = torch.arange(n, dtype=REAL_DTYPE)
    return torch.diag(torch.exp(-2j * math.pi * c * samples.square()).to(COMPLEX_DTYPE))


def afdm_waveform(
    n: int = config.N,
    c1: float = config.AFDM_K3_TUNED_C1,
    c2: float = config.AFDM_C2,
) -> torch.Tensor:
    lambda_1 = chirp_diagonal(c1, n)
    lambda_2 = chirp_diagonal(c2, n)
    return lambda_1.mH @ dft_matrix(n).mH @ lambda_2.mH


def afdm_k3_tuned_waveform(n: int = config.N) -> torch.Tensor:
    return afdm_waveform(n=n, c1=config.AFDM_K3_TUNED_C1, c2=config.AFDM_C2)


def afdm_canonical_waveform(
    n: int,
    alpha_max: int,
    leakage_guard: int = 0,
    c2: float | None = None,
) -> torch.Tensor:
    """Support-rule AFDM reference; exact path law assumes integer delay/Doppler."""
    c1 = (2 * (alpha_max + leakage_guard) + 1) / (2 * n)
    return afdm_waveform(n=n, c1=c1, c2=config.AFDM_C2 if c2 is None else c2)


def random_unitary(n: int = config.N, seed: int = config.RANDOM_UNITARY_SEED) -> torch.Tensor:
    generator = make_generator(seed)
    z = torch.randn((n, n), dtype=REAL_DTYPE, generator=generator) + 1j * torch.randn(
        (n, n), dtype=REAL_DTYPE, generator=generator
    )
    q, r = torch.linalg.qr(z.to(COMPLEX_DTYPE))
    diagonal = torch.diagonal(r)
    phases = torch.where(torch.abs(diagonal) > 0, diagonal / torch.abs(diagonal), torch.ones_like(diagonal))
    # Q diag(phases) makes the corresponding R diagonal real and nonnegative.
    return q * phases[None, :]


def classical_waveforms(c1: float = config.AFDM_K3_TUNED_C1) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=c1),
    }


def diagnostic_waveforms(c1: float = config.AFDM_K3_TUNED_C1) -> dict[str, torch.Tensor]:
    return {**classical_waveforms(c1), "RANDOM": random_unitary()}
