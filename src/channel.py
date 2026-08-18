"""Discrete DD-v1 channel operators using the repository-wide sign convention."""

from dataclasses import dataclass
import math

import torch

from . import config

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128


@dataclass(frozen=True)
class ChannelBatch:
    matrices: torch.Tensor
    delays: torch.Tensor
    dopplers: torch.Tensor
    pre_normalization_gains: torch.Tensor
    normalization_scales: torch.Tensor


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def dft_matrix(n: int = config.N) -> torch.Tensor:
    """Unitary F[k,n] = exp(-j 2 pi k n / N) / sqrt(N)."""
    indices = torch.arange(n, dtype=REAL_DTYPE)
    phase = -2j * math.pi * torch.outer(indices, indices) / n
    return torch.exp(phase).to(COMPLEX_DTYPE) / math.sqrt(n)


def fractional_delay_operator(tau: float | torch.Tensor, n: int = config.N) -> torch.Tensor:
    """Circular fractional delay Pi(tau) = F^H diag(exp(-j2pi k tau/N)) F."""
    tau_tensor = torch.as_tensor(tau, dtype=REAL_DTYPE)
    frequencies = torch.arange(n, dtype=REAL_DTYPE)
    phases = torch.exp(-2j * math.pi * tau_tensor[..., None] * frequencies / n).to(COMPLEX_DTYPE)
    f = dft_matrix(n)
    return f.mH @ (phases[..., :, None] * f)


def doppler_operator(nu: float | torch.Tensor, n: int = config.N) -> torch.Tensor:
    """Delta(nu) = diag(exp(-j 2 pi nu n/N)); the negative sign is fixed."""
    nu_tensor = torch.as_tensor(nu, dtype=REAL_DTYPE)
    samples = torch.arange(n, dtype=REAL_DTYPE)
    diagonal = torch.exp(-2j * math.pi * nu_tensor[..., None] * samples / n).to(COMPLEX_DTYPE)
    return torch.diag_embed(diagonal)


def assemble_dd_channel(
    delays: torch.Tensor,
    dopplers: torch.Tensor,
    gains: torch.Tensor,
    n: int = config.N,
) -> torch.Tensor:
    """Assemble sum_p h_p Delta(nu_p) Pi(tau_p): delay acts first."""
    delays = torch.as_tensor(delays, dtype=REAL_DTYPE)
    dopplers = torch.as_tensor(dopplers, dtype=REAL_DTYPE)
    gains = torch.as_tensor(gains, dtype=COMPLEX_DTYPE)
    single = delays.ndim == 1
    if single:
        delays, dopplers, gains = delays[None], dopplers[None], gains[None]
    delay_ops = fractional_delay_operator(delays, n)
    samples = torch.arange(n, dtype=REAL_DTYPE)
    doppler_diagonals = torch.exp(
        -2j * math.pi * dopplers[..., None] * samples / n
    ).to(COMPLEX_DTYPE)
    path_ops = doppler_diagonals[..., :, None] * delay_ops
    matrices = torch.sum(gains[..., None, None] * path_ops, dim=1)
    return matrices[0] if single else matrices


def sample_dd_v1(
    count: int,
    seed: int,
    n: int = config.N,
    p: int = config.P,
) -> ChannelBatch:
    """Sample normalized DD-v1 channels with continuous iid pre-normalization parameters."""
    generator = make_generator(seed)
    delays = config.DELAY_MIN + (config.DELAY_MAX - config.DELAY_MIN) * torch.rand(
        (count, p), dtype=REAL_DTYPE, generator=generator
    )
    dopplers = config.DOPPLER_MIN + (config.DOPPLER_MAX - config.DOPPLER_MIN) * torch.rand(
        (count, p), dtype=REAL_DTYPE, generator=generator
    )
    gain_scale = math.sqrt(1.0 / (2 * p))
    gains = gain_scale * (
        torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    raw = assemble_dd_channel(delays, dopplers, gains, n)
    scales = torch.sqrt(torch.sum(torch.abs(raw) ** 2, dim=(-2, -1)) / n)
    normalized = raw / scales[:, None, None]
    return ChannelBatch(normalized, delays, dopplers, gains, scales)


def noise_variance(snr_db: float | torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, -torch.as_tensor(snr_db, dtype=REAL_DTYPE) / 10.0)


def complex_awgn(shape: tuple[int, ...], sigma2: float | torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    component_std = torch.sqrt(torch.as_tensor(sigma2, dtype=REAL_DTYPE) / 2.0)
    real = torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
    imag = torch.randn(shape, dtype=REAL_DTYPE, generator=generator)
    return (component_std * (real + 1j * imag)).to(COMPLEX_DTYPE)
