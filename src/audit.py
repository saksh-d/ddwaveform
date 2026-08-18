"""Independent transform references and diagnostics for the Stage 0E audit."""

from __future__ import annotations

import math

import torch

from . import config
from .channel import COMPLEX_DTYPE, REAL_DTYPE, ChannelBatch, assemble_dd_channel, dft_matrix, make_generator
from .receiver import rowwise_topk


def serialize_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Column-major serialization: the first matrix index varies fastest."""
    return matrix.mT.contiguous().reshape(*matrix.shape[:-2], -1)


def deserialize_columns(vector: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    return vector.reshape(*vector.shape[:-1], columns, rows).mT


def otfs_reference_modulate(dd_symbols: torch.Tensor) -> torch.Tensor:
    """Two-stage DD->TF->OFDM construction, serialized column-major."""
    m, n_time = dd_symbols.shape[-2:]
    f_m = dft_matrix(m)
    f_time = dft_matrix(n_time)
    tf_symbols = f_m @ dd_symbols @ f_time.mH
    time_samples = f_m.mH @ tf_symbols
    return serialize_columns(time_samples)


def otfs_reference_demodulate(samples: torch.Tensor, m: int, n_time: int) -> torch.Tensor:
    time_samples = deserialize_columns(samples, m, n_time)
    f_m = dft_matrix(m)
    f_time = dft_matrix(n_time)
    tf_symbols = f_m @ time_samples
    return f_m.mH @ tf_symbols @ f_time


def direct_afdm_idaft(symbols: torch.Tensor, c1: float, c2: float) -> torch.Tensor:
    """Direct IDAFT sum, independent of the factored AFDM matrix implementation."""
    n = symbols.shape[-1]
    output_index = torch.arange(n, dtype=REAL_DTYPE)
    input_index = torch.arange(n, dtype=REAL_DTYPE)
    kernel = torch.exp(
        2j
        * math.pi
        * (
            c1 * output_index[:, None].square()
            + output_index[:, None] * input_index[None, :] / n
            + c2 * input_index[None, :].square()
        )
    ).to(COMPLEX_DTYPE) / math.sqrt(n)
    return symbols @ kernel.mT if symbols.ndim > 1 else kernel @ symbols


def canonical_afdm_c1(n: int, alpha_max: int, leakage_guard: int = 0) -> float:
    if alpha_max < 0 or leakage_guard < 0:
        raise ValueError("alpha_max and leakage_guard must be nonnegative")
    return (2 * (alpha_max + leakage_guard) + 1) / (2 * n)


def expected_afdm_column_offset(delay: int, doppler: int, n: int, c1: float) -> int:
    slope = round(2 * n * c1)
    if not math.isclose(2 * n * c1, slope, abs_tol=1e-12):
        raise ValueError("support law requires integer 2*N*c1")
    return (slope * delay + doppler) % n


def retained_energy(channel_operator: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    retained = rowwise_topk(channel_operator, k)
    denominator = torch.sum(torch.abs(channel_operator) ** 2, dim=(-2, -1))
    eta = torch.sum(torch.abs(retained) ** 2, dim=(-2, -1)) / denominator
    epsilon = torch.sum(torch.abs(channel_operator - retained) ** 2, dim=(-2, -1)) / denominator
    return eta, epsilon


def per_row_retained_energy(channel_operator: torch.Tensor, k: int) -> torch.Tensor:
    retained = rowwise_topk(channel_operator, k)
    denominator = torch.sum(torch.abs(channel_operator) ** 2, dim=-1)
    numerator = torch.sum(torch.abs(retained) ** 2, dim=-1)
    return torch.where(denominator > 0, numerator / denominator, torch.ones_like(denominator))


def normalize_operators(raw: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.sqrt(torch.sum(torch.abs(raw) ** 2, dim=(-2, -1)) / n)
    return raw / scales[..., None, None], scales


def _complex_gains(count: int, p: int, generator: torch.Generator) -> torch.Tensor:
    scale = math.sqrt(1.0 / (2 * p))
    return (
        scale
        * (
            torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
            + 1j * torch.randn((count, p), dtype=REAL_DTYPE, generator=generator)
        )
    ).to(COMPLEX_DTYPE)


def sample_grid_dd(
    count: int,
    seed: int,
    n: int = config.N,
    p: int = config.P,
    max_delay: int = 4,
    max_doppler: int = 2,
) -> ChannelBatch:
    generator = make_generator(seed)
    pairs = torch.tensor(
        [(delay, doppler) for delay in range(max_delay + 1) for doppler in range(-max_doppler, max_doppler + 1)],
        dtype=REAL_DTYPE,
    )
    selected = torch.stack([pairs[torch.randperm(len(pairs), generator=generator)[:p]] for _ in range(count)])
    delays = selected[..., 0]
    dopplers = selected[..., 1]
    gains = _complex_gains(count, p, generator)
    raw = assemble_dd_channel(delays, dopplers, gains, n)
    matrices, scales = normalize_operators(raw, n)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def sample_fractional_doppler(
    count: int,
    seed: int,
    n: int = config.N,
    p: int = config.P,
    max_delay: int = 4,
    nu_max: float = 0.75,
) -> ChannelBatch:
    generator = make_generator(seed)
    delays = torch.randint(0, max_delay + 1, (count, p), generator=generator).to(REAL_DTYPE)
    dopplers = (2 * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator) - 1) * nu_max
    gains = _complex_gains(count, p, generator)
    raw = assemble_dd_channel(delays, dopplers, gains, n)
    matrices, scales = normalize_operators(raw, n)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def sample_continuous_dd(
    count: int,
    seed: int,
    nu_max: float,
    n: int = config.N,
    p: int = config.P,
    delay_max: float = 4.0,
) -> ChannelBatch:
    generator = make_generator(seed)
    delays = delay_max * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator)
    dopplers = (2 * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator) - 1) * nu_max
    gains = _complex_gains(count, p, generator)
    raw = assemble_dd_channel(delays, dopplers, gains, n)
    matrices, scales = normalize_operators(raw, n)
    return ChannelBatch(matrices, delays, dopplers, gains, scales)


def sample_classical_reference(
    count: int,
    seed: int,
    n: int = 256,
    p: int = 3,
    max_delay: int = 2,
    max_doppler: float = 2.0,
) -> ChannelBatch:
    """Raw-gain N=256 reference family; deliberately no per-realization normalization."""
    generator = make_generator(seed)
    delays = torch.stack([torch.randperm(max_delay + 1, generator=generator)[:p] for _ in range(count)]).to(REAL_DTYPE)
    theta = (2 * torch.rand((count, p), dtype=REAL_DTYPE, generator=generator) - 1) * math.pi
    dopplers = max_doppler * torch.cos(theta)
    gains = _complex_gains(count, p, generator)
    matrices = assemble_dd_channel(delays, dopplers, gains, n)
    return ChannelBatch(matrices, delays, dopplers, gains, torch.ones(count, dtype=REAL_DTYPE))
