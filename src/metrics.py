"""QPSK, BER, and empirical normalized-MSE utilities."""

import math

import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE


def random_bits(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, shape, dtype=torch.int64, generator=generator)


def qpsk_encode(bits: torch.Tensor) -> torch.Tensor:
    if bits.shape[-1] != 2:
        raise ValueError("QPSK bits must have final dimension 2")
    bits_real = bits.to(REAL_DTYPE)
    return ((1.0 - 2.0 * bits_real[..., 0]) + 1j * (1.0 - 2.0 * bits_real[..., 1])).to(
        COMPLEX_DTYPE
    ) / math.sqrt(2.0)


def qpsk_decode(symbols: torch.Tensor) -> torch.Tensor:
    return torch.stack((symbols.real < 0, symbols.imag < 0), dim=-1).to(torch.int64)


def bit_error_rate(reference_bits: torch.Tensor, estimated_symbols: torch.Tensor) -> float:
    return torch.mean((qpsk_decode(estimated_symbols) != reference_bits).to(REAL_DTYPE)).item()


def normalized_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return (torch.sum(torch.abs(estimate - reference) ** 2) / torch.sum(torch.abs(reference) ** 2)).item()
