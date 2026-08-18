"""Constellation, BER, and empirical normalized-MSE utilities."""

from dataclasses import dataclass
import math

import torch

from .channel import COMPLEX_DTYPE, REAL_DTYPE


@dataclass(frozen=True)
class Constellation:
    name: str
    points: torch.Tensor
    bit_labels: torch.Tensor

    @property
    def bits_per_symbol(self) -> int:
        return self.bit_labels.shape[-1]


def _binary_indices(bits: torch.Tensor) -> torch.Tensor:
    width = bits.shape[-1]
    weights = 2 ** torch.arange(width - 1, -1, -1, dtype=torch.int64)
    return torch.sum(bits.to(torch.int64) * weights, dim=-1)


def qpsk_constellation() -> Constellation:
    labels = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64
    )
    return Constellation("QPSK", qpsk_encode(labels), labels)


def qam16_constellation() -> Constellation:
    labels = torch.tensor(
        [
            [(index >> 3) & 1, (index >> 2) & 1, (index >> 1) & 1, index & 1]
            for index in range(16)
        ],
        dtype=torch.int64,
    )
    gray_levels = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
    points = torch.tensor(
        [
            complex(
                gray_levels[(int(bits[0]), int(bits[1]))],
                gray_levels[(int(bits[2]), int(bits[3]))],
            )
            for bits in labels
        ],
        dtype=COMPLEX_DTYPE,
    ) / math.sqrt(10.0)
    return Constellation("16QAM", points, labels)


def constellation_encode(bits: torch.Tensor, constellation: Constellation) -> tuple[torch.Tensor, torch.Tensor]:
    if bits.shape[-1] != constellation.bits_per_symbol:
        raise ValueError(
            f"{constellation.name} requires {constellation.bits_per_symbol} bits per symbol"
        )
    indices = _binary_indices(bits)
    return constellation.points[indices], indices


def nearest_neighbor_decode(
    symbols: torch.Tensor, constellation: Constellation
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.abs(symbols[..., None] - constellation.points)
    indices = torch.argmin(distances, dim=-1)
    return constellation.bit_labels[indices], indices


def nearest_neighbor_margin(
    symbols: torch.Tensor,
    true_indices: torch.Tensor,
    constellation: Constellation,
) -> torch.Tensor:
    distances = torch.abs(symbols[..., None] - constellation.points)
    correct = torch.gather(distances, -1, true_indices[..., None]).squeeze(-1)
    competitor_distances = distances.clone()
    competitor_distances.scatter_(-1, true_indices[..., None], torch.inf)
    competitor = torch.min(competitor_distances, dim=-1).values
    return competitor - correct


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
