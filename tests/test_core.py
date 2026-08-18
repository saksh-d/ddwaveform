import math

import pytest
import torch

from src import config
from src.channel import (
    COMPLEX_DTYPE,
    assemble_dd_channel,
    complex_awgn,
    dft_matrix,
    doppler_operator,
    fractional_delay_operator,
    make_generator,
    noise_variance,
    sample_dd_v1,
)
from src.metrics import qpsk_decode, qpsk_encode, random_bits
from src.receiver import effective_operator, finite_budget_receiver, full_information_mse, rowwise_topk
from src.waveforms import diagnostic_waveforms, random_unitary


ATOL = 1e-10
RTOL = 1e-9


def test_dft_unitarity() -> None:
    f = dft_matrix()
    assert torch.allclose(f.mH @ f, torch.eye(config.N, dtype=COMPLEX_DTYPE), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("delay", [0, 1, 4, config.N - 1])
def test_integer_delay_matches_circular_shift(delay: int) -> None:
    generator = make_generator(1)
    x = torch.randn(config.N, dtype=torch.float64, generator=generator).to(COMPLEX_DTYPE)
    assert torch.allclose(fractional_delay_operator(delay) @ x, torch.roll(x, delay), atol=ATOL, rtol=RTOL)


def test_fractional_delay_unitarity_and_composition() -> None:
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    for tau in (0.25, 1.4, 3.75):
        pi = fractional_delay_operator(tau)
        assert torch.allclose(pi.mH @ pi, identity, atol=ATOL, rtol=RTOL)
    lhs = fractional_delay_operator(0.25) @ fractional_delay_operator(1.4)
    assert torch.allclose(lhs, fractional_delay_operator(1.65), atol=ATOL, rtol=RTOL)


def test_doppler_unitarity_and_composition() -> None:
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    delta = doppler_operator(-0.37)
    assert torch.allclose(delta.mH @ delta, identity, atol=ATOL, rtol=RTOL)
    assert torch.allclose(
        doppler_operator(0.2) @ doppler_operator(-0.7),
        doppler_operator(-0.5),
        atol=ATOL,
        rtol=RTOL,
    )


def test_dd_ordering_and_sign_convention() -> None:
    # With circular delay, the global scalar Weyl relation is exact for integer
    # Doppler bins. Fractional nu has a wrap-boundary phase discontinuity.
    nu, ell = 3.0, 4
    delta = doppler_operator(nu)
    pi = fractional_delay_operator(ell)
    phase = torch.exp(torch.tensor(-2j * math.pi * nu * ell / config.N, dtype=COMPLEX_DTYPE))
    assert torch.allclose(delta @ pi, phase * (pi @ delta), atol=ATOL, rtol=RTOL)


def test_multipath_assembly_is_constituent_sum() -> None:
    delays = torch.tensor([0.25, 2.0], dtype=torch.float64)
    dopplers = torch.tensor([-0.4, 0.6], dtype=torch.float64)
    gains = torch.tensor([0.3 + 0.2j, -0.1 + 0.7j], dtype=COMPLEX_DTYPE)
    expected = sum(
        gains[p] * doppler_operator(dopplers[p]) @ fractional_delay_operator(delays[p])
        for p in range(2)
    )
    assert torch.allclose(assemble_dd_channel(delays, dopplers, gains), expected, atol=ATOL, rtol=RTOL)


def test_dd_v1_support_fractionality_and_normalization() -> None:
    batch = sample_dd_v1(32, config.GLOBAL_SEED)
    assert torch.all((batch.delays >= 0.0) & (batch.delays <= 4.0))
    assert torch.all((batch.dopplers >= -0.75) & (batch.dopplers <= 0.75))
    assert torch.any(batch.delays != torch.round(batch.delays))
    assert torch.any(batch.dopplers != torch.round(batch.dopplers))
    energies = torch.sum(torch.abs(batch.matrices) ** 2, dim=(-2, -1)) / config.N
    assert torch.allclose(energies, torch.ones_like(energies), atol=ATOL, rtol=RTOL)


def test_dd_v1_reproducibility() -> None:
    first = sample_dd_v1(3, 41)
    second = sample_dd_v1(3, 41)
    different = sample_dd_v1(3, 42)
    assert torch.equal(first.delays, second.delays)
    assert torch.equal(first.dopplers, second.dopplers)
    assert torch.equal(first.pre_normalization_gains, second.pre_normalization_gains)
    assert torch.equal(first.matrices, second.matrices)
    assert not torch.equal(first.matrices, different.matrices)


def test_awgn_scaling() -> None:
    sigma2 = noise_variance(15.0)
    noise = complex_awgn((200_000,), sigma2, make_generator(10))
    empirical = torch.mean(torch.abs(noise) ** 2)
    assert torch.allclose(empirical, sigma2, rtol=0.01, atol=0.0)


def test_qpsk_round_trip_and_energy() -> None:
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    symbols = qpsk_encode(bits)
    assert torch.equal(qpsk_decode(symbols), bits)
    assert torch.allclose(torch.abs(symbols).square(), torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("name", ["IDENTITY", "OFDM", "OTFS", "AFDM", "RANDOM"])
def test_waveform_unitarity_power_and_identity_channel(name: str) -> None:
    waveform = diagnostic_waveforms()[name]
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    generator = make_generator(54)
    vector = torch.randn(config.N, dtype=torch.float64, generator=generator).to(COMPLEX_DTYPE)
    assert torch.allclose(waveform.mH @ waveform, identity, atol=ATOL, rtol=RTOL)
    assert torch.allclose(torch.linalg.vector_norm(waveform @ vector), torch.linalg.vector_norm(vector), atol=ATOL, rtol=RTOL)
    assert torch.allclose(effective_operator(identity, waveform), identity, atol=ATOL, rtol=RTOL)


def test_random_unitary_is_seed_deterministic() -> None:
    assert torch.equal(random_unitary(seed=88), random_unitary(seed=88))
    assert not torch.equal(random_unitary(seed=88), random_unitary(seed=89))


def test_rowwise_topk_count_ties_and_full_identity() -> None:
    matrix = torch.tensor([[1 + 0j, -1 + 0j, 0.5 + 0j], [0 + 0j, 2 + 0j, 3 + 0j]], dtype=COMPLEX_DTYPE)
    truncated = rowwise_topk(matrix, 1)
    assert torch.count_nonzero(truncated, dim=-1).tolist() == [1, 1]
    assert truncated[0, 0] == matrix[0, 0]
    assert torch.equal(rowwise_topk(matrix, matrix.shape[-1]), matrix)


def test_physical_chain_matches_effective_operator() -> None:
    channel = sample_dd_v1(1, 72).matrices[0]
    waveform = diagnostic_waveforms()["OTFS"]
    symbols = qpsk_encode(random_bits((config.N, 2), make_generator(73)))
    physical = waveform.mH @ (channel @ (waveform @ symbols))
    equivalent = effective_operator(channel, waveform) @ symbols
    assert torch.allclose(physical, equivalent, atol=ATOL, rtol=RTOL)


def test_analytic_receiver_mse_matches_small_monte_carlo() -> None:
    channel = sample_dd_v1(1, 81).matrices[0]
    waveform = diagnostic_waveforms()["OFDM"]
    a = effective_operator(channel, waveform)
    sigma2 = noise_variance(15.0)
    _, equalizer, analytic = finite_budget_receiver(a, sigma2, 3)
    generator = make_generator(82)
    bits = random_bits((2048, config.N, 2), generator)
    symbols = qpsk_encode(bits)
    noise = complex_awgn((2048, config.N), sigma2, generator)
    observations = symbols @ a.T + noise
    estimates = observations @ equalizer.T
    empirical = torch.mean(torch.abs(estimates - symbols) ** 2)
    assert torch.allclose(empirical, analytic, rtol=0.04, atol=0.0)


def test_full_lmmse_cross_waveform_mse_invariance() -> None:
    channel = sample_dd_v1(1, 91).matrices[0]
    sigma2 = noise_variance(18.0)
    bound = full_information_mse(channel, sigma2)
    values = []
    for waveform in diagnostic_waveforms().values():
        a = effective_operator(channel, waveform)
        _, _, mse = finite_budget_receiver(a, sigma2, config.N)
        values.append(mse)
        assert torch.allclose(mse, bound, atol=1e-10, rtol=1e-9)
    assert torch.max(torch.abs(torch.stack(values) - values[0])) <= 1e-10
