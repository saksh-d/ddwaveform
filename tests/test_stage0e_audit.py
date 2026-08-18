import torch

from src import config
from src.audit import (
    canonical_afdm_c1,
    direct_afdm_idaft,
    expected_afdm_column_offset,
    otfs_reference_demodulate,
    otfs_reference_modulate,
    retained_energy,
    serialize_columns,
)
from src.channel import COMPLEX_DTYPE, complex_awgn, doppler_operator, fractional_delay_operator, make_generator, noise_variance
from src.metrics import bit_error_rate, qpsk_encode, random_bits
from src.receiver import effective_operator, finite_budget_receiver
from src.waveforms import afdm_waveform, classical_waveforms, otfs_waveform


def test_identity_replaces_time_label() -> None:
    names = classical_waveforms()
    assert "IDENTITY" in names
    assert "TIME" not in names


def test_two_stage_otfs_matches_kronecker_matrix() -> None:
    generator = make_generator(201)
    dd = (
        torch.randn((config.N_TAU, config.N_NU), dtype=torch.float64, generator=generator)
        + 1j * torch.randn((config.N_TAU, config.N_NU), dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    reference = otfs_reference_modulate(dd)
    matrix_result = otfs_waveform() @ serialize_columns(dd)
    assert torch.allclose(reference, matrix_result, atol=1e-12, rtol=1e-10)


def test_otfs_clean_modulate_demodulate_reconstruction() -> None:
    generator = make_generator(202)
    dd = (
        torch.randn((8, 8), dtype=torch.float64, generator=generator)
        + 1j * torch.randn((8, 8), dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    reconstructed = otfs_reference_demodulate(otfs_reference_modulate(dd), 8, 8)
    assert torch.allclose(reconstructed, dd, atol=1e-12, rtol=1e-10)


def test_direct_afdm_idaft_matches_factored_matrix() -> None:
    generator = make_generator(203)
    symbols = (
        torch.randn(config.N, dtype=torch.float64, generator=generator)
        + 1j * torch.randn(config.N, dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    direct = direct_afdm_idaft(symbols, config.AFDM_C1, config.AFDM_C2)
    factored = afdm_waveform() @ symbols
    assert torch.allclose(direct, factored, atol=1e-12, rtol=1e-10)


def test_canonical_afdm_parameter_rule() -> None:
    assert canonical_afdm_c1(64, alpha_max=2) == 5 / 128
    assert canonical_afdm_c1(64, alpha_max=1, leakage_guard=1) == 5 / 128
    assert canonical_afdm_c1(256, alpha_max=2) == 5 / 512


def test_canonical_afdm_integer_path_support_location() -> None:
    c1 = canonical_afdm_c1(config.N, alpha_max=2)
    waveform = afdm_waveform(c1=c1)
    rows = torch.arange(config.N)
    for delay, doppler in ((0, -2), (1, 0), (2, 2), (4, -1)):
        path = doppler_operator(doppler) @ fractional_delay_operator(delay)
        effective = effective_operator(path, waveform)
        observed = (torch.argmax(torch.abs(effective), dim=-1) - rows) % config.N
        expected = expected_afdm_column_offset(delay, doppler, config.N, c1)
        assert torch.all(observed == expected)
        assert torch.allclose(
            torch.max(torch.abs(effective), dim=-1).values,
            torch.ones(config.N, dtype=torch.float64),
            atol=1e-12,
        )


def test_full_information_ber_improves_strongly_with_snr() -> None:
    n = config.N
    diagonal = torch.linspace(0.8, 1.2, n, dtype=torch.float64).to(COMPLEX_DTYPE)
    channel = torch.diag(diagonal)
    waveform = classical_waveforms()["OFDM"]
    a = effective_operator(channel, waveform)
    generator = make_generator(204)
    bits = random_bits((4096, n, 2), generator)
    symbols = qpsk_encode(bits)
    bers = []
    for snr_db in (0.0, 20.0):
        sigma2 = noise_variance(snr_db)
        noise = complex_awgn((4096, n), sigma2, generator)
        _, equalizer, _ = finite_budget_receiver(a, sigma2, n)
        estimates = (symbols @ a.T + noise) @ equalizer.T
        bers.append(bit_error_rate(bits, estimates))
    assert bers[0] > 0.1
    assert bers[1] < 1e-4


def test_retained_energy_partition_monotonicity_and_full_value() -> None:
    generator = make_generator(205)
    matrix = (
        torch.randn((3, 8, 8), dtype=torch.float64, generator=generator)
        + 1j * torch.randn((3, 8, 8), dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    previous = torch.zeros(3, dtype=torch.float64)
    for k in (1, 2, 3, 5, 8):
        eta, epsilon = retained_energy(matrix, k)
        assert torch.allclose(eta + epsilon, torch.ones_like(eta), atol=1e-12)
        assert torch.all(eta >= previous - 1e-15)
        previous = eta
    assert torch.allclose(previous, torch.ones_like(previous), atol=1e-12)
