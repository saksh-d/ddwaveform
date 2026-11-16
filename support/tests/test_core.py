import math
import torch
from waveforms.core import REAL, COMPLEX, rng, seed
from waveforms.delay_doppler import atoms, bridge, classical, dft, frequency_paths, sample
from waveforms.experiments import ber_point
from waveforms.receivers import constellation, decode, encode, receiver, stream_mse, support
from waveforms.residual_cfo import apply_residual_cfo, effective, operator
from waveforms.supplementary import block_energy, rotate


def test_integer_delay_and_fractional_sign_order():
    n = 8
    x = torch.randn(n, generator=rng("test", "x"), dtype=REAL).to(COMPLEX)
    torch.testing.assert_close(atoms(2., 0., n) @ x, torch.roll(x, 2), atol=3e-15, rtol=1e-12)
    tau, nu = .73, -.41
    k = torch.arange(n, dtype=REAL)
    explicit = torch.diag(torch.exp(-2j * math.pi * nu * k / n)) @ dft(n).mH @ torch.diag(torch.exp(-2j * math.pi * tau * k / n)) @ dft(n)
    torch.testing.assert_close(atoms(tau, nu, n), explicit, atol=3e-15, rtol=1e-12)
    assert torch.linalg.matrix_norm(atoms(tau, nu, n) - atoms(tau, 0, n) @ atoms(0, nu, n)) > .1


def test_path_normalization_and_independent_frequency_sum():
    channels = sample(3, ("test", "paths"), n=8)
    torch.testing.assert_close(channels.h.abs().square().sum((-2, -1)), torch.full((3,), 8.), atol=1e-13, rtol=0)
    expected = dft(8) @ channels.h @ dft(8).mH
    actual = frequency_paths(channels.tau, channels.nu, channels.gains, channels.scale, n=8)
    torch.testing.assert_close(actual, expected, atol=5e-15, rtol=1e-12)


def test_classical_reference_order_and_afdm_parameters():
    ws = classical()
    f = dft()
    torch.testing.assert_close(ws["OTFS"], torch.kron(dft(8).mH.contiguous(), torch.eye(8, dtype=COMPLEX)))
    k = torch.arange(64, dtype=REAL)
    expected = torch.diag(torch.exp(2j * math.pi / 128 * k.square())) @ f.mH @ torch.diag(torch.exp(2j * math.pi * math.sqrt(2) / 6400 * k.square()))
    torch.testing.assert_close(ws["AFDM"], expected)
    for w in ws.values():
        assert torch.linalg.matrix_norm(w.mH @ w - torch.eye(64, dtype=COMPLEX)) < 1e-12


def test_full_mse_invariance_is_separate_from_stream_geometry():
    h = sample(3, ("test", "mse")).h
    means, streams = [], []
    for w in classical().values():
        a = w.mH @ h @ w
        values = stream_mse(a, receiver(a, .03), .03)
        means.append(values.mean(-1))
        streams.append(values)
        b = bridge(w, h)
        torch.testing.assert_close(b["A_spectral"], a, atol=8e-14, rtol=1e-12)
        torch.testing.assert_close(b["distorted_gram"], a.mH @ a, atol=8e-14, rtol=1e-12)
    torch.testing.assert_close(means[0], means[1], atol=2e-15, rtol=0)
    assert (streams[0] - streams[1]).abs().max() > .01


def test_support_stable_ties_and_realized_complex_coefficients():
    scores = torch.ones(4, 4)
    mask = support(scores, 2)
    assert torch.equal(mask, torch.tensor([[True, True, False, False]]).expand(4, 4))
    a = torch.randn(4, 4, dtype=COMPLEX, generator=rng("test", "receiver"))
    g = receiver(a, .1, "fixed", mask, 2)
    model = a * mask
    expected = torch.linalg.solve(model.mH @ model + .1 * torch.eye(4, dtype=COMPLEX), model.mH)
    torch.testing.assert_close(g, expected)
    # True full-A error must retain mismatch; substituting the model is wrong.
    assert (stream_mse(a, g, .1) - stream_mse(model, g, .1)).abs().max() > .01


def test_gray_constellations_unit_energy_and_exact_slicing():
    for mod in ["QPSK", "16QAM"]:
        points, labels = constellation(mod)
        torch.testing.assert_close(points.abs().square().mean(), torch.tensor(1.))
        assert torch.equal(decode(encode(labels, mod), mod), labels)


def test_rng_disjoint_and_replay():
    assert seed("advisor", "ber-smoke", "channel") != seed("global", "structural-channel")
    torch.testing.assert_close(sample(2, ("test", "seed")).h, sample(2, ("test", "seed")).h, atol=0, rtol=0)


def test_physical_pipeline_and_approximate_receiver_counts():
    ws = {"OFDM": classical()["OFDM"]}
    namespace = ("test", "physical")
    errors, _, kinds, ch = ber_point(ws, {}, "QPSK", 12., 2, 2, namespace)
    w = ws["OFDM"]
    bits = torch.randint(0, 2, (2, 2, 64, 2), generator=rng(*namespace, "bits", "QPSK", 12.))
    x = encode(bits, "QPSK")
    gen = rng(*namespace, "noise", "QPSK", 12.)
    variance = 10**(-1.2)
    noise = math.sqrt(variance / 2) * (torch.randn(x.shape, generator=gen) + 1j * torch.randn(x.shape, generator=gen))
    a = w.mH @ ch.h @ w
    z = x @ a.transpose(-1, -2) + noise @ w.conj()
    for ri, kind in enumerate(kinds):
        g = receiver(a, variance, kind)
        expected = (decode(z @ g.transpose(-1, -2), "QPSK") != bits).sum((1, 2, 3))
        assert torch.equal(expected, errors[0, ri])


def test_cfo_sign_orientation_and_colored_analysis_noise():
    w = torch.randn(8, 5, dtype=torch.complex64, generator=rng("test", "cfo-w"))
    r = torch.randn(5, 8, dtype=torch.complex64, generator=rng("test", "cfo-r"))
    eps = torch.tensor([-.1, 0., .1], dtype=torch.float32)
    a = effective(w, r, eps)
    phase = torch.exp(2j * math.pi * eps[:, None] * torch.arange(8, dtype=torch.float32) / 8)
    legacy = torch.einsum("mk,bm,nm->bkn", w, phase, r)
    torch.testing.assert_close(a.transpose(-1, -2), legacy, atol=2e-6, rtol=2e-6)
    symbols = torch.randn(3, 5, dtype=torch.complex64)
    direct = apply_residual_cfo(symbols @ w.T, eps) @ r.T
    predicted = torch.einsum("bi,bji->bj", symbols, a)
    torch.testing.assert_close(direct, predicted, atol=1e-5, rtol=2e-6)
    assert operator(.1)[1, 1].imag > 0
    assert torch.linalg.matrix_norm(r @ r.mH - torch.eye(5)) > 1


def test_within_group_algebra_does_not_fix_topk():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    w = dft(8).mH
    z = torch.randn(8, 8, dtype=COMPLEX, generator=rng("test", "rotation"))
    s = (z - z.mH) * (labels[:, None] == labels[None, :])
    s = s / torch.linalg.matrix_norm(s)
    wp = rotate(w, s, .5)
    h = sample(2, ("test", "rotation-channel"), n=8).h
    a, ap = w.mH @ h @ w, wp.mH @ h @ wp
    torch.testing.assert_close(block_energy(a, labels), block_energy(ap, labels), atol=1e-13, rtol=1e-12)
    assert (a.abs().square() - ap.abs().square()).abs().max() > .01


def test_equal_dft_power_needs_complex_phases():
    f = dft(8)
    w = torch.eye(8, dtype=COMPLEX)
    c = f @ w
    cp = torch.exp(1j * torch.linspace(-2., 1., 8))[:, None] * c
    wp = f.mH @ cp
    torch.testing.assert_close(c.abs().square(), cp.abs().square(), atol=1e-15, rtol=1e-12)
    delay = atoms(1.2, 0., 8)
    torch.testing.assert_close(w.mH @ delay @ w, wp.mH @ delay @ wp, atol=5e-15, rtol=1e-12)
    joint = atoms(1.2, .43, 8)
    assert torch.linalg.matrix_norm(w.mH @ joint @ w - wp.mH @ joint @ wp) > .1
