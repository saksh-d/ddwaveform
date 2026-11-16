import json
import numpy as np
import pandas as pd
import pytest
import torch
from waveforms.artifacts import load_reference, sha, verify, EXPECTED_W
from waveforms.delay_doppler import assemble, bridge, classical, dft, frequency_paths
from waveforms.receivers import support
from waveforms.residual_cfo import effective, load_qualified

pytestmark = pytest.mark.artifacts


def test_all_manifest_files(artifacts):
    manifest = json.loads((artifacts / "LOCAL_MANIFEST.json").read_text())
    assert verify(artifacts, manifest)["files"] > 100
    assert sha(artifacts / "reference/learned_w.npy") == EXPECTED_W


def test_learned_orientation_exact_channel_and_bridge(artifacts):
    w = load_reference(artifacts)
    data = np.load(artifacts / "focused/data/channels.npz", allow_pickle=False)
    bases = np.load(artifacts / "focused/data/bases.npz", allow_pickle=False)
    torch.testing.assert_close(w, torch.from_numpy(bases["Learned"]), atol=0, rtol=0)
    h = torch.from_numpy(data["H"])
    raw = assemble(data["tau"], data["nu"], data["gains"])
    torch.testing.assert_close(raw / float(data["normalization"]), h, atol=1e-14, rtol=1e-12)
    a = w.mH @ h @ w
    torch.testing.assert_close(a, torch.from_numpy(data["Learned_example"]), atol=1e-14, rtol=1e-12)
    b = bridge(w, h)
    hf = frequency_paths(data["tau"], data["nu"], data["gains"], data["normalization"])
    torch.testing.assert_close(hf, b["H_f"], atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(b["A_spectral"], a, atol=1e-13, rtol=1e-11)
    gram = w.mH @ w
    off = gram - torch.diag(gram.diagonal())
    assert off.abs().max() < 2e-15
    assert torch.linalg.matrix_norm(gram - torch.eye(64)) < 3e-14
    assert sorted(bases["permutation"].tolist()) == list(range(64))
    assert np.bincount(bases["labels"]).tolist() == [23, 21, 20]
    controls = classical()
    for name in ["OTFS", "AFDM"]:
        torch.testing.assert_close(controls[name], torch.from_numpy(bases[name]), atol=1e-14, rtol=1e-12)


def test_saved_support_positions_and_counts(artifacts):
    source = np.load(artifacts / "focused/data/supports.npz", allow_pickle=False)
    for name in ["Learned", "OTFS", "AFDM"]:
        expected = support(torch.from_numpy(source[name + "_q"]), 24)
        assert np.array_equal(expected.numpy(), source[name + "_mask"])
        assert (source[name + "_mask"].sum(-1) == 24).all()


def test_every_completed_focused_count_matches_summary(artifacts):
    table = pd.read_csv(artifacts / "focused/ber_summary.csv")
    complete = sorted((artifacts / "focused/communication").glob("*/complete.json"))
    assert len(complete) == 27
    for path in complete:
        record = json.loads(path.read_text())
        data = np.load(path.parent / "per_channel.npz", allow_pickle=False)
        errors = data["errors"]
        assert errors.shape == (3, 2, 2, record["channels"])
        assert np.array_equal(errors.sum(-1), record["errors"])
        assert record["bits"] == record["channels"] * int(data["bits_per_channel"])
        for wi, waveform in enumerate(["Learned", "OTFS", "AFDM"]):
            for ri, receiver in enumerate(["fixed", "full"]):
                for si, slicer in enumerate(["raw", "gain"]):
                    row = table[(table.modulation == record["modulation"]) & (table.snr_db == record["snr_db"]) & (table.waveform == waveform) & (table.receiver == receiver) & (table.slicer == slicer)].iloc[0]
                    assert int(row.errors) == int(errors[wi, ri, si].sum())
                    assert abs(row.ber - row.errors / row.bits) < 1e-15
    # Preserve the unfavorable full-QPSK ordering at 25dB.
    rows = table[(table.modulation == "QPSK") & (table.snr_db == 25) & (table.receiver == "full") & (table.slicer == "raw")].set_index("waveform")
    assert rows.loc["Learned", "ber"] > rows.loc["OTFS", "ber"]


def test_real_cfo_checkpoint_and_stored_operator_diagnostics(artifacts):
    w, r, checks = load_qualified(artifacts)
    assert checks["clean_identity_loss"] < .001
    assert not torch.allclose(r, w.mH, atol=1e-3, rtol=0)
    table = pd.read_csv(artifacts / "residual_cfo/operator_diagnostics.csv")
    table = table[table.method == "Learned"]
    # Stored operator diagnostics provide a second check away from clean identity.
    for _, row in table.iterrows():
        a = effective(w, r, row.eps)
        power = a.abs().square()
        ratio = float((power.sum() - power.diagonal().sum()).clamp_min(0) / power.sum())
        assert abs(ratio - row.offdiag_leakage) < 3e-7
