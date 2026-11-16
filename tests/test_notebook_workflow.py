from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
import waveforms.notebook_workflow as workflow

def make_workflow(config, artifacts, workspace, save_run):
    return workflow.NotebookWorkflow(config, artifacts, workspace, save_run,
        selected_basis=workflow.select_basis("load_existing", artifacts))


MODEL = dict(workflow.NOMINAL)
DISPLAY = {"basis_columns": (1, 5), "snr_max_db": 25.}
FRESH = {"seed": 321, "matrix_channels": 3, "matrix_index": 1,
         "ber_channels": 4, "blocks_per_channel": 3, "snrs_db": (0., 12.)}


@pytest.mark.parametrize("key,value", [("n", 32), ("paths", 2), ("delay_range", (0., 1.)), ("doppler_range", (-.1, .1))])
def test_saved_evidence_rejects_model_changes(key, value):
    model = {**MODEL, key: value}
    with pytest.raises(ValueError, match="Saved BER|N=64"):
        workflow.validate_config("saved_replay", model, DISPLAY)


def test_unsupported_or_unused_configuration_rejected():
    with pytest.raises(ValueError, match="FRESH must be None"):
        workflow.validate_config("saved_replay", MODEL, DISPLAY, FRESH)
    with pytest.raises(ValueError, match="EVIDENCE_MODE"):
        workflow.validate_config("train", MODEL, DISPLAY)
    with pytest.raises(ValueError, match="Unsupported"):
        workflow.validate_config("fresh", {**MODEL, "epochs": 10}, DISPLAY, FRESH)
    with pytest.raises(ValueError, match="unique"):
        workflow.validate_config("fresh", MODEL, {**DISPLAY, "basis_columns": (1, 1)}, FRESH)
    with pytest.raises(ValueError, match="Bounded fresh cap"):
        workflow.validate_config("fresh", MODEL, DISPLAY, {**FRESH, "ber_channels": 64, "blocks_per_channel": 16,
            "snrs_db": (0, 5, 10, 12, 15, 18, 20, 25)})
    with pytest.raises(ValueError, match="two simulated"):
        workflow.validate_config("fresh", MODEL, {**DISPLAY, "snr_max_db": 1}, FRESH)


def test_train_new_fails_before_loading_or_work(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable training attempted I/O or fallback")
    monkeypatch.setattr(workflow, "select_basis", forbidden)
    monkeypatch.setattr(workflow, "provision", forbidden)
    monkeypatch.setattr(workflow, "verify", forbidden)
    with pytest.raises(NotImplementedError, match="train_new is unavailable.*No training started"):
        workflow.start_workflow(mode="train_new", evidence_mode="saved_replay",
            model=MODEL, display=DISPLAY, fresh=None, artifacts=tmp_path/"absent",
            bundle=None, workspace=tmp_path, save_run=True)
    assert list(tmp_path.iterdir()) == []


def test_notebook_training_failure_clears_previous_workflow(tmp_path):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "Waveform_Exploration.ipynb").read_text())
    prepare = next(cell for cell in notebook["cells"] if cell.get("id") == "prepare")
    configuration = next(cell for cell in notebook["cells"] if "configuration" in cell.get("metadata", {}).get("tags", []))
    namespace = {}
    exec("".join(configuration["source"]), namespace)
    namespace.update(workflow=object(), MODE="train_new", EVIDENCE_MODE="saved_replay",
                     ROOT=tmp_path, SAVE_RUN=False)
    with pytest.raises(ValueError, match="requires fresh"):
        exec("".join(prepare["source"]), namespace)
    assert namespace["workflow"] is None and list(tmp_path.iterdir()) == []


@pytest.mark.artifacts
def test_provider_contract_feeds_shared_downstream_and_blocks_saved_curve_reuse(artifacts, tmp_path):
    from waveforms.basis_source import SelectedBasis
    from waveforms.delay_doppler import classical
    # Explicit matrix fixture only: no trainer is called or claimed.
    existing = workflow.select_basis("load_existing", artifacts)
    with np.load(artifacts / "focused/data/supports.npz") as data:
        fixture_support = torch.from_numpy(data["AFDM_mask"].copy())
    supplied = SelectedBasis(classical()["AFDM"], fixture_support,
                             {"source": "unit-test matrix fixture; not training"})
    config = workflow.validate_config("fresh", MODEL, DISPLAY, FRESH)
    run = workflow.NotebookWorkflow(config, artifacts, tmp_path, False, selected_basis=supplied)
    channels, matrices = run.generate_channels()
    torch.testing.assert_close(run.ws["Learned"], supplied.matrix)
    torch.testing.assert_close(matrices["Learned"], supplied.matrix.mH @ channels.h @ supplied.matrix)
    assert not run.saved_reference and (run.reference_views["labels"] == -1).all()
    assert run.summary()["selected_learned"]["source"].startswith("unit-test")
    assert set(run.evaluate_ber().waveform) == {"Learned", "OTFS", "AFDM"}
    replay = workflow.validate_config("saved_replay", MODEL, DISPLAY)
    with pytest.raises(ValueError, match="Saved BER curves belong"):
        workflow.NotebookWorkflow(replay, artifacts, tmp_path, False, selected_basis=supplied)
    changed_support = SelectedBasis(existing.matrix, existing.fixed_q_mask.roll(1, -1),
                                     {"source": "unit-test changed-support fixture"})
    with pytest.raises(ValueError, match="Saved BER curves belong"):
        workflow.NotebookWorkflow(replay, artifacts, tmp_path, False, selected_basis=changed_support)
    with pytest.raises(ValueError, match="no implicit loading fallback"):
        workflow.NotebookWorkflow(config, artifacts, tmp_path, False, selected_basis=None)


@pytest.mark.artifacts
def test_replay_reconstructs_H_A_and_never_simulates(artifacts, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted BER simulation")
    monkeypatch.setattr(workflow, "ber_point", forbidden)
    config = workflow.validate_config("saved_replay", MODEL, DISPLAY)
    run = make_workflow(config, artifacts, tmp_path, False)
    channels, matrices = run.generate_channels()
    saved = np.load(artifacts / "focused/data/channels.npz")
    np.testing.assert_allclose(channels.h[0].numpy(), saved["H"], atol=1e-12, rtol=1e-12)
    for name,w in run.ws.items():
        torch.testing.assert_close(matrices[name], w.mH @ channels.h @ w)
    expected = pd.read_csv(artifacts / "focused/ber_summary.csv")
    expected = expected[(expected.slicer == "raw") & expected.receiver.isin(["fixed", "full"])]
    pd.testing.assert_frame_equal(run.evaluate_ber(), expected)
    assert run.counts == {} and "LOADED" in run.provenance["BER_status"]
    assert channels.h.dtype == torch.complex128 and channels.h.device.type == "cpu"


@pytest.mark.artifacts
def test_fresh_parameters_propagate_to_matrices_and_BER(artifacts, tmp_path):
    model = {**MODEL, "paths": 2, "delay_range": (.2, .9), "doppler_range": (-.1, .2)}
    config = workflow.validate_config("fresh", model, DISPLAY, FRESH)
    run = make_workflow(config, artifacts, tmp_path, False)
    channels, matrices = run.generate_channels()
    assert channels.tau.shape == (3, 2) and run.matrix_index == 1
    assert torch.all((channels.tau >= .2) & (channels.tau <= .9))
    assert torch.all((channels.nu >= -.1) & (channels.nu <= .2))
    torch.testing.assert_close(channels.h.abs().square().sum((-2,-1)), torch.full((3,), 64.), atol=1e-12, rtol=0)
    table = run.evaluate_ber()
    assert len(table) == 24 and set(table.channels) == {4} and set(table.snr_db) == {0., 12.}
    assert set(table[table.modulation == "QPSK"].bits) == {4*3*64*2}
    assert set(table[table.modulation == "16QAM"].bits) == {4*3*64*4}
    for values in run.counts.values():
        assert values["tau"].shape == (4, 2)
        assert np.all((values["tau"] >= .2) & (values["tau"] <= .9))
        assert np.all((values["nu"] >= -.1) & (values["nu"] <= .2))
    repeat = make_workflow(config, artifacts, tmp_path, False)
    torch.testing.assert_close(repeat.generate_channels()[0].h, channels.h, atol=0, rtol=0)
    pd.testing.assert_frame_equal(repeat.evaluate_ber(), table)
    changed = deepcopy(config); changed["fresh"]["seed"] += 1
    other = make_workflow(changed, artifacts, tmp_path, False)
    assert not torch.equal(other.generate_channels()[0].h, channels.h)
    assert "NEW bounded" in run.provenance["BER_status"]
    assert all(torch.equal(run.ws[name], repeat.ws[name]) for name in run.ws)


@pytest.mark.artifacts
def test_inline_output_and_archive_need_no_preexisting_plots(artifacts, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from IPython.display import Image
    import IPython.display
    calls, images = [], []
    def basis(bases, columns, frequency=False):
        calls.append(tuple(columns)); return plt.figure(figsize=(2, 2))
    monkeypatch.setattr(workflow, "plot_basis", basis)
    monkeypatch.setattr(workflow, "plot_channels", lambda data: plt.figure(figsize=(2, 2)))
    def ber(frame, mod, limit, **kwargs):
        calls.append(limit); return plt.figure(figsize=(2, 2))
    monkeypatch.setattr(workflow, "plot_ber", ber)
    monkeypatch.setattr(IPython.display, "display", lambda value: images.append(value))
    config = workflow.validate_config("fresh", MODEL, DISPLAY, FRESH)
    run = make_workflow(config, artifacts, tmp_path, True)
    monkeypatch.setattr(run, "coupling_data", lambda: {})
    monkeypatch.setattr(workflow, "plot_groups", lambda data: plt.figure(figsize=(2,2)))
    monkeypatch.setattr(workflow, "plot_propagation", lambda *args: plt.figure(figsize=(2,2)))
    monkeypatch.setattr(run, "grouped_spectra_data", lambda: {})
    monkeypatch.setattr(workflow, "plot_grouped_spectra", lambda *args: plt.figure(figsize=(2,2)))
    monkeypatch.setattr(workflow, "plot_basis_examples", lambda *args: plt.figure(figsize=(2,2)))
    for name in workflow.ESSENTIALS:
        run.show(name)
    result = run.finish()
    assert len(images) == 9 and all(isinstance(value, Image) for value in images)
    assert calls == [(1, 5), (1, 5), 25., 25.]
    assert result["essential_inline_figures"] == 9
    saved = np.load(run.run_dir / "matrices.npz")
    np.testing.assert_array_equal(saved["H"], run.channels.h.numpy())
    assert saved["Learned_A"].dtype == np.complex128
    assert json.loads((run.run_dir / "run.json").read_text())["config"]["fresh"]["seed"] == 321
    assert len(list(run.run_dir.glob("*.png"))) == 9
