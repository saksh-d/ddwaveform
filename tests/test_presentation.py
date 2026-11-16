"""Presentation boundaries: one selected matrix, no relabeled candidate leakage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from waveforms.selection import INTERNAL_ID, WAVEFORMS, selected_frame, focused_inputs, identity


def test_explicit_identity_filter_never_relabels_other_candidates():
    source = pd.DataFrame({"waveform": ["HISTORICAL", "T0_tail", "T0_q", "OTFS", "AFDM"],
                           "value": [1., 2., 3., 4., 5.]})
    selected = selected_frame(source, source_label=INTERNAL_ID)
    assert selected.set_index("waveform").value.to_dict() == {"Learned": 1., "OTFS": 4., "AFDM": 5.}
    with pytest.raises(ValueError, match="missing"):
        selected_frame(source[source.waveform != INTERNAL_ID], source_label=INTERNAL_ID)
    with pytest.raises(ValueError, match="Ambiguous"):
        selected_frame(pd.concat([source, pd.DataFrame({"waveform": ["Learned"], "value": [9.]})]), source_label=INTERNAL_ID)


def test_research_notebook_separates_internal_candidates():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "Waveform_Exploration.ipynb").read_text())
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for forbidden in ["T0_tail", "T0_q", "HISTORICAL", "training_selected_summary", "load_qualified", "selected_transfer", "waveforms.supplementary", "internal_figures", "internal_studies"]:
        assert forbidden not in text
    internal = (root / "internal/notebooks/variants.ipynb").read_text()
    assert "T0_tail" in internal and "training_selected_summary" in internal
    config_cells = [cell for cell in notebook["cells"] if "configuration" in cell.get("metadata", {}).get("tags", [])]
    assert len(config_cells) == 1
    assert "MODE =" in "".join(config_cells[0]["source"])
    assert 'MODE = "load_existing"' in "".join(config_cells[0]["source"])
    assert "EVIDENCE_MODE =" in "".join(config_cells[0]["source"])
    assert "train_new" in text and "unsupported" in text
    import ast
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell is not config_cells[0]:
            assert not any(isinstance(node, ast.If) for node in ast.walk(ast.parse("".join(cell["source"]))))
    assert "workflow = None" in text
    assert "start_workflow" in text and "workflow.generate_channels()" in text
    assert "show_ber_summary(workflow)" in text and "show_coupling(workflow)" in text and "show_overlay_key(workflow)" in text
    assert "Image(filename=" not in text and "Generate first:" not in text
    assert "saved_replay" in text and "fresh" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])


@pytest.mark.artifacts
def test_selected_matrix_substitution_rejected(artifacts, tmp_path):
    from shutil import copyfile
    bases, channels = focused_inputs(artifacts)
    fake = tmp_path / "artifacts"
    (fake / "reference").mkdir(parents=True)
    (fake / "focused/data").mkdir(parents=True)
    copyfile(artifacts / "reference/learned_w.npy", fake / "reference/learned_w.npy")
    bases["Learned"] = bases["AFDM"]
    np.savez(fake / "focused/data/bases.npz", **bases)
    np.savez(fake / "focused/data/channels.npz", **channels)
    with pytest.raises(ValueError, match="differs from selected"):
        focused_inputs(fake)


@pytest.mark.artifacts
def test_research_figures_are_essentials_only(artifacts, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import waveforms.figures as core
    captured = {}
    def capture(fig, out, name):
        captured[name] = [item.get_text() for item in fig.findobj(Text)]
        plt.close(fig)
        return {"file": name + ".png", "sha256": "captured-in-test"}
    monkeypatch.setattr(core, "save", capture)
    core.figures(artifacts, tmp_path / "core")
    expected = {"channel_matrices_native", "basis_time", "basis_frequency", "ber_QPSK_raw", "ber_16QAM_raw"}
    assert set(captured) == expected
    assert not list((tmp_path / "core").glob("*.npz"))
    for name, labels in captured.items():
        joined = " ".join(labels)
        assert all(label in joined for label in WAVEFORMS), name
        assert not any(label in joined for label in ["T0_", "T1_", "T2_", "HISTORICAL"]), name
    manifest = json.loads((tmp_path / "core/figure_manifest.json").read_text())
    assert manifest["selected_learned"] == identity()
    assert manifest["waveforms"] == list(WAVEFORMS)
    assert manifest["figure_scope"] == "essentials_only"
    assert {Path(item["file"]).stem for item in manifest["exports"]} == expected
    for name in ["ber_QPSK_raw_display.csv", "ber_16QAM_raw_display.csv"]:
        rows = pd.read_csv(tmp_path / "core" / name)
        assert set(rows.waveform) == set(WAVEFORMS)


@pytest.mark.artifacts
def test_original_multi_variant_renderer_remains_internal(artifacts, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import waveforms.internal_studies as internal
    labels = []
    def capture(fig, out, name):
        labels.extend(item.get_text() for item in fig.findobj(Text))
        plt.close(fig)
        return {"file": name + ".png", "sha256": "captured-in-test"}
    monkeypatch.setattr(internal, "save", capture)
    result = internal.figures(artifacts, tmp_path / "internal")
    assert result["PNG_figures"] == 4 and result["audience"] == "internal"
    assert {"HISTORICAL", "T0_tail", "T0_q"} <= set(labels)


def test_cli_has_no_public_supplementary_route(tmp_path, monkeypatch):
    import sys
    import waveforms.cli as cli
    import waveforms.figures as core
    import waveforms.supplementary as studies
    import waveforms.internal_figures as diagnostics
    def forbidden(*args, **kwargs):
        raise AssertionError("Internal generation called by public command")
    monkeypatch.setattr(studies, "figures", forbidden)
    monkeypatch.setattr(diagnostics, "figures", forbidden)
    monkeypatch.setattr(core, "figures", lambda *args: {"figure_scope": "essentials_only"})
    monkeypatch.setattr(sys, "argv", ["waveforms", "figures", "--artifacts", str(tmp_path)])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["waveforms", "supplementary", "--artifacts", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


@pytest.mark.artifacts
def test_selected_studies_and_diagnostics_remain_internal(artifacts, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import waveforms.supplementary as studies
    import waveforms.internal_figures as diagnostics
    captured = []
    def capture(fig, out, name):
        captured.append(name)
        plt.close(fig)
        return {"file": name + ".png", "sha256": "captured-in-test"}
    monkeypatch.setattr(studies, "save", capture)
    monkeypatch.setattr(diagnostics, "save", capture)
    diagnostic = diagnostics.figures(artifacts, tmp_path / "diagnostics")
    supplement = studies.figures(artifacts, tmp_path / "studies")
    assert diagnostic["PNG_figures"] == 9 and diagnostic["audience"] == "internal"
    assert supplement["PNG_figures"] == 2 and supplement["audience"] == "internal"
    assert {"orthogonality_bridge", "channel_matrices_group_view", "ber_QPSK_gain", "structural_transfer", "frequency_groups"} <= set(captured)
    for directory in ["diagnostics", "studies"]:
        manifest = json.loads((tmp_path / directory / "figure_manifest.json").read_text())
        assert manifest["audience"] == "internal"
    original = pd.read_csv(artifacts / "supplementary/main/impairment_transfer.csv")
    chosen = studies.selected_transfer(artifacts)
    expected = original[original.waveform == INTERNAL_ID].drop(columns="waveform").reset_index(drop=True)
    actual = chosen[chosen.waveform == "Learned"].drop(columns="waveform").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
