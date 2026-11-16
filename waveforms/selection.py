"""One frozen DD Learned identity; internal candidates never silently relabeled."""
from pathlib import Path
import numpy as np
from .artifacts import EXPECTED_W, load_reference

LABEL = "Learned"
INTERNAL_ID = "HISTORICAL"
WAVEFORMS = (LABEL, "OTFS", "AFDM")


def identity():
    return {"label": LABEL, "matrix_sha256": EXPECTED_W,
            "scope": "N64 delay-Doppler; one matrix for all modulations, SNRs and receivers",
            "selection": "Retained verified reference after descriptive post-hoc tradeoff review; no universal winner"}


def focused_inputs(root):
    """Verify saved plotting inputs refer to the same original synthesis matrix."""
    root = Path(root)
    w = load_reference(root).numpy()
    with np.load(root / "focused/data/bases.npz", allow_pickle=False) as data:
        bases = {key: data[key] for key in data.files}
    with np.load(root / "focused/data/channels.npz", allow_pickle=False) as data:
        channels = {key: data[key] for key in data.files}
    if not np.array_equal(bases[LABEL], w):
        raise ValueError("Selected basis differs from selected Learned; no per-plot substitution allowed")
    if not np.allclose(w.conj().T @ channels["H"] @ w, channels["Learned_example"], atol=1e-12, rtol=1e-12):
        raise ValueError("Saved selected A does not match selected Learned")
    return bases, channels


def selected_frame(frame, source_label=LABEL):
    """Whitelist a declared source identity; reject collisions instead of blending."""
    if source_label != LABEL and (frame.waveform == LABEL).any():
        raise ValueError("Ambiguous Learned identity in internal table")
    selected = frame[frame.waveform.isin([source_label, *WAVEFORMS[1:]])].copy()
    if set(selected.waveform) != {source_label, *WAVEFORMS[1:]}:
        raise ValueError("Selected Learned or a baseline missing; never substitute another candidate")
    selected.loc[selected.waveform == source_label, "waveform"] = LABEL
    return selected
