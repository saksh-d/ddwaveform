"""INTERNAL ONLY: preserved selected-reference transfer/frequency-group studies.

Multi-variant training/intervention comparisons live in waveforms.internal_studies.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .artifacts import sha, write_json
from .experiments import output_dir
from .figures import COLORS, db, save, style
from .selection import INTERNAL_ID, WAVEFORMS, selected_frame, focused_inputs, identity
import matplotlib.pyplot as plt


def rotate(w, generator, strength):
    """Existing intervention W exp(tS); caller supplies skew-Hermitian S."""
    torch.testing.assert_close(generator.mH, -generator, atol=1e-12, rtol=0)
    return w @ torch.matrix_exp(strength * generator)


def block_energy(a, labels):
    groups = torch.unique(labels, sorted=True)
    return torch.stack([torch.stack([a[..., labels == i, :][..., labels == j].abs().square().sum((-2, -1))
                                   for j in groups], dim=-1) for i in groups], dim=-2)


def selected_transfer(root):
    """Use only the exact selected source ID; never alias a fresh replication."""
    path = Path(root) / "supplementary/main/impairment_transfer.csv"
    return selected_frame(pd.read_csv(path), source_label=INTERNAL_ID)


def figures(root, out):
    root, out = Path(root), output_dir(out)
    bases, _ = focused_inputs(root)
    style()
    exports = []
    transfer = selected_transfer(root)
    transfer.to_csv(out / "selected_transfer.csv", index=False)
    kinds = ["identity", "delay", "doppler", "dd", "multipath"]
    subset = transfer[transfer.strength == 1.]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained", sharey=True)
    for ax, metric, title in zip(axes, ["eta24", "offdiag_leakage"],
            [r"Retained row power $\eta_{24}$", "Across-group / off-diagonal energy"]):
        matrix = np.array([[float(subset[(subset.waveform == name) & (subset.impairment == kind)][metric].iloc[0])
                            for kind in kinds] for name in WAVEFORMS])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(5), ["Identity", "Delay", "Doppler", "One DD", "Multipath"], rotation=20, ha="right")
        ax.set_yticks(np.arange(3), WAVEFORMS)
        ax.set_title(title)
        for i in range(3):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        color="black" if matrix[i,j] > .5 else "white", fontsize=15)
    fig.colorbar(image, ax=axes, shrink=.8)
    exports.append(save(fig, out, "structural_transfer"))
    f = bases["dft"]
    labels, permutation = bases["labels"], bases["permutation"]
    c = f @ bases["Learned"]
    fig, axes = plt.subplots(1, 4, figsize=(23, 6), layout="constrained")
    for ax, name in zip(axes[:3], WAVEFORMS):
        coefficients = f @ bases[name]
        # Native baseline order; only Learned has this frozen discovered partition.
        if name == "Learned":
            coefficients = coefficients[:, permutation]
        image = ax.imshow(db(np.abs(coefficients)**2), origin="lower", vmin=-60, vmax=0, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("Frozen group order" if name == "Learned" else "Native column index")
    axes[0].set_ylabel("DFT frequency bin k")
    for group in range(3):
        axes[3].plot(np.arange(64) / 64, (np.abs(c[:, labels == group])**2).sum(1), label=f"Group {group+1}")
    axes[3].set_title("Learned group projectors")
    axes[3].set_xlabel("Frequency k/N (periodic)")
    axes[3].set_ylabel("Projector diagonal")
    axes[3].legend()
    axes[3].grid(alpha=.2)
    fig.colorbar(image, ax=axes[:3].tolist(), shrink=.8).set_label("Coefficient power (dB)", fontsize=18)
    exports.append(save(fig, out, "frequency_groups"))
    inputs = ["focused/data/bases.npz", "focused/data/channels.npz", "supplementary/main/impairment_transfer.csv"]
    write_json(out / "figure_manifest.json", {"exports": exports, "plot_source_sha256": sha(__file__),
        "audience": "internal", "selected_learned": identity(), "waveforms": list(WAVEFORMS),
        "inputs": {name: sha(root / name) for name in inputs},
        "limits": "Same frozen Learned as core, all strengths retained in CSV. Structural operator transfer only, not BER or historical CFO W/R. No candidate refitting or intervention/training-variant plots."})
    return {"output": str(out), "PNG_figures": len(exports), "audience": "internal", "selected_learned": identity()}
