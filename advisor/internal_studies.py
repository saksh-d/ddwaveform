"""INTERNAL ONLY: all original training/intervention/transfer variant comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .artifacts import sha, write_json
from .experiments import output_dir
from .figures import db, save, style
import matplotlib.pyplot as plt


def rotate(w, generator, strength):
    """Existing intervention W exp(tS); caller supplies skew-Hermitian S."""
    torch.testing.assert_close(generator.mH, -generator, atol=1e-12, rtol=0)
    return w @ torch.matrix_exp(strength * generator)


def block_energy(a, labels):
    groups = torch.unique(labels, sorted=True)
    return torch.stack([torch.stack([a[..., labels == i, :][..., labels == j].abs().square().sum((-2, -1))
                                   for j in groups], dim=-1) for i in groups], dim=-2)


def figures(root, out):
    root, out = Path(root), output_dir(out)
    style()
    main = root / "supplementary/main"
    exports = []
    phases = ["mean", "mean_cont", "tail", "q"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), layout="constrained", sharex="col", sharey="row")
    for col, phase in enumerate(phases):
        for trial in range(3):
            trace = pd.read_csv(main / f"training/trial{trial}/{phase}/trace.csv")
            axes[0, col].plot(trace.step, trace["tail"], label=f"Trial {trial}")
            axes[1, col].plot(trace.step, trace.eta24)
        axes[0, col].set_title({"mean": "Mean parent", "mean_cont": "Mean continuation", "tail": "Tail refinement", "q": "Q-only"}[phase])
        axes[1, col].set_xlabel("Phase validation step")
        for row in range(2):
            axes[row, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Validation top-four MSE")
    axes[1, 0].set_ylabel(r"Validation $\eta_{24}$")
    axes[0, 0].legend()
    exports.append(save(fig, out, "training_validation"))
    intervention = pd.read_csv(main / "interventions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained", sharex=True)
    for row, teacher in enumerate(["HISTORICAL", "T0_tail"]):
        subset = intervention[intervention.teacher == teacher]
        for kind, color in zip(["within", "across", "random"], ["#2166ac", "#d6604d", "#25804b"]):
            data = subset[subset.kind == kind]
            for col, metric in enumerate(["eta24", "fixed_tail"]):
                ax = axes[row, col]
                for _, draw in data.groupby("draw"):
                    ax.plot(draw.t, draw[metric], color=color, alpha=.2)
                means = data.groupby("t")[metric].mean()
                ax.plot(means.index, means, "o-", color=color, label=kind)
                ax.grid(alpha=.2)
        axes[row, 0].set_title(teacher)
        axes[row, 1].set_title(teacher)
        axes[row, 0].set_ylabel(r"$\eta_{24}$")
        axes[row, 1].set_ylabel("Fixed-Q tail MSE")
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Intervention strength t")
    exports.append(save(fig, out, "interventions"))
    transfer = pd.read_csv(main / "impairment_transfer.csv")
    names = ["OFDM", "OTFS", "AFDM", "T0_tail", "T0_q"]
    kinds = ["identity", "delay", "doppler", "dd", "multipath"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained", sharey=True)
    subset = transfer[transfer.strength == 1.]
    for ax, metric, title in zip(axes, ["eta24", "offdiag_leakage"], [r"Retained row power $\eta_{24}$", "Across-group / off-diagonal energy"]):
        matrix = np.array([[float(subset[(subset.waveform == name) & (subset.impairment == kind)][metric].iloc[0]) for kind in kinds] for name in names])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(5), ["Identity", "Delay", "Doppler", "One DD", "Multipath"], rotation=20, ha="right")
        ax.set_yticks(np.arange(5), names)
        ax.set_title(title)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="black" if matrix[i,j] > .5 else "white", fontsize=15)
    fig.colorbar(im, ax=axes, shrink=.8)
    exports.append(save(fig, out, "structural_transfer"))
    bases = np.load(root / "focused/data/bases.npz", allow_pickle=False)
    c = bases["dft"] @ bases["Learned"]
    labels, permutation = bases["labels"], bases["permutation"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    im = axes[0].imshow(db(np.abs(c[:, permutation])**2), origin="lower", vmin=-60, vmax=0, aspect="auto")
    axes[0].set_xlabel("Frozen discovery column order")
    axes[0].set_ylabel("DFT frequency bin k")
    for group in range(3):
        axes[1].plot(np.arange(64) / 64, np.abs(c[:, labels == group])**2 @ np.ones((labels == group).sum()), label=f"Group {group+1}")
    axes[1].set_xlabel("Frequency k/N (periodic)")
    axes[1].set_ylabel("Group projector diagonal")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.colorbar(im, ax=axes[0], shrink=.8).set_label("Coefficient power (dB)", fontsize=18)
    exports.append(save(fig, out, "frequency_groups"))
    inputs = ["training_selected_summary.csv", "interventions.csv", "impairment_transfer.csv"]
    write_json(out / "figure_manifest.json", {"exports": exports, "plot_source_sha256": sha(__file__),
        "audience": "internal", "contains_multiple_learned_variants": True,
        "inputs": {name: sha(main / name) for name in inputs},
        "limits": "Completed N64 results only; transfer panel is structural, not CFO BER. Intervention lines show direction variability, not confidence intervals. Validation phases restart; no BER selection."})
    return {"output": str(out), "PNG_figures": len(exports), "audience": "internal"}
