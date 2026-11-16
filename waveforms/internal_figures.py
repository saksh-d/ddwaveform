"""INTERNAL ONLY: preserved nine-figure diagnostics, including bridge/gain/order views."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
# Keep plotting caches beside local outputs, including in a fresh checkout.
os.environ.setdefault("MPLCONFIGDIR", str(Path("local/.matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import sha, write_json
from .experiments import output_dir
from .selection import WAVEFORMS, selected_frame, focused_inputs, identity
from matplotlib.lines import Line2D

COLORS = {"Learned": "#2166ac", "OTFS": "#d6604d", "AFDM": "#25804b"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 18,
        "axes.titlesize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 16, "legend.title_fontsize": 16,
        "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False})


def save(fig, out, name):
    path = Path(out) / (name + ".png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {"file": path.name, "sha256": sha(path)}


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-6))


def ber_figure(frame, mod, out, snr_max, slicer="raw"):
    frame = selected_frame(frame)
    shown = frame[(frame.modulation == mod) & (frame.slicer == slicer)].copy()
    if snr_max is not None:
        shown = shown[shown.snr_db <= snr_max]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), layout="constrained", sharey=True)
    for ax, kind, title in zip(axes, ["fixed", "full"], ["Fixed discovery Q · K=24", "Full A · LMMSE"]):
        for name in COLORS:
            rows = shown[(shown.waveform == name) & (shown.receiver == kind)].sort_values("snr_db")
            x, y = rows.snr_db.to_numpy(), rows.ber.to_numpy()
            display = np.where(rows.errors > 0, y, rows.detection_limit)
            ax.semilogy(x, display, "o-", color=COLORS[name], label=name, lw=2, ms=5)
            # Saved channel-bootstrap intervals; undefined zero-event intervals stay undefined.
            lo, hi = rows.ci_low.to_numpy(), rows.ci_high.to_numpy()
            ax.fill_between(x, lo, hi, color=COLORS[name], alpha=.15)
            zero = rows.errors.to_numpy() == 0
            ax.scatter(x[zero], display[zero], marker="v", color=COLORS[name], zorder=4)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=.25)
        ax.set_xlabel(r"$E_s/N_0$ (dB)")
        ax.set_xlim(float(shown.snr_db.min()), float(snr_max if snr_max is not None else shown.snr_db.max()))
        ax.legend(loc="best")
    axes[0].set_ylabel("Uncoded bit error rate")
    axes[0].set_ylim(2e-8, .6)
    shown.to_csv(Path(out) / f"ber_{mod}_{slicer}_display.csv", index=False)
    return save(fig, out, f"ber_{mod}_{slicer}")


def figures(root, out, snr_max=25.):
    root, out = Path(root), output_dir(out)
    style()
    bases, channels = focused_inputs(root)
    w, f, columns = bases["Learned"], bases["dft"], bases["selected"]
    labels, permutation = bases["labels"], bases["permutation"]
    exports = []
    n = np.arange(64)
    for frequency in [False, True]:
        fig, axes = plt.subplots(3, 2, figsize=(16, 11.2), layout="constrained", sharex=True)
        for ax, j in zip(axes.flat, columns):
            for name in WAVEFORMS:
                basis = bases[name]
                if frequency:
                    ax.plot(n / 64, np.abs((f @ basis)[:, j])**2, color=COLORS[name], lw=1.5)
                else:
                    ax.plot(n, basis[:, j].real, color=COLORS[name], lw=1.3)
                    ax.plot(n, basis[:, j].imag, color=COLORS[name], ls="--", lw=1.3)
            if frequency:
                extent = max(float((np.abs(f @ bases[name])**2).max()) for name in WAVEFORMS)
                ax.set_ylim(0, extent * 1.08)
            else:
                extent = max(max(float(np.abs(bases[name].real).max()), float(np.abs(bases[name].imag).max())) for name in WAVEFORMS)
                ax.set_ylim(-extent * 1.08, extent * 1.08)
            ax.set_title(f"Learned group {labels[j] + 1} · column {j}")
            ax.grid(alpha=.2)
        handles = [Line2D([], [], color=COLORS[name], label=name) for name in WAVEFORMS]
        if not frequency:
            handles += [Line2D([], [], color="black", label="Real"), Line2D([], [], color="black", ls="--", label="Imaginary")]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
        fig.supxlabel("Frequency bin k/N (periodic)" if frequency else "Time sample n", fontsize=18)
        fig.supylabel("Absolute DFT coefficient energy" if frequency else "Basis coefficient", fontsize=18)
        exports.append(save(fig, out, "basis_frequency" if frequency else "basis_time"))
    for reordered in [False, True]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.8), layout="constrained")
        matrices = [("Physical H", channels["H"])] + [(name + " A", channels[name + "_example"]) for name in COLORS]
        for ax, (name, matrix) in zip(axes, matrices):
            if reordered and name == "Learned A":
                matrix = matrix[np.ix_(permutation, permutation)]
                name += " · frozen order"
            image = ax.imshow(db(np.abs(matrix)**2), vmin=-60, vmax=0, cmap="viridis", origin="lower")
            ax.set_title(name)
            ax.set_xlabel("Input index")
        axes[0].set_ylabel("Output index")
        cb = fig.colorbar(image, ax=axes, shrink=.8)
        cb.set_label("Entry power (dB, reference 1)", fontsize=18)
        exports.append(save(fig, out, "channel_matrices_group_view" if reordered else "channel_matrices_native"))
    frame = selected_frame(pd.read_csv(root / "focused/ber_summary.csv"))
    for mod in ["QPSK", "16QAM"]:
        for slicer in ["raw", "gain"]:
            exports.append(ber_figure(frame, mod, out, snr_max, slicer))
    c = f @ w
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), layout="constrained")
    for row, name in enumerate(WAVEFORMS):
        basis = bases[name]
        matrices = [basis.conj().T @ basis, channels[name + "_example"],
                    basis.conj().T @ channels["H"].conj().T @ channels["H"] @ basis]
        for col, matrix in enumerate(matrices):
            ax = axes[row, col]
            image = ax.imshow(db(np.abs(matrix)**2), vmin=-60, vmax=0, origin="lower")
            if row == 0:
                ax.set_title([r"Transmitted $W^H W$", r"Coupling $A=W^H H W$", r"Distorted $W^H H^H H W$"][col])
            if row == 2:
                ax.set_xlabel("Symbol index")
        axes[row, 0].set_ylabel(name + "\nSymbol index")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8).set_label("Entry power (dB, reference 1)", fontsize=18)
    exports.append(save(fig, out, "orthogonality_bridge"))
    np.savez_compressed(out / "complex_bridge.npz", W=w, C=c, H=channels["H"],
        H_f=f @ channels["H"] @ f.conj().T, A=channels["Learned_example"],
        A_spectral=c.conj().T @ (f @ channels["H"] @ f.conj().T) @ c,
        labels=labels, permutation=permutation)
    inputs = ["focused/data/bases.npz", "focused/data/channels.npz", "focused/ber_summary.csv"]
    write_json(out / "figure_manifest.json", {"inputs": {p: sha(root / p) for p in inputs},
        "plot_source_sha256": sha(__file__),
        "audience": "internal", "selected_learned": identity(), "waveforms": list(WAVEFORMS),
        "exports": exports, "snr_display_max": snr_max,
        "typography": {"font": "DejaVu Sans", "axes": 18, "panel": 17, "ticks": 15, "legend": 16, "dpi": 300},
        "BER": "Saved counts only, approximate channel-bootstrap intervals; downward triangles mark 1/bits for zero errors",
        "matrix_example": "Saved median full-MSE18dB channel 1282 of 2048, no BER selection",
        "gain_slicer": "Supplement only: estimate/complex diag(GA), perfect CSI, guard1e-12",
        "frequency": "All complex C saved; displayed power is absolute 64-bin coefficient energy, not continuous-time bandwidth"})
    (out / "CAPTIONS.md").write_text("""# Figure reading notes

`basis_time/frequency`: the same six columns from the frozen group selection;
native indices, real/imaginary samples and absolute DFT-bin power. Learned,
OTFS and AFDM use the same column indices; group membership describes Learned
only. Solid/dashed time curves denote real/imaginary coefficients. All64 columns
remain in the input array. No peak normalization or claim of hard band limitation.

`channel_matrices_native`: original ordering is authoritative. The group-view
figure permutes only Learned A with its one frozen discovery permutation.
H and each A retain all complex coefficients in the local input/export arrays.
Color is entry power relative to1, clipped for display at-60dB, never a sparse A.

`ber_*_raw`: main QPSK/Gray16QAM curves. Left fixed discovery-Q K24, right full
LMMSE; actual observations always use full A. Common sampling budgets/physical
noise across methods. Axis Es/N0, unit symbol energy. Saved pointwise channel
bootstrap intervals are approximate under adaptive stopping; see source protocol
and per-channel counts. Zero-error markers1/bits are resolution, not confidence
limits. `gain` curves are separate perfect-CSI slicing diagnostics.

Default display ends25dB; full numerical grids through30/40dB remain in bundle.
Use --full-range to inspect high-SNR reversals/floors; no points removed from data.

`orthogonality_bridge`: rows Learned, OTFS, AFDM; W^H W, A and W^H H^H H W
answer different questions. Exactly the same frozen Learned matrix throughout.
Reconstruction uses C=FW, H_f=FHF^H, A=C^H H_f C with complex coefficients;
magnitude spectra alone cannot reconstruct A.
""")
    return {"output": str(out), "PNG_figures": len(exports), "snr_display_max": snr_max, "audience": "internal"}
