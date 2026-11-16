"""Advisor essentials only: H/A, basis samples/spectra and raw BER versus SNR."""
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
from .selection import WAVEFORMS, advisor_frame, focused_inputs, identity
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


def plot_ber(frame, mod, snr_max, slicer="raw", provenance=None, k=24, support_label="Fixed discovery Q"):
    frame = advisor_frame(frame)
    shown = frame[(frame.modulation == mod) & (frame.slicer == slicer)].copy()
    if snr_max is not None:
        shown = shown[shown.snr_db <= snr_max]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), layout="constrained", sharey=True)
    for ax, kind, title in zip(axes, ["fixed", "full"], [f"{support_label} · K={k}", "Full A · LMMSE"]):
        for name in COLORS:
            rows = shown[(shown.waveform == name) & (shown.receiver == kind)].sort_values("snr_db")
            x, y = rows.snr_db.to_numpy(), rows.ber.to_numpy()
            display = np.where(rows.errors > 0, y, rows.detection_limit)
            ax.semilogy(x, display, "o-", color=COLORS[name], label=name, lw=2, ms=5)
            # Interval provenance is declared by the caller; zero-event bands stay undefined.
            lo, hi = rows.ci_low.to_numpy(), rows.ci_high.to_numpy()
            ax.fill_between(x, lo, hi, color=COLORS[name], alpha=.15)
            zero = rows.errors.to_numpy() == 0
            ax.scatter(x[zero], display[zero], marker="v", color=COLORS[name], zorder=4)
        ax.set_title((provenance + "\n" if provenance else "") + title)
        ax.grid(True, which="both", alpha=.25)
        ax.set_xlabel(r"$E_s/N_0$ (dB)")
        ax.set_xlim(float(shown.snr_db.min()), float(snr_max if snr_max is not None else shown.snr_db.max()))
        ax.legend(loc="best")
    axes[0].set_ylabel("Uncoded bit error rate")
    axes[0].set_ylim(2e-8, .6)
    if (shown.errors == 0).any():
        fig.supxlabel("▽ = zero errors, displayed at 1/bits; unresolved BER (bounds in notebook table)", fontsize=15)
    return fig


def ber_figure(frame, mod, out, snr_max, slicer="raw"):
    shown = advisor_frame(frame)
    shown = shown[(shown.modulation == mod) & (shown.slicer == slicer)]
    if snr_max is not None:
        shown = shown[shown.snr_db <= snr_max]
    shown.to_csv(Path(out) / f"ber_{mod}_{slicer}_display.csv", index=False)
    return save(plot_ber(frame, mod, snr_max, slicer), out, f"ber_{mod}_{slicer}")


def plot_basis(bases, columns, frequency=False):
    f = bases["dft"]
    size = bases["Learned"].shape[0]
    n = np.arange(size)
    rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(rows, 2, squeeze=False, figsize=(16, 11.2 if rows == 3 else max(4.5,3.73 * rows)), layout="constrained", sharex=True)
    for ax, j in zip(axes.flat, columns):
        for name in WAVEFORMS:
            basis = bases[name]
            if frequency:
                spectrum = np.fft.fftshift(np.fft.fft(basis[:, j], n=8*size))/np.sqrt(size)
                freq = np.fft.fftshift(np.fft.fftfreq(8*size))
                ax.plot(freq, spectrum.real, color=COLORS[name], lw=1.2)
                ax.plot(freq, spectrum.imag, color=COLORS[name], ls="--", lw=1.2)
            else:
                ax.plot(n, basis[:, j].real, color=COLORS[name], lw=1.3)
                ax.plot(n, basis[:, j].imag, color=COLORS[name], ls="--", lw=1.3)
        if frequency:
            extent = max(float(np.abs(np.fft.fft(bases[name], n=8*size, axis=0)/np.sqrt(size)).max()) for name in WAVEFORMS)
            ax.set_ylim(-extent * 1.08, extent * 1.08)
        else:
            extent = max(max(float(np.abs(bases[name].real).max()), float(np.abs(bases[name].imag).max())) for name in WAVEFORMS)
            ax.set_ylim(-extent * 1.08, extent * 1.08)
        group = int(bases["labels"][j]) if "labels" in bases else -1
        if group >= 0:
            from .coupling import GROUP_COLORS
            ax.set_title(f"Basis column {j} · Learned group {group}", color=GROUP_COLORS[group % len(GROUP_COLORS)])
        else:
            ax.set_title(f"Basis column {j}")
        ax.grid(alpha=.2)
    handles = [Line2D([], [], color=COLORS[name], label=name) for name in WAVEFORMS]
    handles += [Line2D([], [], color="black", label="Real"), Line2D([], [], color="black", ls="--", label="Imaginary")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.supxlabel("Frequency f/Fs (periodic)" if frequency else "Time sample n", fontsize=18)
    fig.supylabel("Complex DTFT coefficient\n(1/√N normalization)" if frequency else "Basis coefficient", fontsize=18)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    return fig


def plot_channels(channels):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.8), layout="constrained")
    matrices = [("Physical H", channels["H"])] + [(name + " A", channels[name + "_example"]) for name in WAVEFORMS]
    for ax, (name, matrix) in zip(axes, matrices):
        image = ax.imshow(db(np.abs(matrix)**2), vmin=-60, vmax=0, cmap="viridis", origin="lower")
        ax.set_title(name)
        ax.set_xlabel("Input index")
    axes[0].set_ylabel("Output index")
    fig.colorbar(image, ax=axes, shrink=.8).set_label("Entry power (dB, reference 1)", fontsize=18)
    return fig


def figures(root, out, snr_max=25.):
    root, out = Path(root), output_dir(out)
    style()
    bases, channels = focused_inputs(root)
    f, columns = bases["dft"], bases["selected"]
    exports = []
    for frequency in [False, True]:
        exports.append(save(plot_basis(bases, columns, frequency), out,
                            "basis_frequency" if frequency else "basis_time"))
    exports.append(save(plot_channels(channels), out, "channel_matrices_native"))
    frame = advisor_frame(pd.read_csv(root / "focused/ber_summary.csv"))
    for mod in ["QPSK", "16QAM"]:
        exports.append(ber_figure(frame, mod, out, snr_max, "raw"))
    inputs = ["focused/data/bases.npz", "focused/data/channels.npz", "focused/ber_summary.csv"]
    write_json(out / "figure_manifest.json", {"inputs": {p: sha(root / p) for p in inputs},
        "plot_source_sha256": sha(__file__),
        "audience": "advisor", "figure_scope": "essentials_only", "selected_learned": identity(), "waveforms": list(WAVEFORMS),
        "exports": exports, "snr_display_max": snr_max,
        "typography": {"font": "DejaVu Sans", "axes": 18, "panel": 17, "ticks": 15, "legend": 16, "dpi": 300},
        "BER": "Saved counts only, approximate channel-bootstrap intervals; downward triangles mark 1/bits for zero errors",
        "matrix_example": "Saved median full-MSE18dB channel 1282 of 2048, no BER selection",
        "frequency": "Complex finite-block DTFT real/imaginary samples, eightfold zero padding, 1/sqrt(N), no window or peak normalization"})
    (out / "CAPTIONS.md").write_text("""# Essential figure reading notes

Exactly five figures: physical/effective-channel matrices, time-domain basis,
frequency-domain basis, QPSK BER and16QAM BER. Same frozen Learned throughout,
with OTFS/AFDM baselines. No analysis figures generated by this command.

`basis_time/frequency`: six previously frozen column indices, unchanged across
waveforms. Solid/dashed time curves denote real/imaginary samples. Frequency curves show real/imaginary finite-block DTFT coefficients with eightfold
zero padding and1/sqrt(N) normalization; no window or peak normalization.
All64complex columns remain available in saved input arrays.

`channel_matrices_native`: physical H and complete A for Learned/OTFS/AFDM,
original ordering. Color is entry power relative to1, clipped for display at
-60dB. Complex H/W/A, path parameters and seeds remain available via export-dd.

`ber_*_raw`: uncoded QPSK/Gray16QAM, historical nearest-point slicing. Left fixed
Q24 model, right full LMMSE; actual observations always use full A. Common
sampling budgets/physical noise. Axis Es/N0, unit symbol energy. Saved pointwise
channel-bootstrap intervals are approximate under adaptive stopping. Zero-error
markers1/bits indicate resolution, not confidence limits.

Default display ends25dB; full numerical grids through30/40dB remain intact.
Use --full-range for the same essential figures including high-SNR reversals.
Internal diagnostics/studies remain separate and never run through this command.
""")
    return {"output": str(out), "PNG_figures": len(exports), "snr_display_max": snr_max, "figure_scope": "essentials_only"}
