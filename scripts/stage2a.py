"""Run Stage 2A frozen-waveform qualification and structural forensics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import assemble_dd_channel, doppler_operator, fractional_delay_operator, sample_dd_v1
from src.metrics import qpsk_constellation
from src.stage2a import (
    FIGURE_WAVEFORMS,
    aggregate_rows,
    analytic_rows,
    communication_rows,
    effective_operator_arrays,
    file_sha256,
    k_sweep_analytic_rows,
    learned_structure,
    load_frozen_waveform,
    paired_ber_bootstrap,
    qualification_waveforms,
    representative_channel_index,
)
from src.receiver import effective_operator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2a"
CHECKPOINT = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
COLORS = {"Learned": "tab:orange", "OFDM": "tab:blue", "OTFS": "tab:green", "AFDM": "tab:red"}
EXPECTED_LEARNED_BIT_ERRORS = {
    ("QPSK", 12): 9659,
    ("QPSK", 15): 3304,
    ("QPSK", 18): 1148,
    ("QPSK", 20): 671,
    ("16QAM", 12): 91174,
    ("16QAM", 15): 50051,
    ("16QAM", 18): 24862,
    ("16QAM", 20): 16560,
}
EXPECTED_LEARNED_MSE = {12: 0.12348702, 15: 0.07758136, 18: 0.04911952, 20: 0.03689084}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def select(rows: list[dict[str, object]], **criteria: object) -> list[dict[str, object]]:
    return [row for row in rows if all(row[key] == value for key, value in criteria.items())]


def verify_reproduction(
    communication_summary: list[dict[str, object]], analytic_summary: list[dict[str, object]]
) -> dict[str, object]:
    for (modulation, snr), expected in EXPECTED_LEARNED_BIT_ERRORS.items():
        row = select(
            communication_summary,
            modulation=modulation,
            snr_db=snr,
            waveform="Learned",
            k=24,
        )[0]
        if int(row["bit_errors"]) != expected or int(row["total_bits"]) != 1_048_576:
            raise RuntimeError(
                f"frozen reproduction mismatch at {modulation}/{snr}: "
                f"{row['bit_errors']}/{row['total_bits']} != {expected}/1048576"
            )
    for snr, expected in EXPECTED_LEARNED_MSE.items():
        row = select(analytic_summary, snr_db=snr, waveform="Learned", k=24)[0]
        if abs(float(row["analytic_mse_mean"]) - expected) > 5e-9:
            raise RuntimeError(
                f"frozen analytic reproduction mismatch at {snr}: "
                f"{row['analytic_mse_mean']} != {expected}"
            )
    return {
        "success": True,
        "comparison": "exact deterministic Stage-1A.2 error counts and reported analytic means",
        "expected_bit_errors": {f"{key[0]}_{key[1]}": value for key, value in EXPECTED_LEARNED_BIT_ERRORS.items()},
    }


def frozen_table(
    communication_summary: list[dict[str, object]], analytic_summary: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for snr in config.STAGE1A_SNRS_DB:
        for waveform in ("IDENTITY", "OFDM", "OTFS", "AFDM", "Learned"):
            qpsk = select(communication_summary, modulation="QPSK", snr_db=snr, waveform=waveform, k=24)[0]
            qam = select(communication_summary, modulation="16QAM", snr_db=snr, waveform=waveform, k=24)[0]
            analytic = select(analytic_summary, snr_db=snr, waveform=waveform, k=24)[0]
            rows.append(
                {
                    "snr_db": snr,
                    "waveform": waveform,
                    "qpsk_ber": qpsk["ber"],
                    "qpsk_ser": qpsk["ser"],
                    "qam16_ber": qam["ber"],
                    "qam16_ser": qam["ser"],
                    "analytic_mse": analytic["analytic_mse_mean"],
                    "mean_per_stream_mse": analytic["mean_per_stream_mse_mean"],
                    "stream_cvar_095": analytic["stream_cvar_095_mean"],
                    "retained_energy_24": analytic["retained_energy_mean"],
                    "limiting_distortion_24_median": analytic["limiting_distortion_median"],
                    "bits_per_modulation_point": qpsk["total_bits"],
                    "n_channels": analytic["n_channels"],
                }
            )
    return rows


def k_table(
    analytic_summary: list[dict[str, object]], ber_summary: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for k in config.STAGE2A_BER_K_VALUES:
        for waveform in ("IDENTITY", "OFDM", "OTFS", "AFDM", "Learned"):
            analytic = select(analytic_summary, snr_db=18, waveform=waveform, k=k)[0]
            ber = select(ber_summary, modulation="QPSK", snr_db=18, waveform=waveform, k=k)[0]
            rows.append(
                {
                    "k": k,
                    "waveform": waveform,
                    "qpsk_ber_18db": ber["ber"],
                    "analytic_mse_18db": analytic["analytic_mse_mean"],
                    "retained_energy": analytic["retained_energy_mean"],
                    "limiting_distortion_median": analytic["limiting_distortion_median"],
                    "condition_number_median": analytic["condition_number_median"],
                    "condition_number_p90": analytic["condition_number_p90"],
                    "condition_number_p95": analytic["condition_number_p95"],
                    "bits": ber["total_bits"],
                    "n_channels": analytic["n_channels"],
                }
            )
    return rows


def plot_ber_vs_snr(rows: list[dict[str, object]], modulation: str, filename: str) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for waveform in FIGURE_WAVEFORMS:
        selected = sorted(select(rows, modulation=modulation, waveform=waveform, k=24), key=lambda row: int(row["snr_db"]))
        axis.semilogy([row["snr_db"] for row in selected], [row["ber"] for row in selected], "o-", label=waveform, color=COLORS[waveform])
    axis.set_xlabel(r"$E_s/N_0$ (dB)")
    axis.set_ylabel("Bit error rate")
    axis.set_title(f"DD-v1, K=24, {modulation}")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def plot_k_curves(analytic: list[dict[str, object]], ber: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for waveform in FIGURE_WAVEFORMS:
        selected = sorted(select(ber, modulation="QPSK", snr_db=18, waveform=waveform), key=lambda row: int(row["k"]))
        axis.semilogy([row["k"] for row in selected], [row["ber"] for row in selected], "o-", label=waveform, color=COLORS[waveform])
    axis.axvline(24, color="0.35", linestyle="--", linewidth=1, label="K=24 discovery point")
    axis.set_xlabel("Receiver budget K"); axis.set_ylabel("Bit error rate")
    axis.set_title("DD-v1 QPSK at 18 dB"); axis.grid(True, which="both", alpha=0.3); axis.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig03_ber_vs_K_18db.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for waveform in FIGURE_WAVEFORMS:
        selected = sorted(select(analytic, snr_db=18, waveform=waveform), key=lambda row: int(row["k"]))
        axis.semilogy([row["k"] for row in selected], [row["analytic_mse_mean"] for row in selected], "o-", label=waveform, color=COLORS[waveform])
    axis.axvline(24, color="0.35", linestyle="--", linewidth=1, label="K=24 discovery point")
    axis.set_xlabel("Receiver budget K"); axis.set_ylabel("Analytic MSE")
    axis.set_title("DD-v1 finite-representation MSE at 18 dB")
    axis.grid(True, which="both", alpha=0.3); axis.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig04_mse_vs_K.png", dpi=180); plt.close(fig)


def plot_effective_heatmaps(arrays: dict[str, np.ndarray]) -> None:
    maximum = max(float(np.abs(value).max()) for value in arrays.values())
    fig = plt.figure(figsize=(8.8, 13.2))
    grid = fig.add_gridspec(4, 3, width_ratios=(1, 1, 0.06), left=0.08, right=0.91, bottom=0.05, top=0.96, hspace=0.28, wspace=0.18)
    axes = np.asarray([[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(4)])
    for row, waveform in enumerate(("OFDM", "OTFS", "AFDM", "Learned")):
        for column, suffix in enumerate(("A", "T24")):
            image = axes[row, column].imshow(np.abs(arrays[f"{waveform}_{suffix}"]), origin="lower", aspect="equal", vmin=0, vmax=maximum, cmap="magma")
            axes[row, column].set_title(f"{waveform}: |{suffix}(A)|" if suffix == "T24" else f"{waveform}: |A|")
            axes[row, column].set_ylabel("Output index")
    axes[-1, 0].set_xlabel("Input index"); axes[-1, 1].set_xlabel("Input index")
    fig.colorbar(image, cax=fig.add_subplot(grid[:, 2]), label="Magnitude")
    fig.suptitle("Same-channel effective operators and row-wise T24", y=0.99)
    fig.savefig(OUT / "fig05_effective_channel_heatmaps.png", dpi=180); plt.close(fig)


def plot_basis(arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    magnitude = axes[0].imshow(arrays["magnitude"].T, origin="lower", aspect="auto", cmap="viridis")
    phase = axes[1].imshow((arrays["unwrapped_phase"] / np.pi).T, origin="lower", aspect="auto", cmap="twilight")
    axes[0].set_title("Magnitude"); axes[1].set_title(r"Unwrapped phase / $\pi$")
    for axis in axes: axis.set_xlabel("Sample index")
    axes[0].set_ylabel("Basis-column index")
    fig.colorbar(magnitude, ax=axes[0], shrink=0.85); fig.colorbar(phase, ax=axes[1], shrink=0.85)
    fig.suptitle("Frozen Learned basis: time-domain structure")
    fig.tight_layout(); fig.savefig(OUT / "fig06_learned_basis_time_structure.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.1, 5.2))
    image = axis.imshow(arrays["fft_magnitude"].T, origin="lower", aspect="auto", extent=(arrays["frequencies"][0], arrays["frequencies"][-1], -0.5, 63.5), cmap="viridis")
    axis.set_xlabel("Normalized frequency (cycles/sample)"); axis.set_ylabel("Basis-column index")
    axis.set_title("Frozen Learned basis: FFT magnitude")
    fig.colorbar(image, ax=axis, label="Magnitude")
    fig.tight_layout(); fig.savefig(OUT / "fig07_learned_basis_spectra.png", dpi=180); plt.close(fig)


def atomic_preview(waveforms: dict[str, torch.Tensor]) -> None:
    paths = ((0, 0), (1, 0), (0, 0.5), (1, 0.5), (2.5, -0.5))
    names = ("OFDM", "OTFS", "AFDM", "Learned")
    operators = {}
    for path_index, (tau, nu) in enumerate(paths):
        physical = doppler_operator(nu) @ fractional_delay_operator(tau)
        for name in names:
            operators[(path_index, name)] = np.abs(effective_operator(physical, waveforms[name]).numpy())
    maximum = max(value.max() for value in operators.values())
    fig = plt.figure(figsize=(12.5, 14.0))
    grid = fig.add_gridspec(len(paths), len(names) + 1, width_ratios=(1, 1, 1, 1, 0.06), left=0.05, right=0.92, bottom=0.04, top=0.95, hspace=0.32, wspace=0.12)
    axes = np.asarray([[fig.add_subplot(grid[row, column]) for column in range(len(names))] for row in range(len(paths))])
    for row, path in enumerate(paths):
        for column, name in enumerate(names):
            image = axes[row, column].imshow(operators[(row, name)], origin="lower", vmin=0, vmax=maximum, cmap="magma")
            axes[row, column].set_title(f"{name}\n(tau, nu)={path}", fontsize=9)
    fig.colorbar(image, cax=fig.add_subplot(grid[:, -1]), label="Magnitude")
    fig.suptitle(r"Atomic preview: $|W^H \Delta(\nu)\Pi(\tau)W|$", y=0.985)
    fig.savefig(OUT / "fig08_atomic_operator_preview.png", dpi=180); plt.close(fig)


def run_stage2a() -> dict[str, object]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, checkpoint = load_frozen_waveform(CHECKPOINT)
    before = frozen.clone()
    waveforms = qualification_waveforms(frozen)
    batch = sample_dd_v1(config.N_STAGE1A2_EVAL_CHANNELS, config.STAGE1A2_EVAL_SEED)
    channels = batch.matrices
    print("Part A: exact frozen reproduction", flush=True)
    analytic = analytic_rows(channels, waveforms)
    communication = communication_rows(channels, waveforms)
    analytic_summary = aggregate_rows(analytic, ("snr_db", "waveform", "k"))
    communication_summary = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k"))
    reproduction = verify_reproduction(communication_summary, analytic_summary)
    table_a = frozen_table(communication_summary, analytic_summary)
    bootstrap = paired_ber_bootstrap(communication)
    write_csv(OUT / "frozen_analytic_per_channel.csv", analytic)
    write_csv(OUT / "frozen_communication_per_channel.csv", communication)
    write_csv(OUT / "table_a_frozen_communication.csv", table_a)
    write_csv(OUT / "paired_ber_confidence_intervals.csv", bootstrap)
    plot_ber_vs_snr(communication_summary, "QPSK", "fig01_qpsk_ber_vs_snr.png")
    plot_ber_vs_snr(communication_summary, "16QAM", "fig02_16qam_ber_vs_snr.png")

    print("Part B: frozen receiver-budget sweep", flush=True)
    k_analytic = k_sweep_analytic_rows(channels, waveforms)
    k_analytic_summary = aggregate_rows(k_analytic, ("snr_db", "waveform", "k"))
    k_ber = communication_rows(
        channels,
        waveforms,
        constellations=(qpsk_constellation(),),
        snrs_db=config.STAGE2A_BER_SNRS_DB,
        k_values=config.STAGE2A_BER_K_VALUES,
        symbol_seed=config.STAGE2A_KSWEEP_SYMBOL_SEED,
        noise_seed=config.STAGE2A_KSWEEP_NOISE_SEED,
    )
    k_ber_summary = aggregate_rows(k_ber, ("modulation", "snr_db", "waveform", "k"))
    table_b = k_table(k_analytic_summary, k_ber_summary)
    write_csv(OUT / "k_sweep_analytic_per_channel.csv", k_analytic)
    write_csv(OUT / "k_sweep_analytic_summary.csv", k_analytic_summary)
    write_csv(OUT / "k_sweep_ber_per_channel.csv", k_ber)
    write_csv(OUT / "k_sweep_ber_summary.csv", k_ber_summary)
    write_csv(OUT / "table_b_k_sweep.csv", table_b)
    plot_k_curves(k_analytic_summary, k_ber_summary)

    print("Parts C/D: operator and frozen-basis structure", flush=True)
    channel_index, difficulty = representative_channel_index(channels)
    physical = channels[channel_index]
    operator_arrays = effective_operator_arrays(physical, waveforms)
    normalized_gains = batch.pre_normalization_gains[channel_index] / batch.normalization_scales[channel_index]
    channel_context = {
        "selection_rule": "nearest to median IDENTITY K=24 analytic MSE at 18 dB",
        "channel_index": channel_index,
        "identity_difficulty": float(difficulty[channel_index]),
        "bank_median_difficulty": float(np.median(difficulty)),
        "delays": batch.delays[channel_index].tolist(),
        "dopplers": batch.dopplers[channel_index].tolist(),
        "normalized_gain_magnitudes": torch.abs(normalized_gains).tolist(),
    }
    np.savez(OUT / "representative_effective_operators.npz", **operator_arrays)
    write_json(OUT / "representative_channel.json", channel_context)
    plot_effective_heatmaps(operator_arrays)
    structure_rows, structure_summary, structure_arrays = learned_structure(frozen)
    write_csv(OUT / "learned_basis_column_metrics.csv", structure_rows)
    write_json(OUT / "learned_basis_structure_summary.json", structure_summary)
    np.savez(OUT / "learned_basis_arrays.npz", **structure_arrays)
    plot_basis(structure_arrays)
    atomic_preview(waveforms)

    if not torch.equal(before, frozen):
        raise RuntimeError("frozen waveform mutated during Stage 2A")
    diagnostics = {
        "checkpoint": checkpoint,
        "checkpoint_unchanged": file_sha256(CHECKPOINT) == checkpoint["sha256"],
        "waveform_tensor_unchanged": bool(torch.equal(before, frozen)),
        "reproduction": reproduction,
        "channel_count": len(channels),
        "snrs_db": list(config.STAGE1A_SNRS_DB),
        "k_values": list(config.STAGE2A_K_VALUES),
        "ber_k_values": list(config.STAGE2A_BER_K_VALUES),
        "ber_sweep_snrs_db": list(config.STAGE2A_BER_SNRS_DB),
        "representative_channel": channel_context,
        "learned_structure": structure_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage2a_diagnostics.json", diagnostics)
    print(f"STAGE 2A COMPLETE in {diagnostics['runtime_seconds']:.1f} s", flush=True)
    return diagnostics


if __name__ == "__main__":
    run_stage2a()
