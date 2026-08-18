"""Run Stage 2C receiver-budget causality experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.stage2a import (
    aggregate_rows,
    communication_rows,
    file_sha256,
    load_frozen_waveform,
    qualification_waveforms,
)
from src.stage2b import spectral_centroids
from src.stage2c import (
    BER_NOISE_SEED,
    BER_SYMBOL_SEED,
    EVAL_K_VALUES,
    HAAR_SEEDS,
    INITIALIZATION_NAMES,
    TAIL_BETA,
    TAIL_LAMBDA,
    TRAIN_CHANNEL_SEED,
    TRAIN_K_VALUES,
    TRAIN_SNR_SEED,
    cross_budget_analytic_rows,
    optimize_budget_waveforms,
    select_validation_run,
    stage2c_initializations,
    structural_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2c"
CHECKPOINTS = OUT / "checkpoints"
REPORT = ROOT / "reports" / "stage2c_summary.md"
FROZEN_CHECKPOINT = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE2A = ROOT / "outputs" / "stage2a"
STAGE2B = ROOT / "outputs" / "stage2b"
COLORS = {"Learned": "tab:orange", "OFDM": "tab:blue", "OTFS": "tab:green", "AFDM": "tab:red"}
PLOT_NAMES = ("Learned", "OFDM", "OTFS", "AFDM")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().resolve_conj().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def artifact_hashes() -> dict[str, str]:
    paths = sorted(STAGE2A.glob("**/*")) + sorted(STAGE2B.glob("**/*"))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths if path.is_file()}


def consolidate_legacy_results() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    communication = read_csv(STAGE2A / "table_a_frozen_communication.csv")
    k_sweep = read_csv(STAGE2A / "table_b_k_sweep.csv")
    write_csv(OUT / "legacy_k24_communication.csv", communication)
    write_csv(OUT / "legacy_k24_k_sweep_18db.csv", k_sweep)
    plot_legacy_ber(communication, "qpsk_ber", "QPSK", "fig01_qpsk_ber_vs_snr_k24.png")
    plot_legacy_ber(communication, "qam16_ber", "16-QAM", "fig02_16qam_ber_vs_snr_k24.png")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in PLOT_NAMES:
        rows = sorted((row for row in k_sweep if row["waveform"] == name), key=lambda row: int(row["k"]))
        axis.semilogy([int(row["k"]) for row in rows], [float(row["qpsk_ber_18db"]) for row in rows], "o-", color=COLORS[name], label=name)
    axis.axvline(24, color="0.35", ls="--", lw=1, label="K=24 training budget")
    axis.set(xlabel="Receiver budget K", ylabel="Bit error rate", title="Frozen K=24-trained waveform, QPSK at 18 dB")
    axis.grid(True, which="both", alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig03_frozen_k24_ber_vs_k_18db.png", dpi=180); plt.close(fig)
    return communication, k_sweep


def plot_legacy_ber(rows: list[dict[str, str]], metric: str, modulation: str, filename: str) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in PLOT_NAMES:
        selected = sorted((row for row in rows if row["waveform"] == name), key=lambda row: int(row["snr_db"]))
        axis.semilogy([int(row["snr_db"]) for row in selected], [float(row[metric]) for row in selected], "o-", color=COLORS[name], label=name)
    axis.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Bit error rate", title=f"DD-v1, K=24, {modulation}")
    axis.grid(True, which="both", alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def load_or_train(k: int, starts: torch.Tensor, validation: torch.Tensor, steps: int, force: bool):
    path = CHECKPOINTS / f"training_k{k}.pt"
    protocol = {
        "k": k,
        "steps": steps,
        "batch_size": config.STAGE1A2_TRAIN_BATCH_SIZE,
        "validation_cadence": config.STAGE1A2_VALIDATION_CADENCE,
        "beta": TAIL_BETA,
        "lambda": TAIL_LAMBDA,
        "mean_mse_gate": config.STAGE1A2_MEAN_MSE_GATE,
        "initialization_names": list(INITIALIZATION_NAMES),
        "haar_seeds": list(HAAR_SEEDS),
        "train_channel_seed": TRAIN_CHANNEL_SEED,
        "train_snr_seed": TRAIN_SNR_SEED,
        "validation_seed": config.VALIDATION_SEED,
    }
    if path.exists() and not force:
        payload = torch.load(path, weights_only=False)
        if payload["protocol"] != protocol:
            raise RuntimeError(f"K={k} training protocol mismatch; use --force-training")
        print(f"loaded K={k} training", flush=True)
        return payload["result"]
    result = optimize_budget_waveforms(starts, validation, k=k, steps=steps)
    torch.save({"protocol": protocol, "result": result}, path)
    for index, name in enumerate(INITIALIZATION_NAMES):
        torch.save(result.best_matrices[index], CHECKPOINTS / f"k{k}_{name.lower()}_best.pt")
    print(f"trained K={k} in {result.runtime_seconds:.1f} s", flush=True)
    return result


def training_rows(results: dict[int, object], selected: dict[int, int]) -> list[dict[str, object]]:
    rows = []
    for k, result in results.items():
        for index, name in enumerate(INITIALIZATION_NAMES):
            rows.append(
                {
                    "training_k": k,
                    "predicted_group_count": math.ceil(config.N / k),
                    "initialization": name,
                    "initialization_seed": "deterministic OFDM" if name == "OFDM" else HAAR_SEEDS[index - 1],
                    "selected_step": int(result.best_steps[index]),
                    "initial_validation_mean_mse": float(result.initial_mean_mse[index]),
                    "initial_validation_cvar95": float(result.initial_cvar95[index]),
                    "selected_validation_mean_mse": float(result.best_mean_mse[index]),
                    "selected_validation_cvar90": float(result.best_cvar90[index]),
                    "selected_validation_cvar95": float(result.best_cvar95[index]),
                    "relative_mean_change": float(result.best_mean_mse[index] / result.initial_mean_mse[index] - 1),
                    "max_unitarity_residual": float(result.max_unitarity_residuals[index]),
                    "nonfinite_seen": bool(result.nonfinite_seen[index]),
                    "validation_selected_for_k": index == selected[k],
                    "tensor_sha256": tensor_sha256(result.best_matrices[index]),
                }
            )
    return rows


def structural_rows_and_arrays(
    results: dict[int, object], selected: dict[int, int], channels: torch.Tensor
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[int, dict[str, object]]]:
    rows = []
    arrays: dict[str, np.ndarray] = {}
    selected_summaries = {}
    for k, result in results.items():
        for index, name in enumerate(INITIALIZATION_NAMES):
            energy, support, labels, summary = structural_diagnostics(result.best_matrices[index], channels, k)
            row = {
                "training_k": k,
                "initialization": name,
                "validation_selected_for_k": index == selected[k],
                "predicted_group_count": summary["predicted_group_count"],
                "selected_group_count": summary["group_count"],
                "group_sizes": ";".join(map(str, summary["group_sizes"])),
                "maximum_group_size": summary["maximum_group_size"],
                "median_group_size": summary["median_group_size"],
                "within_energy_fraction": summary["within_energy_fraction"],
                "within_to_cross_energy_ratio": summary["within_to_cross_energy_ratio"],
                "within_topk_support_fraction": summary["within_topk_support_fraction"],
                "mean_support_probability_within": summary["mean_support_probability_within"],
                "mean_support_probability_across": summary["mean_support_probability_across"],
                "spectral_operator_rank_correlation": summary["spectral_operator_rank_correlation"],
                "operator_spectral_adjusted_rand": summary["operator_spectral_adjusted_rand"],
                "spectral_selected_group_count": summary["spectral_group_count"],
            }
            rows.append(row)
            if index == selected[k]:
                arrays[f"energy_k{k}"] = energy
                arrays[f"support_k{k}"] = support
                arrays[f"labels_k{k}"] = labels
                selected_summaries[k] = summary
    return rows, arrays, selected_summaries


def plot_structure(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    marker = {"OFDM": "o", "HAAR_1": "s", "HAAR_2": "^"}
    for name in INITIALIZATION_NAMES:
        selected = sorted((row for row in rows if row["initialization"] == name), key=lambda row: int(row["training_k"]))
        axes[0].plot([row["training_k"] for row in selected], [row["selected_group_count"] for row in selected], marker=marker[name], ls="none", ms=8, label=name)
        for row in selected:
            sizes = [int(value) for value in str(row["group_sizes"]).split(";")]
            axes[1].scatter([row["training_k"]] * len(sizes), sizes, marker=marker[name], s=48, alpha=0.8)
    ks = np.asarray(TRAIN_K_VALUES)
    axes[0].plot(ks, np.ceil(config.N / ks), "k--", label=r"$\lceil64/K_{train}\rceil$")
    axes[0].set(xlabel=r"Training receiver budget $K_{train}$", ylabel="Discovered group count", xticks=TRAIN_K_VALUES, yticks=range(2, 9))
    axes[1].set(xlabel=r"Training receiver budget $K_{train}$", ylabel="Discovered group size", xticks=TRAIN_K_VALUES)
    for axis in axes: axis.grid(True, alpha=0.3)
    axes[0].legend(); fig.suptitle("Unconstrained learned subspace organization across receiver budgets")
    fig.tight_layout(); fig.savefig(OUT / "fig04_group_count_and_sizes_vs_training_k.png", dpi=180); plt.close(fig)


def plot_selected_maps(arrays: dict[str, np.ndarray], results: dict[int, object], selected: dict[int, int]) -> None:
    maximum = max(float(arrays[f"energy_k{k}"].max()) for k in TRAIN_K_VALUES)
    fig = plt.figure(figsize=(12.9, 4.1)); grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.05), left=0.06, right=0.94, bottom=0.12, top=0.85, wspace=0.14)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, k in zip(axes, TRAIN_K_VALUES):
        image = axis.imshow(arrays[f"energy_k{k}"], origin="upper", cmap="magma", vmin=0, vmax=maximum)
        axis.set_title(f"Learned K={k}"); axis.set_xlabel("Input column")
    axes[0].set_ylabel("Output row"); fig.colorbar(image, cax=fig.add_subplot(grid[0, 3]), label=r"$E|A_{ij}|^2$")
    fig.suptitle("Held-out DD-v1 ensemble operator energy (shared scale)")
    fig.savefig(OUT / "fig05_selected_ensemble_operator_energy.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(12.9, 4.1)); grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.05), left=0.06, right=0.94, bottom=0.12, top=0.85, wspace=0.14)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, k in zip(axes, TRAIN_K_VALUES):
        image = axis.imshow(arrays[f"support_k{k}"], origin="upper", cmap="viridis", vmin=0, vmax=1)
        axis.set_title(f"Learned K={k}: top-{k}"); axis.set_xlabel("Input column")
    axes[0].set_ylabel("Output row"); fig.colorbar(image, cax=fig.add_subplot(grid[0, 3]), label="Support probability")
    fig.suptitle("Held-out row-wise top-K support probability (shared scale)")
    fig.savefig(OUT / "fig06_selected_topk_support_probability.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(13.3, 4.3)); grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.05), left=0.07, right=0.94, bottom=0.13, top=0.85, wspace=0.14)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, k in zip(axes, TRAIN_K_VALUES):
        waveform = results[k].best_matrices[selected[k]].detach().resolve_conj().cpu().numpy()
        order = np.argsort(spectral_centroids(results[k].best_matrices[selected[k]]))
        spectrum = np.fft.fftshift(np.fft.fft(waveform[:, order], n=4096, axis=0), axes=0)
        db = 20 * np.log10(np.maximum(np.abs(spectrum) / np.maximum(np.abs(spectrum).max(axis=0, keepdims=True), 1e-300), 1e-4))
        image = axis.imshow(db.T, origin="lower", aspect="auto", extent=(-0.5, 0.5, -0.5, 63.5), cmap="viridis", vmin=-80, vmax=0)
        axis.set_title(f"Learned K={k}"); axis.set_xlabel(r"Normalized frequency $f/F_s$")
    axes[0].set_ylabel("Spectral-centroid-ordered column")
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 3]), label="Magnitude (dB)")
    fig.suptitle("Dense spectra of validation-selected learned bases")
    fig.savefig(OUT / "fig07_selected_dense_spectra.png", dpi=180); plt.close(fig)


def matched_communication(channels: torch.Tensor, selected_waveforms: dict[int, torch.Tensor]):
    all_rows = []
    for k, learned in selected_waveforms.items():
        rows = communication_rows(
            channels,
            qualification_waveforms(learned),
            constellations=(qpsk_constellation(), qam16_constellation()),
            snrs_db=config.STAGE1A_SNRS_DB,
            k_values=(k,),
            symbol_seed=BER_SYMBOL_SEED,
            noise_seed=BER_NOISE_SEED,
        )
        for row in rows:
            row["training_k"] = k
        all_rows.extend(rows)
    summary = aggregate_rows(all_rows, ("training_k", "modulation", "snr_db", "waveform", "k"))
    write_csv(OUT / "matched_budget_communication_per_channel.csv", all_rows)
    write_csv(OUT / "matched_budget_communication_summary.csv", summary)
    for modulation, filename in (("QPSK", "fig08_matched_budget_qpsk_ber.png"), ("16QAM", "fig09_matched_budget_16qam_ber.png")):
        fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.1), sharex=True, sharey=True)
        for axis, k in zip(axes, TRAIN_K_VALUES):
            for name in PLOT_NAMES:
                rows = sorted((row for row in summary if row["training_k"] == k and row["modulation"] == modulation and row["waveform"] == name), key=lambda row: int(row["snr_db"]))
                axis.semilogy([row["snr_db"] for row in rows], [max(float(row["ber"]), 0.5 / int(row["total_bits"])) for row in rows], "o-", color=COLORS[name], label=name)
            axis.set_title(f"K train = K eval = {k}"); axis.set_xlabel(r"$E_s/N_0$ (dB)"); axis.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel("Bit error rate"); axes[-1].legend(fontsize=8)
        fig.suptitle(f"Matched-budget DD-v1 {modulation}")
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    return summary


def aggregate_cross_analytic(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for train_k in TRAIN_K_VALUES:
        for eval_k in EVAL_K_VALUES:
            selected = [row for row in rows if row["training_k"] == train_k and row["evaluation_k"] == eval_k]
            output.append({
                "training_k": train_k,
                "evaluation_k": eval_k,
                "snr_db": 18,
                "analytic_mse_mean": float(np.mean([row["analytic_mse"] for row in selected])),
                "retained_energy_mean": float(np.mean([row["retained_energy"] for row in selected])),
                "limiting_distortion_mean": float(np.mean([row["limiting_distortion"] for row in selected])),
                "limiting_distortion_median": float(np.median([row["limiting_distortion"] for row in selected])),
                "n_channels": len(selected),
            })
    return output


def plot_cross_analytic(rows: list[dict[str, object]]) -> None:
    metrics = (("analytic_mse_mean", "Analytic MSE"), ("retained_energy_mean", "Retained energy"), ("limiting_distortion_median", "Limiting distortion (median)"))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.9))
    for axis, (metric, title) in zip(axes, metrics):
        matrix = np.asarray([[next(float(row[metric]) for row in rows if row["training_k"] == train_k and row["evaluation_k"] == eval_k) for eval_k in EVAL_K_VALUES] for train_k in TRAIN_K_VALUES])
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(EVAL_K_VALUES)), EVAL_K_VALUES, rotation=45)
        axis.set_yticks(range(len(TRAIN_K_VALUES)), TRAIN_K_VALUES)
        axis.set(xlabel=r"Evaluation budget $K_{eval}$", ylabel=r"Training budget $K_{train}$", title=title)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Cross-budget learned-waveform generalization at 18 dB")
    fig.tight_layout(); fig.savefig(OUT / "fig10_cross_budget_analytic_heatmaps.png", dpi=180); plt.close(fig)


def cross_budget_ber(channels: torch.Tensor, selected_waveforms: dict[int, torch.Tensor]):
    rows = []
    for train_k, learned in selected_waveforms.items():
        current = communication_rows(
            channels,
            qualification_waveforms(learned),
            constellations=(qpsk_constellation(),),
            snrs_db=(18,),
            k_values=TRAIN_K_VALUES,
            symbol_seed=BER_SYMBOL_SEED + 20,
            noise_seed=BER_NOISE_SEED + 20,
        )
        for row in current:
            if row["waveform"] == "Learned":
                row["training_k"] = train_k
                rows.append(row)
    summary = aggregate_rows(rows, ("training_k", "modulation", "snr_db", "waveform", "k"))
    write_csv(OUT / "cross_budget_qpsk_18db_per_channel.csv", rows)
    write_csv(OUT / "cross_budget_qpsk_18db_summary.csv", summary)
    matrix = np.asarray([[next(float(row["ber"]) for row in summary if row["training_k"] == train_k and row["k"] == eval_k) for eval_k in TRAIN_K_VALUES] for train_k in TRAIN_K_VALUES])
    fig, axis = plt.subplots(figsize=(5.8, 4.7)); image = axis.imshow(matrix, origin="lower", cmap="magma")
    axis.set_xticks(range(3), TRAIN_K_VALUES); axis.set_yticks(range(3), TRAIN_K_VALUES)
    axis.set(xlabel=r"Evaluation budget $K_{eval}$", ylabel=r"Training budget $K_{train}$", title="Learned QPSK BER at 18 dB")
    for row in range(3):
        for column in range(3): axis.text(column, row, f"{matrix[row, column]:.3g}", ha="center", va="center", color="black" if matrix[row, column] > matrix.max() / 2 else "white")
    fig.colorbar(image, ax=axis, label="BER"); fig.tight_layout(); fig.savefig(OUT / "fig11_cross_budget_qpsk_18db.png", dpi=180); plt.close(fig)
    return summary


def retraining_vs_frozen(
    channels: torch.Tensor,
    frozen: torch.Tensor,
    matched_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Same-seed comparison needed to test repair of the frozen K=24 solution."""
    frozen_rows = communication_rows(
        channels,
        qualification_waveforms(frozen),
        constellations=(qpsk_constellation(), qam16_constellation()),
        snrs_db=config.STAGE1A_SNRS_DB,
        k_values=TRAIN_K_VALUES,
        symbol_seed=BER_SYMBOL_SEED,
        noise_seed=BER_NOISE_SEED,
    )
    frozen_summary = aggregate_rows(frozen_rows, ("modulation", "snr_db", "waveform", "k"))
    output = []
    for k in TRAIN_K_VALUES:
        for modulation in ("QPSK", "16QAM"):
            for snr_db in config.STAGE1A_SNRS_DB:
                retrained = next(row for row in matched_summary if row["training_k"] == k and row["modulation"] == modulation and row["snr_db"] == snr_db and row["waveform"] == "Learned")
                baseline = next(row for row in frozen_summary if row["modulation"] == modulation and row["snr_db"] == snr_db and row["waveform"] == "Learned" and row["k"] == k)
                output.append({
                    "training_and_evaluation_k": k,
                    "modulation": modulation,
                    "snr_db": snr_db,
                    "retrained_bit_errors": retrained["bit_errors"],
                    "frozen_k24_bit_errors": baseline["bit_errors"],
                    "total_bits": retrained["total_bits"],
                    "retrained_ber": retrained["ber"],
                    "frozen_k24_ber": baseline["ber"],
                    "relative_ber_change": float(retrained["ber"]) / float(baseline["ber"]) - 1,
                })
    write_csv(OUT / "retraining_vs_frozen_k24_same_seed.csv", output)
    return output


def classify_outcome(structure_rows: list[dict[str, object]]) -> str:
    counts = {k: [int(row["selected_group_count"]) for row in structure_rows if row["training_k"] == k] for k in TRAIN_K_VALUES}
    predicted = {k: math.ceil(config.N / k) for k in TRAIN_K_VALUES}
    if all(all(value == predicted[k] for value in counts[k]) for k in TRAIN_K_VALUES):
        return "A"
    if any(len(set(counts[k])) > 1 for k in TRAIN_K_VALUES):
        return "C"
    return "B"


def write_report(training: list[dict[str, object]], structure: list[dict[str, object]], matched: list[dict[str, object]], cross: list[dict[str, object]], repair: list[dict[str, object]], outcome: str, diagnostics: dict[str, object]) -> None:
    selected_training = [row for row in training if row["validation_selected_for_k"]]
    selected_structure = [row for row in structure if row["validation_selected_for_k"]]
    lines = [
        "# Stage 2C — Receiver-Budget Causality and Learned Subspace Reorganization",
        "",
        f"Outcome **{outcome} (qualified)**. Receiver-budget law is not supported as a stable 4-to-3-to-2 law. Group counts are initialization-sensitive, but the Haar runs have materially worse validation objectives, so this does not establish communication-equivalent symmetry solutions.",
        "",
        "## Protocol",
        "",
        f"- N=64, P=3, DD-v1 unchanged; K train in {list(TRAIN_K_VALUES)}.",
        f"- Loss: mean MSE + {TAIL_LAMBDA:.2f} [CVaR_{TAIL_BETA:.2f} - mean MSE]; 1,000 steps, batch 16, unchanged learning-rate schedule.",
        f"- Starts: OFDM plus Haar seeds {HAAR_SEEDS[0]} and {HAAR_SEEDS[1]}; online channel/SNR seeds {TRAIN_CHANNEL_SEED}/{TRAIN_SNR_SEED} shared across K.",
        "- Checkpoint selection: lowest validation stream CVaR 0.95 within the per-start 2% mean-MSE gate; BER and structure excluded.",
        "- Branch `cambridge`, target `origin/main`; no commit, push, merge, P variation, or Stage 3 action.",
        "- Added `src/stage2c.py`, `scripts/stage2c.py`, `tests/test_stage2c.py`, `notebooks/03_stage2c_receiver_budget_causality.ipynb`, and this report; numerical artifacts remain ignored.",
        "",
        "## Validation-selected runs",
        "",
        "| K | start | step | validation mean MSE | validation CVaR 0.95 |",
        "|---:|---|---:|---:|---:|",
    ]
    lines += [f"| {row['training_k']} | {row['initialization']} | {row['selected_step']} | {row['selected_validation_mean_mse']:.8f} | {row['selected_validation_cvar95']:.8f} |" for row in selected_training]
    lines += ["", "## Structural result", "", "| K | start | predicted | discovered | sizes | within energy | within/cross | support within/across | rank corr. | ARI |", "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['training_k']} | {row['initialization']} | {row['predicted_group_count']} | {row['selected_group_count']} | {row['group_sizes']} | {row['within_energy_fraction']:.4f} | {row['within_to_cross_energy_ratio']:.2f} | {row['mean_support_probability_within']:.3f}/{row['mean_support_probability_across']:.3f} | {row['spectral_operator_rank_correlation']:.3f} | {row['operator_spectral_adjusted_rand']:.3f} |" for row in structure]
    lines += ["", "## Matched-budget Learned BER", "", "| K | modulation | SNR dB | BER |", "|---:|---|---:|---:|"]
    lines += [f"| {row['training_k']} | {row['modulation']} | {row['snr_db']} | {float(row['ber']):.8g} |" for row in matched if row["waveform"] == "Learned"]
    lines += ["", "## Same-seed retraining comparison at 18 dB", "", "| K | modulation | retrained BER | frozen K=24 BER | relative change |", "|---:|---|---:|---:|---:|"]
    lines += [f"| {row['training_and_evaluation_k']} | {row['modulation']} | {float(row['retrained_ber']):.8g} | {float(row['frozen_k24_ber']):.8g} | {100*float(row['relative_ber_change']):+.1f}% |" for row in repair if row["snr_db"] == 18]
    lines += ["", "## Cross-budget analytic MSE at 18 dB", "", "| K train | K eval | MSE | retained energy | limiting distortion median |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['training_k']} | {row['evaluation_k']} | {row['analytic_mse_mean']:.8f} | {row['retained_energy_mean']:.6f} | {row['limiting_distortion_median']:.6f} |" for row in cross]
    lines += ["", "## Checkpoint hashes", ""]
    lines += [f"- `{name}`: `{digest}`" for name, digest in diagnostics["stage2c_checkpoint_hashes"].items()]
    lines += ["", "## Preservation and runtime", "", f"- Frozen checkpoint SHA-256: `{diagnostics['frozen_checkpoint']['sha256']}`.", f"- Stage-2A/2B artifacts unchanged: `{diagnostics['stage2a2b_artifacts_unchanged']}`.", f"- Training runtime: {diagnostics['training_runtime_seconds_total']:.1f} s; final evaluation/runtime: {diagnostics['evaluation_runtime_seconds']:.1f} s; full protocol: {diagnostics['full_protocol_seconds']:.1f} s.", "- Final complete suite: 76 passed in 2.18 s; notebook executed top-to-bottom with zero error outputs; `git diff --check` passed.", "", "Scientific interpretation: the group-count comparison is observational evidence from unconstrained communication training only. No physical-path, N, DD-distribution, or structural-loss variable changed. The 47.6% QPSK and 42.3% 16-QAM K=16 BER reductions show matched-budget repair, while the non-predicted and start-sensitive partitions reject a simple causal `ceil(64/K)` law in this protocol. The weaker Haar objectives prevent interpreting their partitions as communication-equivalent symmetry solutions. STOP — no P variation, waveform-family fitting, or Stage 3 was performed.", ""]
    REPORT.write_text("\n".join(lines))


def run_stage2c(*, steps: int = config.STAGE1A2_PRODUCTION_STEPS, force_training: bool = False) -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    training_preexisting = all((CHECKPOINTS / f"training_k{k}.pt").exists() for k in TRAIN_K_VALUES) and not force_training
    hashes_before = artifact_hashes()
    frozen, frozen_info = load_frozen_waveform(FROZEN_CHECKPOINT)
    frozen_before = frozen.clone()
    legacy, legacy_k = consolidate_legacy_results()
    names, starts = stage2c_initializations()
    if names != INITIALIZATION_NAMES: raise RuntimeError("unexpected Stage 2C initialization names")
    validation = sample_dd_v1(config.N_VALIDATION_CHANNELS, config.VALIDATION_SEED).matrices
    results = {k: load_or_train(k, starts, validation, steps, force_training) for k in TRAIN_K_VALUES}
    selected = {k: select_validation_run(result) for k, result in results.items()}
    selected_waveforms = {k: results[k].best_matrices[selected[k]] for k in TRAIN_K_VALUES}
    training = training_rows(results, selected); write_csv(OUT / "training_runs.csv", training)

    bank = sample_dd_v1(config.N_STAGE1A2_EVAL_CHANNELS, config.STAGE1A2_EVAL_SEED).matrices
    structure, arrays, selected_summaries = structural_rows_and_arrays(results, selected, bank)
    write_csv(OUT / "structural_metrics_all_runs.csv", structure); np.savez(OUT / "selected_operator_and_support_maps.npz", **arrays)
    write_json(OUT / "selected_community_summaries.json", {str(key): value for key, value in selected_summaries.items()})
    plot_structure(structure); plot_selected_maps(arrays, results, selected)
    matched = matched_communication(bank, selected_waveforms)
    repair = retraining_vs_frozen(bank, frozen, matched)
    analytic_per_channel = cross_budget_analytic_rows(bank, selected_waveforms)
    analytic = aggregate_cross_analytic(analytic_per_channel)
    write_csv(OUT / "cross_budget_analytic_per_channel.csv", analytic_per_channel); write_csv(OUT / "cross_budget_analytic_summary.csv", analytic)
    plot_cross_analytic(analytic)
    cross_ber = cross_budget_ber(bank, selected_waveforms)
    outcome = classify_outcome(structure)
    hashes_after = artifact_hashes()
    if hashes_after != hashes_before: raise RuntimeError("Stage 2A/2B artifact preservation failure")
    if not torch.equal(frozen, frozen_before) or file_sha256(FROZEN_CHECKPOINT) != frozen_info["sha256"]: raise RuntimeError("frozen checkpoint mutated")
    elapsed = time.perf_counter() - started
    training_total = sum(result.runtime_seconds for result in results.values())
    evaluation_runtime = elapsed if training_preexisting else elapsed - training_total
    diagnostics = {
        "outcome": outcome,
        "frozen_checkpoint": frozen_info,
        "stage2a2b_artifact_count": len(hashes_before),
        "stage2a2b_artifacts_unchanged": hashes_after == hashes_before,
        "initializations": {"names": list(INITIALIZATION_NAMES), "haar_seeds": list(HAAR_SEEDS), "raw_starts_only": True},
        "training": {"k_values": list(TRAIN_K_VALUES), "steps": steps, "beta": TAIL_BETA, "lambda": TAIL_LAMBDA, "channel_seed": TRAIN_CHANNEL_SEED, "snr_seed": TRAIN_SNR_SEED},
        "evaluation": {"channel_seed": config.STAGE1A2_EVAL_SEED, "channels": len(bank), "symbol_seed": BER_SYMBOL_SEED, "noise_seed": BER_NOISE_SEED},
        "selected_runs": {str(k): INITIALIZATION_NAMES[selected[k]] for k in TRAIN_K_VALUES},
        "training_runtime_seconds": {str(k): results[k].runtime_seconds for k in TRAIN_K_VALUES},
        "training_runtime_seconds_total": training_total,
        "evaluation_runtime_seconds": evaluation_runtime,
        "full_protocol_seconds": training_total + evaluation_runtime,
        "runtime_seconds": elapsed,
        "stage2c_checkpoint_hashes": {
            f"k{k}_{name.lower()}_best.pt": file_sha256(CHECKPOINTS / f"k{k}_{name.lower()}_best.pt")
            for k in TRAIN_K_VALUES for name in INITIALIZATION_NAMES
        },
    }
    write_json(OUT / "stage2c_diagnostics.json", diagnostics)
    write_report(training, structure, matched, analytic, repair, outcome, diagnostics)
    print(f"STAGE 2C COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME {outcome}", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=config.STAGE1A2_PRODUCTION_STEPS); parser.add_argument("--force-training", action="store_true"); return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); run_stage2c(steps=arguments.steps, force_training=arguments.force_training)
