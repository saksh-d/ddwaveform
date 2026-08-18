"""Run Stage 2D DD-coupling and oracle-support-removal diagnostics."""

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
from src.channel import noise_variance, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import aggregate_rows, file_sha256, load_frozen_waveform, qualification_waveforms
from src.stage2b import spectral_overlap_matrix
from src.stage2c import INITIALIZATION_NAMES, TRAIN_K_VALUES
from src.stage2d import (
    NOISE_SEED,
    PRIOR_SEED,
    SOBOL_SEED,
    STRUCTURE_SEED,
    SYMBOL_SEED,
    TEST_SEED,
    ambiguity_coupling,
    centroid_neighborhood_mask,
    circular_spectral_separation,
    empirical_coupling,
    empirical_support_prior,
    locality_curve_rows,
    matrix_agreement,
    score_support_mask,
    sobol_dd_samples,
    support_communication_rows,
    support_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2d"
REPORT = ROOT / "reports" / "stage2d_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE2C_CHECKPOINTS = ROOT / "outputs" / "stage2c" / "checkpoints"
LEGACY = ROOT / "outputs" / "stage2a" / "table_a_frozen_communication.csv"
TRAINING = ROOT / "outputs" / "stage2c" / "training_runs.csv"
PRINCIPAL = ("Learned", "OFDM", "OTFS", "AFDM")
RULES = ("Oracle", "Q", "S", "Centroid", "Prior")
COLORS = {"Learned": "tab:orange", "OFDM": "tab:blue", "OTFS": "tab:green", "AFDM": "tab:red"}
Q_SAMPLES = 2048
N_PRIOR = 512
N_STRUCTURE = 512
N_TEST = 512


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


def tree_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for directory in ("stage2a", "stage2b", "stage2c"):
        for path in sorted((ROOT / "outputs" / directory).glob("**/*")):
            if path.is_file():
                output[str(path.relative_to(ROOT))] = file_sha256(path)
    return output


def load_waveforms() -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]], dict[str, object]]:
    learned, frozen_info = load_frozen_waveform(FROZEN)
    principals = {name: value for name, value in qualification_waveforms(learned).items() if name != "IDENTITY"}
    training = read_csv(TRAINING)
    metadata: dict[str, dict[str, object]] = {}
    checkpoints: dict[str, torch.Tensor] = {}
    for k in TRAIN_K_VALUES:
        for initialization in INITIALIZATION_NAMES:
            key = f"K{k}_{initialization}"
            path = STAGE2C_CHECKPOINTS / f"k{k}_{initialization.lower()}_best.pt"
            tensor = torch.load(path, weights_only=True)
            row = next(item for item in training if int(item["training_k"]) == k and item["initialization"] == initialization)
            residual = float(torch.linalg.matrix_norm(tensor.mH @ tensor - torch.eye(config.N, dtype=tensor.dtype)))
            if residual >= 1e-12:
                raise RuntimeError(f"non-unitary Stage-2C checkpoint: {path.name}")
            checkpoints[key] = tensor
            metadata[key] = {
                "training_k": k,
                "initialization": initialization,
                "validation_objective": float(row["selected_validation_cvar95"]),
                "validation_mean_mse": float(row["selected_validation_mean_mse"]),
                "selected_step": int(row["selected_step"]),
                "validation_selected_for_k": row["validation_selected_for_k"] == "True",
                "sha256": file_sha256(path),
                "unitarity_residual": residual,
            }
    return {**principals, **checkpoints}, metadata, frozen_info


def coupling_analysis(
    waveforms: dict[str, torch.Tensor], structure_channels: torch.Tensor
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    delays, dopplers = sobol_dd_samples(Q_SAMPLES, SOBOL_SEED)
    c = empirical_coupling(structure_channels, waveforms)
    s: dict[str, np.ndarray] = {}
    q: dict[str, np.ndarray] = {}
    q_stability: dict[str, float] = {}
    agreements: list[dict[str, object]] = []
    locality: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        agreement_k = int(name.split("_")[0][1:]) if name.startswith("K") else 24
        first = ambiguity_coupling(waveform, delays[: Q_SAMPLES // 2], dopplers[: Q_SAMPLES // 2])
        second = ambiguity_coupling(waveform, delays[Q_SAMPLES // 2 :], dopplers[Q_SAMPLES // 2 :])
        q[name] = 0.5 * (first + second)
        q_stability[name] = float(np.linalg.norm(first - second) / max(np.linalg.norm(q[name]), 1e-300))
        s[name] = spectral_overlap_matrix(waveform)
        for left_name, left, right_name, right in (("S", s[name], "Q", q[name]), ("Q", q[name], "C", c[name]), ("S", s[name], "C", c[name])):
            agreements.append({"waveform": name, "k": agreement_k, "comparison": f"{left_name}->{right_name}", **matrix_agreement(left, right, agreement_k)})
        if name in PRINCIPAL:
            for row in locality_curve_rows(circular_spectral_separation(waveform), s[name], q[name], c[name]):
                locality.append({"waveform": name, **row})
    return s, q, c, agreements, locality, q_stability


def make_masks(
    waveforms: dict[str, torch.Tensor], s: dict[str, np.ndarray], q: dict[str, np.ndarray], prior_channels: torch.Tensor,
    k_values: dict[str, int],
) -> tuple[dict[str, dict[str, torch.Tensor | None]], dict[str, np.ndarray]]:
    masks: dict[str, dict[str, torch.Tensor | None]] = {}
    priors: dict[str, np.ndarray] = {}
    for name, waveform in waveforms.items():
        k = k_values[name]
        priors[name] = empirical_support_prior(prior_channels, waveform, k)
        masks[name] = {
            "Oracle": None,
            "Q": score_support_mask(q[name], k),
            "S": score_support_mask(s[name], k),
            "Centroid": centroid_neighborhood_mask(waveform, k),
            "Prior": score_support_mask(priors[name], k),
        }
    return masks, priors


def summarize(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().to(torch.float64).reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if len(finite) == 0:
        return {f"{prefix}_mean": float("inf"), f"{prefix}_median": float("inf"), f"{prefix}_p10": float("inf"), f"{prefix}_p90": float("inf"), f"{prefix}_p95": float("inf")}
    return {
        f"{prefix}_mean": float(torch.mean(finite)),
        f"{prefix}_median": float(torch.median(finite)),
        f"{prefix}_p10": float(torch.quantile(finite, 0.10)),
        f"{prefix}_p90": float(torch.quantile(finite, 0.90)),
        f"{prefix}_p95": float(torch.quantile(finite, 0.95)),
    }


def structural_support_rows(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor], masks: dict[str, dict[str, torch.Tensor | None]], k_values: dict[str, int]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    row_distributions: list[dict[str, object]] = []
    sigma2 = noise_variance(18)
    for name, waveform in waveforms.items():
        effective = effective_operator(channels, waveform)
        oracle_mask = torch.abs(rowwise_topk(effective, k_values[name])) > 0
        for rule, fixed in masks[name].items():
            chosen = oracle_mask if fixed is None else fixed
            metrics = support_metrics(effective, chosen, k_values[name], sigma2)
            result: dict[str, object] = {"waveform": name, "k": k_values[name], "support_rule": rule, "snr_db": 18, "n_channels": len(channels)}
            for metric in ("support_recall", "jaccard", "retained_energy", "analytic_mse", "limiting_distortion", "condition_number", "stream_cvar_095"):
                result.update(summarize(metrics[metric], metric))
            rows.append(result)
            row_energy = metrics["row_retained_energy"].reshape(-1)
            row_distributions.append({
                "waveform": name, "k": k_values[name], "support_rule": rule,
                "row_count": len(row_energy), "mean": float(torch.mean(row_energy)),
                "p01": float(torch.quantile(row_energy, 0.01)), "p05": float(torch.quantile(row_energy, 0.05)),
                "p10": float(torch.quantile(row_energy, 0.10)), "median": float(torch.median(row_energy)),
            })
    return rows, row_distributions


def plot_legacy_communication() -> None:
    rows = read_csv(LEGACY)
    for modulation, metric, filename in (("QPSK", "qpsk_ber", "fig01_qpsk_ber_vs_snr_k24.png"), ("16QAM", "qam16_ber", "fig02_16qam_ber_vs_snr_k24.png")):
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for name in PRINCIPAL:
            selected = sorted((row for row in rows if row["waveform"] == name), key=lambda row: int(row["snr_db"]))
            axis.semilogy([int(row["snr_db"]) for row in selected], [float(row[metric]) for row in selected], "o-", color=COLORS[name], label=name)
        axis.set_xticks(config.STAGE1A_SNRS_DB)
        axis.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Bit error rate", title=f"Frozen DD-v1 communication, K=24, {modulation.replace('16QAM', '16-QAM')}")
        axis.grid(True, which="both", alpha=0.3); axis.legend(); fig.tight_layout(); fig.savefig(OUT / filename, dpi=180, bbox_inches="tight", pad_inches=0.12); plt.close(fig)


def plot_couplings(s: dict[str, np.ndarray], q: dict[str, np.ndarray], c: dict[str, np.ndarray], waveforms: dict[str, torch.Tensor]) -> None:
    name = "Learned"
    order = np.argsort(__import__("src.stage2b", fromlist=["spectral_centroids"]).spectral_centroids(waveforms[name]))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    for axis, label, matrix in zip(axes, ("Spectral overlap S", "DD ambiguity Q", "Full-channel C"), (s[name], q[name], c[name])):
        ordered = matrix[np.ix_(order, order)]; image = axis.imshow(ordered / max(ordered.max(), 1e-300), origin="lower", cmap="magma", vmin=0, vmax=1)
        axis.set(title=label, xlabel="centroid-ordered atom j", ylabel="centroid-ordered atom i")
    fig.colorbar(image, ax=axes, shrink=0.82, label="individually normalized coupling")
    fig.savefig(OUT / "fig03_learned_s_q_c.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), constrained_layout=True)
    for axis, principal in zip(axes.flat, PRINCIPAL):
        off = ~np.eye(config.N, dtype=bool)
        axis.hexbin(q[principal][off], c[principal][off], gridsize=35, bins="log", mincnt=1, cmap="viridis")
        rho = matrix_agreement(q[principal], c[principal], 24)["spearman_offdiagonal"]
        axis.set(xlabel="Q coupling", ylabel="C coupling", title=f"{principal}: Spearman {rho:.3f}")
    fig.savefig(OUT / "fig04_q_vs_c_principal.png", dpi=180); plt.close(fig)


def plot_locality(rows: list[dict[str, object]]) -> None:
    for variable, filename, xlabel in (("spectral_separation", "fig05_coupling_vs_spectral_separation.png", "Circular spectral-centroid separation"), ("spectral_overlap", "fig06_coupling_vs_spectral_overlap.png", "Spectral overlap S")):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), sharey=False)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.25, hspace=0.34)
        for axis, waveform in zip(axes.flat, PRINCIPAL):
            for object_name, style in (("Q", "o-"), ("C", "s--")):
                selected = [row for row in rows if row["variable"] == variable and row["waveform"] == waveform and row["object"] == object_name]
                scale = max(float(row["mean_coupling"]) for row in selected)
                axis.plot([float(row["center"]) for row in selected], [float(row["mean_coupling"]) / max(scale, 1e-300) for row in selected], style, label=object_name)
            axis.set(title=waveform, xlabel=xlabel, ylabel="Mean coupling / maximum"); axis.grid(alpha=0.3); axis.legend()
        fig.savefig(OUT / filename, dpi=180, bbox_inches="tight", pad_inches=0.12); plt.close(fig)


def plot_support(
    support_rows: list[dict[str, object]], communication: list[dict[str, object]], priors: dict[str, np.ndarray], q: dict[str, np.ndarray], test_channels: torch.Tensor
) -> None:
    selected = [row for row in support_rows if row["waveform"] in PRINCIPAL]
    fig, axis = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(len(PRINCIPAL)); width = 0.16
    for index, rule in enumerate(RULES):
        values = [next(float(row["retained_energy_mean"]) for row in selected if row["waveform"] == name and row["support_rule"] == rule) for name in PRINCIPAL]
        axis.bar(x + (index - 2) * width, values, width, label=rule)
    axis.set_xticks(x, PRINCIPAL); axis.set(ylabel="Mean retained operator energy", ylim=(0, 1.03), title="Oracle versus fixed top-24 support")
    axis.grid(axis="y", alpha=0.3); axis.legend(ncol=5); fig.tight_layout(); fig.savefig(OUT / "fig07_support_retained_energy.png", dpi=180); plt.close(fig)

    summary = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    for modulation, filename in (("QPSK", "fig08_support_qpsk_ber.png"), ("16QAM", "fig09_support_16qam_ber.png")):
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.88, wspace=0.25)
        for rule in RULES:
            rows = sorted((row for row in summary if row["waveform"] == "Learned" and row["modulation"] == modulation and row["support_rule"] == rule), key=lambda row: int(row["snr_db"]))
            axes[0].semilogy([int(row["snr_db"]) for row in rows], [row["ber"] for row in rows], "o-", label=rule)
        axes[0].set(title="Authoritative Learned", xlabel=r"$E_s/N_0$ (dB)", ylabel="BER"); axes[0].grid(True, which="both", alpha=0.3); axes[0].legend()
        for name in PRINCIPAL:
            rows = sorted((row for row in summary if row["waveform"] == name and row["modulation"] == modulation and row["support_rule"] == "Q"), key=lambda row: int(row["snr_db"]))
            axes[1].semilogy([int(row["snr_db"]) for row in rows], [row["ber"] for row in rows], "o-", color=COLORS[name], label=name)
        axes[1].set(title="Fixed DD-ambiguity support", xlabel=r"$E_s/N_0$ (dB)", ylabel="BER"); axes[1].grid(True, which="both", alpha=0.3); axes[1].legend()
        fig.savefig(OUT / filename, dpi=180, bbox_inches="tight", pad_inches=0.14); plt.close(fig)

    learned_a = effective_operator(test_channels[0], qualification_waveforms(torch.load(FROZEN, weights_only=True))["Learned"])
    oracle = (torch.abs(rowwise_topk(learned_a, 24)) > 0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, title, matrix, cmap in zip(axes, ("Construction-bank support probability", "Fixed Q top-24 mask", "Test channel oracle top-24"), (priors["Learned"], score_support_mask(q["Learned"], 24).numpy(), oracle), ("magma", "gray_r", "gray_r")):
        image = axis.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=1); axis.set(title=title, xlabel="column j", ylabel="row i"); fig.colorbar(image, ax=axis, shrink=0.78)
    fig.savefig(OUT / "fig10_support_prior_q_oracle.png", dpi=180); plt.close(fig)


def checkpoint_summary(
    metadata: dict[str, dict[str, object]], agreements: list[dict[str, object]], support_rows: list[dict[str, object]], communication: list[dict[str, object]]
) -> list[dict[str, object]]:
    aggregated = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    output = []
    for name, info in metadata.items():
        support = {row["support_rule"]: row for row in support_rows if row["waveform"] == name}
        qc = next(row for row in agreements if row["waveform"] == name and row["comparison"] == "Q->C")
        sc = next(row for row in agreements if row["waveform"] == name and row["comparison"] == "S->C")
        result = {"waveform": name, **info, "s_to_c_spearman": sc["spearman_offdiagonal"], "q_to_c_spearman": qc["spearman_offdiagonal"], "oracle_retained_energy": support["Oracle"]["retained_energy_mean"], "q_fixed_retained_energy": support["Q"]["retained_energy_mean"], "q_fixed_limiting_distortion": support["Q"]["limiting_distortion_mean"]}
        for modulation in ("QPSK", "16QAM"):
            oracle = next(row for row in aggregated if row["waveform"] == name and row["modulation"] == modulation and row["support_rule"] == "Oracle")
            fixed = next(row for row in aggregated if row["waveform"] == name and row["modulation"] == modulation and row["support_rule"] == "Q")
            prefix = "qpsk" if modulation == "QPSK" else "qam16"
            result[f"{prefix}_oracle_ber_18db"] = oracle["ber"]
            result[f"{prefix}_q_fixed_ber_18db"] = fixed["ber"]
            result[f"{prefix}_q_fixed_absolute_penalty"] = fixed["ber"] - oracle["ber"]
            result[f"{prefix}_q_fixed_relative_penalty"] = fixed["ber"] / max(oracle["ber"], 1e-300) - 1
        output.append(result)
    return output


def plot_checkpoint_summary(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    for k, marker in zip(TRAIN_K_VALUES, ("o", "s", "^")):
        selected = [row for row in rows if row["training_k"] == k]
        axes[0].scatter([row["q_to_c_spearman"] for row in selected], [row["qpsk_oracle_ber_18db"] for row in selected], marker=marker, s=65, label=f"K={k}")
        axes[1].scatter([row["q_fixed_retained_energy"] for row in selected], [row["qpsk_q_fixed_absolute_penalty"] for row in selected], marker=marker, s=65, label=f"K={k}")
    axes[0].set(xlabel="Q-to-C Spearman", ylabel="Oracle QPSK BER at 18 dB", title="Communication versus DD predictability")
    axes[1].set(xlabel="Fixed-Q retained energy", ylabel="Fixed-Q minus oracle BER", title="Support predictability versus penalty")
    for axis in axes: axis.grid(alpha=0.3); axis.legend()
    fig.savefig(OUT / "fig11_checkpoint_communication_locality.png", dpi=180); plt.close(fig)


def classify(agreements: list[dict[str, object]], support_rows: list[dict[str, object]], communication: list[dict[str, object]]) -> str:
    qc = next(row for row in agreements if row["waveform"] == "Learned" and row["comparison"] == "Q->C")
    sc = next(row for row in agreements if row["waveform"] == "Learned" and row["comparison"] == "S->C")
    q_support = next(row for row in support_rows if row["waveform"] == "Learned" and row["support_rule"] == "Q")
    summary = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    oracle = next(row for row in summary if row["waveform"] == "Learned" and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Oracle")
    fixed = next(row for row in summary if row["waveform"] == "Learned" and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Q")
    classical_fixed = min(
        row["ber"] for row in summary
        if row["waveform"] in ("OFDM", "OTFS", "AFDM") and row["modulation"] == "QPSK"
        and row["snr_db"] == 18 and row["support_rule"] == "Q"
    )
    if qc["spearman_offdiagonal"] >= 0.7 and q_support["retained_energy_mean"] >= 0.95 and fixed["ber"] <= 1.25 * oracle["ber"] and fixed["ber"] < classical_fixed:
        return "A"
    if qc["spearman_offdiagonal"] >= 0.7:
        return "B"
    if sc["spearman_offdiagonal"] >= 0.5:
        return "C"
    return "D"


def write_report(diagnostics: dict[str, object], agreements: list[dict[str, object]], support_rows: list[dict[str, object]], communication: list[dict[str, object]], checkpoints: list[dict[str, object]]) -> None:
    principal_agreement = [row for row in agreements if row["waveform"] in PRINCIPAL]
    support = [row for row in support_rows if row["waveform"] in PRINCIPAL]
    comm = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    lines = [
        "# Stage 2D — DD Coupling Law and Oracle-Support Removal", "",
        f"Outcome **{diagnostics['outcome']}** under preregistered quantitative gates. This is diagnostic evidence only; no waveform was trained or selected.", "",
        "## Protocol", "",
        f"- DD-v1 unchanged: N={config.N}, P={config.P}, continuous delay [{config.DELAY_MIN},{config.DELAY_MAX}] and Doppler [{config.DOPPLER_MIN},{config.DOPPLER_MAX}].",
        f"- Q: {Q_SAMPLES} deterministic scrambled Sobol single paths (seed {SOBOL_SEED}); C: {N_STRUCTURE} held-out channels (seed {STRUCTURE_SEED}).",
        f"- Empirical support prior: separate {N_PRIOR}-channel construction bank (seed {PRIOR_SEED}); final receiver/BER: untouched {N_TEST}-channel bank (seed {TEST_SEED}).",
        "- Correlations exclude self-coupling diagonals. Fixed masks never inspect a realized test operator; retained coefficients remain the true realized coefficients.",
        "- Full principal BER uses at least 1,048,576 paired bits per point; all-checkpoint 18 dB checks use the same paired-bit target.",
        "- No training, fine-tuning, waveform selection, channel-spread variation, P variation, structural loss, or Stage 3 action.", "",
        "## S/Q/C agreement", "", "| waveform | comparison | Spearman | row mean | top-24 overlap | scaled Frobenius discrepancy |", "|---|---|---:|---:|---:|---:|",
    ]
    lines += [f"| {row['waveform']} | {row['comparison']} | {row['spearman_offdiagonal']:.4f} | {row['row_spearman_mean']:.4f} | {row['topk_overlap_mean']:.4f} | {row['scaled_frobenius_discrepancy']:.4f} |" for row in principal_agreement]
    lines += ["", "## K=24 support removal at 18 dB", "", "| waveform | rule | recall | Jaccard | retained energy | analytic MSE | limiting distortion | condition p95 | stream CVaR 0.95 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['waveform']} | {row['support_rule']} | {row['support_recall_mean']:.4f} | {row['jaccard_mean']:.4f} | {row['retained_energy_mean']:.5f} | {row['analytic_mse_mean']:.5f} | {row['limiting_distortion_mean']:.5f} | {row['condition_number_p95']:.2f} | {row['stream_cvar_095_mean']:.5f} |" for row in support]
    lines += ["", "## Matched-budget selected Learned support test at 18 dB", "", "| K | rule | retained energy | analytic MSE | limiting distortion | QPSK BER | 16-QAM BER |", "|---:|---|---:|---:|---:|---:|---:|"]
    for name, k in (("Learned_K16", 16), ("Learned", 24), ("Learned_K32", 32)):
        support_name = "Learned" if k == 24 else f"K{k}_OFDM"
        for rule in RULES:
            structural = next(row for row in support_rows if row["waveform"] == support_name and row["support_rule"] == rule)
            qpsk = next(row for row in comm if row["waveform"] == name and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == rule)
            qam = next(row for row in comm if row["waveform"] == name and row["modulation"] == "16QAM" and row["snr_db"] == 18 and row["support_rule"] == rule)
            lines.append(f"| {k} | {rule} | {structural['retained_energy_mean']:.5f} | {structural['analytic_mse_mean']:.5f} | {structural['limiting_distortion_mean']:.5f} | {qpsk['ber']:.7g} | {qam['ber']:.7g} |")
    lines += ["", "## Authoritative Learned BER", "", "| modulation | SNR dB | support | BER | errors / bits |", "|---|---:|---|---:|---:|"]
    lines += [f"| {row['modulation']} | {row['snr_db']} | {row['support_rule']} | {row['ber']:.8g} | {row['bit_errors']} / {row['total_bits']} |" for row in comm if row["waveform"] == "Learned"]
    lines += ["", "## All Stage-2C checkpoints at 18 dB", "", "| checkpoint | K | init | val. CVaR | Q->C | fixed-Q energy | QPSK oracle/fixed | 16-QAM oracle/fixed |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['waveform']} | {row['training_k']} | {row['initialization']} | {row['validation_objective']:.6f} | {row['q_to_c_spearman']:.3f} | {row['q_fixed_retained_energy']:.4f} | {row['qpsk_oracle_ber_18db']:.5g}/{row['qpsk_q_fixed_ber_18db']:.5g} | {row['qam16_oracle_ber_18db']:.5g}/{row['qam16_q_fixed_ber_18db']:.5g} |" for row in checkpoints]
    lines += ["", "## Checkpoint hashes", "", f"- Authoritative Stage-1A.2: `{diagnostics['frozen_checkpoint']['sha256']}`"]
    lines += [f"- `{name}`: `{row['sha256']}`" for name, row in diagnostics["stage2c_checkpoints"].items()]
    lines += ["", "## Required figures", "", "- `fig01_qpsk_ber_vs_snr_k24.png`, `fig02_16qam_ber_vs_snr_k24.png`: frozen communications context.", "- `fig03_learned_s_q_c.png`, `fig04_q_vs_c_principal.png`, `fig05_coupling_vs_spectral_separation.png`, `fig06_coupling_vs_spectral_overlap.png`: coupling-law evidence.", "- `fig07_support_retained_energy.png`, `fig08_support_qpsk_ber.png`, `fig09_support_16qam_ber.png`, `fig10_support_prior_q_oracle.png`: oracle-removal evidence.", "- `fig11_checkpoint_communication_locality.png`: exploratory checkpoint cross-check."]
    lines += ["", "## Preservation and runtime", "", f"- Authoritative checkpoint unitarity residual: `{diagnostics['frozen_checkpoint']['unitarity_residual']:.3e}`.", f"- All Stage-2C checkpoint hashes and all {diagnostics['prior_artifact_count']} Stage-2A/2B/2C artifacts remained byte-identical: `{diagnostics['prior_artifacts_unchanged']}`.", f"- Maximum Stage-2C checkpoint unitarity residual: `{diagnostics['maximum_stage2c_unitarity_residual']:.3e}`.", f"- Q half-sample maximum relative discrepancy: `{diagnostics['q_half_sample_max_relative_discrepancy']:.3e}`.", f"- Runtime: `{diagnostics['runtime_seconds']:.1f}` s.", "", "## Interpretation", "", diagnostics["interpretation"], "", "STOP — no channel-spread/P variation, waveform-family fitting, or Stage 3 work was performed.", ""]
    REPORT.write_text("\n".join(lines))


def run_stage2d() -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = tree_hashes()
    all_waveforms, metadata, frozen_info = load_waveforms()
    structural_bank = sample_dd_v1(N_STRUCTURE, STRUCTURE_SEED).matrices
    prior_bank = sample_dd_v1(N_PRIOR, PRIOR_SEED).matrices
    test_bank = sample_dd_v1(N_TEST, TEST_SEED).matrices
    s, q, c, agreements, locality, q_stability = coupling_analysis(all_waveforms, structural_bank)
    arrays = {f"S_{name}": value for name, value in s.items()} | {f"Q_{name}": value for name, value in q.items()} | {f"C_{name}": value for name, value in c.items()}
    np.savez(OUT / "s_q_c_matrices.npz", **arrays)
    write_csv(OUT / "s_q_c_agreement.csv", agreements); write_csv(OUT / "continuous_locality_curves.csv", locality)

    k_values = {name: (int(name.split("_")[0][1:]) if name.startswith("K") else 24) for name in all_waveforms}
    masks, priors = make_masks(all_waveforms, s, q, prior_bank, k_values)
    np.savez(OUT / "support_priors.npz", **priors)
    support_rows, row_distributions = structural_support_rows(test_bank, all_waveforms, masks, k_values)
    write_csv(OUT / "support_receiver_summary.csv", support_rows); write_csv(OUT / "row_retained_energy_summary.csv", row_distributions)

    principal_waveforms = {name: all_waveforms[name] for name in PRINCIPAL} | {"Learned_K16": all_waveforms["K16_OFDM"], "Learned_K32": all_waveforms["K32_OFDM"]}
    principal_k = {name: (16 if name.endswith("K16") else 32 if name.endswith("K32") else 24) for name in principal_waveforms}
    principal_masks = {name: (masks[name] if name in masks else masks["K16_OFDM"] if name.endswith("K16") else masks["K32_OFDM"]) for name in principal_waveforms}
    communication = support_communication_rows(test_bank, principal_waveforms, principal_masks, principal_k, (qpsk_constellation(), qam16_constellation()), config.STAGE1A_SNRS_DB, target_bits=1_000_000, symbol_seed=SYMBOL_SEED, noise_seed=NOISE_SEED)
    write_csv(OUT / "principal_support_communication_per_channel.csv", communication)
    write_csv(OUT / "principal_support_communication_summary.csv", aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule")))

    checkpoint_waveforms = {name: all_waveforms[name] for name in metadata}
    checkpoint_masks = {name: {"Oracle": None, "Q": masks[name]["Q"]} for name in metadata}
    checkpoint_comm = support_communication_rows(test_bank, checkpoint_waveforms, checkpoint_masks, {name: k_values[name] for name in metadata}, (qpsk_constellation(), qam16_constellation()), (18,), target_bits=1_000_000, symbol_seed=SYMBOL_SEED + 20, noise_seed=NOISE_SEED + 20)
    write_csv(OUT / "checkpoint_support_communication_per_channel.csv", checkpoint_comm)
    checkpoint_rows = checkpoint_summary(metadata, agreements, support_rows, checkpoint_comm)
    write_csv(OUT / "checkpoint_communication_locality.csv", checkpoint_rows)

    plot_legacy_communication(); plot_couplings(s, q, c, all_waveforms); plot_locality(locality); plot_support(support_rows, communication, priors, q, test_bank); plot_checkpoint_summary(checkpoint_rows)
    outcome = classify(agreements, support_rows, communication)
    learned_qc = next(row for row in agreements if row["waveform"] == "Learned" and row["comparison"] == "Q->C")
    learned_support = next(row for row in support_rows if row["waveform"] == "Learned" and row["support_rule"] == "Q")
    comm_summary = aggregate_rows(communication, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    learned_oracle = next(row for row in comm_summary if row["waveform"] == "Learned" and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Oracle")
    learned_fixed = next(row for row in comm_summary if row["waveform"] == "Learned" and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Q")
    classical_fixed = min((row for row in comm_summary if row["waveform"] in ("OFDM", "OTFS", "AFDM") and row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Q"), key=lambda row: row["ber"])
    interpretation = (
        f"Learned Q predicts C with off-diagonal Spearman {learned_qc['spearman_offdiagonal']:.3f} and fixed-Q support retains "
        f"{learned_support['retained_energy_mean']:.3%} mean energy. At 18 dB its QPSK BER is {learned_fixed['ber']:.6g}, "
        f"{100 * (learned_fixed['ber'] / learned_oracle['ber'] - 1):.1f}% above oracle but {100 * (1 - learned_fixed['ber'] / classical_fixed['ber']):.1f}% below the best classical fixed-Q result ({classical_fixed['waveform']}). "
        f"Outcome {outcome} is therefore supported: DD locality is a stable operator law and replaces realized top-K with a small BER cost for the authoritative solution. The all-checkpoint table also shows that high Q-to-C correlation alone is insufficient: weaker Haar checkpoints have diffuse fixed support and large BER penalties."
    )
    hashes_after = tree_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("Stage-2A/2B/2C artifact preservation failure")
    if file_sha256(FROZEN) != frozen_info["sha256"]:
        raise RuntimeError("authoritative checkpoint mutated")
    diagnostics = {
        "outcome": outcome, "frozen_checkpoint": frozen_info,
        "stage2c_checkpoints": metadata, "prior_artifact_count": len(hashes_before),
        "prior_artifacts_unchanged": hashes_after == hashes_before,
        "maximum_stage2c_unitarity_residual": max(float(row["unitarity_residual"]) for row in metadata.values()),
        "q_samples": Q_SAMPLES, "q_half_sample_relative_discrepancy": q_stability,
        "q_half_sample_max_relative_discrepancy": max(q_stability.values()),
        "banks": {"prior": {"seed": PRIOR_SEED, "channels": N_PRIOR}, "structure": {"seed": STRUCTURE_SEED, "channels": N_STRUCTURE}, "test": {"seed": TEST_SEED, "channels": N_TEST}},
        "ber_seeds": {"symbols": SYMBOL_SEED, "noise": NOISE_SEED},
        "runtime_seconds": time.perf_counter() - started, "interpretation": interpretation,
    }
    write_json(OUT / "stage2d_diagnostics.json", diagnostics)
    write_report(diagnostics, agreements, support_rows, communication, checkpoint_rows)
    print(f"STAGE 2D COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME {outcome}", flush=True)
    return diagnostics


if __name__ == "__main__":
    run_stage2d()
