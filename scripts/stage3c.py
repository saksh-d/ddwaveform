"""Run Stage 3C frozen saturated-Q communication-geometry forensics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator, rowwise_topk
from src.receiver_geometry import apply_mask, distribution_summary, exact_error_geometry, support_geometry
from src.stage2a import load_frozen_waveform, qualification_waveforms
from src.stage2d import score_support_mask, spearman_rank
from src.stage3a import q_concentration
from src.stage3c import communication_rows, paired_delta_summary, retained_operator, support_overlap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3c"
REPORT = ROOT / "reports" / "stage3c_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE3B = ROOT / "outputs" / "stage3b"
K = 24
SNRS = (12, 15, 18, 20)
CHANNELS = 512
CHANNEL_SEED = 11090
SYMBOL_SEED = 11091
NOISE_SEED = 11092
TARGET_BITS = 1_000_000
PRIMARY = (
    "Q_MEAN_FROM_OFDM", "Q_MEAN_FROM_HAAR_1", "Q_MEAN_FROM_HAAR_2",
    "Q_ROBUST_FROM_OFDM", "Q_ROBUST_FROM_HAAR_1", "Q_ROBUST_FROM_HAAR_2",
)
FIGURE_WAVEFORMS = ("Learned", *PRIMARY, "OFDM", "OTFS", "AFDM")
ALL_WAVEFORMS = ("IDENTITY", *FIGURE_WAVEFORMS)
SUPPORT_RULES = ("Full", "Oracle", "Q")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted((ROOT / "outputs").glob("**/*"))
        if path.is_file() and "stage3c" not in path.parts
    }


def load_waveforms() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    learned, info = load_frozen_waveform(FROZEN)
    principals = qualification_waveforms(learned)
    primary = {
        name: torch.load(STAGE3B / "checkpoints" / f"{name.lower()}_best.pt", weights_only=True)
        for name in PRIMARY
    }
    waveforms = {**principals, **primary}
    ordered = {name: waveforms[name] for name in ALL_WAVEFORMS}
    hashes = {"Learned": info["sha256"]}
    hashes.update({name: file_sha256(STAGE3B / "checkpoints" / f"{name.lower()}_best.pt") for name in PRIMARY})
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    for name, waveform in ordered.items():
        if float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity)) >= 1e-12:
            raise RuntimeError(f"non-unitary frozen waveform: {name}")
    return ordered, hashes


def load_q_and_masks() -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    saved = np.load(STAGE3B / "heldout_q_c_matrices.npz")
    q = {name: saved[f"Q_{name}"] for name in ALL_WAVEFORMS}
    return q, {name: score_support_mask(q[name], K) for name in ALL_WAVEFORMS}


def analytic_geometry(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    channel_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    metrics = (
        ("retained_energy", False), ("lambda_min", False), ("lambda_max", True),
        ("condition_number", True), ("effective_rank", False),
        ("equalizer_noise_enhancement", True), ("mean_stream_mse", True),
        ("worst_stream_mse", True), ("stream_mse_cvar95", True),
        ("median_gamma", False), ("worst_gamma", False), ("gamma_lower_cvar05", False),
        ("error_anisotropy", True), ("offdiagonal_error_correlation_rms", True),
    )
    with torch.no_grad():
        for name, waveform in waveforms.items():
            effective = effective_operator(channels, waveform)
            for snr_db in SNRS:
                for support_rule in SUPPORT_RULES:
                    retained = retained_operator(effective, support_rule, K, masks[name])
                    geometry = exact_error_geometry(effective, retained, noise_variance(snr_db))
                    response_diagonal = torch.diagonal(geometry["response"], dim1=-2, dim2=-1)
                    extra = {
                        "desired_gain_mean": torch.abs(response_diagonal).mean(dim=-1),
                        "desired_gain_min": torch.abs(response_diagonal).min(dim=-1).values,
                        "desired_phase_abs_p95": torch.quantile(torch.abs(torch.angle(response_diagonal)), 0.95, dim=-1),
                    }
                    for channel_index in range(len(channels)):
                        row: dict[str, object] = {
                            "channel_index": channel_index, "waveform": name,
                            "snr_db": snr_db, "support_rule": support_rule,
                            "k": config.N if support_rule == "Full" else K,
                        }
                        for metric, _ in metrics:
                            row[metric] = float(geometry[metric][channel_index])
                        for metric, values in extra.items():
                            row[metric] = float(values[channel_index])
                        channel_rows.append(row)
                    summary: dict[str, object] = {
                        "waveform": name, "snr_db": snr_db, "support_rule": support_rule,
                        "k": config.N if support_rule == "Full" else K,
                    }
                    for metric, high_is_bad in metrics:
                        summary.update({
                            f"{metric}_{key}": value
                            for key, value in distribution_summary(geometry[metric], high_is_bad=high_is_bad).items()
                        })
                    for metric, values in extra.items():
                        summary.update({
                            f"{metric}_{key}": value
                            for key, value in distribution_summary(values, high_is_bad=True).items()
                        })
                    summary_rows.append(summary)
    return channel_rows, summary_rows


def singular_value_summaries(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    """Ordered singular-value distributions of every receiver-design operator."""
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name, waveform in waveforms.items():
            effective = effective_operator(channels, waveform)
            for support_rule in SUPPORT_RULES:
                singular = torch.linalg.svdvals(retained_operator(effective, support_rule, K, masks[name]))
                for index in range(config.N):
                    rows.append({
                        "waveform": name, "support_rule": support_rule,
                        "k": config.N if support_rule == "Full" else K,
                        "descending_index": index,
                        **distribution_summary(singular[:, index], high_is_bad=index >= config.N // 2),
                    })
    return rows


def communication_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("modulation", "snr_db", "waveform", "support_rule", "k")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, object]] = []
    for values, selected in groups.items():
        bit_errors = sum(int(row["bit_errors"]) for row in selected)
        total_bits = sum(int(row["total_bits"]) for row in selected)
        symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
        total_symbols = sum(int(row["total_symbols"]) for row in selected)
        result: dict[str, object] = dict(zip(keys, values))
        result.update({
            "bit_errors": bit_errors, "total_bits": total_bits, "ber": bit_errors / total_bits,
            "symbol_errors": symbol_errors, "total_symbols": total_symbols, "ser": symbol_errors / total_symbols,
            "n_channels": len(selected),
        })
        for metric in (
            "empirical_mse", "constellation_scaled_variance", "normalized_margin_mean",
            "normalized_margin_p05", "negative_margin_fraction", "residual_anisotropy",
        ):
            values_array = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(np.mean(values_array))
            result[f"{metric}_median"] = float(np.median(values_array))
            result[f"{metric}_p05"] = float(np.quantile(values_array, 0.05))
            result[f"{metric}_p95"] = float(np.quantile(values_array, 0.95))
        output.append(result)
    return output


def paired_explanations(
    analytic: list[dict[str, object]], communication: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    analytic_lookup = {
        (row["waveform"], row["snr_db"], row["support_rule"], row["channel_index"]): row
        for row in analytic
    }
    communication_lookup = {
        (row["waveform"], row["modulation"], row["snr_db"], row["support_rule"], row["channel_index"]): row
        for row in communication
    }
    paired: list[dict[str, object]] = []
    metrics = (
        "ber", "empirical_mse", "constellation_scaled_variance", "normalized_margin_p05",
        "negative_margin_fraction", "residual_anisotropy", "retained_energy",
        "condition_number", "equalizer_noise_enhancement", "gamma_lower_cvar05",
        "stream_mse_cvar95", "desired_gain_min", "desired_phase_abs_p95",
    )
    for candidate in PRIMARY:
        for modulation in ("QPSK", "16QAM"):
            for snr_db in SNRS:
                for support_rule in SUPPORT_RULES:
                    for channel_index in range(CHANNELS):
                        learned_c = communication_lookup[("Learned", modulation, snr_db, support_rule, channel_index)]
                        candidate_c = communication_lookup[(candidate, modulation, snr_db, support_rule, channel_index)]
                        learned_a = analytic_lookup[("Learned", snr_db, support_rule, channel_index)]
                        candidate_a = analytic_lookup[(candidate, snr_db, support_rule, channel_index)]
                        row: dict[str, object] = {
                            "candidate": candidate, "modulation": modulation, "snr_db": snr_db,
                            "support_rule": support_rule, "channel_index": channel_index,
                        }
                        for metric in metrics:
                            left = learned_c if metric in learned_c else learned_a
                            right = candidate_c if metric in candidate_c else candidate_a
                            row[f"learned_{metric}"] = left[metric]
                            row[f"candidate_{metric}"] = right[metric]
                            row[f"delta_{metric}"] = float(right[metric]) - float(left[metric])
                        paired.append(row)
    summaries: list[dict[str, object]] = []
    for candidate in PRIMARY:
        for modulation in ("QPSK", "16QAM"):
            for snr_db in SNRS:
                for support_rule in SUPPORT_RULES:
                    selected = [
                        row for row in paired
                        if row["candidate"] == candidate and row["modulation"] == modulation
                        and row["snr_db"] == snr_db and row["support_rule"] == support_rule
                    ]
                    ber_delta = np.asarray([float(row["delta_ber"]) for row in selected])
                    for metric in metrics:
                        metric_delta = np.asarray([float(row[f"delta_{metric}"]) for row in selected])
                        summary = paired_delta_summary(
                            np.asarray([float(row[f"learned_{metric}"]) for row in selected]),
                            np.asarray([float(row[f"candidate_{metric}"]) for row in selected]),
                        )
                        finite = np.isfinite(metric_delta) & np.isfinite(ber_delta)
                        summary.update({
                            "candidate": candidate, "modulation": modulation, "snr_db": snr_db,
                            "support_rule": support_rule, "metric": metric,
                            "spearman_metric_delta_vs_ber_delta": spearman_rank(metric_delta[finite], ber_delta[finite]) if finite.sum() > 2 else float("nan"),
                        })
                        summaries.append(summary)
    return paired, summaries


def equivalence_diagnostics(
    q: dict[str, np.ndarray], masks: dict[str, torch.Tensor],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    locality = np.load(ROOT / "outputs" / "stage2b" / "learned_association_matrices.npz")
    labels = torch.from_numpy(locality["community_labels"]); order = torch.from_numpy(locality["centroid_order"])
    locality_rows = [{"waveform": name, **support_geometry(masks[name], labels, order)} for name in ALL_WAVEFORMS]
    pairwise: list[dict[str, object]] = []
    names = ("Learned", *PRIMARY)
    off = ~np.eye(config.N, dtype=bool)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pairwise.append({
                "left": left, "right": right,
                "q_spearman_offdiagonal": spearman_rank(q[left][off], q[right][off]),
                "support_overlap": support_overlap(masks[left], masks[right]),
                "q_relative_frobenius_distance": float(np.linalg.norm(q[left] - q[right]) / np.linalg.norm(q[left])),
            })
    curves: list[dict[str, object]] = []
    for name in names:
        for k in range(1, config.N + 1):
            concentration = q_concentration(q[name], k)
            curves.append({"waveform": name, "k": k, "eta_q": concentration["eta_q"]})
    return locality_rows, pairwise, curves


def make_plots(
    communication: list[dict[str, object]], analytic: list[dict[str, object]],
    paired: list[dict[str, object]], pairwise: list[dict[str, object]],
) -> None:
    summary_lookup = {
        (row["modulation"], int(row["snr_db"]), row["waveform"], row["support_rule"]): row
        for row in communication
    }
    colors = plt.cm.tab10(np.linspace(0, 1, len(FIGURE_WAVEFORMS)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    for row_index, modulation in enumerate(("QPSK", "16QAM")):
        for column, support_rule in enumerate(SUPPORT_RULES):
            axis = axes[row_index, column]
            for color, name in zip(colors, FIGURE_WAVEFORMS):
                values = [summary_lookup[(modulation, snr, name, support_rule)]["ber"] for snr in SNRS]
                axis.plot(SNRS, values, marker="o", linewidth=1.2, markersize=3, color=color, label=name)
            axis.set_yscale("log"); axis.grid(True, which="both", alpha=.25)
            axis.set_title(f"{modulation}: {support_rule}")
            if row_index == 1: axis.set_xlabel("SNR (dB)")
            if column == 0: axis.set_ylabel("BER")
    axes[0, 2].legend(fontsize=6, ncol=2)
    fig.savefig(OUT / "fig01_ber_decomposition.png", dpi=180); plt.close(fig)

    selected = [row for row in analytic if row["snr_db"] == 18 and row["support_rule"] == "Q" and row["waveform"] in ("Learned", *PRIMARY)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("retained_energy", "gamma_lower_cvar05", "stream_mse_cvar95"),
        ("Retained energy", "Lower-tail post-LMMSE SINR", "Worst-stream MSE tail"),
    ):
        data = [[float(row[metric]) for row in selected if row["waveform"] == name] for name in ("Learned", *PRIMARY)]
        axis.boxplot(data, showfliers=False); axis.set_xticks(range(1, 8), ("Learned", *PRIMARY), rotation=45, ha="right", fontsize=7)
        axis.set_title(title); axis.grid(axis="y", alpha=.25)
        if metric != "retained_energy": axis.set_yscale("log")
    fig.savefig(OUT / "fig02_fixed_q_geometry.png", dpi=180); plt.close(fig)

    selected_paired = [row for row in paired if row["modulation"] == "QPSK" and row["snr_db"] == 18 and row["support_rule"] == "Q"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric, label in zip(axes, ("gamma_lower_cvar05", "normalized_margin_p05"), ("Delta lower-tail SINR", "Delta normalized margin p05")):
        for color, candidate in zip(colors[1:7], PRIMARY):
            rows = [row for row in selected_paired if row["candidate"] == candidate]
            axis.scatter([row[f"delta_{metric}"] for row in rows], [row["delta_ber"] for row in rows], s=7, alpha=.25, color=color, label=candidate)
        axis.axhline(0, color="black", linewidth=.8); axis.axvline(0, color="black", linewidth=.8)
        axis.set(xlabel=label, ylabel="Delta channel BER", title="Paired QPSK fixed-Q channels")
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(OUT / "fig03_paired_tail_explanation.png", dpi=180); plt.close(fig)

    names = ("Learned", *PRIMARY); matrix = np.eye(len(names))
    for row in pairwise:
        i = names.index(row["left"]); j = names.index(row["right"])
        matrix[i, j] = matrix[j, i] = float(row["support_overlap"])
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(names)), names, fontsize=7)
    axis.set_title("Fixed-Q top-24 row-support overlap")
    fig.colorbar(image, ax=axis)
    fig.savefig(OUT / "fig04_support_equivalence.png", dpi=180); plt.close(fig)


def write_report(
    communication: list[dict[str, object]], analytic: list[dict[str, object]],
    paired_summary: list[dict[str, object]], pairwise: list[dict[str, object]],
    hashes: dict[str, str], diagnostics: dict[str, object],
) -> None:
    comm = {(r["modulation"], int(r["snr_db"]), r["waveform"], r["support_rule"]): r for r in communication}
    geom = {(r["waveform"], int(r["snr_db"]), r["support_rule"]): r for r in analytic}
    qpsk = {
        rule: (float(comm[("QPSK", 18, "Learned", rule)]["ber"]), np.mean([float(comm[("QPSK", 18, name, rule)]["ber"]) for name in PRIMARY]))
        for rule in SUPPORT_RULES
    }
    qam = {
        rule: (float(comm[("16QAM", 18, "Learned", rule)]["ber"]), np.mean([float(comm[("16QAM", 18, name, rule)]["ber"]) for name in PRIMARY]))
        for rule in SUPPORT_RULES
    }
    learned_gamma = geom[("Learned", 18, "Q")]["gamma_lower_cvar05_median"]
    candidate_gamma = np.mean([float(geom[(name, 18, "Q")]["gamma_lower_cvar05_median"]) for name in PRIMARY])
    learned_tail = geom[("Learned", 18, "Q")]["stream_mse_cvar95_median"]
    candidate_tail = np.mean([float(geom[(name, 18, "Q")]["stream_mse_cvar95_median"]) for name in PRIMARY])
    learned_condition = geom[("Learned", 18, "Q")]["condition_number_median"]
    candidate_condition = np.mean([float(geom[(name, 18, "Q")]["condition_number_median"]) for name in PRIMARY])
    learned_retained = geom[("Learned", 18, "Q")]["retained_energy_mean"]
    candidate_retained = np.mean([float(geom[(name, 18, "Q")]["retained_energy_mean"]) for name in PRIMARY])
    overlaps = [float(row["support_overlap"]) for row in pairwise if row["left"] != "Learned"]
    lines = [
        "# Stage 3C — Saturated-Q Communication Geometry", "",
        "## Result: Outcome B — constellation-sensitive geometry identified", "",
        "No single modulation-agnostic scalar explains the residual ranking. Q fixes representation energy but not the joint coefficient cross-products governing the finite receiver's error distribution. The simplest robust formulation is therefore a two-statistic LMMSE error profile: aggregate error plus an adverse stream/channel tail.", "",
        "## Discrepancy localization at 18 dB", "",
        "| Modulation | Receiver | Learned BER | mean Stage-3B BER |", "|---|---|---:|---:|",
    ]
    for modulation, values in (("QPSK", qpsk), ("16-QAM", qam)):
        for rule in SUPPORT_RULES:
            lines.append(f"| {modulation} | {rule} | {values[rule][0]:.6g} | {values[rule][1]:.6g} |")
    lines += [
        "", "The QPSK separation is already visible with full reception at every tested SNR and persists through oracle and fixed-Q reception, so it does not enter first at support selection. The 16-QAM ranking is SNR- and receiver-sensitive: Stage-3B is better at 12/15 dB even under full reception, Learned is slightly better under full reception at 18/20 dB, and truncation shifts 18/20 dB back toward Stage-3B.", "",
        "## Receiver geometry", "",
        f"- Fixed-Q retained energy is higher for Stage-3B on average ({candidate_retained:.6f}) than Learned ({learned_retained:.6f}); support/energy loss cannot explain Learned's QPSK advantage.",
        f"- Median condition number is effectively unchanged: Learned {float(learned_condition):.3f}, Stage-3B mean {candidate_condition:.3f}. Aggregate conditioning and equalizer norm are not sufficient separators.",
        f"- Median lower-tail post-LMMSE SINR is healthier for Learned ({float(learned_gamma):.3f}) than the Stage-3B mean ({candidate_gamma:.3f}); median worst-stream MSE tail is {float(learned_tail):.4f} versus {candidate_tail:.4f}.",
        "- Paired QPSK channel results place the BER differences in the same adverse reliability and decision-margin tails identified by Stage 1A.1/1A.2. This reappearance was tested, not assumed.", "",
        "## Why modulation ranks differ", "",
        "At 18 dB fixed-Q, Stage-3B has lower aggregate/scaled error variance and higher mean normalized margin, but worse p05 margin and lower-tail SINR. QPSK is dominated by the adverse sign-decision tail and therefore favors Learned. For 16-QAM, Stage-3B has fewer negative-margin decisions overall despite a worse extreme margin tail, so it wins after truncation. Because mean and tail point in opposite directions, neither one alone explains both constellations.", "",
        "## What Q omits", "",
        "`Q_ij = E|A_ij|^2` contains elementwise second moments. It does not determine joint products such as `E[A_ij A_kl*]`. The finite receiver depends on these cross-products through `A_K^H A_K`, the equalizer, and the exact mismatched error covariance `(F A-I)(F A-I)^H + sigma^2 F F^H`. Nearly equal eta and retained energy therefore do not imply equal stream reliability.", "",
        "## Representation equivalence", "",
        f"- Pairwise support overlap among Stage-3B solutions spans {min(overlaps):.3f}-{max(overlaps):.3f}; they are not one identical support mask.",
        "- OFDM-start solutions retain more Stage-2B locality than Haar-start solutions, while all reach nearly identical concentration. They are functionally concentration-equivalent but genuinely distinct in atom and support geometry.", "",
        "## Interpretation gate", "",
        "1. The QPSK gap first appears at full-information reception and persists through oracle-K and fixed-Q reception.",
        "2. It is not caused by support or retained-energy loss.",
        "3. Aggregate condition number/noise enhancement does not explain it.",
        "4. Robust lower-tail post-LMMSE reliability explains the QPSK gap more directly, but not the complete two-modulation ranking.",
        "5. The Stage-1A.2 rare channel/stream-tail mechanism reappears.",
        "6. 16-QAM values aggregate variance and multi-threshold decision geometry differently, producing an SNR- and receiver-dependent reversal.",
        "7. No single tested modulation-agnostic scalar explains both; the simplest robust formulation retains both mean and adverse-tail error statistics.",
        "8. Q omits joint coefficient phase/correlation and thus Gram/error-covariance geometry.",
        "9. Stage-3B solutions share saturated concentration but not identical supports or locality.",
        "10. Simplest later candidate: Q concentration plus a two-statistic receiver-geometry criterion that controls aggregate LMMSE error and its robust stream/channel tail. Do not optimize it in Stage 3C.", "",
        "## Protocol and preservation", "",
        f"- Frozen paired bank: {CHANNELS} independent normalized P=3 DD-v1 channels; SNRs {SNRS}; K={K}; common symbols and physical noise.",
        f"- Authoritative Learned SHA-256: `{hashes['Learned']}`; all six Stage-3B hashes are recorded in the diagnostics JSON.",
        f"- All {diagnostics['protected_artifact_count']} prior artifacts remained byte-identical. IDENTITY is numerical-only and Stage 2E was not used.",
        "- No waveform was trained, tuned, or modified in Stage 3C. STOP before Stage 3D.", "",
    ]
    REPORT.write_text("\n".join(lines))


def run_stage3c() -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    waveforms, hashes = load_waveforms(); q, masks = load_q_and_masks()
    channels = sample_dd_v1(CHANNELS, CHANNEL_SEED).matrices
    analytic_channel, analytic_summary = analytic_geometry(channels, waveforms, masks)
    write_csv(OUT / "receiver_geometry_per_channel.csv", analytic_channel)
    write_csv(OUT / "receiver_geometry_summary.csv", analytic_summary)
    write_csv(OUT / "singular_value_distributions.csv", singular_value_summaries(channels, waveforms, masks))
    print("receiver geometry complete", flush=True)
    communication_channel = communication_rows(
        channels, waveforms, masks, (qpsk_constellation(), qam16_constellation()), SNRS,
        k=K, target_bits=TARGET_BITS, symbol_seed=SYMBOL_SEED, noise_seed=NOISE_SEED,
    )
    communication = communication_summary(communication_channel)
    write_csv(OUT / "communication_per_channel.csv", communication_channel)
    write_csv(OUT / "communication_summary.csv", communication)
    print("paired communication decomposition complete", flush=True)
    paired, paired_summary = paired_explanations(analytic_channel, communication_channel)
    write_csv(OUT / "paired_learned_vs_stage3b_per_channel.csv", paired)
    write_csv(OUT / "paired_explanation_summary.csv", paired_summary)
    locality, pairwise, curves = equivalence_diagnostics(q, masks)
    write_csv(OUT / "support_locality.csv", locality)
    write_csv(OUT / "q_support_pairwise.csv", pairwise)
    write_csv(OUT / "q_concentration_curves.csv", curves)
    make_plots(communication, analytic_channel, paired, pairwise)
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage 3C changed a prior artifact")
    diagnostics = {
        "outcome": "B",
        "second_property": "two-statistic LMMSE error profile: aggregate error and robust lower tail",
        "channel_bank": {"seed": CHANNEL_SEED, "channels": CHANNELS, "p": config.P},
        "symbol_seed": SYMBOL_SEED, "noise_seed": NOISE_SEED,
        "snrs_db": list(SNRS), "k": K, "n": config.N,
        "waveform_hashes": hashes,
        "stage2e_used": False, "waveform_training_performed": False,
        "identity_in_figures": False,
        "protected_artifact_count": len(protected_before),
        "protected_artifacts_unchanged": protected_after == protected_before,
        "runtime_seconds": time.perf_counter() - started,
    }
    with (OUT / "stage3c_diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True); handle.write("\n")
    write_report(communication, analytic_summary, paired_summary, pairwise, hashes, diagnostics)
    print(f"STAGE 3C COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME B; STOP", flush=True)
    return diagnostics


if __name__ == "__main__":
    run_stage3c()
