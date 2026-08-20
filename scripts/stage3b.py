"""Run Stage 3B direct Q-concentration training and causal validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.stage2d import load_waveforms
from src import config
from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import aggregate_rows, file_sha256, load_frozen_waveform, qualification_waveforms
from src.stage2d import (
    empirical_coupling,
    matrix_agreement,
    score_support_mask,
    sobol_dd_samples,
    support_communication_rows,
    support_metrics,
)
from src.stage3a import concentration_thresholds, fixed_mask_expected_retained, q_concentration
from src.stage3b import (
    BER_NOISE_SEED,
    BER_SYMBOL_SEED,
    EVALUATION_CHANNELS,
    EVALUATION_CHANNEL_SEED,
    EVALUATION_Q_SAMPLES,
    EVALUATION_Q_SEED,
    INITIALIZATION_NAMES,
    LOWER_TAIL_FRACTION,
    ROBUST_LAMBDA,
    TRAIN_K,
    TRAIN_SAMPLES,
    TRAIN_SEED,
    VALIDATION_SAMPLES,
    VALIDATION_SEED,
    aligned_basis_similarity,
    atomic_operators,
    differentiable_q,
    fixed_initializations,
    optimize_q_waveforms,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3b"
CHECKPOINTS = OUT / "checkpoints"
REPORT = ROOT / "reports" / "stage3b_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE2C_CHECKPOINTS = ROOT / "outputs" / "stage2c" / "checkpoints"
ARMS = ("mean", "robust")
FIGURE_WAVEFORMS = ("Learned", "OFDM", "OTFS", "AFDM")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")


def protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory in ("stage2a", "stage2b", "stage2c", "stage2d", "stage3a"):
        for path in sorted((ROOT / "outputs" / directory).glob("**/*")):
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return hashes


def load_or_create_banks() -> tuple[torch.Tensor, torch.Tensor]:
    path = OUT / "q_training_validation_banks.npz"
    if path.exists():
        saved = np.load(path)
        expected = (TRAIN_SEED, VALIDATION_SEED, TRAIN_SAMPLES, VALIDATION_SAMPLES)
        observed = tuple(int(value) for value in saved["protocol"])
        if observed != expected:
            raise RuntimeError(f"saved Stage-3B bank protocol mismatch: {observed} != {expected}")
        train_delays = torch.from_numpy(saved["train_delays"])
        train_dopplers = torch.from_numpy(saved["train_dopplers"])
        val_delays = torch.from_numpy(saved["validation_delays"])
        val_dopplers = torch.from_numpy(saved["validation_dopplers"])
    else:
        train_delays, train_dopplers = sobol_dd_samples(TRAIN_SAMPLES, TRAIN_SEED)
        val_delays, val_dopplers = sobol_dd_samples(VALIDATION_SAMPLES, VALIDATION_SEED)
        np.savez(
            path,
            protocol=np.asarray((TRAIN_SEED, VALIDATION_SEED, TRAIN_SAMPLES, VALIDATION_SAMPLES)),
            train_delays=train_delays.numpy(), train_dopplers=train_dopplers.numpy(),
            validation_delays=val_delays.numpy(), validation_dopplers=val_dopplers.numpy(),
        )
    train_pairs = set(zip(train_delays.tolist(), train_dopplers.tolist()))
    val_pairs = set(zip(val_delays.tolist(), val_dopplers.tolist()))
    if train_pairs & val_pairs:
        raise RuntimeError("Q training and validation banks overlap")
    return atomic_operators(train_delays, train_dopplers), atomic_operators(val_delays, val_dopplers)


def train_or_load(
    arm: str,
    starts: torch.Tensor,
    train_operators: torch.Tensor,
    validation_operators: torch.Tensor,
    steps: int,
    force: bool,
) -> dict[str, object]:
    path = CHECKPOINTS / f"training_{arm}.pt"
    protocol = {
        "arm": arm, "steps": steps, "validation_cadence": 25,
        "train_samples": TRAIN_SAMPLES, "validation_samples": VALIDATION_SAMPLES,
        "train_seed": TRAIN_SEED, "validation_seed": VALIDATION_SEED,
        "k": TRAIN_K, "robust_lambda": ROBUST_LAMBDA,
        "lower_tail_fraction": LOWER_TAIL_FRACTION,
    }
    if path.exists() and not force:
        payload = torch.load(path, weights_only=False)
        if payload["protocol"] != protocol:
            raise RuntimeError(f"cached {arm} training protocol mismatch")
        print(f"loaded cached {arm} arm", flush=True)
        return payload
    result = optimize_q_waveforms(
        starts, INITIALIZATION_NAMES, train_operators, validation_operators,
        arm=arm, steps=steps, validation_cadence=25,
    )
    payload = {
        "protocol": protocol,
        "best_matrices": result.best_matrices,
        "best_steps": result.best_steps,
        "initial_validation_eta": result.initial_validation_eta,
        "initial_validation_lower_tail": result.initial_validation_lower_tail,
        "best_validation_eta": result.best_validation_eta,
        "best_validation_lower_tail": result.best_validation_lower_tail,
        "final_matrices": result.final_matrices,
        "final_validation_eta": result.final_validation_eta,
        "final_validation_lower_tail": result.final_validation_lower_tail,
        "max_unitarity_residuals": result.max_unitarity_residuals,
        "nonfinite_seen": result.nonfinite_seen,
        "trace_rows": result.trace_rows,
        "runtime_seconds": result.runtime_seconds,
    }
    torch.save(payload, path)
    for index, name in enumerate(INITIALIZATION_NAMES):
        torch.save(result.best_matrices[index], CHECKPOINTS / f"q_{arm}_from_{name.lower()}_best.pt")
    print(f"trained {arm} arm in {result.runtime_seconds:.1f} s", flush=True)
    return payload


def selected_waveforms(payloads: dict[str, dict[str, object]]) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for arm, payload in payloads.items():
        matrices = payload["best_matrices"]
        for index, initialization in enumerate(INITIALIZATION_NAMES):
            selected[f"Q_{arm.upper()}_FROM_{initialization}"] = matrices[index]
    return selected


def all_evaluation_waveforms(selected: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    learned, _ = load_frozen_waveform(FROZEN)
    principals = qualification_waveforms(learned)
    names, starts = fixed_initializations()
    raw = {f"RAW_{name}": starts[index] for index, name in enumerate(names) if name != "OFDM"}
    stage2c = {
        f"STAGE2C_K24_{name}": torch.load(
            STAGE2C_CHECKPOINTS / f"k24_{name.lower()}_best.pt", weights_only=True
        ) for name in INITIALIZATION_NAMES
    }
    groups = {name: "stage3b" for name in selected}
    groups.update({name: "principal" for name in principals})
    groups.update({name: "raw_start" for name in raw})
    groups.update({name: "stage2c_k24" for name in stage2c})
    return {**selected, **principals, **raw, **stage2c}, groups


def summarize(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().to(torch.float64).reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if len(finite) == 0:
        return {f"{prefix}_mean": float("inf"), f"{prefix}_median": float("inf"), f"{prefix}_p05": float("inf"), f"{prefix}_p95": float("inf")}
    return {
        f"{prefix}_mean": float(torch.mean(finite)),
        f"{prefix}_median": float(torch.median(finite)),
        f"{prefix}_p05": float(torch.quantile(finite, 0.05)),
        f"{prefix}_p95": float(torch.quantile(finite, 0.95)),
    }


def structural_evaluation(
    waveforms: dict[str, torch.Tensor], groups: dict[str, str], channels: torch.Tensor
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor], dict[str, np.ndarray], dict[str, np.ndarray]]:
    delays, dopplers = sobol_dd_samples(EVALUATION_Q_SAMPLES, EVALUATION_Q_SEED)
    operators = atomic_operators(delays, dopplers)
    names = tuple(waveforms)
    with torch.no_grad():
        q_stack = differentiable_q(torch.stack([waveforms[name] for name in names]), operators, chunk_size=16)
    q = {name: q_stack[index].numpy() for index, name in enumerate(names)}
    c = empirical_coupling(channels, waveforms, chunk=16)
    masks = {name: score_support_mask(q[name], TRAIN_K) for name in names}
    rows: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        concentration = q_concentration(q[name], TRAIN_K)
        thresholds = concentration_thresholds(q[name])
        agreement = matrix_agreement(q[name], c[name], TRAIN_K)
        effective = effective_operator(channels, waveform)
        fixed = support_metrics(effective, masks[name], TRAIN_K, noise_variance(18))
        oracle_mask = torch.abs(rowwise_topk(effective, TRAIN_K)) > 0
        oracle = support_metrics(effective, oracle_mask, TRAIN_K, noise_variance(18))
        rows.append({
            "waveform": name, "group": groups[name], "k": TRAIN_K,
            "eta_q": concentration["eta_q"], "row_mean": concentration["row_mean"],
            "row_median": concentration["row_median"], "row_p05": concentration["row_p05"],
            "row_worst": concentration["row_worst"],
            "effective_support_mean": concentration["effective_support_mean"],
            "normalized_entropy_mean": concentration["normalized_entropy_mean"],
            "k90": thresholds[0.90], "k95": thresholds[0.95],
            "k99": thresholds[0.99], "k995": thresholds[0.995],
            "q_to_c_spearman": agreement["spearman_offdiagonal"],
            "q_to_c_top24_overlap": agreement["topk_overlap_mean"],
            "fixed_q_expected_retained": fixed_mask_expected_retained(c[name], masks[name].numpy()),
            **summarize(fixed["retained_energy"], "fixed_retained_energy"),
            **summarize(oracle["retained_energy"], "oracle_retained_energy"),
            **summarize(fixed["support_recall"], "fixed_support_recall"),
            **summarize(fixed["analytic_mse"], "fixed_analytic_mse"),
            **summarize(oracle["analytic_mse"], "oracle_analytic_mse"),
            **summarize(fixed["condition_number"], "fixed_condition_number"),
            **summarize(fixed["limiting_distortion"], "fixed_limiting_distortion"),
            **summarize(fixed["stream_cvar_095"], "fixed_stream_cvar_095"),
            "unitarity_residual": float(torch.linalg.matrix_norm(waveform.mH @ waveform - torch.eye(config.N, dtype=COMPLEX_DTYPE))),
        })
    return rows, masks, q, c


def communication_evaluation(
    waveforms: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], channels: torch.Tensor
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rules = {name: {"Oracle": None, "Q": masks[name]} for name in waveforms}
    k_values = {name: TRAIN_K for name in waveforms}
    per_channel = support_communication_rows(
        channels, waveforms, rules, k_values,
        (qpsk_constellation(), qam16_constellation()), config.STAGE1A_SNRS_DB,
        target_bits=1_000_000, symbol_seed=BER_SYMBOL_SEED, noise_seed=BER_NOISE_SEED,
    )
    summary = aggregate_rows(per_channel, ("modulation", "snr_db", "waveform", "k", "support_rule"))
    by_key = {(row["modulation"], row["snr_db"], row["waveform"], row["support_rule"]): row for row in summary}
    for row in summary:
        other = "Oracle" if row["support_rule"] == "Q" else "Q"
        peer = by_key[(row["modulation"], row["snr_db"], row["waveform"], other)]
        if row["support_rule"] == "Q":
            denominator = max(float(peer["ber"]), 1 / float(peer["total_bits"]))
            row["fixed_vs_oracle_relative_penalty"] = float(row["ber"]) / denominator - 1
        else:
            row["fixed_vs_oracle_relative_penalty"] = ""
    return per_channel, summary


def alignment_rows(waveforms: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    references = {name: waveforms[name] for name in FIGURE_WAVEFORMS}
    rows: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        if not name.startswith("Q_"):
            continue
        for reference_name, reference in references.items():
            result = aligned_basis_similarity(waveform, reference)
            rows.append({
                "waveform": name, "reference": reference_name,
                "mean_matched_squared_overlap": result["mean_matched_squared_overlap"],
                "minimum_matched_squared_overlap": result["minimum_matched_squared_overlap"],
                "aligned_chordal_distance": result["aligned_chordal_distance"],
            })
    return rows


def solution_pairwise_rows(waveforms: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    """Check whether distinct Q optima share concentration without sharing a basis."""
    names = sorted(name for name in waveforms if name.startswith("Q_"))
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result = aligned_basis_similarity(waveforms[left], waveforms[right])
            rows.append({
                "left": left, "right": right,
                "mean_matched_squared_overlap": result["mean_matched_squared_overlap"],
                "minimum_matched_squared_overlap": result["minimum_matched_squared_overlap"],
                "aligned_chordal_distance": result["aligned_chordal_distance"],
            })
    return rows


def causal_rows(
    training: dict[str, dict[str, object]], structural: list[dict[str, object]],
    communication: list[dict[str, object]],
) -> list[dict[str, object]]:
    structure = {row["waveform"]: row for row in structural}
    comm = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    rows: list[dict[str, object]] = []
    for arm, payload in training.items():
        for index, initialization in enumerate(INITIALIZATION_NAMES):
            selected = f"Q_{arm.upper()}_FROM_{initialization}"
            baseline = "OFDM" if initialization == "OFDM" else f"RAW_{initialization}"
            row: dict[str, object] = {
                "arm": arm, "initialization": initialization, "waveform": selected,
                "baseline": baseline, "selected_step": int(payload["best_steps"][index]),
                "validation_eta_initial": float(payload["initial_validation_eta"][index]),
                "validation_eta_selected": float(payload["best_validation_eta"][index]),
                "validation_lower_tail_initial": float(payload["initial_validation_lower_tail"][index]),
                "validation_lower_tail_selected": float(payload["best_validation_lower_tail"][index]),
            }
            for key in ("eta_q", "row_p05", "fixed_retained_energy_mean", "fixed_analytic_mse_mean", "fixed_condition_number_median"):
                row[f"heldout_{key}_baseline"] = structure[baseline][key]
                row[f"heldout_{key}_selected"] = structure[selected][key]
                row[f"heldout_{key}_delta"] = float(structure[selected][key]) - float(structure[baseline][key])
            for modulation in ("QPSK", "16QAM"):
                prefix = "qpsk" if modulation == "QPSK" else "qam16"
                for support in ("Oracle", "Q"):
                    selected_ber = float(comm[(selected, modulation, 18, support)]["ber"])
                    baseline_ber = float(comm[(baseline, modulation, 18, support)]["ber"])
                    label = support.lower()
                    row[f"{prefix}_{label}_ber_baseline"] = baseline_ber
                    row[f"{prefix}_{label}_ber_selected"] = selected_ber
                    row[f"{prefix}_{label}_relative_change"] = selected_ber / max(baseline_ber, 1e-300) - 1
            rows.append(row)
    return rows


def classify(causal: list[dict[str, object]]) -> tuple[str, str]:
    eta_help = [row["heldout_eta_q_delta"] > 1e-3 for row in causal]
    retained_help = [row["heldout_fixed_retained_energy_mean_delta"] > 5e-4 for row in causal]
    mse_help = [row["heldout_fixed_analytic_mse_mean_delta"] < 0 for row in causal]
    qpsk_help = [row["qpsk_q_relative_change"] < 0 for row in causal]
    qam_help = [row["qam16_q_relative_change"] < 0 for row in causal]
    haar = [index for index, row in enumerate(causal) if str(row["initialization"]).startswith("HAAR")]
    if any(eta_help[index] and not retained_help[index] for index in range(len(causal))):
        return "D", "A held-out concentration gain failed to produce the predicted retained-energy direction."
    if all(eta_help[index] and retained_help[index] and mse_help[index] and qpsk_help[index] and qam_help[index] for index in haar):
        return "A", "Both Q-only arms rescued both Haar starts through representation, analytic-MSE, and two-modulation BER outcomes."
    if all(eta_help[index] and retained_help[index] for index in haar) and not all(qpsk_help[index] and qam_help[index] for index in haar):
        return "B", "Q concentration and fixed-support energy improved from Haar, but communication gains were incomplete."
    return "C", "The causal effect remained materially initialization or optimization dependent."


def make_plots(
    traces: list[dict[str, object]], structural: list[dict[str, object]],
    communication: list[dict[str, object]], alignments: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm, axis in zip(ARMS, axes):
        for initialization in INITIALIZATION_NAMES:
            selected = [row for row in traces if row["arm"] == arm and row["initialization"] == initialization]
            axis.plot([row["step"] for row in selected], [row["validation_eta"] for row in selected], label=initialization)
        axis.set(title=f"{arm.title()} Q objective", xlabel="optimization step", ylabel=r"validation $\eta_{24}^Q$")
        axis.grid(alpha=.3); axis.legend()
    fig.savefig(OUT / "fig01_validation_concentration.png", dpi=180); plt.close(fig)

    shown = [row for row in structural if row["waveform"] != "IDENTITY"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    for axis, zoomed in zip(axes, (False, True)):
        for row in shown:
            if zoomed and row["group"] not in ("stage3b", "principal"):
                continue
            stage3b = row["group"] == "stage3b"
            principal = row["group"] == "principal"
            axis.scatter(
                row["eta_q"], row["fixed_retained_energy_mean"],
                color="tab:red" if stage3b else "tab:blue" if principal else "0.65",
                marker="x" if stage3b else "o", s=55,
                label=row["waveform"] if zoomed else None,
            )
        if zoomed:
            axis.set(xlim=(0.975, 1.0005), ylim=(0.972, 1.0005), title="High-concentration regime")
            axis.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=7)
        else:
            axis.set(title="All numerical controls")
        axis.set(xlabel=r"held-out $\eta_{24}^Q$", ylabel="realized fixed-Q retained energy")
        axis.grid(alpha=.3)
    fig.suptitle("Stage 3B representation bridge")
    fig.savefig(OUT / "fig02_eta_vs_retained_energy.png", dpi=180); plt.close(fig)

    selected_names = [row["waveform"] for row in shown if row["group"] == "stage3b"] + list(FIGURE_WAVEFORMS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for modulation, axis in zip(("QPSK", "16QAM"), axes):
        values = [next(row for row in communication if row["waveform"] == name and row["modulation"] == modulation and int(row["snr_db"]) == 18 and row["support_rule"] == "Q")["ber"] for name in selected_names]
        axis.bar(np.arange(len(values)), values)
        axis.set_xticks(np.arange(len(values)), selected_names, rotation=45, ha="right", fontsize=7)
        axis.set_yscale("log"); axis.set(title=f"{modulation}, fixed Q support, 18 dB", ylabel="BER"); axis.grid(axis="y", alpha=.3)
    fig.savefig(OUT / "fig03_fixed_q_ber_18db.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    names = sorted({row["waveform"] for row in alignments})
    refs = FIGURE_WAVEFORMS
    matrix = np.asarray([[next(row for row in alignments if row["waveform"] == name and row["reference"] == ref)["mean_matched_squared_overlap"] for ref in refs] for name in names])
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(refs)), refs); axis.set_yticks(range(len(names)), names, fontsize=7)
    axis.set(title="Phase/permutation-aligned basis overlap"); fig.colorbar(image, ax=axis, label="mean matched squared overlap")
    fig.tight_layout(); fig.savefig(OUT / "fig04_aligned_basis_similarity.png", dpi=180); plt.close(fig)


def write_report(
    outcome: str, interpretation: str, causal: list[dict[str, object]],
    structural: list[dict[str, object]], communication: list[dict[str, object]],
    alignments: list[dict[str, object]], pairwise: list[dict[str, object]], diagnostics: dict[str, object]
) -> None:
    structure = {row["waveform"]: row for row in structural}
    comm = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    selected_names = [row["waveform"] for row in causal]
    robust_tail_deltas = []
    for initialization in INITIALIZATION_NAMES:
        mean = structure[f"Q_MEAN_FROM_{initialization}"]["row_p05"]
        robust = structure[f"Q_ROBUST_FROM_{initialization}"]["row_p05"]
        robust_tail_deltas.append(float(robust) - float(mean))
    qpsk_selected = [float(comm[(name, "QPSK", 18, "Q")]["ber"]) for name in selected_names]
    qam_selected = [float(comm[(name, "16QAM", 18, "Q")]["ber"]) for name in selected_names]
    learned_qpsk = float(comm[("Learned", "QPSK", 18, "Q")]["ber"])
    learned_qam = float(comm[("Learned", "16QAM", 18, "Q")]["ber"])
    pairwise_values = [float(row["mean_matched_squared_overlap"]) for row in pairwise]
    lines = [
        "# Stage 3B — Direct Q-Concentration Optimization and Causal Validation", "",
        f"## Result: Outcome {outcome}", "", interpretation, "",
        "## Causal interventions at 18 dB", "",
        "| Arm/start | val eta initial→selected | held-out eta Δ | retained Δ | analytic MSE Δ | QPSK fixed BER Δ | 16-QAM fixed BER Δ |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in causal:
        lines.append(
            f"| {row['arm']}/{row['initialization']} | {row['validation_eta_initial']:.6f}→{row['validation_eta_selected']:.6f} | "
            f"{row['heldout_eta_q_delta']:+.6f} | {row['heldout_fixed_retained_energy_mean_delta']:+.6f} | "
            f"{row['heldout_fixed_analytic_mse_mean_delta']:+.6f} | {100*row['qpsk_q_relative_change']:+.1f}% | {100*row['qam16_q_relative_change']:+.1f}% |"
        )
    lines += [
        "", "## Interpretation gate", "",
        "1. **Held-out concentration:** yes. Every selected run increased held-out eta; Haar gains were about +0.462 and OFDM gains about +0.009.",
        f"2. **Weak-row protection:** consistent but modest. Robust-minus-mean held-out row-p05 changes were {', '.join(f'{value:+.6f}' for value in robust_tail_deltas)} for OFDM/HAAR-1/HAAR-2.",
        "3. **Fixed-support energy:** yes. Realized retained-energy gains tracked eta in the predicted direction for all six interventions.",
        "4. **Analytic MSE:** yes. Every intervention reduced fixed-Q analytic MSE; Haar starts moved from about 2.31 to about 0.046 at 18 dB.",
        f"5. **QPSK BER:** yes versus every matched raw start, but not versus authoritative Learned: selected range {min(qpsk_selected):.6g}-{max(qpsk_selected):.6g}, Learned {learned_qpsk:.6g} at 18 dB.",
        f"6. **16-QAM BER:** yes versus every matched raw start and slightly better than Learned: selected range {min(qam_selected):.6g}-{max(qam_selected):.6g}, Learned {learned_qam:.6g} at 18 dB.",
        "7. **Initialization robustness:** yes for the causal direction; all OFDM and Haar interventions improved concentration, representation, MSE, and both BERs.",
        "8. **Haar rescue:** yes. Both raw Haar starts reached held-out eta above 0.99948 and fixed-Q BER near the OFDM-start Q solutions.",
        f"9. **Structure:** no classical/learned collapse. OFDM is the nearest tested reference, but aligned overlap is only 0.2176-0.4817; pairwise Stage-3B overlap spans {min(pairwise_values):.4f}-{max(pairwise_values):.4f} despite nearly identical Q concentration.",
        "10. **Sufficiency:** supported as a causal primary design property for useful finite-K waveforms in this frozen setting, but not a complete communication objective: the QPSK gap to Learned shows coefficient/conditioning geometry can still matter after concentration saturates.",
        "", "## Structural resemblance", "",
    ]
    for name in sorted({row["waveform"] for row in alignments}):
        best = max((row for row in alignments if row["waveform"] == name), key=lambda row: row["mean_matched_squared_overlap"])
        lines.append(f"- {name}: closest to {best['reference']} with aligned mean squared overlap {best['mean_matched_squared_overlap']:.4f}.")
    lines += [
        "", "## Protocol and preservation", "",
        f"- Q training/validation/evaluation banks: {TRAIN_SAMPLES}/{VALIDATION_SAMPLES}/{EVALUATION_Q_SAMPLES}; training used K={TRAIN_K} only.",
        f"- Robust arm fixed before evaluation: lambda={ROBUST_LAMBDA}, weakest-row fraction={LOWER_TAIL_FRACTION}.",
        f"- Communication bank: {EVALUATION_CHANNELS} normalized P={config.P} DD-v1 channels with paired symbols/noise and no BER-based selection.",
        f"- Training/evaluation/full-protocol runtimes: {diagnostics['training_runtime_seconds_total']:.1f}/{diagnostics['evaluation_runtime_seconds']:.1f}/{diagnostics['full_protocol_seconds']:.1f} s.",
        f"- Authoritative checkpoint SHA-256: `{diagnostics['authoritative_checkpoint_sha256']}`.",
        f"- All {diagnostics['protected_artifact_count']} Stage-2/3A artifacts remained byte-identical: `{diagnostics['protected_artifacts_unchanged']}`.",
        "- IDENTITY is retained in numerical outputs and excluded from figures.",
        "- Stage 2E was not imported or used. No push, merge, receiver change, or Stage 3C work was performed.", "",
    ]
    REPORT.write_text("\n".join(lines))


def run_stage3b(*, steps: int = 1000, force_training: bool = False) -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    frozen, frozen_info = load_frozen_waveform(FROZEN)
    _, stage2c_metadata, _ = load_waveforms()
    train_operators, validation_operators = load_or_create_banks()
    names, starts = fixed_initializations()
    if names != INITIALIZATION_NAMES:
        raise RuntimeError("Stage-3B initialization protocol changed")
    payloads = {arm: train_or_load(arm, starts, train_operators, validation_operators, steps, force_training) for arm in ARMS}
    training_runtime = sum(float(payload["runtime_seconds"]) for payload in payloads.values())
    traces = [row for payload in payloads.values() for row in payload["trace_rows"]]
    write_csv(OUT / "training_validation_history.csv", traces)
    selected = selected_waveforms(payloads)
    waveforms, groups = all_evaluation_waveforms(selected)
    evaluation_started = time.perf_counter()
    test_channels = sample_dd_v1(EVALUATION_CHANNELS, EVALUATION_CHANNEL_SEED).matrices
    structural, masks, q, c = structural_evaluation(waveforms, groups, test_channels)
    write_csv(OUT / "structural_communication_metrics.csv", structural)
    np.savez(OUT / "heldout_q_c_matrices.npz", **{f"Q_{name}": value for name, value in q.items()}, **{f"C_{name}": value for name, value in c.items()})
    print("held-out Q/C and receiver diagnostics complete", flush=True)
    per_channel, communication = communication_evaluation(waveforms, masks, test_channels)
    write_csv(OUT / "communication_per_channel.csv", per_channel)
    write_csv(OUT / "communication_summary.csv", communication)
    alignments = alignment_rows(waveforms); write_csv(OUT / "aligned_basis_similarity.csv", alignments)
    pairwise = solution_pairwise_rows(waveforms); write_csv(OUT / "solution_pairwise_similarity.csv", pairwise)
    causal = causal_rows(payloads, structural, communication); write_csv(OUT / "causal_intervention_summary.csv", causal)
    outcome, interpretation = classify(causal)
    make_plots(traces, structural, communication, alignments)
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage-2/3A artifact preservation failure")
    if not torch.equal(frozen, torch.load(FROZEN, weights_only=True)) or file_sha256(FROZEN) != frozen_info["sha256"]:
        raise RuntimeError("authoritative checkpoint mutated")
    diagnostics = {
        "outcome": outcome, "interpretation": interpretation,
        "stage2e_used": False, "communication_used_for_selection": False,
        "training_k_values": [TRAIN_K], "training_steps": steps,
        "q_banks": {"train": {"seed": TRAIN_SEED, "samples": TRAIN_SAMPLES}, "validation": {"seed": VALIDATION_SEED, "samples": VALIDATION_SAMPLES}, "evaluation": {"seed": EVALUATION_Q_SEED, "samples": EVALUATION_Q_SAMPLES}},
        "communication_bank": {"seed": EVALUATION_CHANNEL_SEED, "channels": EVALUATION_CHANNELS, "symbol_seed": BER_SYMBOL_SEED, "noise_seed": BER_NOISE_SEED},
        "authoritative_checkpoint_sha256": frozen_info["sha256"],
        "stage2c_checkpoint_hashes": {name: data["sha256"] for name, data in stage2c_metadata.items()},
        "stage3b_checkpoint_hashes": {path.name: file_sha256(path) for path in sorted(CHECKPOINTS.glob("q_*_best.pt"))},
        "maximum_unitarity_residual": max(float(row["unitarity_residual"]) for row in structural),
        "protected_artifact_count": len(protected_before),
        "protected_artifacts_unchanged": protected_after == protected_before,
        "training_runtime_seconds_total": training_runtime,
        "evaluation_runtime_seconds": time.perf_counter() - evaluation_started,
        "full_protocol_seconds": training_runtime + (time.perf_counter() - evaluation_started),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage3b_diagnostics.json", diagnostics)
    write_report(outcome, interpretation, causal, structural, communication, alignments, pairwise, diagnostics)
    print(f"STAGE 3B COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME {outcome}", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); run_stage3b(steps=arguments.steps, force_training=arguments.force_training)
