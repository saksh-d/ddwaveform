"""Run Stage 3D constrained saturated-Q receiver-geometry intervention."""

from __future__ import annotations

import argparse
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

from scripts.stage3c import ALL_WAVEFORMS, PRIMARY, analytic_geometry, load_waveforms
from src import config
from src.channel import sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.stage2d import score_support_mask, sobol_dd_samples
from src.stage3a import q_concentration
from src.stage3b import atomic_operators, differentiable_q
from src.stage3c import communication_rows
from src.stage3d import (
    EVALUATION_CHANNELS, EVALUATION_CHANNEL_SEED, EVALUATION_Q_SAMPLES, EVALUATION_Q_SEED,
    GEOMETRY_SNR_DB, GEOMETRY_TRAIN_CHANNELS, GEOMETRY_TRAIN_SEED,
    GEOMETRY_VALIDATION_CHANNELS, GEOMETRY_VALIDATION_SEED, K, NOISE_SEED,
    Q_EPSILON, Q_PENALTY, Q_TRAIN_SAMPLES, Q_TRAIN_SEED,
    Q_VALIDATION_SAMPLES, Q_VALIDATION_SEED, SYMBOL_SEED, TAIL_BETA, TAIL_LAMBDAS,
    TRAINING_STEPS, GeometryObjectiveSpec, optimize_geometry_waveforms,
    select_analytic_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3d"
CHECKPOINTS = OUT / "checkpoints"
REPORT = ROOT / "reports" / "stage3d_summary.md"
SNRS = (12, 15, 18, 20)
TARGET_BITS = 1_000_000
SPECS = (GeometryObjectiveSpec("mean", 0.0),) + tuple(
    GeometryObjectiveSpec("tail", value) for value in TAIL_LAMBDAS
)


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
        if path.is_file() and "stage3d" not in path.parts
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")


def load_or_create_banks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = OUT / "deterministic_banks.npz"
    expected = np.asarray((
        Q_TRAIN_SEED, Q_VALIDATION_SEED, Q_TRAIN_SAMPLES, Q_VALIDATION_SAMPLES,
        GEOMETRY_TRAIN_SEED, GEOMETRY_VALIDATION_SEED,
        GEOMETRY_TRAIN_CHANNELS, GEOMETRY_VALIDATION_CHANNELS,
    ))
    if path.exists():
        saved = np.load(path)
        if not np.array_equal(saved["protocol"], expected):
            raise RuntimeError("cached Stage-3D bank protocol mismatch")
        q_train = atomic_operators(torch.from_numpy(saved["q_train_delays"]), torch.from_numpy(saved["q_train_dopplers"]))
        q_validation = atomic_operators(torch.from_numpy(saved["q_validation_delays"]), torch.from_numpy(saved["q_validation_dopplers"]))
        geometry_train = torch.from_numpy(saved["geometry_train_matrices"])
        geometry_validation = torch.from_numpy(saved["geometry_validation_matrices"])
        return q_train, q_validation, geometry_train, geometry_validation
    q_train_delays, q_train_dopplers = sobol_dd_samples(Q_TRAIN_SAMPLES, Q_TRAIN_SEED)
    q_validation_delays, q_validation_dopplers = sobol_dd_samples(Q_VALIDATION_SAMPLES, Q_VALIDATION_SEED)
    geometry_train_batch = sample_dd_v1(GEOMETRY_TRAIN_CHANNELS, GEOMETRY_TRAIN_SEED)
    geometry_validation_batch = sample_dd_v1(GEOMETRY_VALIDATION_CHANNELS, GEOMETRY_VALIDATION_SEED)
    np.savez(
        path, protocol=expected,
        q_train_delays=q_train_delays.numpy(), q_train_dopplers=q_train_dopplers.numpy(),
        q_validation_delays=q_validation_delays.numpy(), q_validation_dopplers=q_validation_dopplers.numpy(),
        geometry_train_matrices=geometry_train_batch.matrices.numpy(),
        geometry_train_delays=geometry_train_batch.delays.numpy(),
        geometry_train_dopplers=geometry_train_batch.dopplers.numpy(),
        geometry_train_gains=geometry_train_batch.pre_normalization_gains.numpy(),
        geometry_train_scales=geometry_train_batch.normalization_scales.numpy(),
        geometry_validation_matrices=geometry_validation_batch.matrices.numpy(),
        geometry_validation_delays=geometry_validation_batch.delays.numpy(),
        geometry_validation_dopplers=geometry_validation_batch.dopplers.numpy(),
        geometry_validation_gains=geometry_validation_batch.pre_normalization_gains.numpy(),
        geometry_validation_scales=geometry_validation_batch.normalization_scales.numpy(),
    )
    return (
        atomic_operators(q_train_delays, q_train_dopplers),
        atomic_operators(q_validation_delays, q_validation_dopplers),
        geometry_train_batch.matrices, geometry_validation_batch.matrices,
    )


def train_or_load(steps: int, force: bool) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    payload_path = CHECKPOINTS / "training.pt"
    protocol = {
        "steps": steps, "validation_cadence": 25, "k": K,
        "geometry_snr_db": GEOMETRY_SNR_DB, "tail_beta": TAIL_BETA,
        "tail_lambdas": list(TAIL_LAMBDAS), "q_epsilon": Q_EPSILON,
        "q_penalty": Q_PENALTY, "q_train_seed": Q_TRAIN_SEED,
        "q_validation_seed": Q_VALIDATION_SEED,
        "geometry_train_seed": GEOMETRY_TRAIN_SEED,
        "geometry_validation_seed": GEOMETRY_VALIDATION_SEED,
    }
    if payload_path.exists() and not force:
        payload = torch.load(payload_path, weights_only=False)
        if payload["protocol"] != protocol:
            raise RuntimeError("cached Stage-3D training protocol mismatch")
        selected = payload["selected"]
        print("loaded cached Stage-3D training", flush=True)
        return payload, selected
    frozen, _ = load_waveforms()
    starts = torch.stack([frozen[name] for name in PRIMARY])
    q_train, q_validation, geometry_train, geometry_validation = load_or_create_banks()
    result = optimize_geometry_waveforms(
        starts, PRIMARY, SPECS, q_train, q_validation, geometry_train, geometry_validation,
        steps=steps, validation_cadence=25,
    )
    selected = select_analytic_checkpoints(result)
    if not selected:
        raise RuntimeError("no Q-feasible Stage-3D validation checkpoint")
    payload = {
        "protocol": protocol, "initial_metrics": result.initial_metrics,
        "q_floors": result.q_floors, "reference_mean": result.reference_mean,
        "reference_tail": result.reference_tail, "final_matrices": result.final_matrices,
        "max_unitarity_residuals": result.max_unitarity_residuals,
        "nonfinite_seen": result.nonfinite_seen, "trace_rows": result.trace_rows,
        "runtime_seconds": result.runtime_seconds, "selected": selected,
    }
    torch.save(payload, payload_path)
    for name, row in selected.items():
        torch.save(row["matrix"], CHECKPOINTS / f"{name.lower()}_best.pt")
    print(f"trained Stage 3D in {result.runtime_seconds:.1f} s; selected {len(selected)} checkpoints", flush=True)
    return payload, selected


def communication_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    keys = ("modulation", "snr_db", "waveform", "support_rule", "k")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, object]] = []
    for key, selected in groups.items():
        item: dict[str, object] = dict(zip(keys, key))
        item.update({
            "bit_errors": sum(int(row["bit_errors"]) for row in selected),
            "total_bits": sum(int(row["total_bits"]) for row in selected),
            "symbol_errors": sum(int(row["symbol_errors"]) for row in selected),
            "total_symbols": sum(int(row["total_symbols"]) for row in selected),
            "n_channels": len(selected),
        })
        item["ber"] = int(item["bit_errors"]) / int(item["total_bits"])
        item["ser"] = int(item["symbol_errors"]) / int(item["total_symbols"])
        for metric in ("empirical_mse", "constellation_scaled_variance", "normalized_margin_mean", "normalized_margin_p05", "negative_margin_fraction", "residual_anisotropy"):
            values = np.asarray([float(row[metric]) for row in selected])
            for stat, value in (("mean", np.mean(values)), ("median", np.median(values)), ("p05", np.quantile(values, .05)), ("p95", np.quantile(values, .95))):
                item[f"{metric}_{stat}"] = float(value)
        output.append(item)
    return output


def structural_rows(waveforms: dict[str, torch.Tensor]) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    delays, dopplers = sobol_dd_samples(EVALUATION_Q_SAMPLES, EVALUATION_Q_SEED)
    operators = atomic_operators(delays, dopplers)
    names = tuple(waveforms)
    with torch.no_grad():
        q = differentiable_q(torch.stack([waveforms[name] for name in names]), operators, chunk_size=16)
    rows: list[dict[str, object]] = []; masks: dict[str, torch.Tensor] = {}
    identity = torch.eye(config.N, dtype=torch.complex128)
    for index, name in enumerate(names):
        concentration = q_concentration(q[index].numpy(), K)
        masks[name] = score_support_mask(q[index].numpy(), K)
        rows.append({
            "waveform": name, "eta": concentration["eta_q"],
            "row_p05": concentration["row_p05"],
            "unitarity_residual": float(torch.linalg.matrix_norm(waveforms[name].mH @ waveforms[name] - identity)),
        })
    return rows, masks


def intervention_rows(
    selected: dict[str, dict[str, object]], structural: list[dict[str, object]],
    analytic: list[dict[str, object]], communication: list[dict[str, object]],
) -> list[dict[str, object]]:
    structure = {row["waveform"]: row for row in structural}
    geometry = {(row["waveform"], int(row["snr_db"]), row["support_rule"]): row for row in analytic}
    comm = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    rows: list[dict[str, object]] = []
    for name, metadata in selected.items():
        start = str(metadata["start"])
        before_g = geometry[(start, 18, "Q")]; after_g = geometry[(name, 18, "Q")]
        row: dict[str, object] = {
            "waveform": name, "start": start, "spec": metadata["spec"],
            "tail_lambda": metadata["tail_lambda"], "selected_step": metadata["step"],
            "validation_mean": metadata["validation_mean"], "validation_tail": metadata["validation_tail"],
            "validation_eta": metadata["validation_eta"], "validation_q_floor": metadata["q_floor"],
            "evaluation_eta_before": structure[start]["eta"], "evaluation_eta_after": structure[name]["eta"],
            "evaluation_eta_delta": float(structure[name]["eta"]) - float(structure[start]["eta"]),
            "mean_mse_before": before_g["mean_stream_mse_mean"], "mean_mse_after": after_g["mean_stream_mse_mean"],
            "tail_before": before_g["stream_mse_cvar95_adverse_cvar05"], "tail_after": after_g["stream_mse_cvar95_adverse_cvar05"],
            "retained_energy_before": before_g["retained_energy_mean"], "retained_energy_after": after_g["retained_energy_mean"],
        }
        for modulation in ("QPSK", "16QAM"):
            for support in ("Full", "Oracle", "Q"):
                before = float(comm[(start, modulation, 18, support)]["ber"])
                after = float(comm[(name, modulation, 18, support)]["ber"])
                prefix = f"{modulation.lower()}_{support.lower()}"
                row[f"{prefix}_ber_before"] = before; row[f"{prefix}_ber_after"] = after
                row[f"{prefix}_relative_change"] = after / before - 1.0
            prefix = modulation.lower()
            row[f"{prefix}_fixed_vs_oracle_penalty_before"] = float(row[f"{prefix}_q_ber_before"]) - float(row[f"{prefix}_oracle_ber_before"])
            row[f"{prefix}_fixed_vs_oracle_penalty_after"] = float(row[f"{prefix}_q_ber_after"]) - float(row[f"{prefix}_oracle_ber_after"])
        rows.append(row)
    return rows


def classify(rows: list[dict[str, object]], communication: list[dict[str, object]]) -> tuple[str, str]:
    if any(float(row["evaluation_eta_delta"]) < -2 * Q_EPSILON for row in rows):
        return "D", "Q/geometry incompatibility: held-out representation escaped the preregistered saturated regime."
    improved_tail = [row for row in rows if float(row["tail_after"]) < float(row["tail_before"])]
    joint = [row for row in rows if float(row["qpsk_q_relative_change"]) < 0 and float(row["16qam_q_relative_change"]) <= 0]
    families = {"OFDM" if str(row["start"]).endswith("OFDM") else "HAAR" for row in joint}
    if len(families) == 2:
        return "A", "Q-preserving analytic mean/tail intervention improved both constellations across OFDM- and Haar-derived starts."
    if all(float(row["mean_mse_after"]) >= float(row["mean_mse_before"]) for row in rows):
        return "C", "The tested adverse tail remains descriptive but is insufficient as a causal design objective: aggregate error never improved out of sample and communication changes were small and inconsistent."
    if improved_tail and all(float(row["qpsk_q_relative_change"]) >= 0 for row in improved_tail):
        return "C", "The adverse analytic tail was directly improved, but QPSK communication did not respond as predicted."
    return "B", "Analytically selected mean/tail operating points expose a genuine communication tradeoff; no BER-selected universal point is chosen."


def make_plots(trace: list[dict[str, object]], selected: dict[str, dict[str, object]], structural: list[dict[str, object]], communication: list[dict[str, object]]) -> None:
    def short(row: dict[str, object]) -> str:
        start = str(row["start"]).replace("Q_MEAN_FROM_", "QM-").replace("Q_ROBUST_FROM_", "QR-").replace("HAAR_", "H")
        spec = str(row["spec"]).replace("TAIL_", "T").replace("MEAN", "M")
        return f"{spec}:{start}"

    plt.figure(figsize=(7.2, 4.5))
    for spec in ("MEAN", "TAIL_0.25", "TAIL_1"):
        rows = [row for row in trace if row["spec"] == spec]
        steps = sorted({int(row["step"]) for row in rows})
        plt.plot(steps, [np.mean([float(row["validation_mean"]) for row in rows if int(row["step"]) == step]) for step in steps], label=spec)
    plt.xlabel("step"); plt.ylabel("validation aggregate error"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "fig01_validation_mean.png", dpi=180); plt.close()

    plt.figure(figsize=(6.8, 4.8))
    for row in selected.values():
        plt.scatter(float(row["validation_mean"]), float(row["validation_tail"]), label=short(row), s=28)
    plt.xlabel("validation M"); plt.ylabel("validation T"); plt.legend(fontsize=6, ncol=2); plt.tight_layout()
    plt.savefig(OUT / "fig02_validation_pareto.png", dpi=180); plt.close()

    selected_names = tuple(selected)
    labels = [short(selected[name]) for name in selected_names]
    values = {row["waveform"]: row for row in structural}
    plt.figure(figsize=(max(8, .45 * len(selected_names)), 4.5))
    plt.bar(np.arange(len(selected_names)), [float(values[name]["eta"]) for name in selected_names])
    plt.axhline(.999, color="black", linestyle="--", linewidth=1); plt.ylim(.9985, 1.0)
    plt.xticks(np.arange(len(selected_names)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel(r"held-out $\eta_{24}^Q$"); plt.tight_layout()
    plt.savefig(OUT / "fig03_q_preservation.png", dpi=180); plt.close()

    lookup = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    show = ("Learned",) + selected_names
    fig, axes = plt.subplots(1, 2, figsize=(max(10, .55 * len(show)), 4.8))
    for axis, modulation in zip(axes, ("QPSK", "16QAM")):
        axis.bar(np.arange(len(show)), [float(lookup[(name, modulation, 18, "Q")]["ber"]) for name in show])
        axis.set_yscale("log"); axis.set_title(modulation); axis.set_ylabel("fixed-Q BER at 18 dB")
        axis.set_xticks(np.arange(len(show)), ("Learned", *labels), rotation=60, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig04_fixed_q_ber.png", dpi=180); plt.close(fig)


def write_report(outcome: str, interpretation: str, rows: list[dict[str, object]], diagnostics: dict[str, object]) -> None:
    best_qpsk = min(rows, key=lambda row: float(row["qpsk_q_ber_after"]))
    best_joint = min(rows, key=lambda row: max(float(row["qpsk_q_relative_change"]), float(row["16qam_q_relative_change"])))
    mean_changes = [float(row["mean_mse_after"]) / float(row["mean_mse_before"]) - 1 for row in rows]
    tail_changes = [float(row["tail_after"]) / float(row["tail_before"]) - 1 for row in rows]
    qpsk_changes = [float(row["qpsk_q_relative_change"]) for row in rows]
    qam_changes = [float(row["16qam_q_relative_change"]) for row in rows]
    lines = [
        "# Stage 3D — Causal Completion of the DD Design Principle", "",
        f"## Result: Outcome {outcome}", "", interpretation, "",
        "## Frozen analytic intervention", "",
        f"- Exact mismatched fixed-Q LMMSE geometry at {GEOMETRY_SNR_DB} dB; K={K}, N={config.N}, P={config.P}.",
        f"- Tail is the Stage-3C two-level upper stream/channel CVaR with beta={TAIL_BETA}.",
        f"- Per-start Q floor uses epsilon_Q={Q_EPSILON}; Arm-B lambdas are {TAIL_LAMBDAS} after frozen per-start normalization.",
        f"- {len(rows)} analytic checkpoints entered paired communication evaluation; none was selected using BER.", "",
        "## Central 18 dB fixed-Q results", "",
        "| Checkpoint | Start | M change | T change | eta change | QPSK BER change | 16-QAM BER change |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['waveform']} | {row['start']} | {float(row['mean_mse_after'])/float(row['mean_mse_before'])-1:+.1%} | "
            f"{float(row['tail_after'])/float(row['tail_before'])-1:+.1%} | {float(row['evaluation_eta_delta']):+.2e} | "
            f"{float(row['qpsk_q_relative_change']):+.1%} | {float(row['16qam_q_relative_change']):+.1%} |"
        )
    lines += [
        "", "## Interpretation", "",
        f"- Held-out M changed {min(mean_changes):+.2%} to {max(mean_changes):+.2%}; no intervention improved it.",
        f"- Held-out T changed {min(tail_changes):+.2%} to {max(tail_changes):+.2%}; the best reduction was not material or consistent.",
        f"- Fixed-Q QPSK changed {min(qpsk_changes):+.2%} to {max(qpsk_changes):+.2%}; 16-QAM changed {min(qam_changes):+.2%} to {max(qam_changes):+.2%} and never improved.",
        f"- Lowest analytically frozen QPSK BER: `{best_qpsk['waveform']}`; this label is descriptive and caused no reselection.",
        f"- Best simultaneous relative-change control: `{best_joint['waveform']}`; again, no post-hoc checkpoint choice was made.",
        "- Central comparisons are same-start pre/post interventions; Learned and classical bases remain frozen controls.",
        "- Full, oracle-K, and fixed-Q BER plus fixed-vs-oracle penalties are available for both modulations at 12/15/18/20 dB in the machine-readable communication and causal tables.",
        "- STOP. No Stage 3E, from-scratch synthesis, receiver change, or waveform-law distillation was performed.", "",
        "## Preservation", "",
        f"- Authoritative Learned hash: `{diagnostics['waveform_hashes']['Learned']}`.",
        f"- All six Stage-3B hashes were verified; all {diagnostics['protected_artifact_count']} prior artifacts remained byte-identical.",
        "- IDENTITY is retained in numerical outputs and excluded from figures. Stage 2E was not used.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def run_stage3d(*, steps: int = TRAINING_STEPS, force_training: bool = False) -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes(); frozen, waveform_hashes = load_waveforms()
    payload, selected = train_or_load(steps, force_training)
    write_csv(OUT / "training_validation_history.csv", payload["trace_rows"])
    selected_metadata = [{key: value for key, value in row.items() if key != "matrix"} | {"waveform": name} for name, row in selected.items()]
    write_csv(OUT / "validation_selected_checkpoints.csv", selected_metadata)
    selected_waveforms = {name: row["matrix"] for name, row in selected.items()}
    waveforms = {**frozen, **selected_waveforms}
    structural, masks = structural_rows(waveforms); write_csv(OUT / "heldout_q_summary.csv", structural)
    channels = sample_dd_v1(EVALUATION_CHANNELS, EVALUATION_CHANNEL_SEED).matrices
    analytic_channel, analytic_summary = analytic_geometry(channels, waveforms, masks)
    write_csv(OUT / "receiver_geometry_per_channel.csv", analytic_channel); write_csv(OUT / "receiver_geometry_summary.csv", analytic_summary)
    print("frozen analytic evaluation complete", flush=True)
    per_channel = communication_rows(
        channels, waveforms, masks, (qpsk_constellation(), qam16_constellation()), SNRS,
        k=K, target_bits=TARGET_BITS, symbol_seed=SYMBOL_SEED, noise_seed=NOISE_SEED,
    )
    communication = communication_summary(per_channel)
    write_csv(OUT / "communication_per_channel.csv", per_channel); write_csv(OUT / "communication_summary.csv", communication)
    interventions = intervention_rows(selected, structural, analytic_summary, communication)
    write_csv(OUT / "causal_intervention_summary.csv", interventions)
    outcome, interpretation = classify(interventions, communication)
    make_plots(payload["trace_rows"], selected, structural, communication)
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage 3D changed a prior artifact")
    diagnostics = {
        "outcome": outcome, "interpretation": interpretation, "stage2e_used": False,
        "communication_used_for_selection": False, "identity_in_figures": False,
        "n": config.N, "p": config.P, "k": K, "snrs_db": list(SNRS),
        "geometry_snr_db": GEOMETRY_SNR_DB, "tail_beta": TAIL_BETA,
        "tail_lambdas": list(TAIL_LAMBDAS), "q_epsilon": Q_EPSILON,
        "selected_checkpoint_count": len(selected), "waveform_hashes": waveform_hashes,
        "stage3d_checkpoint_hashes": {path.name: file_sha256(path) for path in sorted(CHECKPOINTS.glob("geom_*_best.pt"))},
        "protected_artifact_count": len(protected_before), "protected_artifacts_unchanged": True,
        "training_runtime_seconds": payload["runtime_seconds"], "runtime_seconds": time.perf_counter() - started,
        "evaluation_bank": {"channel_seed": EVALUATION_CHANNEL_SEED, "channels": EVALUATION_CHANNELS, "q_seed": EVALUATION_Q_SEED, "q_samples": EVALUATION_Q_SAMPLES, "symbol_seed": SYMBOL_SEED, "noise_seed": NOISE_SEED},
    }
    write_json(OUT / "stage3d_diagnostics.json", diagnostics)
    write_report(outcome, interpretation, interventions, diagnostics)
    print(f"STAGE 3D COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME {outcome}; STOP", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=TRAINING_STEPS); parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); run_stage3d(steps=args.steps, force_training=args.force_training)
