"""Close mandatory Stage-3B evidence before beginning Stage 3C."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.stage3b import load_or_create_banks
from src import config
from src.channel import COMPLEX_DTYPE, noise_variance, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator
from src.receiver_geometry import apply_mask, distribution_summary, exact_error_geometry, support_geometry
from src.stage2a import aggregate_rows, load_frozen_waveform
from src.stage2d import score_support_mask, sobol_dd_samples, support_communication_rows
from src.stage3a import q_concentration
from src.stage3b import (
    BER_NOISE_SEED,
    BER_SYMBOL_SEED,
    EVALUATION_CHANNELS,
    EVALUATION_CHANNEL_SEED,
    EVALUATION_Q_SEED,
    TRAIN_K,
    aligned_basis_similarity,
    atomic_operators,
    differentiable_q,
    optimize_q_waveforms,
)
from src.waveforms import identity_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3b_completion"
CHECKPOINTS = OUT / "checkpoints"
STAGE3B = ROOT / "outputs" / "stage3b"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
ARMS = ("mean", "robust")
PRIMARY = (
    "Q_MEAN_FROM_OFDM", "Q_MEAN_FROM_HAAR_1", "Q_MEAN_FROM_HAAR_2",
    "Q_ROBUST_FROM_OFDM", "Q_ROBUST_FROM_HAAR_1", "Q_ROBUST_FROM_HAAR_2",
)
BANK_SIZES = (128, 256, 512, 1024, 2048)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / "outputs").glob("**/*"))
        if path.is_file() and "stage3b_completion" not in path.parts
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_primary() -> dict[str, torch.Tensor]:
    waveforms: dict[str, torch.Tensor] = {}
    for name in PRIMARY:
        filename = name.lower() + "_best.pt"
        waveforms[name] = torch.load(STAGE3B / "checkpoints" / filename, weights_only=True)
    return waveforms


def train_learned_arms(
    learned: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], list[dict[str, object]]]:
    train_operators, validation_operators = load_or_create_banks()
    selected: dict[str, torch.Tensor] = {}
    traces: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for arm in ARMS:
        path = CHECKPOINTS / f"training_{arm}_from_learned.pt"
        if path.exists():
            payload = torch.load(path, weights_only=False)
        else:
            result = optimize_q_waveforms(
                learned[None], ("LEARNED",), train_operators, validation_operators,
                arm=arm, steps=1000, validation_cadence=25,
            )
            payload = {
                "arm": arm,
                "best_matrices": result.best_matrices,
                "best_steps": result.best_steps,
                "initial_validation_eta": result.initial_validation_eta,
                "best_validation_eta": result.best_validation_eta,
                "initial_validation_lower_tail": result.initial_validation_lower_tail,
                "best_validation_lower_tail": result.best_validation_lower_tail,
                "trace_rows": result.trace_rows,
                "runtime_seconds": result.runtime_seconds,
                "max_unitarity_residuals": result.max_unitarity_residuals,
                "nonfinite_seen": result.nonfinite_seen,
            }
            torch.save(payload, path)
            torch.save(result.best_matrices[0], CHECKPOINTS / f"q_{arm}_from_learned_best.pt")
        selected[f"Q_{arm.upper()}_FROM_LEARNED"] = payload["best_matrices"][0]
        traces.extend(payload["trace_rows"])
        summaries.append({
            "arm": arm,
            "initialization": "Learned",
            "selected_step": int(payload["best_steps"][0]),
            "initial_validation_eta": float(payload["initial_validation_eta"][0]),
            "selected_validation_eta": float(payload["best_validation_eta"][0]),
            "initial_validation_lower_tail": float(payload["initial_validation_lower_tail"][0]),
            "selected_validation_lower_tail": float(payload["best_validation_lower_tail"][0]),
            "maximum_unitarity_residual": float(payload["max_unitarity_residuals"][0]),
            "nonfinite_seen": bool(payload["nonfinite_seen"][0]),
            "runtime_seconds": float(payload["runtime_seconds"]),
        })
    return selected, traces, summaries


def final_quarter_flip_rows(traces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        names = sorted({str(row["initialization"]) for row in traces if row["arm"] == arm})
        for name in names:
            selected = [row for row in traces if row["arm"] == arm and row["initialization"] == name]
            maximum = max(int(row["step"]) for row in selected)
            final = [float(row["support_churn"]) for row in selected if int(row["step"]) > 0.75 * maximum]
            rows.append({
                "arm": arm, "initialization": name, "start_step_exclusive": int(0.75 * maximum),
                "observations": len(final), "mean_flip_fraction": float(np.mean(final)),
                "p95_flip_fraction": float(np.quantile(final, 0.95)), "maximum_flip_fraction": max(final),
            })
    return rows


def run_stage3b_completion() -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    protected_before = tree_hashes()
    learned, frozen_info = load_frozen_waveform(FROZEN)
    primary = load_primary()
    warm, warm_traces, warm_summaries = train_learned_arms(learned)
    write_csv(OUT / "learned_warm_start_training.csv", warm_summaries)
    waveforms = {"Learned": learned, **primary, **warm}

    original_traces = list(csv.DictReader((STAGE3B / "training_validation_history.csv").open()))
    flip_rows = final_quarter_flip_rows([*original_traces, *warm_traces])
    write_csv(OUT / "final_quarter_support_flip_rate.csv", flip_rows)

    delays, dopplers = sobol_dd_samples(max(BANK_SIZES), EVALUATION_Q_SEED)
    operators = atomic_operators(delays, dopplers)
    names = tuple(waveforms)
    bank_rows: list[dict[str, object]] = []
    q_full: dict[str, np.ndarray] = {}
    for size in BANK_SIZES:
        with torch.no_grad():
            q_values = differentiable_q(torch.stack([waveforms[name] for name in names]), operators[:size], chunk_size=16)
        for index, name in enumerate(names):
            q = q_values[index].cpu().numpy()
            concentration = q_concentration(q, TRAIN_K)
            bank_rows.append({
                "waveform": name, "bank_size": size, "k": TRAIN_K,
                "eta_q": concentration["eta_q"],
                "row_mean": concentration["row_mean"],
                "row_median": concentration["row_median"],
                "row_p05": concentration["row_p05"],
                "row_worst": concentration["row_worst"],
                "effective_support_mean": concentration["effective_support_mean"],
                "normalized_entropy_mean": concentration["normalized_entropy_mean"],
            })
            if size == max(BANK_SIZES):
                q_full[name] = q
    write_csv(OUT / "q_bank_size_saturation.csv", bank_rows)

    channels = sample_dd_v1(EVALUATION_CHANNELS, EVALUATION_CHANNEL_SEED).matrices
    locality = np.load(ROOT / "outputs" / "stage2b" / "learned_association_matrices.npz")
    labels = torch.from_numpy(locality["community_labels"])
    order = torch.from_numpy(locality["centroid_order"])
    geometry_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    masks: dict[str, torch.Tensor] = {}
    identity = identity_waveform()
    for name, waveform in waveforms.items():
        mask = score_support_mask(q_full[name], TRAIN_K); masks[name] = mask
        effective = effective_operator(channels, waveform)
        receiver = exact_error_geometry(effective, apply_mask(effective, mask), noise_variance(18))
        row: dict[str, object] = {"waveform": name, "snr_db": 18}
        for metric, high_is_bad in (
            ("retained_energy", False), ("condition_number", True), ("lambda_min", False),
            ("equalizer_noise_enhancement", True), ("worst_gamma", False),
            ("gamma_lower_cvar05", False), ("stream_mse_cvar95", True),
        ):
            row.update({f"{metric}_{key}": value for key, value in distribution_summary(receiver[metric], high_is_bad=high_is_bad).items()})
        geometry_rows.append(row)
        aligned = aligned_basis_similarity(waveform, identity)
        mask_rows.append({
            "waveform": name,
            "aligned_similarity_to_generic_permutation": aligned["mean_matched_squared_overlap"],
            **support_geometry(mask, labels, order),
        })
    write_csv(OUT / "finite_receiver_geometry.csv", geometry_rows)
    write_csv(OUT / "permutation_and_locality.csv", mask_rows)

    communication_waveforms = {"Learned": learned, **warm}
    communication_masks = {name: {"Q": masks[name]} for name in communication_waveforms}
    per_channel = support_communication_rows(
        channels, communication_waveforms, communication_masks,
        {name: TRAIN_K for name in communication_waveforms},
        (qpsk_constellation(), qam16_constellation()), (18,),
        target_bits=1_000_000, symbol_seed=BER_SYMBOL_SEED, noise_seed=BER_NOISE_SEED,
    )
    communication = aggregate_rows(per_channel, ("modulation", "snr_db", "waveform", "support_rule"))
    write_csv(OUT / "learned_warm_start_communication.csv", communication)
    write_csv(OUT / "learned_warm_start_per_channel.csv", per_channel)

    protected_after = tree_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage-3B completion modified a frozen prior artifact")
    diagnostics = {
        "stage": "3B completion",
        "learned_sha256": frozen_info["sha256"],
        "primary_stage3b_hashes": {name: sha256(STAGE3B / "checkpoints" / (name.lower() + "_best.pt")) for name in PRIMARY},
        "learned_warm_hashes": {path.name: sha256(path) for path in sorted(CHECKPOINTS.glob("q_*_best.pt"))},
        "mandatory_items": {
            "learned_warm_start_arms": True,
            "finite_conditioning_noise_enhancement": True,
            "q_bank_size_saturation": True,
            "final_quarter_support_flip_rate": True,
            "generic_permutation_similarity": True,
            "stage2b_locality_comparison": True,
        },
        "protected_prior_artifacts_unchanged": True,
        "runtime_seconds": time.perf_counter() - started,
    }
    with (OUT / "stage3b_completion_diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True); handle.write("\n")
    print(f"STAGE 3B COMPLETION PASS in {diagnostics['runtime_seconds']:.1f} s", flush=True)
    return diagnostics


if __name__ == "__main__":
    run_stage3b_completion()
