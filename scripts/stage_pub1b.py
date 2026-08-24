"""Resumable Stage Pub-1B sparse-compute and dimension-scaling runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance
from src.metrics import (
    Constellation,
    constellation_encode,
    nearest_neighbor_decode,
    qam16_constellation,
    qam64_constellation,
    qpsk_constellation,
    random_bits,
)
from src.oracle import polar_project
from src.pub1a.core import REGIME_BY_KEY, fixed_statistical_masks, waveform_set
from src.pub1b.core import (
    BER_THRESHOLDS,
    COMPUTE_K,
    ITERATION_BUDGETS,
    K_FRACTIONS,
    N64_CHECKPOINT_SHA256,
    PARENT_COMMIT,
    PUB1A_TREE_SHA256,
    TAIL_BETA,
    TAIL_LAMBDA,
    afdm_candidate_multipliers,
    cg_mac_count,
    clean_reconstruction_error,
    directory_tree_sha256,
    file_sha256,
    fixed_scaled_masks,
    full_mse_invariance,
    sample_scaled_dd,
    scaled_waveforms,
    scaling_definition,
    scaling_rule_record,
    sparse_cg_batch_snapshots,
    sparse_cg_lmmse,
    sparse_from_mask,
    stream_cvar,
    stream_mse_scaled,
    validation_metrics_scaled,
)
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer
from src.stage2a import load_frozen_waveform
from src.stage2b import operator_offset_metrics
from src.stage2d import apply_support
from src.stage2e import sample_regime
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage_pub1b"
DATA_COMPLEXITY = OUT / "data" / "complexity"
DATA_SCALING = OUT / "data" / "scaling"
FIG_COMPLEXITY = OUT / "figures" / "complexity"
FIG_SCALING = OUT / "figures" / "scaling"
CHECKPOINTS = OUT / "checkpoints"
PROVENANCE = OUT / "provenance"
FROZEN64 = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
PUB1A = ROOT / "outputs" / "stage_pub1a"
PUB1A_KSWEEP = PUB1A / "data" / "k_sweep_summary.csv"
COLORS = {"Learned": "#d97706", "OTFS": "#15803d", "AFDM": "#dc2626", "OFDM": "#2563eb"}
MARKERS = {"Learned": "o", "OTFS": "s", "AFDM": "^", "OFDM": "D"}
SEEDS = {
    "complexity_validation": 14000,
    "complexity_support": 12280,
    "complexity_test": 12380,
    "complexity_symbols": 12480,
    "complexity_noise": 12580,
    "scaling_validation": 14100,
    "scaling_support": 14200,
    "scaling_test": 14300,
    "scaling_symbols": 14400,
    "scaling_noise": 14500,
    "training_channels": 14600,
    "training_snr": 14700,
}
COMPUTE_POINTS = (
    (qpsk_constellation(), 18),
    (qam16_constellation(), 18),
    (qam64_constellation(), 25),
)
SCALING_SNRS = (12, 15, 18, 20, 25)


def ensure_dirs() -> None:
    for path in (DATA_COMPLEXITY, DATA_SCALING, FIG_COMPLEXITY, FIG_SCALING, CHECKPOINTS, PROVENANCE):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def canonical_name(name: str) -> str:
    return "Learned" if name.startswith("Learned-N") else name


def preservation_gate() -> dict[str, object]:
    if file_sha256(FROZEN64) != N64_CHECKPOINT_SHA256:
        raise RuntimeError("authoritative N=64 Learned checkpoint changed")
    tree = directory_tree_sha256(PUB1A)
    if tree != PUB1A_TREE_SHA256:
        raise RuntimeError(f"Pub-1A result package changed: {tree}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_COMMIT, head], cwd=ROOT).returncode == 0:
        raise RuntimeError("Pub-1B HEAD does not descend from the exact Pub-1A parent")
    record = {
        "pub1a_parent_commit": PARENT_COMMIT,
        "current_commit": head,
        "target_ref": "origin/pub",
        "n64_checkpoint_sha256": N64_CHECKPOINT_SHA256,
        "pub1a_tree_sha256": tree,
        "pub1a_file_count": sum(path.is_file() for path in PUB1A.rglob("*")),
        "preserved": True,
    }
    write_json(PROVENANCE / "pub1a_preservation.json", record)
    return record


def environment_record() -> dict[str, object]:
    import scipy
    return {
        "pub1a_parent_commit": PARENT_COMMIT,
        "hardware": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pytorch": torch.__version__,
        "sparse_backend": "custom torch complex128 exact-K row storage with gather/scatter-add",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_policy": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


def provenance_record() -> dict[str, object]:
    return {
        "pub1a_parent_commit": PARENT_COMMIT,
        "target_ref": "origin/pub",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "seeds": SEEDS,
        "compute": {
            "k": list(COMPUTE_K), "iteration_budgets": list(ITERATION_BUDGETS),
            "tolerance": 1e-10, "preconditioner": None,
            "support_cost": "offline channel-class statistic; excluded from per-packet detector MACs",
            "setup_cost": "A_K^H z counted separately as N*K complex MACs per received vector",
            "execution_cost": "2*N*K sparse plus 6*N vector/dot complex-equivalent MACs per completed CG iteration; initial r^H r costs N",
        },
        "environment": environment_record(),
    }


def load_n64_waveforms() -> dict[str, torch.Tensor]:
    learned, _ = load_frozen_waveform(FROZEN64)
    return waveform_set(learned, 3)


def phase_complexity_validate(smoke: bool = False) -> None:
    preservation_gate()
    waveforms = load_n64_waveforms()
    regime = REGIME_BY_KEY["dd_v1"]
    masks, _ = fixed_statistical_masks(waveforms, regime, (24,), 64 if smoke else 256, SEEDS["complexity_support"])
    channels = sample_regime(regime.stage2e(), 1 if smoke else 3, SEEDS["complexity_validation"]).matrices
    generator = make_generator(SEEDS["complexity_validation"] + 1)
    rows = []
    for name in ("Learned", "OTFS", "AFDM"):
        actual = effective_operator(channels, waveforms[name])
        for channel_index, matrix in enumerate(actual):
            retained = apply_support(matrix, masks[name]["fixedStatK24"])
            sparse = sparse_from_mask(matrix, masks[name]["fixedStatK24"])
            symbols = torch.randn(64, dtype=COMPLEX_DTYPE, generator=generator)
            noise = torch.randn(64, dtype=COMPLEX_DTYPE, generator=generator) * math.sqrt(float(noise_variance(18)))
            observation = matrix @ symbols + noise
            direct = mismatched_lmmse_equalizer(retained, noise_variance(18)) @ observation
            iterative = sparse_cg_lmmse(sparse, observation, noise_variance(18), max_iterations=256, tolerance=1e-12)
            relative = float(torch.linalg.vector_norm(iterative.estimate - direct) / torch.linalg.vector_norm(direct))
            rows.append({
                "waveform": name, "channel_index": channel_index, "k": 24,
                "relative_solution_error": relative,
                "relative_residual": iterative.relative_residual,
                "iterations": iterative.iterations, "converged": iterative.converged,
                "dtype": str(iterative.estimate.dtype), "nnz": sparse.nnz,
                "same_estimator": relative < 1e-10 and iterative.relative_residual < 1e-12,
            })
    if not all(row["same_estimator"] for row in rows):
        write_csv(DATA_COMPLEXITY / "numerical_validation.csv", rows)
        raise RuntimeError("HARD STOP: sparse iterative solver differs from direct K-LMMSE")
    write_csv(DATA_COMPLEXITY / "numerical_validation.csv", rows)
    write_json(PROVENANCE / "run.json", provenance_record())


def _error_counts(bits: torch.Tensor, indices: torch.Tensor, estimates: torch.Tensor, constellation: Constellation) -> tuple[int, int]:
    decoded, decoded_indices = nearest_neighbor_decode(estimates, constellation)
    return int(torch.sum(decoded != bits)), int(torch.sum(decoded_indices != indices))


def _complexity_inputs(smoke: bool):
    waveforms = load_n64_waveforms()
    regime = REGIME_BY_KEY["dd_v1"]
    count = 4 if smoke else 64
    support_samples = 64 if smoke else 2048
    channels = sample_regime(regime.stage2e(), count, SEEDS["complexity_test"]).matrices
    masks, _ = fixed_statistical_masks(waveforms, regime, COMPUTE_K, support_samples, SEEDS["complexity_support"])
    return waveforms, channels, masks


def phase_complexity_run(smoke: bool = False) -> None:
    validation_path = DATA_COMPLEXITY / "numerical_validation.csv"
    if not validation_path.exists() or not all(row["same_estimator"] == "True" for row in read_csv(validation_path)):
        raise RuntimeError("run and pass --phase complexity-validate first")
    preservation_gate()
    waveforms, channels, masks = _complexity_inputs(smoke)
    target_bits = 4096 if smoke else 262144
    budgets = (1, 4, 16) if smoke else ITERATION_BUDGETS
    rows: list[dict[str, object]] = []
    runtime_inputs: dict[str, object] = {}
    with torch.no_grad():
        for constellation, snr_db in COMPUTE_POINTS:
            blocks = math.ceil(target_bits / (len(channels) * 64 * constellation.bits_per_symbol))
            bits = random_bits(
                (len(channels), blocks, 64, constellation.bits_per_symbol),
                make_generator(SEEDS["complexity_symbols"] + constellation.bits_per_symbol),
            )
            symbols, indices = constellation_encode(bits, constellation)
            noise = complex_awgn(
                (len(channels), blocks, 64), noise_variance(snr_db),
                make_generator(SEEDS["complexity_noise"] + constellation.bits_per_symbol),
            )
            for name in ("Learned", "OTFS", "AFDM", "OFDM"):
                waveform = waveforms[name]
                transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
                physical = torch.einsum("cij,cbj->cbi", channels, transmitted) + noise
                received = torch.einsum("ij,cbj->cbi", waveform.mH, physical)
                actual = effective_operator(channels, waveform)
                for k in COMPUTE_K if not smoke else (24, 64):
                    totals = {
                        budget: {"bit_errors": 0, "symbol_errors": 0, "iterations": 0, "residual": 0.0,
                                 "converged": 0, "setup_seconds": 0.0, "execution_seconds": 0.0}
                        for budget in budgets
                    }
                    direct_bit_errors = 0
                    direct_symbol_errors = 0
                    analytic_mse = 0.0
                    systems = 0
                    for channel_index in range(len(channels)):
                        mode = "full" if k == 64 else f"fixedStatK{k}"
                        mask = torch.ones((64, 64), dtype=torch.bool) if k == 64 else masks[name][mode]
                        retained = apply_support(actual[channel_index], mask)
                        sparse = sparse_from_mask(actual[channel_index], mask)
                        equalizer = mismatched_lmmse_equalizer(retained, noise_variance(snr_db))
                        direct = torch.einsum("ij,bj->bi", equalizer, received[channel_index])
                        bit_errors, symbol_errors = _error_counts(bits[channel_index], indices[channel_index], direct, constellation)
                        direct_bit_errors += bit_errors
                        direct_symbol_errors += symbol_errors
                        analytic_mse += float(analytic_communication_mse(actual[channel_index], equalizer, noise_variance(snr_db)))
                        snapshots, setup_seconds, execution_seconds = sparse_cg_batch_snapshots(
                            sparse, received[channel_index], noise_variance(snr_db), budgets, tolerance=1e-10,
                        )
                        systems += blocks
                        for budget, snapshot in snapshots.items():
                            bit_errors, symbol_errors = _error_counts(bits[channel_index], indices[channel_index], snapshot.estimate, constellation)
                            total = totals[budget]
                            total["bit_errors"] += bit_errors
                            total["symbol_errors"] += symbol_errors
                            total["iterations"] += int(snapshot.iterations.sum())
                            total["residual"] += float(snapshot.relative_residual.sum())
                            total["converged"] += int(snapshot.converged.sum())
                            total["setup_seconds"] += setup_seconds
                            total["execution_seconds"] += execution_seconds * budget / budgets[-1]
                        if k == 24 and channel_index == 0:
                            runtime_inputs[f"{constellation.name}|{name}"] = {
                                "columns": sparse.columns, "values": sparse.values,
                                "observation": received[channel_index, 0], "sigma2": float(noise_variance(snr_db)),
                            }
                    total_bits = bits.numel()
                    total_symbols = indices.numel()
                    for budget in budgets:
                        total = totals[budget]
                        iterations = total["iterations"]
                        mac = cg_mac_count(64, k, 0)
                        sparse_macs = 2 * 64 * k * iterations
                        vector_macs = 64 * systems + 6 * 64 * iterations
                        setup_macs = 64 * k * systems
                        rows.append({
                            "modulation": constellation.name, "snr_db": snr_db, "waveform": name,
                            "k": k, "iteration_budget": budget, "systems": systems,
                            "bit_errors": total["bit_errors"], "total_bits": total_bits,
                            "ber": total["bit_errors"] / total_bits,
                            "symbol_errors": total["symbol_errors"], "total_symbols": total_symbols,
                            "ser": total["symbol_errors"] / total_symbols,
                            "mean_relative_residual": total["residual"] / systems,
                            "direct_bit_errors": direct_bit_errors, "direct_ber": direct_bit_errors / total_bits,
                            "direct_symbol_errors": direct_symbol_errors, "direct_ser": direct_symbol_errors / total_symbols,
                            "direct_analytic_mse": analytic_mse / len(channels),
                            "mean_actual_iterations": iterations / systems,
                            "convergence_fraction": total["converged"] / systems,
                            "sparse_detector_macs": sparse_macs,
                            "vector_dot_macs": vector_macs,
                            "preconditioner_macs": 0,
                            "setup_macs": setup_macs,
                            "execution_macs": sparse_macs + vector_macs,
                            "total_macs": sparse_macs + vector_macs + setup_macs,
                            "mean_execution_macs": (sparse_macs + vector_macs) / systems,
                            "mean_total_macs": (sparse_macs + vector_macs + setup_macs) / systems,
                            "raw_setup_seconds": total["setup_seconds"],
                            "raw_execution_seconds": total["execution_seconds"],
                        })
    write_csv(DATA_COMPLEXITY / "operating_points.csv", rows)
    torch.save(runtime_inputs, DATA_COMPLEXITY / "runtime_inputs.pt")
    phase_complexity_runtime(smoke)
    phase_complexity_plots(smoke)


def phase_complexity_runtime(smoke: bool = False) -> None:
    path = DATA_COMPLEXITY / "runtime_inputs.pt"
    if not path.exists():
        raise RuntimeError("run --phase complexity-run first")
    inputs = torch.load(path, weights_only=True)
    repeats = 3 if smoke else 25
    warmups = 1 if smoke else 5
    budgets = (1, 4, 16) if smoke else ITERATION_BUDGETS
    rows = []
    # Runtime alone uses a controlled single-thread PyTorch policy. Training and
    # Monte Carlo retain the environment's normal CPU parallelism.
    torch.set_num_threads(1)
    for key, payload in inputs.items():
        modulation, waveform = key.split("|")
        from src.pub1b.core import SparseRowMatrix
        sparse = SparseRowMatrix(payload["columns"], payload["values"])
        for budget in budgets:
            for _ in range(warmups):
                sparse_cg_lmmse(sparse, payload["observation"], payload["sigma2"], max_iterations=budget, tolerance=1e-10)
            setup = []
            execution = []
            total = []
            for _ in range(repeats):
                result = sparse_cg_lmmse(sparse, payload["observation"], payload["sigma2"], max_iterations=budget, tolerance=1e-10)
                setup.append(result.setup_seconds)
                execution.append(result.execution_seconds)
                total.append(result.setup_seconds + result.execution_seconds)
            rows.append({
                "modulation": modulation, "waveform": waveform, "k": 24, "iteration_budget": budget,
                "warmups": warmups, "repeats": repeats,
                "median_setup_seconds": float(np.median(setup)),
                "median_execution_seconds": float(np.median(execution)),
                "median_detector_seconds": float(np.median(total)),
                "p25_detector_seconds": float(np.quantile(total, .25)),
                "p75_detector_seconds": float(np.quantile(total, .75)),
            })
    write_csv(DATA_COMPLEXITY / "runtime.csv", rows)
    write_json(PROVENANCE / "runtime_environment.json", environment_record())


def pareto_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for modulation in sorted({row["modulation"] for row in rows}):
        for waveform in sorted({row["waveform"] for row in rows}):
            selected = [row for row in rows if row["modulation"] == modulation and row["waveform"] == waveform]
            selected.sort(key=lambda row: (float(row["mean_total_macs"]), float(row["ber"])))
            best = math.inf
            for row in selected:
                ber = float(row["ber"])
                if ber < best:
                    output.append({**row, "pareto": True})
                    best = ber
    return output


def _compute_classification(rows: list[dict[str, str]]) -> tuple[str, dict[str, object]]:
    evidence = {}
    passing = 0
    for modulation in ("QPSK", "16QAM", "64QAM"):
        current = [row for row in rows if row["modulation"] == modulation]
        learned = [row for row in current if row["waveform"] == "Learned"]
        classical = [row for row in current if row["waveform"] in ("OTFS", "AFDM")]
        wins = 0
        for row in learned:
            budget = float(row["mean_total_macs"])
            competitors = [float(value["ber"]) for value in classical if float(value["mean_total_macs"]) <= budget]
            if competitors and float(row["ber"]) < min(competitors):
                wins += 1
        nontrivial = wins >= 2
        evidence[modulation] = {"strict_common_or_lower_mac_wins": wins, "nontrivial": nontrivial}
        passing += int(nontrivial)
    classification = "SUPPORTED" if passing == 3 else "PARTIALLY SUPPORTED" if passing else "NOT SUPPORTED"
    return classification, evidence


def phase_complexity_plots(smoke: bool = False) -> None:
    rows = read_csv(DATA_COMPLEXITY / "operating_points.csv")
    runtime = read_csv(DATA_COMPLEXITY / "runtime.csv")
    pareto = pareto_rows(rows)
    write_csv(DATA_COMPLEXITY / "pareto_frontiers.csv", pareto)
    thresholds = []
    for modulation in sorted({row["modulation"] for row in rows}):
        for waveform in ("Learned", "OTFS", "AFDM"):
            selected = [row for row in rows if row["modulation"] == modulation and row["waveform"] == waveform]
            for threshold in BER_THRESHOLDS:
                reached = [row for row in selected if float(row["ber"]) <= threshold]
                if reached:
                    best = min(reached, key=lambda row: float(row["mean_total_macs"]))
                    thresholds.append({"modulation": modulation, "waveform": waveform, "ber_threshold": threshold,
                                       "minimum_mean_total_macs": float(best["mean_total_macs"]),
                                       "k": int(best["k"]), "iteration_budget": int(best["iteration_budget"]),
                                       "observed_ber": float(best["ber"])})
    if thresholds:
        write_csv(DATA_COMPLEXITY / "threshold_table.csv", thresholds)

    for xkey, name, xlabel in (
        ("iteration_budget", "B1_ber_vs_iterations_k24", "CG iteration budget"),
        ("mean_total_macs", "B2_ber_vs_macs_k24", "mean detector complex MACs (setup + execution)"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
            for waveform in ("Learned", "OTFS", "AFDM"):
                selected = [row for row in rows if row["modulation"] == modulation and row["waveform"] == waveform and row["k"] == "24"]
                selected.sort(key=lambda row: float(row[xkey]))
                axis.semilogy([float(row[xkey]) for row in selected], [max(float(row["ber"]), .5 / float(row["total_bits"])) for row in selected],
                              marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
                axis.axhline(max(float(selected[0]["direct_ber"]), .5 / float(selected[0]["total_bits"])), color=COLORS[waveform], ls="--", alpha=.45)
            axis.set_title(modulation); axis.set_xlabel(xlabel); axis.grid(True, which="both", alpha=.25)
        axes[0].set_ylabel("BER"); axes[-1].legend(fontsize=8)
        save_figure(fig, FIG_COMPLEXITY, name)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        for waveform in ("Learned", "OTFS", "AFDM"):
            selected = [row for row in pareto if row["modulation"] == modulation and row["waveform"] == waveform]
            selected.sort(key=lambda row: float(row["mean_total_macs"]))
            axis.semilogy([float(row["mean_total_macs"]) for row in selected], [max(float(row["ber"]), .5 / float(row["total_bits"])) for row in selected],
                          marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
        axis.set_title(modulation); axis.set_xlabel("mean detector complex MACs"); axis.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("Pareto BER"); axes[-1].legend(fontsize=8)
    save_figure(fig, FIG_COMPLEXITY, "B3_pareto_ber_vs_macs")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        for waveform in ("Learned", "OTFS", "AFDM"):
            timed = [row for row in runtime if row["modulation"] == modulation and row["waveform"] == waveform]
            timed.sort(key=lambda row: int(row["iteration_budget"]))
            lookup = {(row["waveform"], row["modulation"], row["iteration_budget"]): row for row in rows if row["k"] == "24"}
            axis.semilogy([1e3 * float(row["median_detector_seconds"]) for row in timed],
                          [max(float(lookup[(waveform, modulation, row["iteration_budget"])]["ber"]), 1e-8) for row in timed],
                          marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
        axis.set_title(modulation); axis.set_xlabel("median detector runtime (ms)"); axis.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("BER"); axes[-1].legend(fontsize=8)
    save_figure(fig, FIG_COMPLEXITY, "B4_ber_vs_runtime")

    classification, evidence = _compute_classification(rows)
    write_json(DATA_COMPLEXITY / "claim.json", {"classification": classification, "rule": "SUPPORTED iff all three modulations have at least two strict Learned BER wins over every OTFS/AFDM point at common-or-lower MACs; PARTIAL iff one or two", "evidence": evidence})


def phase_scaling_config(smoke: bool = False) -> None:
    preservation_gate()
    record = scaling_rule_record()
    record.update({
        "pub1a_parent_commit": PARENT_COMMIT,
        "afdm_rule": "alpha=floor(nu+1/2); candidates 2*alpha_max+1 and 2*(alpha_max+1)+1; AFDM-only validation MSE selects",
        "afdm_candidates": {str(n): list(afdm_candidate_multipliers(n)) for n in (128, 256)},
        "afdm_c2_rule": "sqrt(2)/(100*N), identical repository scaling convention",
        "training": {
            "initialization": "OFDM",
            "stage1": "2000-step exact hard-K mean-MSE projected Adam; lr 1e-2/3e-3/1e-3",
            "tail": "1000-step beta=.95 lambda=.40 tail refinement; lr 3e-3/1e-3/3e-4",
            "batch_size": 16, "validation_channels": 256, "validation_cadence": 25,
            "snr_db": [12, 15, 18, 20], "selection": "validation only; no test BER",
        },
        "n128_hard_gate": "unitarity, reconstruction, full-MSE invariance, classical/AFDM/OTFS sanity, finite training, and interpretable constrained test before N256",
        "sublinear_k_rule": "smallest K/N with BER <= 2x full BER when full point has at least 20 errors; otherwise analytic MSE <= 2x full MSE",
    })
    write_json(DATA_SCALING / "scaling_config.json", record)
    write_json(PROVENANCE / "run.json", provenance_record())


def _unitarity_residual(waveform: torch.Tensor) -> float:
    n = waveform.shape[-1]
    return float(torch.linalg.matrix_norm(waveform.mH @ waveform - torch.eye(n, dtype=COMPLEX_DTYPE)))


def _stage1_lr(step: int) -> float:
    if step <= 1000:
        return 1e-2
    if step <= 1600:
        return 3e-3
    return 1e-3


def _tail_lr(step: int) -> float:
    if step <= 500:
        return 3e-3
    if step <= 800:
        return 1e-3
    return 3e-4


def _train_one_stage(
    *,
    n: int,
    initial: torch.Tensor,
    validation: torch.Tensor,
    stage: str,
    steps: int,
    batch_size: int,
    cadence: int,
    state_path: Path,
    reference_mean: float | None = None,
) -> dict[str, object]:
    k = scaling_definition(n).primary_k
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([parameter], lr=_stage1_lr(1) if stage == "mean" else _tail_lr(1))
    validation_chunk = 8 if n == 256 else 32
    initial_mean, initial_cvar = validation_metrics_scaled(parameter.detach(), validation, k, chunk=validation_chunk)
    state: dict[str, object] | None = None
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        if state["protocol"] != {"n": n, "stage": stage, "steps": steps, "batch_size": batch_size, "cadence": cadence, "k": k}:
            raise RuntimeError(f"incompatible resumable state: {state_path}")
        parameter.data.copy_(state["parameter"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best = state["best"].clone()
        best_mean = float(state["best_mean"])
        best_cvar = float(state["best_cvar"])
        best_step = int(state["best_step"])
        trace = list(state["trace"])
        max_unitarity = float(state["max_unitarity"])
        runtime_seconds = float(state["runtime_seconds"])
    else:
        start_step = 1
        best = parameter.detach().clone()
        best_mean, best_cvar, best_step = initial_mean, initial_cvar, 0
        trace = [{"step": 0, "validation_mean_mse": initial_mean, "validation_cvar95": initial_cvar}]
        max_unitarity = _unitarity_residual(parameter.detach())
        runtime_seconds = 0.0
    protocol = {"n": n, "stage": stage, "steps": steps, "batch_size": batch_size, "cadence": cadence, "k": k}
    started = time.perf_counter()
    nonfinite = False
    for step in range(start_step, steps + 1):
        optimizer.param_groups[0]["lr"] = _stage1_lr(step) if stage == "mean" else _tail_lr(step)
        channels = sample_scaled_dd(batch_size, SEEDS["training_channels"] + 100 * n + step, n).matrices
        snr_generator = make_generator(SEEDS["training_snr"] + 100 * n + step)
        snrs_db = 12.0 + 8.0 * torch.rand(batch_size, dtype=torch.float64, generator=snr_generator)
        streams = stream_mse_scaled(parameter, channels, noise_variance(snrs_db), k)
        means = streams.mean(dim=-1)
        if stage == "mean":
            objective = means.mean()
        else:
            objective = (means + TAIL_LAMBDA * (stream_cvar(streams, TAIL_BETA) - means)).mean()
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(objective):
            nonfinite = True
            raise RuntimeError(f"non-finite N={n} {stage} objective at step {step}")
        objective.backward()
        if not torch.isfinite(parameter.grad.real).all() or not torch.isfinite(parameter.grad.imag).all():
            nonfinite = True
            raise RuntimeError(f"non-finite N={n} {stage} gradient at step {step}")
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(polar_project(parameter))
            max_unitarity = max(max_unitarity, _unitarity_residual(parameter))
        if step % cadence == 0 or step == steps:
            mean, cvar = validation_metrics_scaled(parameter.detach(), validation, k, chunk=validation_chunk)
            feasible = stage == "mean" or mean <= float(reference_mean) * 1.02
            improved = mean < best_mean if stage == "mean" else feasible and (cvar, mean) < (best_cvar, best_mean)
            if improved:
                best = parameter.detach().clone()
                best_mean, best_cvar, best_step = mean, cvar, step
            trace.append({"step": step, "training_objective": float(objective),
                          "validation_mean_mse": mean, "validation_cvar95": cvar,
                          "feasible": feasible, "selected_so_far": best_step})
            elapsed = runtime_seconds + time.perf_counter() - started
            torch.save({
                "protocol": protocol, "step": step, "parameter": parameter.detach(),
                "optimizer": optimizer.state_dict(), "best": best, "best_mean": best_mean,
                "best_cvar": best_cvar, "best_step": best_step, "trace": trace,
                "max_unitarity": max_unitarity, "runtime_seconds": elapsed,
            }, state_path)
            print(f"N={n} {stage} step={step}/{steps} val_mean={mean:.8f} val_cvar={cvar:.8f} best={best_step}", flush=True)
            started = time.perf_counter()
            runtime_seconds = elapsed
    return {
        "waveform": best, "initial_mean": initial_mean, "initial_cvar": initial_cvar,
        "best_mean": best_mean, "best_cvar": best_cvar, "best_step": best_step,
        "trace": trace, "max_unitarity": max_unitarity, "runtime_seconds": runtime_seconds,
        "nonfinite": nonfinite, "protocol": protocol,
    }


def _afdm_audit(n: int, smoke: bool) -> dict[str, object]:
    definition = scaling_definition(n)
    candidates = afdm_candidate_multipliers(n)
    alpha = int(math.floor(definition.doppler_max + .5))
    delays = (0, 1, int(definition.delay_max // 2), int(definition.delay_max))
    dopplers = (-alpha, alpha, 0, -alpha)
    records = []
    for multiplier in candidates:
        waveform = afdm_waveform(n=n, c1=multiplier / (2 * n), c2=math.sqrt(2.0) / (100 * n))
        paths = []
        observed = []
        for delay, doppler in zip(delays, dopplers):
            from src.channel import assemble_dd_channel
            path = assemble_dd_channel(
                torch.tensor([delay]), torch.tensor([doppler]),
                torch.tensor([1 + 0j], dtype=COMPLEX_DTYPE), n=n,
            )
            summary = operator_offset_metrics(effective_operator(path, waveform))
            predicted = (multiplier * delay + doppler) % n
            observed.append(int(summary["dominant_offset"]))
            paths.append({"delay": delay, "doppler": doppler, "predicted_offset": predicted,
                          "observed_offset": int(summary["dominant_offset"])})
        records.append({
            "multiplier_2Nc1": multiplier, "c1": multiplier / (2 * n),
            "c2": math.sqrt(2.0) / (100 * n), "paths": paths,
            "predicted_equals_observed": all(row["predicted_offset"] == row["observed_offset"] for row in paths),
            "distinct": len(set(observed)) == len(observed),
            "unitarity_residual": _unitarity_residual(waveform),
        })
    if not all(row["predicted_equals_observed"] and row["distinct"] for row in records):
        raise RuntimeError(f"N={n} AFDM path-separation hard gate failed")
    waveforms = {str(multiplier): afdm_waveform(n=n, c1=multiplier / (2 * n), c2=math.sqrt(2.0) / (100 * n)) for multiplier in candidates}
    masks, _ = fixed_scaled_masks(waveforms, n, (definition.primary_k,), 16 if smoke else 256, SEEDS["scaling_support"] + n)
    validation = sample_scaled_dd(4 if smoke else 64, SEEDS["scaling_validation"] + n, n).matrices
    for record in records:
        name = str(record["multiplier_2Nc1"])
        waveform = waveforms[name]
        actual = effective_operator(validation, waveform)
        retained = apply_support(actual, masks[name][definition.primary_k])
        values = []
        for snr in (12, 15, 18, 20):
            equalizer = mismatched_lmmse_equalizer(retained, noise_variance(snr))
            values.append(float(analytic_communication_mse(actual, equalizer, noise_variance(snr)).mean()))
        record["validation_mean_analytic_mse"] = float(np.mean(values))
    selected = min(records, key=lambda row: (row["validation_mean_analytic_mse"], row["multiplier_2Nc1"]))
    return {
        "n": n, "delay_max": definition.delay_max, "doppler_max": definition.doppler_max,
        "alpha_max": alpha, "candidates": records,
        "selected_multiplier_2Nc1": selected["multiplier_2Nc1"],
        "selection_rule": "minimum fixed-statistical-K validation analytic MSE; AFDM only; no Learned/test/BER input",
        "hard_gate": True,
    }


def _train_scaled(n: int, smoke: bool) -> dict[str, object]:
    if not (DATA_SCALING / "scaling_config.json").exists():
        raise RuntimeError("run --phase scaling-config before training")
    directory = CHECKPOINTS / (f"N{n}_smoke" if smoke else f"N{n}")
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"learned_n{n}.pt"
    metadata_path = directory / "metadata.json"
    if final_path.exists() and metadata_path.exists() and not smoke:
        return read_json(metadata_path)
    validation = sample_scaled_dd(4 if smoke else 256, SEEDS["scaling_validation"] + n, n).matrices
    mean_steps, tail_steps = (2, 2) if smoke else (2000, 1000)
    batch_size = 2 if smoke else 16
    cadence = 1 if smoke else 25
    mean = _train_one_stage(
        n=n, initial=ofdm_waveform(n), validation=validation, stage="mean", steps=mean_steps,
        batch_size=batch_size, cadence=cadence, state_path=directory / "mean_state.pt",
    )
    tail = _train_one_stage(
        n=n, initial=mean["waveform"], validation=validation, stage="tail", steps=tail_steps,
        batch_size=batch_size, cadence=cadence, state_path=directory / "tail_state.pt",
        reference_mean=float(mean["best_mean"]),
    )
    waveform = tail["waveform"]
    torch.save(waveform, final_path)
    write_csv(DATA_SCALING / f"training_trace_n{n}.csv", [
        {"stage": "mean", **row} for row in mean["trace"]
    ] + [{"stage": "tail", **row} for row in tail["trace"]])
    metadata = {
        "pub1a_parent_commit": PARENT_COMMIT, "label": f"Learned-N{n}", "n": n,
        "k": scaling_definition(n).primary_k, "checkpoint": str(final_path.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(final_path), "dtype": str(waveform.dtype),
        "unitarity_residual": _unitarity_residual(waveform),
        "clean_reconstruction_error": clean_reconstruction_error(waveform),
        "mean_stage": {key: value for key, value in mean.items() if key not in ("waveform", "trace")},
        "tail_stage": {key: value for key, value in tail.items() if key not in ("waveform", "trace")},
        "trainable_complex_entries": n * n, "checkpoint_bytes": final_path.stat().st_size,
        "training_wall_seconds": float(mean["runtime_seconds"]) + float(tail["runtime_seconds"]),
        "smoke": smoke,
    }
    write_json(metadata_path, metadata)
    return metadata


def _aggregate_counts(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, dict[int, torch.Tensor]],
    constellation: Constellation,
    snr_db: int,
    ks: tuple[int, ...],
    target_bits: int,
    seed_offset: int,
    experiment: str,
) -> list[dict[str, object]]:
    n = channels.shape[-1]
    blocks = math.ceil(target_bits / (len(channels) * n * constellation.bits_per_symbol))
    bits = random_bits(
        (len(channels), blocks, n, constellation.bits_per_symbol),
        make_generator(SEEDS["scaling_symbols"] + seed_offset + constellation.bits_per_symbol),
    )
    symbols, indices = constellation_encode(bits, constellation)
    noise = complex_awgn(
        (len(channels), blocks, n), noise_variance(snr_db),
        make_generator(SEEDS["scaling_noise"] + seed_offset + constellation.bits_per_symbol),
    )
    rows = []
    with torch.no_grad():
        for name, waveform in waveforms.items():
            transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
            physical = torch.einsum("cij,cbj->cbi", channels, transmitted) + noise
            received = torch.einsum("ij,cbj->cbi", waveform.mH, physical)
            actual = effective_operator(channels, waveform)
            for k in ks:
                mask = masks[name][k]
                retained = apply_support(actual, mask)
                equalizer = mismatched_lmmse_equalizer(retained, noise_variance(snr_db))
                estimate = torch.einsum("cij,cbj->cbi", equalizer, received)
                decoded, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                bit_errors = int(torch.sum(decoded != bits))
                symbol_errors = int(torch.sum(decoded_indices != indices))
                mse = float(analytic_communication_mse(actual, equalizer, noise_variance(snr_db)).mean())
                rows.append({
                    "n": n, "modulation": constellation.name, "snr_db": snr_db,
                    "experiment": experiment,
                    "waveform": canonical_name(name), "waveform_label": name, "k": k, "k_fraction": k / n,
                    "bit_errors": bit_errors, "total_bits": bits.numel(), "ber": bit_errors / bits.numel(),
                    "symbol_errors": symbol_errors, "total_symbols": indices.numel(), "ser": symbol_errors / indices.numel(),
                    "analytic_mse": mse, "channels": len(channels), "blocks_per_channel": blocks,
                })
    return rows


def _scaling_gate(n: int, metadata: dict[str, object], audit: dict[str, object], waveforms: dict[str, torch.Tensor], channels: torch.Tensor, rows: list[dict[str, object]]) -> dict[str, object]:
    primary = scaling_definition(n).primary_k
    representative = [row for row in rows if row["experiment"] == "snr_sweep" and row["snr_db"] == 18 and row["k"] == primary and row["modulation"] in ("QPSK", "16QAM")]
    wins = {}
    for modulation in ("QPSK", "16QAM"):
        selected = [row for row in representative if row["modulation"] == modulation]
        learned = next(row for row in selected if row["waveform"] == "Learned")
        classical = [row for row in selected if row["waveform"] in ("OTFS", "AFDM")]
        wins[modulation] = {
            "learned_ber": learned["ber"],
            "best_classical_ber": min(row["ber"] for row in classical),
            "learned_wins": learned["ber"] < min(row["ber"] for row in classical),
        }
    checks = {
        "learned_unitarity": float(metadata["unitarity_residual"]) < 1e-11,
        "clean_reconstruction": float(metadata["clean_reconstruction_error"]) < 1e-11,
        "training_finite": not bool(metadata["mean_stage"]["nonfinite"]) and not bool(metadata["tail_stage"]["nonfinite"]),
        "afdm_hard_gate": bool(audit["hard_gate"]),
        "classical_unitarity": all(_unitarity_residual(waveforms[name]) < 1e-11 for name in ("OTFS", "AFDM", "OFDM")),
        "otfs_dimensions": waveforms["OTFS"].shape == (n, n),
        "full_mse_invariance": full_mse_invariance(channels[:2], waveforms) < 1e-11,
        "finite_representation_phenomenon": any(value["learned_wins"] for value in wins.values()),
    }
    return {"n": n, "checks": checks, "primary_results": wins, "passed": all(checks.values()),
            "stop_rule": "N256 prohibited unless every N128 check passes; constrained phenomenon requires a strict Learned BER win over both primary classical baselines in at least one primary modulation"}


def _evaluate_scaled(n: int, smoke: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata = _train_scaled(n, smoke)
    checkpoint_path = ROOT / str(metadata["checkpoint"])
    waveform = torch.load(checkpoint_path, weights_only=True)
    if file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise RuntimeError(f"N={n} checkpoint hash mismatch")
    audit_path = DATA_SCALING / f"afdm_audit_n{n}.json"
    audit = _afdm_audit(n, smoke)
    write_json(audit_path, audit)
    waveforms = scaled_waveforms(n, waveform, int(audit["selected_multiplier_2Nc1"]))
    definition = scaling_definition(n)
    support_samples = 16 if smoke else 256
    masks, _ = fixed_scaled_masks(waveforms, n, definition.k_values, support_samples, SEEDS["scaling_support"] + 10 * n)
    channels = sample_scaled_dd(4 if smoke else 64, SEEDS["scaling_test"] + n, n).matrices
    target_bits = 8192 if smoke else 262144
    rows: list[dict[str, object]] = []
    for modulation in (qpsk_constellation(), qam16_constellation()):
        for snr_index, snr in enumerate((18,) if smoke else SCALING_SNRS):
            rows.extend(_aggregate_counts(
                channels, waveforms, masks, modulation, snr, (definition.primary_k,), target_bits,
                seed_offset=1000 * n + 10 * snr_index,
                experiment="snr_sweep",
            ))
        rows.extend(_aggregate_counts(
            channels, waveforms, masks, modulation, 18, definition.k_values, target_bits,
            seed_offset=1000 * n + 500,
            experiment="k_sweep",
        ))
    write_csv(DATA_SCALING / f"communication_n{n}.csv", rows)
    gate = _scaling_gate(n, metadata, audit, waveforms, channels, rows)
    write_json(DATA_SCALING / f"hard_gate_n{n}.json", gate)
    return rows, gate


def phase_scaling_n128(smoke: bool = False) -> None:
    preservation_gate()
    _, gate = _evaluate_scaled(128, smoke)
    if not gate["passed"]:
        print("N=128 hard gate did not pass; STOP before N=256", flush=True)


def phase_scaling_n256(smoke: bool = False) -> None:
    gate_path = DATA_SCALING / "hard_gate_n128.json"
    if not gate_path.exists() or not bool(read_json(gate_path)["passed"]):
        raise RuntimeError("N=128 hard gate did not pass; N=256 is prohibited")
    preservation_gate()
    _evaluate_scaled(256, smoke)


def _n64_scaling_rows() -> list[dict[str, object]]:
    output = []
    communication = read_csv(PUB1A / "data" / "communication_summary.csv")
    for row in communication:
        if row["regime"] != "dd_v1" or row["modulation"] not in ("QPSK", "16QAM"):
            continue
        if row["waveform"] not in ("Learned", "OTFS", "AFDM") or row["receiver_mode"] != "fixedStatK24":
            continue
        if int(row["snr_db"]) not in SCALING_SNRS:
            continue
        output.append({
            "n": 64, "experiment": "snr_sweep", "modulation": row["modulation"],
            "snr_db": int(row["snr_db"]), "waveform": row["waveform"], "waveform_label": row["waveform"],
            "k": 24, "k_fraction": .375, "bit_errors": int(row["bit_errors"]),
            "total_bits": int(row["total_bits"]), "ber": float(row["ber"]),
            "symbol_errors": int(row["symbol_errors"]), "total_symbols": int(row["total_symbols"]),
            "ser": float(row["ser"]), "analytic_mse": math.nan,
        })
    for row in read_csv(PUB1A_KSWEEP):
        if row["modulation"] not in ("QPSK", "16QAM") or row["waveform"] not in ("Learned", "OTFS", "AFDM"):
            continue
        k = int(row["k"])
        if k not in scaling_definition(64).k_values:
            continue
        output.append({
            "n": 64, "experiment": "k_sweep", "modulation": row["modulation"],
            "snr_db": int(row["snr_db"]), "waveform": row["waveform"], "waveform_label": row["waveform"],
            "k": k, "k_fraction": k / 64, "bit_errors": int(row["bit_errors"]),
            "total_bits": int(row["total_bits"]), "ber": float(row["ber"]),
            "symbol_errors": int(row["symbol_errors"]), "total_symbols": int(row["total_symbols"]),
            "ser": float(row["ser"]), "analytic_mse": math.nan,
        })
    analytic = read_csv(PUB1A / "data" / "communication_analytic_mse.csv")
    lookup = {}
    for row in analytic:
        if row["regime"] == "dd_v1" and int(row["snr_db"]) == 18:
            lookup[(row["waveform"], int(row["k"]))] = float(row["analytic_mse"])
    for row in output:
        if row["experiment"] == "k_sweep" and (row["waveform"], row["k"]) in lookup:
            row["analytic_mse"] = lookup[(row["waveform"], row["k"])]
    return output


def _available_scaling_rows() -> list[dict[str, object]]:
    rows = _n64_scaling_rows()
    for n in (128, 256):
        path = DATA_SCALING / f"communication_n{n}.csv"
        if path.exists():
            for row in read_csv(path):
                rows.append({
                    **row, "n": int(row["n"]), "snr_db": int(row["snr_db"]), "k": int(row["k"]),
                    "k_fraction": float(row["k_fraction"]), "bit_errors": int(row["bit_errors"]),
                    "total_bits": int(row["total_bits"]), "ber": float(row["ber"]),
                    "symbol_errors": int(row["symbol_errors"]), "total_symbols": int(row["total_symbols"]),
                    "ser": float(row["ser"]), "analytic_mse": float(row["analytic_mse"]),
                })
    return rows


def _scaling_classification(rows: list[dict[str, object]]) -> dict[str, object]:
    evidence = {}
    positive_dimensions = 0
    available = sorted({int(row["n"]) for row in rows if int(row["n"]) > 64})
    for n in available:
        primary = scaling_definition(n).primary_k
        wins = 0
        details = {}
        for modulation in ("QPSK", "16QAM"):
            selected = [row for row in rows if int(row["n"]) == n and row["experiment"] == "snr_sweep" and row["modulation"] == modulation and int(row["snr_db"]) == 18 and int(row["k"]) == primary]
            learned = next(row for row in selected if row["waveform"] == "Learned")
            classical = [row for row in selected if row["waveform"] in ("OTFS", "AFDM")]
            win = float(learned["ber"]) < min(float(row["ber"]) for row in classical)
            wins += int(win)
            details[modulation] = {"learned_ber": learned["ber"], "best_classical_ber": min(float(row["ber"]) for row in classical), "win": win}
        evidence[str(n)] = {"primary_modulation_wins": wins, "details": details}
        positive_dimensions += int(wins > 0)
    if available == [128, 256] and positive_dimensions == 2:
        classification = "SUPPORTED"
    elif positive_dimensions:
        classification = "PARTIALLY SUPPORTED"
    else:
        classification = "NOT SUPPORTED"
    return {"classification": classification, "evidence": evidence,
            "rule": "SUPPORTED iff both N128 and N256 retain a strict primary-K BER win over both OTFS and AFDM in at least one primary modulation; PARTIAL iff only one tested larger dimension does"}


def phase_scaling_plots(smoke: bool = False) -> None:
    rows = _available_scaling_rows()
    dimensions = sorted({int(row["n"]) for row in rows})
    for modulation in ("QPSK", "16QAM"):
        fig, axes = plt.subplots(1, len(dimensions), figsize=(4.4 * len(dimensions), 3.8), squeeze=False)
        for axis, n in zip(axes[0], dimensions):
            for waveform in ("Learned", "OTFS", "AFDM"):
                selected = [row for row in rows if int(row["n"]) == n and row["experiment"] == "snr_sweep" and row["modulation"] == modulation and row["waveform"] == waveform]
                selected.sort(key=lambda row: int(row["snr_db"]))
                if selected:
                    axis.semilogy([row["snr_db"] for row in selected], [max(float(row["ber"]), .5 / int(row["total_bits"])) for row in selected],
                                  marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
            axis.set_title(f"N={n}, K/N=0.375"); axis.set_xlabel("Es/N0 (dB)"); axis.grid(True, which="both", alpha=.25)
        axes[0, 0].set_ylabel(f"{modulation} BER"); axes[0, -1].legend(fontsize=8)
        save_figure(fig, FIG_SCALING, f"S1_{modulation.lower()}_ber_vs_snr")

    fig, axes = plt.subplots(2, len(dimensions), figsize=(4.4 * len(dimensions), 7.2), squeeze=False)
    for row_index, modulation in enumerate(("QPSK", "16QAM")):
        for axis, n in zip(axes[row_index], dimensions):
            for waveform in ("Learned", "OTFS", "AFDM"):
                selected = [row for row in rows if int(row["n"]) == n and row["experiment"] == "k_sweep" and row["modulation"] == modulation and row["waveform"] == waveform]
                selected.sort(key=lambda row: float(row["k_fraction"]))
                axis.semilogy([row["k_fraction"] for row in selected], [max(float(row["ber"]), .5 / int(row["total_bits"])) for row in selected],
                              marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
            axis.set_title(f"{modulation}, N={n}"); axis.set_xlabel("K/N"); axis.grid(True, which="both", alpha=.25)
    axes[0, 0].set_ylabel("BER"); axes[1, 0].set_ylabel("BER"); axes[0, -1].legend(fontsize=8)
    save_figure(fig, FIG_SCALING, "S2_ber_vs_k_fraction")

    ratio_rows = []
    required_rows = []
    for n in dimensions:
        for modulation in ("QPSK", "16QAM"):
            for waveform in ("Learned", "OTFS", "AFDM"):
                selected = [row for row in rows if int(row["n"]) == n and row["experiment"] == "k_sweep" and row["modulation"] == modulation and row["waveform"] == waveform]
                full = next((row for row in selected if int(row["k"]) == n), None)
                if full is None:
                    continue
                stable_ber = int(full["bit_errors"]) >= 20 and float(full["ber"]) > 0
                for row in selected:
                    if stable_ber:
                        ratio_rows.append({"n": n, "modulation": modulation, "waveform": waveform, "k_fraction": row["k_fraction"],
                                           "ratio": float(row["ber"]) / float(full["ber"]), "basis": "BER"})
                    elif math.isfinite(float(row["analytic_mse"])) and float(full["analytic_mse"]) > 0:
                        ratio_rows.append({"n": n, "modulation": modulation, "waveform": waveform, "k_fraction": row["k_fraction"],
                                           "ratio": float(row["analytic_mse"]) / float(full["analytic_mse"]), "basis": "analytic_MSE_fallback"})
                eligible = [row for row in ratio_rows if row["n"] == n and row["modulation"] == modulation and row["waveform"] == waveform and row["ratio"] <= 2]
                if eligible:
                    required_rows.append({"n": n, "modulation": modulation, "waveform": waveform,
                                          "required_k_fraction": min(float(row["k_fraction"]) for row in eligible),
                                          "basis": eligible[0]["basis"]})
    if ratio_rows:
        write_csv(DATA_SCALING / "constrained_to_full_ratio.csv", ratio_rows)
    if required_rows:
        write_csv(DATA_SCALING / "required_k_fraction.csv", required_rows)
    fig, axes = plt.subplots(2, len(dimensions), figsize=(4.4 * len(dimensions), 7.2), squeeze=False)
    for row_index, modulation in enumerate(("QPSK", "16QAM")):
        for axis, n in zip(axes[row_index], dimensions):
            for waveform in ("Learned", "OTFS", "AFDM"):
                selected = [row for row in ratio_rows if row["n"] == n and row["modulation"] == modulation and row["waveform"] == waveform]
                selected.sort(key=lambda row: float(row["k_fraction"]))
                if selected:
                    axis.semilogy([row["k_fraction"] for row in selected], [row["ratio"] for row in selected],
                                  marker=MARKERS[waveform], color=COLORS[waveform], label=waveform)
            axis.axhline(2, color="black", ls="--", alpha=.4); axis.set_title(f"{modulation}, N={n}"); axis.set_xlabel("K/N"); axis.grid(True, which="both", alpha=.25)
    axes[0, 0].set_ylabel("constrained/full ratio"); axes[1, 0].set_ylabel("constrained/full ratio"); axes[0, -1].legend(fontsize=8)
    save_figure(fig, FIG_SCALING, "S3_constrained_to_full")

    cost_rows = [{
        "n": 64, "trainable_complex_entries": 64 ** 2, "checkpoint_bytes": FROZEN64.stat().st_size,
        "training_wall_seconds": 571.4171181668062, "training_time_note": "Stage-1 mean stage only; restored final tail checkpoint lacks full tail runtime metadata",
    }]
    for n in (128, 256):
        path = CHECKPOINTS / f"N{n}" / "metadata.json"
        if path.exists():
            metadata = read_json(path)
            cost_rows.append({"n": n, "trainable_complex_entries": metadata["trainable_complex_entries"],
                              "checkpoint_bytes": metadata["checkpoint_bytes"], "training_wall_seconds": metadata["training_wall_seconds"],
                              "training_time_note": "mean plus tail stages on recorded CPU backend"})
    write_csv(DATA_SCALING / "dense_learning_cost.csv", cost_rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].plot([row["n"] for row in cost_rows], [row["trainable_complex_entries"] for row in cost_rows], marker="o")
    axes[0].set_yscale("log"); axes[0].set_xlabel("N"); axes[0].set_ylabel("trainable complex entries")
    axes[1].plot([row["n"] for row in cost_rows], [row["checkpoint_bytes"] / 2**20 for row in cost_rows], marker="o")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("checkpoint size (MiB)")
    for axis in axes: axis.grid(True, which="both", alpha=.25)
    save_figure(fig, FIG_SCALING, "S4_dense_learning_scaling")
    write_json(DATA_SCALING / "claim.json", _scaling_classification(rows))


def _result_line(rows: list[dict[str, str]], modulation: str, waveform: str, k: int = 24, iterations: int = 16) -> str:
    selected = [row for row in rows if row["modulation"] == modulation and row["waveform"] == waveform and int(row["k"]) == k and int(row["iteration_budget"]) == iterations]
    row = selected[0] if selected else None
    return "not available" if row is None else f"BER {float(row['ber']):.6g} at {float(row['mean_total_macs']):.0f} mean total MACs"


def phase_report(smoke: bool = False) -> None:
    preservation = preservation_gate()
    compute_rows = read_csv(DATA_COMPLEXITY / "operating_points.csv") if (DATA_COMPLEXITY / "operating_points.csv").exists() else []
    compute_claim = read_json(DATA_COMPLEXITY / "claim.json") if (DATA_COMPLEXITY / "claim.json").exists() else {"classification": "NOT RUN", "evidence": {}}
    scaling_claim = read_json(DATA_SCALING / "claim.json") if (DATA_SCALING / "claim.json").exists() else {"classification": "NOT RUN", "evidence": {}}
    n128_gate = read_json(DATA_SCALING / "hard_gate_n128.json") if (DATA_SCALING / "hard_gate_n128.json").exists() else None
    n256_gate = read_json(DATA_SCALING / "hard_gate_n256.json") if (DATA_SCALING / "hard_gate_n256.json").exists() else None
    n128_metadata = read_json(CHECKPOINTS / "N128" / "metadata.json") if (CHECKPOINTS / "N128" / "metadata.json").exists() else None
    n256_metadata = read_json(CHECKPOINTS / "N256" / "metadata.json") if (CHECKPOINTS / "N256" / "metadata.json").exists() else None
    n128_afdm = read_json(DATA_SCALING / "afdm_audit_n128.json") if (DATA_SCALING / "afdm_audit_n128.json").exists() else None
    n256_afdm = read_json(DATA_SCALING / "afdm_audit_n256.json") if (DATA_SCALING / "afdm_audit_n256.json").exists() else None
    config_record = read_json(DATA_SCALING / "scaling_config.json") if (DATA_SCALING / "scaling_config.json").exists() else scaling_rule_record()
    validation = read_csv(DATA_COMPLEXITY / "numerical_validation.csv") if (DATA_COMPLEXITY / "numerical_validation.csv").exists() else []
    max_solution = max((float(row["relative_solution_error"]) for row in validation), default=math.nan)
    max_residual = max((float(row["relative_residual"]) for row in validation), default=math.nan)
    n128_text = "not run" if n128_gate is None else ("PASSED" if n128_gate["passed"] else "FAILED; N=256 stopped")
    n256_text = "not run" if n256_gate is None else ("PASSED" if n256_gate["passed"] else "FAILED")
    text = f"""# Stage Pub-1B — Receiver Compute and N-Scaling

## A. Provenance

- Publication parent `{PARENT_COMMIT}`; target `origin/pub`; no `origin/main` or Stage-3/4 mutation.
- N=64 Learned SHA-256 `{N64_CHECKPOINT_SHA256}`.

## B. Pub-1A preservation proof

- `{preservation['pub1a_file_count']}` Pub-1A files retain tree SHA-256 `{preservation['pub1a_tree_sha256']}`.
- Preservation status: `{preservation['preserved']}`.

## C. Sparse iterative LMMSE definition

CG solves `(A_K^H A_K + sigma^2 I)s=A_K^H z` with sparse `A_K`/`A_K^H` matvecs. The measured path never forms `A_K^H A_K`. Reception uses the true full `A` to generate `z`; no learned receiver or changed estimator is used.

## D. MAC accounting convention

Each iteration charges `2NK` sparse complex MACs plus `6N` complex-equivalent vector/dot MACs; the initial residual dot costs `N`. RHS setup `A_K^H z` costs `NK` separately. Fixed channel-class support construction is offline and excluded from per-packet detector cost. No preconditioner is used.

## E. Numerical equivalence to direct LMMSE

Maximum tight-gate relative solution error `{max_solution:.3e}`; maximum relative residual `{max_residual:.3e}`. The direct dense solution remains authoritative.

## F. BER-vs-MAC results

- At the representative K=24, 16-iteration point, every waveform uses `56,896` mean total MACs.
- QPSK: Learned {_result_line(compute_rows, 'QPSK', 'Learned')}; best classical {_result_line(compute_rows, 'QPSK', 'AFDM')}.
- 16-QAM: Learned {_result_line(compute_rows, '16QAM', 'Learned')}; best classical {_result_line(compute_rows, '16QAM', 'AFDM')}.
- 64-QAM: Learned {_result_line(compute_rows, '64QAM', 'Learned')}; best classical {_result_line(compute_rows, '64QAM', 'AFDM')}.
- Raw `(K,iteration)` points, exact references, residuals, actual iterations, and disaggregated MACs are in `data/complexity/operating_points.csv`.

## G. Runtime results

Runtime includes only prepared sparse detector setup/solve. Warmups, repeats, median, p25/p75, hardware, backend, and thread policy are saved. Runtime is implementation evidence, not hardware-independent complexity.

## H. Compute-claim classification

**{compute_claim['classification']}** — “The Learned finite representation translates into an actual receiver arithmetic advantage under a common sparse iterative LMMSE detector.” Classification follows the predeclared non-trivial common-or-lower-MAC Pareto rule.

## I. N-scaling rule and physical interpretation

`Pi(tau)=F^H diag(exp(-j2pi k tau/N))F` and `Delta(nu)=diag(exp(-j2pi nu n/N))`; linear bounds preserve circular delay fraction `tau/N` and normalized Doppler `nu/N`. Frozen bounds are `(4,.75)`, `(8,1.5)`, `(16,3)` for N=64/128/256 with P=3.

## J. N128 training/configuration/result

N=128 gate: **{n128_text}**. Learned-N128 uses the N=64 methodology: OFDM-start 2,000-step hard-K mean-MSE projected Adam followed by fixed beta `{TAIL_BETA}`, lambda `{TAIL_LAMBDA}` 1,000-step tail refinement and validation-only selection. Mean/tail selected steps are `{n128_metadata['mean_stage']['best_step'] if n128_metadata else 'NA'}`/`{n128_metadata['tail_stage']['best_step'] if n128_metadata else 'NA'}`; final SHA-256 `{n128_metadata['checkpoint_sha256'] if n128_metadata else 'NA'}`. At K=48/18 dB, Learned QPSK/16-QAM BER is `{n128_gate['primary_results']['QPSK']['learned_ber'] if n128_gate else 'NA'}`/`{n128_gate['primary_results']['16QAM']['learned_ber'] if n128_gate else 'NA'}` versus best classical `{n128_gate['primary_results']['QPSK']['best_classical_ber'] if n128_gate else 'NA'}`/`{n128_gate['primary_results']['16QAM']['best_classical_ber'] if n128_gate else 'NA'}`.

## K. N256 result or explicit stop

N=256 gate: **{n256_text}**. It was launched only after the saved N=128 gate passed. Mean/tail selected steps are `{n256_metadata['mean_stage']['best_step'] if n256_metadata else 'NA'}`/`{n256_metadata['tail_stage']['best_step'] if n256_metadata else 'NA'}`; final SHA-256 `{n256_metadata['checkpoint_sha256'] if n256_metadata else 'NA'}`. At K=96/18 dB, Learned QPSK/16-QAM BER is `{n256_gate['primary_results']['QPSK']['learned_ber'] if n256_gate else 'NA'}`/`{n256_gate['primary_results']['16QAM']['learned_ber'] if n256_gate else 'NA'}` versus best classical `{n256_gate['primary_results']['QPSK']['best_classical_ber'] if n256_gate else 'NA'}`/`{n256_gate['primary_results']['16QAM']['best_classical_ber'] if n256_gate else 'NA'}`.

## L. AFDM/OTFS scaling definitions

AFDM candidate slopes derive from scaled Doppler `alpha_max`, pass predicted/observed integer offsets, and use AFDM-only validation MSE. N=128 selected `2Nc1={n128_afdm['selected_multiplier_2Nc1'] if n128_afdm else 'NA'}`; N=256 selected `2Nc1={n256_afdm['selected_multiplier_2Nc1'] if n256_afdm else 'NA'}`. `c2=sqrt(2)/(100N)`. OTFS uses frozen delay-fastest grids 16x8 at N=128 and 16x16 at N=256.

## M. BER-vs-SNR and BER-vs-K/N scaling results

Saved direct-LMMSE tables and S1–S3 figures separate N and modulation. The primary matched points are K/N=.375; all dimensions include normalized K fractions `{list(K_FRACTIONS)}` and full K=N.

## N. Dense-learning memory/training-cost scaling

S4 reports N-squared trainable entries, checkpoint bytes, and measured training time where available. This dense implementation is a limitation, not a compact scalable architecture.

## O. Scaling-claim classification

**{scaling_claim['classification']}** — “The Learned constrained-representation advantage observed at N=64 persists when the same learning methodology is scaled to larger N.” Compute and scaling classifications are independent.

## P. Limitations

- Fixed support is offline channel-class information; channel estimation and support search are outside detector accounting.
- Runtime is backend/hardware dependent; MACs are the primary compute evidence.
- One scaling seed per N; no N=1024, multi-seed, multi-K, PAPR/OOB, learned receiver, or compact-family claim.
- BER ratios fall back to analytic MSE when full-BER error counts are statistically unstable.

## Q. Conference-publication implications

Pub-1B bounds whether Pub-1A's representation result has an arithmetic interpretation and dimensional persistence. It does not broaden the channel or spectral-efficiency claims and does not use parallel Stage-3/4 results.

**STOP after Pub-1B.**
"""
    (OUT / "report.md").write_text(text)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "overview", "metadata": {}, "source": ["# Pub-1B saved-result summary\n", "Loads saved CSV/JSON only; no simulation or training.\n"]},
            {"cell_type": "code", "id": "load", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from pathlib import Path\n", "import pandas as pd, json\n", "ROOT=Path('..')\n",
                "compute=pd.read_csv(ROOT/'outputs/stage_pub1b/data/complexity/operating_points.csv')\n",
                "scaling={n:pd.read_csv(ROOT/f'outputs/stage_pub1b/data/scaling/communication_n{n}.csv') for n in (128,256) if (ROOT/f'outputs/stage_pub1b/data/scaling/communication_n{n}.csv').exists()}\n",
                "compute.head()\n",
            ]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    write_json(ROOT / "notebooks" / "07_stage_pub1b_receiver_compute_scaling.ipynb", notebook)
    manifest_path = PROVENANCE / "dataset_manifest.json"
    artifacts = []
    for path in sorted(value for value in OUT.rglob("*") if value.is_file() and value != manifest_path):
        artifacts.append({
            "path": str(path.relative_to(OUT)), "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    write_json(manifest_path, {
        "schema": "ddwaveform.stage-pub1b.manifest.v1",
        "pub1a_parent_commit": PARENT_COMMIT,
        "n64_checkpoint_sha256": N64_CHECKPOINT_SHA256,
        "pub1a_tree_sha256": PUB1A_TREE_SHA256,
        "artifacts": artifacts,
    })


def run(phase: str, smoke: bool) -> None:
    ensure_dirs()
    phases = {
        "complexity-validate": phase_complexity_validate,
        "complexity-run": phase_complexity_run,
        "complexity-plots": phase_complexity_plots,
        "scaling-config": phase_scaling_config,
        "scaling-n128": phase_scaling_n128,
        "scaling-n256": phase_scaling_n256,
        "scaling-plots": phase_scaling_plots,
        "report": phase_report,
    }
    if phase == "all":
        phase_complexity_validate(smoke)
        phase_complexity_run(smoke)
        phase_scaling_config(smoke)
        phase_scaling_n128(smoke)
        gate = read_json(DATA_SCALING / "hard_gate_n128.json")
        if gate["passed"]:
            phase_scaling_n256(smoke)
        else:
            print("N=128 gate failed; --phase all correctly skips N=256", flush=True)
        phase_scaling_plots(smoke)
        phase_report(smoke)
    else:
        phases[phase](smoke)
    preservation_gate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=(
        "complexity-validate", "complexity-run", "complexity-plots", "scaling-config",
        "scaling-n128", "scaling-n256", "scaling-plots", "report", "all",
    ), default="all")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run(arguments.phase, arguments.smoke)
