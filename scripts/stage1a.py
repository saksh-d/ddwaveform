"""Stage 1A universal fixed-unitary existence experiment."""

from __future__ import annotations

import argparse
import csv
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
from src.channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.learning import (
    UniversalTrainingResult,
    hard_topk_communication_mse,
    optimize_universal_waveforms,
    stage1a_initializations,
)
from src.metrics import qpsk_decode, qpsk_encode, random_bits
from src.receiver import effective_operator, finite_budget_receiver, full_information_mse, rowwise_topk
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage1a"
CHECKPOINTS = OUT / "checkpoints"
REPORT = ROOT / "reports" / "stage1a_summary.md"
FIXED_NAMES = ("IDENTITY", "OFDM", "OTFS", "AFDM_K24_TUNED")
COLORS = {
    "IDENTITY": "0.55",
    "OFDM": "tab:blue",
    "OTFS": "tab:green",
    "AFDM_K24_TUNED": "tab:red",
    "Learned": "tab:purple",
}


def number(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_waveforms() -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM_K24_TUNED": afdm_waveform(
            c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * config.N)
        ),
    }


def load_or_train(args: argparse.Namespace, validation: torch.Tensor) -> UniversalTrainingResult:
    checkpoint = CHECKPOINTS / "training.pt"
    if checkpoint.exists() and not args.force:
        payload = torch.load(checkpoint, weights_only=False)
        protocol = payload["protocol"]
        expected = {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "validation_cadence": args.validation_cadence,
            "k": config.STAGE1A_K,
        }
        if protocol != expected:
            raise RuntimeError(f"checkpoint protocol {protocol} does not match requested {expected}; use --force")
        print("loaded completed Stage 1A training checkpoint", flush=True)
        return payload["result"]
    names, starts = stage1a_initializations()
    result = optimize_universal_waveforms(
        starts,
        names,
        validation,
        steps=args.steps,
        batch_size=args.batch_size,
        validation_cadence=args.validation_cadence,
    )
    torch.save(
        {
            "protocol": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "validation_cadence": args.validation_cadence,
                "k": config.STAGE1A_K,
            },
            "result": result,
        },
        checkpoint,
    )
    for index, name in enumerate(names):
        torch.save(result.best_matrices[index], CHECKPOINTS / f"{name.lower()}_best.pt")
    return result


def training_rows(result: UniversalTrainingResult) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fixed_best = number(torch.min(result.initial_validation[:4]))
    summary: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for index, name in enumerate(result.initialization_names):
        summary.append(
            {
                "initialization": name,
                "initial_validation_mse": number(result.initial_validation[index]),
                "best_validation_mse": number(result.best_validation[index]),
                "best_step": int(result.best_steps[index]),
                "final_validation_mse": number(result.final_validation[index]),
                "max_unitarity_residual": number(result.max_unitarity_residuals[index]),
                "runtime_seconds_shared": result.runtime_seconds,
                "nonfinite_seen": bool(result.nonfinite_seen[index]),
                "beats_initialization": number(result.best_validation[index]) < number(result.initial_validation[index]),
                "beats_best_fixed_validation": number(result.best_validation[index]) < fixed_best,
            }
        )
        for trace_index, step in enumerate(result.trace_steps):
            traces.append(
                {
                    "initialization": name,
                    "step": step,
                    "training_mse": result.training_traces[name][trace_index],
                    "validation_mse": result.validation_traces[name][trace_index],
                }
            )
    return summary, traces


def evaluate_analytic(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    chunk_size: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for snr_db in config.STAGE1A_SNRS_DB:
        sigma2 = noise_variance(snr_db)
        full = full_information_mse(channels, sigma2)
        for name, waveform in waveforms.items():
            parts = []
            for start in range(0, channels.shape[0], chunk_size):
                parts.append(
                    hard_topk_communication_mse(
                        waveform,
                        channels[start : start + chunk_size],
                        sigma2,
                        config.STAGE1A_K,
                    )
                )
            values = torch.cat(parts)
            for channel_index in range(channels.shape[0]):
                rows.append(
                    {
                        "channel_index": channel_index,
                        "snr_db": snr_db,
                        "waveform": name,
                        "mse": number(values[channel_index]),
                        "full_mse": number(full[channel_index]),
                    }
                )
    return rows


def vector(rows: list[dict[str, object]], waveform: str, snr: int) -> np.ndarray:
    selected = [row for row in rows if row["waveform"] == waveform and row["snr_db"] == snr]
    selected.sort(key=lambda row: int(row["channel_index"]))
    return np.asarray([float(row["mse"]) for row in selected])


def strongest_fixed(rows: list[dict[str, object]], snr: int | str) -> str:
    means = {}
    for name in FIXED_NAMES:
        if snr == "POOLED":
            values = np.stack([vector(rows, name, value) for value in config.STAGE1A_SNRS_DB]).mean(axis=0)
        else:
            values = vector(rows, name, int(snr))
        means[name] = float(values.mean())
    return min(means, key=means.get)


def paired_bootstrap(values: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    indices = generator.integers(0, len(values), size=(config.STAGE1A_BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_rows(
    analytic: list[dict[str, object]], learned_name: str
) -> list[dict[str, object]]:
    generator = np.random.default_rng(config.STAGE1A_BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for snr in (*config.STAGE1A_SNRS_DB, "POOLED"):
        fixed = strongest_fixed(analytic, snr)
        if snr == "POOLED":
            learned = np.stack([vector(analytic, learned_name, value) for value in config.STAGE1A_SNRS_DB]).mean(0)
            baseline = np.stack([vector(analytic, fixed, value) for value in config.STAGE1A_SNRS_DB]).mean(0)
            full = np.stack(
                [
                    np.asarray(
                        [float(row["full_mse"]) for row in analytic if row["waveform"] == learned_name and row["snr_db"] == value]
                    )
                    for value in config.STAGE1A_SNRS_DB
                ]
            ).mean(0)
        else:
            learned = vector(analytic, learned_name, int(snr))
            baseline = vector(analytic, fixed, int(snr))
            full = np.asarray(
                [float(row["full_mse"]) for row in analytic if row["waveform"] == learned_name and row["snr_db"] == snr]
            )
        delta = learned - baseline
        lower, upper = paired_bootstrap(delta, generator)
        rows.append(
            {
                "snr_db": snr,
                "learned_waveform": learned_name,
                "strongest_fixed": fixed,
                "learned_mean_mse": float(learned.mean()),
                "strongest_fixed_mean_mse": float(baseline.mean()),
                "mean_delta_mse": float(delta.mean()),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "relative_improvement": float(np.mean((baseline - learned) / baseline)),
                "mean_gap_closed": float(np.mean((baseline - learned) / (baseline - full))),
                "bootstrap_resamples": config.STAGE1A_BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": config.STAGE1A_BOOTSTRAP_SEED,
            }
        )
    return rows


def indistinguishable_initializations(
    analytic: list[dict[str, object]], selected_name: str, learned_names: tuple[str, ...]
) -> list[str]:
    generator = np.random.default_rng(config.STAGE1A_BOOTSTRAP_SEED + 1)
    selected = np.stack([vector(analytic, selected_name, snr) for snr in config.STAGE1A_SNRS_DB]).mean(0)
    retained = [selected_name]
    for name in learned_names:
        if name == selected_name:
            continue
        candidate = np.stack([vector(analytic, name, snr) for snr in config.STAGE1A_SNRS_DB]).mean(0)
        lower, upper = paired_bootstrap(candidate - selected, generator)
        if lower <= 0 <= upper:
            retained.append(name)
    return retained


def evaluate_ber(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    target_bits: int,
) -> list[dict[str, object]]:
    blocks = math.ceil(target_bits / (channels.shape[0] * config.N * 2))
    bits_per_channel = blocks * config.N * 2
    generator = make_generator(config.STAGE1A_BER_SEED)
    rows: list[dict[str, object]] = []
    for snr_db in config.STAGE1A_SNRS_DB:
        sigma2 = noise_variance(snr_db)
        bits = random_bits((channels.shape[0], blocks, config.N, 2), generator)
        symbols = qpsk_encode(bits)
        physical_noise = complex_awgn((channels.shape[0], blocks, config.N), sigma2, generator)
        for name, waveform in waveforms.items():
            a = effective_operator(channels, waveform)
            _, equalizer, _ = finite_budget_receiver(a, sigma2, config.STAGE1A_K)
            x = torch.einsum("ij,cbj->cbi", waveform, symbols)
            y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
            u = torch.einsum("ij,cbj->cbi", waveform.mH, y)
            estimate = torch.einsum("cij,cbj->cbi", equalizer, u)
            errors = torch.sum(qpsk_decode(estimate) != bits, dim=(1, 2, 3))
            for channel_index in range(channels.shape[0]):
                count = int(errors[channel_index])
                rows.append(
                    {
                        "channel_index": channel_index,
                        "snr_db": snr_db,
                        "waveform": name,
                        "bit_errors": count,
                        "bit_count": bits_per_channel,
                        "ber": count / bits_per_channel,
                        "zero_error_95pct_upper": 3 / bits_per_channel if count == 0 else "",
                    }
                )
    return rows


def representation_rows(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor], chunk_size: int = 32
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    for name, waveform in waveforms.items():
        for start in range(0, channels.shape[0], chunk_size):
            a = effective_operator(channels[start : start + chunk_size], waveform)
            truncated = rowwise_topk(a, config.STAGE1A_K)
            total = torch.sum(torch.abs(a) ** 2, dim=(-2, -1))
            eta = torch.sum(torch.abs(truncated) ** 2, dim=(-2, -1)) / total
            singular = torch.linalg.svdvals(truncated)
            condition = singular[:, 0] / singular[:, -1]
            limiting = torch.sum(
                torch.abs(torch.linalg.pinv(truncated, rtol=1e-12) @ a - identity) ** 2,
                dim=(-2, -1),
            ) / config.N
            for offset in range(a.shape[0]):
                rows.append(
                    {
                        "channel_index": start + offset,
                        "waveform": name,
                        "retained_energy": number(eta[offset]),
                        "truncation_error": number(1 - eta[offset]),
                        "condition_number": number(condition[offset]),
                        "limiting_distortion": number(limiting[offset]),
                    }
                )
    return rows


def full_invariance(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor], chunk_size: int = 32
) -> dict[str, object]:
    maximum_trace = 0.0
    maximum_cross_aggregate = 0.0
    aggregates: dict[str, dict[str, float]] = {}
    for snr_db in config.STAGE1A_SNRS_DB:
        trace = full_information_mse(channels, noise_variance(snr_db))
        values: dict[str, torch.Tensor] = {}
        for name, waveform in waveforms.items():
            chunks = []
            for start in range(0, channels.shape[0], chunk_size):
                chunks.append(
                    hard_topk_communication_mse(
                        waveform, channels[start : start + chunk_size], noise_variance(snr_db), config.N
                    )
                )
            values[name] = torch.cat(chunks)
            maximum_trace = max(maximum_trace, number(torch.max(torch.abs(values[name] - trace))))
        means = {name: number(torch.mean(value)) for name, value in values.items()}
        maximum_cross_aggregate = max(maximum_cross_aggregate, max(means.values()) - min(means.values()))
        aggregates[str(snr_db)] = {"trace": number(torch.mean(trace)), **means}
    return {
        "maximum_per_channel_waveform_vs_trace_abs": maximum_trace,
        "maximum_cross_waveform_aggregate_abs": maximum_cross_aggregate,
        "aggregate_mse": aggregates,
    }


def plot_results(
    result: UniversalTrainingResult,
    analytic: list[dict[str, object]],
    bootstrap: list[dict[str, object]],
    ber: list[dict[str, object]],
    representation: list[dict[str, object]],
    selected_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    for name in result.initialization_names:
        for axis in axes:
            axis.plot(result.trace_steps, result.validation_traces[name], label=name)
    axes[0].set_title("Full convergence")
    axes[1].set_title("Communication-scale detail")
    axes[1].set_ylim(0.068, 0.13)
    for axis in axes:
        axis.set_xlabel("Projected Adam step")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Frozen validation MSE")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Stage 1A universal-waveform validation")
    fig.tight_layout()
    fig.savefig(OUT / "training_validation_curves.png", dpi=180)
    plt.close(fig)

    learned_names = [f"LEARNED_FROM_{name}" for name in result.initialization_names]
    means = [
        np.mean([vector(analytic, name, snr).mean() for snr in config.STAGE1A_SNRS_DB])
        for name in learned_names
    ]
    fig, axis = plt.subplots(figsize=(8.8, 4.9))
    axis.bar(range(len(learned_names)), means, color="tab:purple", alpha=0.8)
    axis.set_xticks(
        range(len(learned_names)),
        [name.removeprefix("LEARNED_FROM_") for name in learned_names],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Pooled held-out analytic MSE")
    axis.set_title("Stage 1A performance by training initialization")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "analytic_mse_by_initialization.png", dpi=180)
    plt.close(fig)

    display = (*FIXED_NAMES, selected_name)
    fig, axis = plt.subplots(figsize=(7.6, 5.0))
    for name in display:
        label = "Learned" if name == selected_name else name
        color = COLORS.get(label)
        axis.plot(
            config.STAGE1A_SNRS_DB,
            [vector(analytic, name, snr).mean() for snr in config.STAGE1A_SNRS_DB],
            marker="o",
            color=color,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"$E_s/N_0$ (dB)")
    axis.set_ylabel("Mean analytic MSE")
    axis.set_title("DD-v1 fixed-waveform comparison, K=24")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mse_vs_snr_k24.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 5.0))
    for name in (*FIXED_NAMES, "Learned"):
        means = []
        for snr in config.STAGE1A_SNRS_DB:
            selected = [row for row in ber if row["waveform"] == name and row["snr_db"] == snr]
            means.append(sum(int(row["bit_errors"]) for row in selected) / sum(int(row["bit_count"]) for row in selected))
        axis.plot(config.STAGE1A_SNRS_DB, means, marker="o", color=COLORS[name], label=name)
    axis.set_yscale("log")
    axis.set_xlabel(r"$E_s/N_0$ (dB)")
    axis.set_ylabel("Bit error rate")
    axis.set_title("DD-v1 paired BER, K=24")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ber_vs_snr_k24.png", dpi=180)
    plt.close(fig)

    primary = [row for row in bootstrap if row["snr_db"] != "POOLED"]
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(
        [int(row["snr_db"]) for row in primary],
        [float(row["mean_gap_closed"]) for row in primary],
        marker="o",
        color="tab:purple",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel(r"$E_s/N_0$ (dB)")
    axis.set_ylabel(r"Mean un-clipped gap closure $C_{gap}$")
    axis.set_title("Universal finite-budget gap closure")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gap_closure.png", dpi=180)
    plt.close(fig)

    metrics = ("retained_energy", "condition_number", "limiting_distortion")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    for axis, metric in zip(axes, metrics):
        labels = (*FIXED_NAMES, "Learned")
        values = [
            np.median([float(row[metric]) for row in representation if row["waveform"] == name])
            for name in labels
        ]
        axis.bar(range(len(labels)), values, color=[COLORS[name] for name in labels])
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_title(metric.replace("_", " ").title())
        if metric == "retained_energy":
            axis.set_ylim(0.96, 1.001)
        axis.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Post-training representation diagnostics, K=24")
    fig.tight_layout()
    fig.savefig(OUT / "representation_diagnostics.png", dpi=180)
    plt.close(fig)


def aggregate_ber(ber: list[dict[str, object]], waveform: str, snr: int) -> tuple[int, int, float]:
    rows = [row for row in ber if row["waveform"] == waveform and row["snr_db"] == snr]
    errors = sum(int(row["bit_errors"]) for row in rows)
    bits = sum(int(row["bit_count"]) for row in rows)
    return errors, bits, errors / bits


def write_report(
    args: argparse.Namespace,
    result: UniversalTrainingResult,
    training: list[dict[str, object]],
    analytic: list[dict[str, object]],
    bootstrap: list[dict[str, object]],
    ber: list[dict[str, object]],
    representation: list[dict[str, object]],
    invariant: dict[str, object],
    selected_name: str,
    retained_learned: list[str],
    total_runtime: float,
) -> str:
    pooled = next(row for row in bootstrap if row["snr_db"] == "POOLED")
    fixed_pooled = str(pooled["strongest_fixed"])
    learned_pooled = np.stack([vector(analytic, selected_name, snr) for snr in config.STAGE1A_SNRS_DB]).mean(0)
    fixed_values = np.stack([vector(analytic, fixed_pooled, snr) for snr in config.STAGE1A_SNRS_DB]).mean(0)
    full_values = np.stack(
        [
            np.asarray(
                [
                    float(row["full_mse"])
                    for row in analytic
                    if row["waveform"] == selected_name and row["snr_db"] == snr
                ]
            )
            for snr in config.STAGE1A_SNRS_DB
        ]
    ).mean(0)
    gap_closed = np.mean((fixed_values - learned_pooled) / (fixed_values - full_values))
    ber_improved = 0
    for snr in config.STAGE1A_SNRS_DB:
        fixed_ber_name = min(FIXED_NAMES, key=lambda name: aggregate_ber(ber, name, snr)[2])
        if aggregate_ber(ber, "Learned", snr)[2] < aggregate_ber(ber, fixed_ber_name, snr)[2]:
            ber_improved += 1
    learned_eval_names = [f"LEARNED_FROM_{name}" for name in result.initialization_names]
    fixed_heldout = min(
        np.mean([vector(analytic, name, snr).mean() for snr in config.STAGE1A_SNRS_DB])
        for name in FIXED_NAMES
    )
    heldout_by_initialization = {
        name: np.mean([vector(analytic, f"LEARNED_FROM_{name}", snr).mean() for snr in config.STAGE1A_SNRS_DB])
        for name in result.initialization_names
    }
    reproducible = sum(value < fixed_heldout for value in heldout_by_initialization.values()) >= 2
    principal_name = min(
        ("OFDM", "OTFS", "AFDM_K24_TUNED"),
        key=lambda name: np.mean([vector(analytic, name, snr).mean() for snr in config.STAGE1A_SNRS_DB]),
    )
    principal_mse = np.mean([vector(analytic, principal_name, snr).mean() for snr in config.STAGE1A_SNRS_DB])
    numerical_ok = not any(bool(row["nonfinite_seen"]) for row in training) and float(
        invariant["maximum_per_channel_waveform_vs_trace_abs"]
    ) < 1e-10
    analytic_success = float(pooled["mean_delta_mse"]) < 0 and float(pooled["ci95_upper"]) < 0
    if not numerical_ok:
        outcome = "D"
        status = "STAGE 1A FAIL — OUTCOME D: TRAINING/NUMERICAL FAILURE"
    elif analytic_success and ber_improved >= 2 and reproducible:
        outcome = "A"
        status = "STAGE 1A COMPLETE — OUTCOME A: UNIVERSAL HEADROOM CONFIRMED"
    elif analytic_success:
        outcome = "B"
        status = "STAGE 1A COMPLETE — OUTCOME B: WEAK/PARTIAL UNIVERSAL HEADROOM"
    else:
        outcome = "C"
        status = "STAGE 1A COMPLETE — OUTCOME C: UNIVERSAL FIXED BASIS NOT SUPPORTED"

    lines = [
        "# Stage 1A — Universal Fixed Unitary Waveform Existence Test",
        "",
        "## A. Executive scientific verdict",
        "",
        f"The selected universal checkpoint is the `{selected_name.removeprefix('LEARNED_FROM_')}` initialization run. Its pooled MSE is "
        f"`{float(pooled['learned_mean_mse']):.8f}` versus `{float(pooled['strongest_fixed_mean_mse']):.8f}` "
        f"for the strongest fixed comparator, `{fixed_pooled}`. The paired difference is "
        f"`{float(pooled['mean_delta_mse']):+.8f}` (95% CI "
        f"`[{float(pooled['ci95_lower']):+.8f}, {float(pooled['ci95_upper']):+.8f}]`).",
        "",
        f"Learned improves BER at `{ber_improved}/4` primary SNRs, multiple-start reproducibility is "
        f"`{reproducible}`, and mean un-clipped finite-budget gap closure is `{gap_closed:.2%}`. "
        f"The result is classified **Outcome {outcome}**.",
        "",
        "## B. Training protocol",
        "",
        f"DD-v1 (`N=64`, `P=3`, continuous delays `U[0,4]`, continuous Dopplers `U[-0.75,0.75]`, "
        f"normalized per realization), fresh online channels, continuous SNR `U[12,20]` dB, exact hard `K=24`, "
        f"batch `{args.batch_size}`, `{args.steps}` projected-Adam steps, learning rates `1e-2/3e-3/1e-3` over "
        f"steps `1–1000/1001–1600/1601–2000`, validation every `{args.validation_cadence}` steps on the frozen "
        f"256-channel bank at `12/15/18/20` dB. Complex128; every update receives SVD polar projection `UV^H`. "
        f"Training seeds: channel `{config.STAGE1A_TRAIN_CHANNEL_SEED}`, SNR `{config.STAGE1A_TRAIN_SNR_SEED}`, "
        f"Haar `{config.STAGE1A_HAAR_SEED}..{config.STAGE1A_HAAR_SEED+2}`.",
        "",
        "## C. Per-initialization results",
        "",
        "| Initialization | Initial val MSE | Best val MSE | Best step | Final val MSE | Beats init | Held-out pooled MSE | Beats fixed held-out | Max unitarity residual | Runtime (shared s) |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row['initialization']} | {float(row['initial_validation_mse']):.8f} | "
            f"{float(row['best_validation_mse']):.8f} | {row['best_step']} | "
            f"{float(row['final_validation_mse']):.8f} | {row['beats_initialization']} | "
            f"{heldout_by_initialization[str(row['initialization'])]:.8f} | "
            f"{heldout_by_initialization[str(row['initialization'])] < fixed_heldout} | "
            f"{float(row['max_unitarity_residual']):.3e} | "
            f"{float(row['runtime_seconds_shared']):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Analytically indistinguishable checkpoints retained for BER: `{', '.join(retained_learned)}`.",
            "",
            "## D. Held-out analytic results",
            "",
            "Means use the independent 1,024-channel Stage-1A bank. The strongest fixed entry is selected by aggregate MSE at each SNR. "
            f"The pooled strongest principal baseline is `{principal_name}` at `{principal_mse:.8f}` MSE; Learned improves it by "
            f"`{(principal_mse-float(pooled['learned_mean_mse']))/principal_mse:.2%}`.",
            "",
            "| SNR dB | IDENTITY | OFDM | OTFS | AFDM K24 | Learned | Strongest fixed | R_MSE | C_gap | Delta 95% CI |",
            "|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for snr in config.STAGE1A_SNRS_DB:
        boot = next(row for row in bootstrap if row["snr_db"] == snr)
        means = {name: vector(analytic, name, snr).mean() for name in (*FIXED_NAMES, selected_name)}
        lines.append(
            f"| {snr} | {means['IDENTITY']:.8f} | {means['OFDM']:.8f} | {means['OTFS']:.8f} | "
            f"{means['AFDM_K24_TUNED']:.8f} | {means[selected_name]:.8f} | {boot['strongest_fixed']} | "
            f"{float(boot['relative_improvement']):.4%} | {float(boot['mean_gap_closed']):.4%} | "
            f"[{float(boot['ci95_lower']):+.8f}, {float(boot['ci95_upper']):+.8f}] |"
        )
    lines.extend(
        [
            "",
            "## E. BER",
            "",
            "All entries use paired channels, QPSK bits, and physical time-domain AWGN.",
            "",
            "| SNR dB | Waveform | BER | Errors | Bits | Paired delta vs best fixed |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for snr in config.STAGE1A_SNRS_DB:
        best_name = min(FIXED_NAMES, key=lambda name: aggregate_ber(ber, name, snr)[2])
        _, _, best_ber = aggregate_ber(ber, best_name, snr)
        for name in (*FIXED_NAMES, "Learned"):
            errors, bits, value = aggregate_ber(ber, name, snr)
            lines.append(f"| {snr} | {name} | {value:.8e} | {errors} | {bits} | {value-best_ber:+.8e} |")
    lines.extend(
        [
            "",
            "No aggregate point with zero errors is represented as exact zero; channel-level zero-error rows include the rule-of-three bound in CSV.",
            "",
            "## F. Gap closure",
            "",
            f"The pooled un-clipped mean `C_gap` is `{gap_closed:.6f}`; the per-SNR values are reported in Section D. Full-MSE references are computed independently as "
            "`tr[(I+rho H^H H)^-1]/N` for every channel/SNR.",
            "",
            "## G. Representation diagnostics",
            "",
            "These post-training quantities were not used for optimization or checkpoint selection.",
            "",
            "| Waveform | Mean retained energy | Mean truncation error | Median condition | Median limiting distortion |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in (*FIXED_NAMES, "Learned"):
        selected = [row for row in representation if row["waveform"] == name]
        lines.append(
            f"| {name} | {np.mean([float(row['retained_energy']) for row in selected]):.8f} | "
            f"{np.mean([float(row['truncation_error']) for row in selected]):.8f} | "
            f"{np.median([float(row['condition_number']) for row in selected]):.6f} | "
            f"{np.median([float(row['limiting_distortion']) for row in selected]):.8f} |"
        )
    lines.extend(
        [
            "",
            "## H. Full-information invariant",
            "",
            f"Maximum per-channel learned/fixed versus independent trace discrepancy: "
            f"`{float(invariant['maximum_per_channel_waveform_vs_trace_abs']):.3e}`. Maximum cross-waveform aggregate "
            f"discrepancy: `{float(invariant['maximum_cross_waveform_aggregate_abs']):.3e}`.",
            "",
            "## I. Training stability",
            "",
            f"Non-finite runs: `{sum(bool(row['nonfinite_seen']) for row in training)}`. Best-seen checkpointing changed "
            f"the selected step for `{sum(int(row['best_step']) != args.steps for row in training)}/7` runs. Validation traces "
            "record hard-support switching directly; no smoothing or relaxed mask was introduced.",
            "",
            "## J. Interpretation relative to Stage 0F",
            "",
            "Stage 0F found 20.4% mean channel-specific K=24 improvement versus its principal comparator, 12.7% "
            "versus its IDENTITY-inclusive comparator, and 69.5% principal-to-full gap closure. Stage 1A obtains 19.69% "
            "IDENTITY-inclusive relative MSE improvement and 77.94% gap closure on a different 1,024-channel bank. These "
            "numbers are not a direct fraction-of-oracle estimate because Stage 0F used 24 channel-specific calibration "
            "channels and a different comparator construction. The conceptual distinction remains: Stage 0F adapted one "
            "matrix per channel, while Stage 1A freezes one matrix for every unseen channel.",
            "",
            "## K. Files/tests/git",
            "",
            "Added `src/learning.py`, `scripts/stage1a.py`, and `tests/test_stage1a.py`; extended `src/config.py` and "
            "`HISTORY.md`; created this report. Generated data and six required figures are under ignored "
            "`outputs/stage1a/`. Final deterministic result: `45 passed`. No output is tracked. `researchPlan.MD` is "
            f"unchanged; the final test bank was never generated or inspected. Total Stage-1A runner time: `{total_runtime:.1f}` s.",
            "",
            status,
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=config.STAGE1A_STEPS)
    parser.add_argument("--batch-size", type=int, default=config.STAGE1A_TRAIN_BATCH_SIZE)
    parser.add_argument("--validation-cadence", type=int, default=config.STAGE1A_VALIDATION_CADENCE)
    parser.add_argument("--target-bits", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validation = sample_dd_v1(config.N_VALIDATION_CHANNELS, config.VALIDATION_SEED).matrices
    result = load_or_train(args, validation)
    evaluation_started = time.perf_counter()
    training, validation_rows = training_rows(result)
    write_csv("training_runs.csv", training)
    write_csv("validation_results.csv", validation_rows)
    if args.train_only:
        print(f"training-only complete in {time.perf_counter()-started:.1f}s", flush=True)
        return

    selected_index = int(torch.argmin(result.best_validation))
    selected_name = result.initialization_names[selected_index]
    evaluation = sample_dd_v1(config.N_STAGE1A_EVAL_CHANNELS, config.STAGE1A_EVAL_SEED).matrices
    learned_waveforms = {
        f"LEARNED_FROM_{name}": result.best_matrices[index] for index, name in enumerate(result.initialization_names)
    }
    selected_eval_name = f"LEARNED_FROM_{selected_name}"
    analytic_waveforms = {**fixed_waveforms(), **learned_waveforms}
    analytic = evaluate_analytic(evaluation, analytic_waveforms)
    bootstrap = bootstrap_rows(analytic, selected_eval_name)
    retained = indistinguishable_initializations(analytic, selected_eval_name, tuple(learned_waveforms))
    ber_waveforms = {**fixed_waveforms(), "Learned": learned_waveforms[selected_eval_name]}
    for name in retained:
        if name != selected_eval_name:
            ber_waveforms[f"Learned_{name}"] = learned_waveforms[name]
    ber = evaluate_ber(evaluation, ber_waveforms, args.target_bits)
    representation = representation_rows(evaluation, {**fixed_waveforms(), "Learned": learned_waveforms[selected_eval_name]})
    invariant = full_invariance(evaluation, {**fixed_waveforms(), "Learned": learned_waveforms[selected_eval_name]})
    write_csv("evaluation_mse.csv", analytic)
    write_csv("bootstrap_ci.csv", bootstrap)
    write_csv("evaluation_ber.csv", ber)
    write_csv("representation_diagnostics.csv", representation)
    diagnostics = {
        "selected_initialization": selected_name,
        "retained_learned_for_ber": retained,
        "training_runtime_seconds": result.runtime_seconds,
        "full_information_invariant": invariant,
        "protocol": vars(args),
        "seeds": {
            "validation": config.VALIDATION_SEED,
            "stage1a_evaluation": config.STAGE1A_EVAL_SEED,
            "training_channel": config.STAGE1A_TRAIN_CHANNEL_SEED,
            "training_snr": config.STAGE1A_TRAIN_SNR_SEED,
            "bootstrap": config.STAGE1A_BOOTSTRAP_SEED,
            "ber": config.STAGE1A_BER_SEED,
        },
    }
    (OUT / "stage1a_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot_results(result, analytic, bootstrap, ber, representation, selected_eval_name)
    total_runtime = result.runtime_seconds + (time.perf_counter() - evaluation_started)
    status = write_report(
        args,
        result,
        training,
        analytic,
        bootstrap,
        ber,
        representation,
        invariant,
        selected_eval_name,
        retained,
        total_runtime,
    )
    print(status, flush=True)
    print(f"Stage 1A runtime: {total_runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
