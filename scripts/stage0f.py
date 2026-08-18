"""Stage 0F channel-specific unitary headroom and receiver-budget calibration."""

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
from src.audit import per_row_retained_energy, retained_energy, sample_fractional_doppler, sample_grid_dd
from src.channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.metrics import qpsk_decode, qpsk_encode, random_bits
from src.oracle import OracleBatchResult, finite_budget_objective, optimize_oracle_batch, tune_afdm_for_budgets
from src.receiver import effective_operator, finite_budget_receiver, full_information_mse, rowwise_topk
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage0f"
CHECKPOINTS = OUT / "checkpoints"
COLORS = {
    "IDENTITY": "0.55",
    "OFDM": "tab:blue",
    "OTFS": "tab:green",
    "AFDM": "tab:red",
    "CHANNEL_SPECIFIC_ORACLE": "tab:purple",
    "MULTIK_ORACLE": "tab:brown",
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


def afdm_c1(multiplier: int) -> float:
    return multiplier / (2 * config.N)


def fixed_waveforms(multiplier: int) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=afdm_c1(multiplier)),
    }


def run_or_load(
    name: str,
    channels: torch.Tensor,
    multiplier: int,
    k_values: tuple[int, ...],
    force: bool,
    random_seed: int,
) -> OracleBatchResult:
    path = CHECKPOINTS / f"{name}.pt"
    if path.exists() and not force:
        result = torch.load(path, weights_only=False)
        print(f"loaded {name}: mean objective={number(torch.mean(result.best_objectives)):.8e}", flush=True)
        return result
    started = time.perf_counter()
    result = optimize_oracle_batch(
        channels,
        afdm_c1(multiplier),
        k_values,
        config.ORACLE_SNRS_DB,
        config.ORACLE_STEPS,
        random_seed,
    )
    torch.save(result, path)
    print(
        f"optimized {name} in {time.perf_counter()-started:.1f}s: "
        f"mean objective={number(torch.mean(result.best_objectives)):.8e}, "
        f"unitarity={result.max_unitarity_residual:.3e}",
        flush=True,
    )
    return result


def initialization_rows(label: str, result: OracleBatchResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for channel_index in range(result.initialization_objectives.shape[0]):
        for restart_index, name in enumerate(result.initialization_names):
            rows.append(
                {
                    "objective": label,
                    "channel_index": channel_index,
                    "initialization": name,
                    "initial_objective": number(result.initialization_objectives[channel_index, restart_index]),
                    "best_objective": number(result.best_objectives_by_restart[channel_index, restart_index]),
                    "final_objective": number(result.final_objectives[channel_index, restart_index]),
                    "best_step": int(result.best_steps_by_restart[channel_index, restart_index]),
                    "unitarity_residual": number(
                        torch.linalg.matrix_norm(
                            result.best_matrices_by_restart[channel_index, restart_index].mH
                            @ result.best_matrices_by_restart[channel_index, restart_index]
                            - torch.eye(config.N, dtype=COMPLEX_DTYPE)
                        )
                    ),
                    "nonfinite_seen": bool(result.nonfinite_seen[channel_index, restart_index]),
                    "won_channel": restart_index == int(result.winning_restart_indices[channel_index]),
                }
            )
    return rows


def independent_full(channel: torch.Tensor, snr_db: int) -> torch.Tensor:
    return full_information_mse(channel, noise_variance(snr_db))


def analytic_rows(
    channels: torch.Tensor,
    oracle_results: dict[int, OracleBatchResult],
    multik_result: OracleBatchResult,
    selected_multipliers: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in config.ORACLE_K_VALUES:
        waveforms = fixed_waveforms(selected_multipliers[f"K{k}"])
        oracle = oracle_results[k].best_matrices
        multik = multik_result.best_matrices
        for snr_db in config.ORACLE_SNRS_DB:
            baseline_values = {
                name: finite_budget_objective(waveform, channels, (k,), (snr_db,))
                for name, waveform in waveforms.items()
            }
            principal_stack = torch.stack([baseline_values[name] for name in ("OFDM", "OTFS", "AFDM")], dim=-1)
            best_values, best_indices = torch.min(principal_stack, dim=-1)
            oracle_values = finite_budget_objective(oracle, channels, (k,), (snr_db,))
            multik_values = finite_budget_objective(multik, channels, (k,), (snr_db,))
            full_values = independent_full(channels, snr_db)
            for channel_index in range(channels.shape[0]):
                baseline = number(best_values[channel_index])
                oracle_value = number(oracle_values[channel_index])
                full_value = number(full_values[channel_index])
                denominator = baseline - full_value
                rows.append(
                    {
                        "channel_index": channel_index,
                        "k": k,
                        "snr_db": snr_db,
                        "identity_mse": number(baseline_values["IDENTITY"][channel_index]),
                        "ofdm_mse": number(baseline_values["OFDM"][channel_index]),
                        "otfs_mse": number(baseline_values["OTFS"][channel_index]),
                        "afdm_mse": number(baseline_values["AFDM"][channel_index]),
                        "best_principal_name": ("OFDM", "OTFS", "AFDM")[int(best_indices[channel_index])],
                        "best_principal_mse": baseline,
                        "oracle_mse": oracle_value,
                        "multik_oracle_mse": number(multik_values[channel_index]),
                        "full_mse": full_value,
                        "delta_mse": oracle_value - baseline,
                        "relative_improvement": (baseline - oracle_value) / baseline,
                        "gap_closed": (baseline - oracle_value) / denominator if denominator != 0 else math.nan,
                        "identity_beats_principal": number(baseline_values["IDENTITY"][channel_index]) < baseline,
                    }
                )
        pooled_baselines = {
            name: finite_budget_objective(waveform, channels, (k,), config.ORACLE_SNRS_DB)
            for name, waveform in waveforms.items()
        }
        principal_stack = torch.stack([pooled_baselines[name] for name in ("OFDM", "OTFS", "AFDM")], dim=-1)
        best_values, best_indices = torch.min(principal_stack, dim=-1)
        oracle_values = finite_budget_objective(oracle, channels, (k,), config.ORACLE_SNRS_DB)
        multik_values = finite_budget_objective(multik, channels, (k,), config.ORACLE_SNRS_DB)
        full_values = torch.mean(
            torch.stack([independent_full(channels, snr) for snr in config.ORACLE_SNRS_DB]), dim=0
        )
        for channel_index in range(channels.shape[0]):
            baseline = number(best_values[channel_index])
            oracle_value = number(oracle_values[channel_index])
            full_value = number(full_values[channel_index])
            denominator = baseline - full_value
            rows.append(
                {
                    "channel_index": channel_index,
                    "k": k,
                    "snr_db": "POOLED",
                    "identity_mse": number(pooled_baselines["IDENTITY"][channel_index]),
                    "ofdm_mse": number(pooled_baselines["OFDM"][channel_index]),
                    "otfs_mse": number(pooled_baselines["OTFS"][channel_index]),
                    "afdm_mse": number(pooled_baselines["AFDM"][channel_index]),
                    "best_principal_name": ("OFDM", "OTFS", "AFDM")[int(best_indices[channel_index])],
                    "best_principal_mse": baseline,
                    "oracle_mse": oracle_value,
                    "multik_oracle_mse": number(multik_values[channel_index]),
                    "full_mse": full_value,
                    "delta_mse": oracle_value - baseline,
                    "relative_improvement": (baseline - oracle_value) / baseline,
                    "gap_closed": (baseline - oracle_value) / denominator if denominator != 0 else math.nan,
                    "identity_beats_principal": number(pooled_baselines["IDENTITY"][channel_index]) < baseline,
                }
            )
    return rows


def bootstrap_rows(analytic: list[dict[str, object]]) -> list[dict[str, object]]:
    generator = np.random.default_rng(config.BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for k in config.ORACLE_K_VALUES:
        for snr in (*config.ORACLE_SNRS_DB, "POOLED"):
            values = np.asarray(
                [float(row["delta_mse"]) for row in analytic if row["k"] == k and row["snr_db"] == snr]
            )
            indices = generator.integers(0, len(values), size=(config.ORACLE_BOOTSTRAP_RESAMPLES, len(values)))
            means = values[indices].mean(axis=1)
            rows.append(
                {
                    "k": k,
                    "snr_db": snr,
                    "mean_delta_mse": float(values.mean()),
                    "ci95_lower": float(np.quantile(means, 0.025)),
                    "ci95_upper": float(np.quantile(means, 0.975)),
                    "bootstrap_resamples": config.ORACLE_BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": config.BOOTSTRAP_SEED,
                }
            )
    return rows


def representation_metrics(matrix: torch.Tensor, channel: torch.Tensor, k: int) -> dict[str, float]:
    a = effective_operator(channel, matrix)
    a_tilde = rowwise_topk(a, k)
    eta, epsilon = retained_energy(a, k)
    singular = torch.linalg.svdvals(a_tilde)
    condition = singular[0] / singular[-1]
    limiting = torch.sum(
        torch.abs(torch.linalg.pinv(a_tilde, rtol=1e-12) @ a - torch.eye(config.N, dtype=COMPLEX_DTYPE)) ** 2
    ) / config.N
    return {
        "retained_energy": number(eta),
        "truncation_error": number(epsilon),
        "condition_number": number(condition),
        "limiting_distortion": number(limiting),
        "row_retained_min": number(torch.min(per_row_retained_energy(a, k))),
    }


def representation_rows(
    channels: torch.Tensor,
    analytic: list[dict[str, object]],
    oracle_results: dict[int, OracleBatchResult],
    selected_multipliers: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in config.ORACLE_K_VALUES:
        waveforms = fixed_waveforms(selected_multipliers[f"K{k}"])
        pooled = [row for row in analytic if row["k"] == k and row["snr_db"] == "POOLED"]
        for channel_index, channel in enumerate(channels):
            best_name = str(pooled[channel_index]["best_principal_name"])
            for name, matrix in (
                ("BEST_PRINCIPAL_CLASSICAL", waveforms[best_name]),
                ("CHANNEL_SPECIFIC_ORACLE", oracle_results[k].best_matrices[channel_index]),
            ):
                rows.append(
                    {
                        "channel_index": channel_index,
                        "k": k,
                        "entity": name,
                        "classical_name": best_name if name == "BEST_PRINCIPAL_CLASSICAL" else "",
                        **representation_metrics(matrix, channel, k),
                    }
                )
            for name in ("OFDM", "OTFS", "AFDM"):
                rows.append(
                    {
                        "channel_index": channel_index,
                        "k": k,
                        "entity": name,
                        "classical_name": name,
                        **representation_metrics(waveforms[name], channel, k),
                    }
                )
    return rows


def multik_rows(
    channels: torch.Tensor,
    oracle_results: dict[int, OracleBatchResult],
    multik_result: OracleBatchResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in config.ORACLE_K_VALUES:
        single = finite_budget_objective(oracle_results[k].best_matrices, channels, (k,), config.ORACLE_SNRS_DB)
        multi = finite_budget_objective(multik_result.best_matrices, channels, (k,), config.ORACLE_SNRS_DB)
        for channel_index in range(channels.shape[0]):
            rows.append(
                {
                    "channel_index": channel_index,
                    "k": k,
                    "single_k_objective": number(single[channel_index]),
                    "multik_objective": number(multi[channel_index]),
                    "ratio_multik_to_single": number(multi[channel_index] / single[channel_index]),
                }
            )
    return rows


def waveform_ber_by_channel(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    bits: torch.Tensor,
    physical_noise: torch.Tensor,
    sigma2: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if waveform.ndim == 2:
        waveform = waveform.expand(channels.shape[0], -1, -1)
    a = waveform.mH @ channels @ waveform
    _, equalizer, analytic = finite_budget_receiver(a, sigma2, k)
    symbols = qpsk_encode(bits)
    x = torch.einsum("cij,cbj->cbi", waveform, symbols)
    y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
    u = torch.einsum("cij,cbj->cbi", waveform.mH, y)
    estimate = torch.einsum("cij,cbj->cbi", equalizer, u)
    errors = torch.sum(qpsk_decode(estimate) != bits, dim=(1, 2, 3))
    return errors, analytic


def ber_rows(
    channels: torch.Tensor,
    oracle_results: dict[int, OracleBatchResult],
    multik_result: OracleBatchResult,
    selected_multipliers: dict[str, int],
) -> list[dict[str, object]]:
    blocks = math.ceil(1_000_000 / (channels.shape[0] * config.N * 2))
    generator = make_generator(config.ORACLE_CALIBRATION_SEED + 900)
    rows: list[dict[str, object]] = []
    bits_per_channel = blocks * config.N * 2
    for k in config.ORACLE_K_VALUES:
        waveforms = fixed_waveforms(selected_multipliers[f"K{k}"])
        evaluated = {
            "OFDM": waveforms["OFDM"],
            "OTFS": waveforms["OTFS"],
            "AFDM": waveforms["AFDM"],
            "CHANNEL_SPECIFIC_ORACLE": oracle_results[k].best_matrices,
            "MULTIK_ORACLE": multik_result.best_matrices,
        }
        for snr_db in config.ORACLE_BER_SNRS_DB:
            sigma2 = noise_variance(snr_db)
            bits = random_bits((channels.shape[0], blocks, config.N, 2), generator)
            noise = complex_awgn((channels.shape[0], blocks, config.N), sigma2, generator)
            for name, waveform in evaluated.items():
                errors, analytic = waveform_ber_by_channel(channels, waveform, bits, noise, sigma2, k)
                for channel_index in range(channels.shape[0]):
                    rows.append(
                        {
                            "channel_index": channel_index,
                            "k": k,
                            "snr_db": snr_db,
                            "waveform": name,
                            "bit_errors": int(errors[channel_index]),
                            "bit_count": bits_per_channel,
                            "ber": int(errors[channel_index]) / bits_per_channel,
                            "analytic_mse": number(analytic[channel_index]),
                        }
                    )
    return rows


def control_rows(
    label: str,
    channels: torch.Tensor,
    k: int,
    oracle: OracleBatchResult,
    multiplier: int,
) -> list[dict[str, object]]:
    waveforms = fixed_waveforms(multiplier)
    rows: list[dict[str, object]] = []
    for channel_index, channel in enumerate(channels):
        full = number(
            torch.mean(torch.stack([independent_full(channel, snr) for snr in config.ORACLE_SNRS_DB]))
        )
        baselines = {
            name: number(finite_budget_objective(waveform, channel, (k,), config.ORACLE_SNRS_DB))
            for name, waveform in waveforms.items()
        }
        principal = min(baselines[name] for name in ("OFDM", "OTFS", "AFDM"))
        oracle_value = number(oracle.best_objectives[channel_index])
        rows.append(
            {
                "control": label,
                "channel_index": channel_index,
                "k": k,
                "full_mse": full,
                "identity_mse": baselines["IDENTITY"],
                "best_principal_mse": principal,
                "oracle_mse": oracle_value,
                "oracle_minus_full": oracle_value - full,
                "oracle_relative_vs_principal": (principal - oracle_value) / principal,
                "violates_full_reference": oracle_value < full - 1e-10,
            }
        )
    return rows


def plot_convergence(label: str, result: OracleBatchResult) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name, trace in result.traces.items():
        axis.plot(range(len(trace)), trace, label=name)
    axis.set_xlabel("Projected Adam step")
    axis.set_ylabel("Design objective")
    axis.set_title(f"Representative channel convergence: {label}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "convergence" / f"{label}.png", dpi=180)
    plt.close(fig)


def plot_primary(analytic: list[dict[str, object]], ber: list[dict[str, object]], representation: list[dict[str, object]], multik: list[dict[str, object]]) -> None:
    for snr in config.ORACLE_BER_SNRS_DB:
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for name, column in (
            ("OFDM", "ofdm_mse"), ("OTFS", "otfs_mse"), ("AFDM", "afdm_mse"), ("CHANNEL_SPECIFIC_ORACLE", "oracle_mse")
        ):
            means = [
                np.mean([float(row[column]) for row in analytic if row["k"] == k and row["snr_db"] == snr])
                for k in config.ORACLE_K_VALUES
            ]
            axis.plot(config.ORACLE_K_VALUES, means, marker="o", color=COLORS[name], label=name.replace("_", " "))
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
        axis.set_xticks(config.ORACLE_K_VALUES)
        axis.set_xlabel("Retained interactions per row, K")
        axis.set_ylabel("Mean analytic MSE")
        axis.set_title(f"DD-v1 channel-specific headroom at {snr} dB")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"oracle_mse_vs_K_{snr}dB.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for name in ("OFDM", "OTFS", "AFDM", "CHANNEL_SPECIFIC_ORACLE"):
            means = []
            for k in config.ORACLE_K_VALUES:
                selected = [row for row in ber if row["k"] == k and row["snr_db"] == snr and row["waveform"] == name]
                means.append(sum(int(row["bit_errors"]) for row in selected) / sum(int(row["bit_count"]) for row in selected))
            axis.plot(config.ORACLE_K_VALUES, means, marker="o", color=COLORS[name], label=name.replace("_", " "))
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
        axis.set_xticks(config.ORACLE_K_VALUES)
        axis.set_xlabel("Retained interactions per row, K")
        axis.set_ylabel("Paired QPSK BER")
        axis.set_title(f"DD-v1 oracle BER validation at {snr} dB")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"oracle_ber_vs_K_{snr}dB.png", dpi=180)
        plt.close(fig)

    pooled = [row for row in analytic if row["snr_db"] == "POOLED"]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    data = [[float(row["gap_closed"]) for row in pooled if row["k"] == k] for k in config.ORACLE_K_VALUES]
    axis.boxplot(data, tick_labels=[str(k) for k in config.ORACLE_K_VALUES], showmeans=True)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("K")
    axis.set_ylabel("Fraction of classical-to-full gap closed")
    axis.set_title("Channel-specific oracle gap closure, pooled design SNRs")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "oracle_gap_closure_vs_K.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    data = [[100 * float(row["relative_improvement"]) for row in pooled if row["k"] == k] for k in config.ORACLE_K_VALUES]
    axis.boxplot(data, tick_labels=[f"K={k}" for k in config.ORACLE_K_VALUES], showmeans=True)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Analytic MSE improvement (%)")
    axis.set_title("Channel-specific oracle relative-gain distribution")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "oracle_relative_gain_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in ("OFDM", "OTFS", "AFDM", "CHANNEL_SPECIFIC_ORACLE"):
        means = [
            np.mean([float(row["retained_energy"]) for row in representation if row["k"] == k and row["entity"] == name])
            for k in config.ORACLE_K_VALUES
        ]
        axis.plot(config.ORACLE_K_VALUES, means, marker="o", color=COLORS[name], label=name.replace("_", " "))
    axis.set_xticks(config.ORACLE_K_VALUES)
    axis.set_ylim(0.9, 1.002)
    axis.set_xlabel("K")
    axis.set_ylabel("Mean retained energy")
    axis.set_title("Post hoc representation concentration")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "oracle_retained_energy_vs_K.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    data = [[float(row["ratio_multik_to_single"]) for row in multik if row["k"] == k] for k in config.ORACLE_K_VALUES]
    axis.boxplot(data, tick_labels=[str(k) for k in config.ORACLE_K_VALUES], showmeans=True)
    axis.axhline(1, color="black", linewidth=0.8)
    axis.set_xlabel("K")
    axis.set_ylabel("Multi-K objective / single-K objective")
    axis.set_title("Penalty for one channel-specific basis across budgets")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "multik_oracle_penalty.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="ignore cached oracle checkpoints")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (OUT / "convergence").mkdir(parents=True, exist_ok=True)

    validation = sample_dd_v1(config.N_VALIDATION_CHANNELS, config.VALIDATION_SEED).matrices
    tuning_rows, selected = tune_afdm_for_budgets(validation)
    expected = {
        "K16": config.AFDM_K16_TUNED_MULTIPLIER,
        "K24": config.AFDM_K24_TUNED_MULTIPLIER,
        "K32": config.AFDM_K32_TUNED_MULTIPLIER,
        "MULTIK": config.AFDM_MULTIK_TUNED_MULTIPLIER,
    }
    if selected != expected:
        raise RuntimeError(f"validation selections {selected} do not match locked config {expected}")
    write_csv("afdm_budget_tuning.csv", tuning_rows)
    print(f"AFDM validation selections: {selected}", flush=True)

    channels = sample_dd_v1(config.N_ORACLE_CHANNELS, config.ORACLE_CALIBRATION_SEED).matrices
    oracle_results = {
        k: run_or_load(
            f"ddv1_k{k}", channels, selected[f"K{k}"], (k,), args.force, config.ORACLE_RANDOM_SEED
        )
        for k in config.ORACLE_K_VALUES
    }
    multik_result = run_or_load(
        "ddv1_multik", channels, selected["MULTIK"], config.ORACLE_K_VALUES, args.force, config.ORACLE_RANDOM_SEED
    )

    grid_channels = sample_grid_dd(config.N_GRID_CONTROL_CHANNELS, config.ORACLE_CONTROL_SEED).matrices
    grid_result = run_or_load(
        "grid_k3", grid_channels, config.AFDM_K3_TUNED_C1_MULTIPLIER, (3,), args.force, config.ORACLE_RANDOM_SEED + 10_000
    )
    fractional_channels = sample_fractional_doppler(
        config.N_FRAC_DOPPLER_CONTROL_CHANNELS, config.ORACLE_CONTROL_SEED + 1
    ).matrices
    fractional_result = run_or_load(
        "frac_doppler_k24", fractional_channels, selected["K24"], (24,), args.force, config.ORACLE_RANDOM_SEED + 20_000
    )

    init_rows: list[dict[str, object]] = []
    for k, result in oracle_results.items():
        init_rows.extend(initialization_rows(f"K{k}", result))
        plot_convergence(f"k{k}", result)
    init_rows.extend(initialization_rows("MULTIK", multik_result))
    init_rows.extend(initialization_rows("GRID_K3", grid_result))
    init_rows.extend(initialization_rows("FRAC_DOPPLER_K24", fractional_result))
    plot_convergence("multik", multik_result)
    write_csv("oracle_initialization_results.csv", init_rows)

    analytic = analytic_rows(channels, oracle_results, multik_result, selected)
    bootstrap = bootstrap_rows(analytic)
    representation = representation_rows(channels, analytic, oracle_results, selected)
    multi = multik_rows(channels, oracle_results, multik_result)
    ber = ber_rows(channels, oracle_results, multik_result, selected)
    controls = control_rows("GRID-DD", grid_channels, 3, grid_result, config.AFDM_K3_TUNED_C1_MULTIPLIER)
    controls.extend(control_rows("FRAC-DOPPLER", fractional_channels, 24, fractional_result, selected["K24"]))

    if any(bool(row["violates_full_reference"]) for row in controls):
        raise RuntimeError("oracle violated the independent full-information MSE reference")
    for result in (*oracle_results.values(), multik_result, grid_result, fractional_result):
        if torch.any(result.best_objectives > torch.min(result.initialization_objectives, dim=1).values + 1e-12):
            raise RuntimeError("best-seen oracle is worse than the best initialization")

    write_csv("oracle_per_channel_results.csv", analytic)
    write_csv("oracle_bootstrap_ci.csv", bootstrap)
    write_csv("oracle_representation_diagnostics.csv", representation)
    write_csv("oracle_multik_results.csv", multi)
    write_csv("oracle_ber_results.csv", ber)
    write_csv("oracle_control_results.csv", controls)
    plot_primary(analytic, ber, representation, multi)

    pooled_ci = [row for row in bootstrap if row["snr_db"] == "POOLED"]
    headroom_confirmed = any(float(row["ci95_upper"]) < 0 for row in pooled_ci)
    all_results = (*oracle_results.values(), multik_result, grid_result, fractional_result)
    diagnostics = {
        "runtime_seconds": time.perf_counter() - started,
        "afdm_selected_multipliers": selected,
        "oracle_channels": config.N_ORACLE_CHANNELS,
        "steps_per_restart": config.ORACLE_STEPS,
        "restarts_per_channel": 6,
        "max_unitarity_residual": max(result.max_unitarity_residual for result in all_results),
        "nonfinite_run_count": sum(int(torch.count_nonzero(result.nonfinite_seen)) for result in all_results),
        "grid_max_oracle_minus_full": max(float(row["oracle_minus_full"]) for row in controls if row["control"] == "GRID-DD"),
        "grid_min_oracle_minus_full": min(float(row["oracle_minus_full"]) for row in controls if row["control"] == "GRID-DD"),
        "headroom_confirmed_by_pooled_ci": headroom_confirmed,
        "ber_blocks_per_channel": math.ceil(1_000_000 / (config.N_ORACLE_CHANNELS * config.N * 2)),
        "ber_bits_per_k_snr": math.ceil(1_000_000 / (config.N_ORACLE_CHANNELS * config.N * 2))
        * config.N_ORACLE_CHANNELS
        * config.N
        * 2,
    }
    (OUT / "stage0f_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
