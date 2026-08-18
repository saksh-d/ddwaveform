"""Stage 1A.1 frozen communication-error forensic audit."""

from __future__ import annotations

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
from src.channel import COMPLEX_DTYPE, complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.diagnostics import (
    distribution_summary,
    mse_decomposition,
    per_stream_diagnostics,
    representation_diagnostics,
)
from src.metrics import (
    Constellation,
    constellation_encode,
    nearest_neighbor_decode,
    nearest_neighbor_margin,
    qam16_constellation,
    qpsk_constellation,
    random_bits,
)
from src.receiver import effective_operator, finite_budget_receiver
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage1a1"
CHECKPOINT = ROOT / "outputs" / "stage1a" / "checkpoints" / "haar_2_best.pt"
TRAINING_CHECKPOINT = ROOT / "outputs" / "stage1a" / "checkpoints" / "training.pt"
SNRS = config.STAGE1A_SNRS_DB
K = config.STAGE1A_K
FIGURE_WAVEFORMS = ("OFDM", "OTFS", "AFDM", "Learned")
ALL_WAVEFORMS = ("IDENTITY", *FIGURE_WAVEFORMS)
COLORS = {
    "OFDM": "tab:blue",
    "OTFS": "tab:green",
    "AFDM": "tab:red",
    "Learned": "tab:purple",
}
QPSK_EPSILON = 0.1
BOOTSTRAP_RESAMPLES = 10_000


def number(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_waveforms() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    learned = torch.load(CHECKPOINT, weights_only=True)
    payload = torch.load(TRAINING_CHECKPOINT, weights_only=False)
    result = payload["result"]
    index = result.initialization_names.index("HAAR_2")
    stored = result.best_matrices[index]
    exact_match = bool(torch.equal(learned, stored))
    if not exact_match:
        raise RuntimeError("standalone HAAR_2 checkpoint does not match the Stage-1A training payload")
    waveforms = {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * config.N)),
        "Learned": learned,
    }
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    residual = number(torch.linalg.matrix_norm(learned.mH @ learned - identity))
    return waveforms, {
        "path": str(CHECKPOINT.relative_to(ROOT)),
        "sha256": file_sha256(CHECKPOINT),
        "training_payload_sha256": file_sha256(TRAINING_CHECKPOINT),
        "exact_tensor_match_to_training_payload": exact_match,
        "selected_initialization": "LEARNED_FROM_HAAR_2",
        "unitarity_residual": residual,
    }


def analytic_forensics(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, int], torch.Tensor],
    float,
]:
    mse_rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    gamma_cache: dict[tuple[str, int], torch.Tensor] = {}
    maximum_decomposition_residual = 0.0
    for waveform_name, waveform in waveforms.items():
        effective = effective_operator(channels, waveform)
        representation = representation_diagnostics(effective, K)
        for snr_db in SNRS:
            sigma2 = noise_variance(snr_db)
            _, equalizer, receiver_total = finite_budget_receiver(effective, sigma2, K)
            parts = mse_decomposition(effective, equalizer, sigma2)
            maximum_decomposition_residual = max(
                maximum_decomposition_residual,
                number(torch.max(torch.abs(receiver_total - parts["total"]))),
            )
            stream = per_stream_diagnostics(parts["b"], equalizer, sigma2)
            gamma_cache[(waveform_name, snr_db)] = stream["gamma"].detach()
            for metric_name, values in stream.items():
                summary = distribution_summary(values)
                stream_rows.append(
                    {
                        "waveform": waveform_name,
                        "snr_db": snr_db,
                        "metric": metric_name,
                        **summary,
                    }
                )
            for channel_index in range(channels.shape[0]):
                mse_rows.append(
                    {
                        "channel_index": channel_index,
                        "waveform": waveform_name,
                        "snr_db": snr_db,
                        "total": number(parts["total"][channel_index]),
                        "diagonal": number(parts["diagonal"][channel_index]),
                        "offdiagonal": number(parts["offdiagonal"][channel_index]),
                        "noise": number(parts["noise"][channel_index]),
                        "sum_residual": number(
                            parts["total"][channel_index]
                            - parts["diagonal"][channel_index]
                            - parts["offdiagonal"][channel_index]
                            - parts["noise"][channel_index]
                        ),
                    }
                )
                channel_gamma = stream["gamma"][channel_index]
                channel_rows.append(
                    {
                        "channel_index": channel_index,
                        "waveform": waveform_name,
                        "snr_db": snr_db,
                        "analytic_mse": number(parts["total"][channel_index]),
                        "median_stream_gamma": number(torch.median(channel_gamma)),
                        "p05_stream_gamma": number(torch.quantile(channel_gamma, 0.05)),
                        "retained_energy": number(representation["retained_energy"][channel_index]),
                        "limiting_distortion": number(representation["limiting_distortion"][channel_index]),
                        "condition_number": number(representation["condition_number"][channel_index]),
                        "qpsk_ber": math.nan,
                    }
                )
    return mse_rows, stream_rows, channel_rows, gamma_cache, maximum_decomposition_residual


def margin_row(
    modulation: str,
    waveform: str,
    snr_db: int,
    metric: str,
    values: torch.Tensor,
    subset: str = "ALL_CHANNELS",
) -> dict[str, object]:
    summary = distribution_summary(values)
    flat = values.reshape(-1)
    return {
        "modulation": modulation,
        "waveform": waveform,
        "snr_db": snr_db,
        "subset": subset,
        "metric": metric,
        **summary,
        "probability_negative": number(torch.mean((flat < 0).to(torch.float64))),
        "probability_near_positive": number(
            torch.mean(((flat >= 0) & (flat < QPSK_EPSILON)).to(torch.float64))
        ),
        "near_positive_epsilon": QPSK_EPSILON,
    }


def simulate_modulations(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, int, str], torch.Tensor],
    dict[tuple[str, int, str], torch.Tensor],
]:
    modulation_rows: list[dict[str, object]] = []
    qpsk_margin_rows: list[dict[str, object]] = []
    constellation_margin_rows: list[dict[str, object]] = []
    margin_cache: dict[tuple[str, int, str], torch.Tensor] = {}
    empirical_cache: dict[tuple[str, int, str], torch.Tensor] = {}
    constellations = (qpsk_constellation(), qam16_constellation())
    for modulation_index, constellation in enumerate(constellations):
        blocks = math.ceil(
            1_000_000
            / (channels.shape[0] * config.N * constellation.bits_per_symbol)
        )
        bits_per_channel = blocks * config.N * constellation.bits_per_symbol
        symbols_per_channel = blocks * config.N
        for snr_index, snr_db in enumerate(SNRS):
            symbol_generator = make_generator(
                config.STAGE1A1_SYMBOL_SEED + 100 * modulation_index + snr_index
            )
            noise_generator = make_generator(
                config.STAGE1A1_NOISE_SEED + 100 * modulation_index + snr_index
            )
            bits = random_bits(
                (channels.shape[0], blocks, config.N, constellation.bits_per_symbol),
                symbol_generator,
            )
            symbols, true_indices = constellation_encode(bits, constellation)
            sigma2 = noise_variance(snr_db)
            physical_noise = complex_awgn(
                (channels.shape[0], blocks, config.N), sigma2, noise_generator
            )
            for waveform_name, waveform in waveforms.items():
                effective = effective_operator(channels, waveform)
                _, equalizer, analytic = finite_budget_receiver(effective, sigma2, K)
                x = torch.einsum("ij,cbj->cbi", waveform, symbols)
                y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
                u = torch.einsum("ij,cbj->cbi", waveform.mH, y)
                estimate = torch.einsum("cij,cbj->cbi", equalizer, u)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                empirical_mse = torch.mean(torch.abs(estimate - symbols) ** 2, dim=(1, 2))
                nn_margin = nearest_neighbor_margin(estimate, true_indices, constellation)
                empirical_cache[(constellation.name, snr_db, waveform_name)] = empirical_mse
                if snr_db in (12, 20) and waveform_name in FIGURE_WAVEFORMS:
                    margin_cache[(constellation.name, snr_db, waveform_name)] = nn_margin.detach()
                if constellation.name == "QPSK" and waveform_name in ("IDENTITY", "Learned"):
                    margin_cache[("QPSK_NN_ALL", snr_db, waveform_name)] = nn_margin.detach()
                constellation_margin_rows.append(
                    margin_row(
                        constellation.name,
                        waveform_name,
                        snr_db,
                        "nearest_neighbor_margin",
                        nn_margin,
                    )
                )
                if constellation.name == "QPSK":
                    signed_real = torch.sign(symbols.real) * estimate.real
                    signed_imag = torch.sign(symbols.imag) * estimate.imag
                    signed = torch.stack((signed_real, signed_imag), dim=-1)
                    if waveform_name in ("IDENTITY", "Learned"):
                        margin_cache[("QPSK_SIGNED_ALL", snr_db, waveform_name)] = signed.detach()
                    qpsk_margin_rows.append(
                        margin_row("QPSK", waveform_name, snr_db, "signed_bit_margin", signed)
                    )
                for channel_index in range(channels.shape[0]):
                    modulation_rows.append(
                        {
                            "channel_index": channel_index,
                            "modulation": constellation.name,
                            "waveform": waveform_name,
                            "snr_db": snr_db,
                            "analytic_mse": number(analytic[channel_index]),
                            "empirical_mse": number(empirical_mse[channel_index]),
                            "bit_errors": int(bit_errors[channel_index]),
                            "bit_count": bits_per_channel,
                            "ber": int(bit_errors[channel_index]) / bits_per_channel,
                            "symbol_errors": int(symbol_errors[channel_index]),
                            "symbol_count": symbols_per_channel,
                            "ser": int(symbol_errors[channel_index]) / symbols_per_channel,
                        }
                    )
    return modulation_rows, qpsk_margin_rows, constellation_margin_rows, margin_cache, empirical_cache


def channel_comparisons_and_conditionals(
    channel_rows: list[dict[str, object]],
    modulation_rows: list[dict[str, object]],
    qpsk_margin_rows: list[dict[str, object]],
    constellation_margin_rows: list[dict[str, object]],
    margin_cache: dict[tuple[str, int, str], torch.Tensor],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    qpsk_lookup = {
        (str(row["waveform"]), int(row["snr_db"]), int(row["channel_index"])): row
        for row in modulation_rows
        if row["modulation"] == "QPSK"
    }
    for row in channel_rows:
        qpsk = qpsk_lookup[(str(row["waveform"]), int(row["snr_db"]), int(row["channel_index"]))]
        row["qpsk_ber"] = qpsk["ber"]
    comparisons: list[dict[str, object]] = []
    for snr_db in SNRS:
        identity = sorted(
            [row for row in channel_rows if row["waveform"] == "IDENTITY" and row["snr_db"] == snr_db],
            key=lambda row: int(row["channel_index"]),
        )
        learned = sorted(
            [row for row in channel_rows if row["waveform"] == "Learned" and row["snr_db"] == snr_db],
            key=lambda row: int(row["channel_index"]),
        )
        mse_delta = np.asarray([float(l["analytic_mse"]) - float(i["analytic_mse"]) for i, l in zip(identity, learned)])
        ber_delta = np.asarray([float(l["qpsk_ber"]) - float(i["qpsk_ber"]) for i, l in zip(identity, learned)])
        lower_mse = mse_delta < 0
        lower_ber = ber_delta < 0
        discordant = lower_mse & (ber_delta > 0)
        positive_penalty = np.maximum(ber_delta, 0)
        top_count = max(1, math.ceil(0.10 * len(ber_delta)))
        top_share = (
            float(np.sort(positive_penalty)[-top_count:].sum() / positive_penalty.sum())
            if positive_penalty.sum() > 0
            else 0.0
        )
        comparisons.append(
            {
                "snr_db": snr_db,
                "fraction_lower_mse": float(lower_mse.mean()),
                "fraction_lower_ber": float(lower_ber.mean()),
                "fraction_lower_mse_higher_ber": float(discordant.mean()),
                "fraction_lower_mse_lower_ber": float((lower_mse & lower_ber).mean()),
                "mse_delta_mean": float(mse_delta.mean()),
                "mse_delta_median": float(np.median(mse_delta)),
                "mse_delta_p05": float(np.quantile(mse_delta, 0.05)),
                "mse_delta_p95": float(np.quantile(mse_delta, 0.95)),
                "ber_delta_mean": float(ber_delta.mean()),
                "ber_delta_median": float(np.median(ber_delta)),
                "ber_delta_p05": float(np.quantile(ber_delta, 0.05)),
                "ber_delta_p95": float(np.quantile(ber_delta, 0.95)),
                "top_10pct_channel_share_of_positive_ber_penalty": top_share,
            }
        )
        indices = torch.as_tensor(np.flatnonzero(discordant), dtype=torch.int64)
        for waveform_name in ("IDENTITY", "Learned"):
            signed = margin_cache[("QPSK_SIGNED_ALL", snr_db, waveform_name)]
            nn_margin = margin_cache[("QPSK_NN_ALL", snr_db, waveform_name)]
            if indices.numel() > 0:
                subset_name = "LEARNED_LOWER_MSE_HIGHER_BER_CHANNELS"
                qpsk_margin_rows.append(
                    margin_row(
                        "QPSK",
                        waveform_name,
                        snr_db,
                        "signed_bit_margin",
                        signed[indices],
                        subset_name,
                    )
                )
                constellation_margin_rows.append(
                    margin_row(
                        "QPSK",
                        waveform_name,
                        snr_db,
                        "nearest_neighbor_margin",
                        nn_margin[indices],
                        subset_name,
                    )
                )
    return channel_rows, comparisons


def aggregate_modulation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    for modulation in ("QPSK", "16QAM"):
        for waveform in ALL_WAVEFORMS:
            for snr_db in SNRS:
                selected = [
                    row for row in rows
                    if row["modulation"] == modulation and row["waveform"] == waveform and row["snr_db"] == snr_db
                ]
                bits = sum(int(row["bit_count"]) for row in selected)
                bit_errors = sum(int(row["bit_errors"]) for row in selected)
                symbols = sum(int(row["symbol_count"]) for row in selected)
                symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
                analytic = np.mean([float(row["analytic_mse"]) for row in selected])
                empirical = np.mean([float(row["empirical_mse"]) for row in selected])
                aggregates.append(
                    {
                        "modulation": modulation,
                        "waveform": waveform,
                        "snr_db": snr_db,
                        "analytic_mse": analytic,
                        "empirical_mse": empirical,
                        "empirical_minus_analytic": empirical - analytic,
                        "relative_mse_discrepancy": (empirical - analytic) / analytic,
                        "bit_errors": bit_errors,
                        "total_bits": bits,
                        "ber": bit_errors / bits,
                        "symbol_errors": symbol_errors,
                        "total_symbols": symbols,
                        "ser": symbol_errors / symbols,
                        "zero_error_95pct_upper": 3 / bits if bit_errors == 0 else "",
                    }
                )
    return aggregates


def paired_bootstrap(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    generator = np.random.default_rng(config.STAGE1A1_BOOTSTRAP_SEED)
    output: list[dict[str, object]] = []
    for modulation in ("QPSK", "16QAM"):
        for snr_db in SNRS:
            identity = sorted(
                [row for row in rows if row["modulation"] == modulation and row["waveform"] == "IDENTITY" and row["snr_db"] == snr_db],
                key=lambda row: int(row["channel_index"]),
            )
            learned = sorted(
                [row for row in rows if row["modulation"] == modulation and row["waveform"] == "Learned" and row["snr_db"] == snr_db],
                key=lambda row: int(row["channel_index"]),
            )
            for metric in ("analytic_mse", "ber", "ser"):
                delta = np.asarray([float(l[metric]) - float(i[metric]) for i, l in zip(identity, learned)])
                indices = generator.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
                means = delta[indices].mean(axis=1)
                output.append(
                    {
                        "modulation": modulation,
                        "snr_db": snr_db,
                        "metric": metric,
                        "mean_learned_minus_identity": float(delta.mean()),
                        "ci95_lower": float(np.quantile(means, 0.025)),
                        "ci95_upper": float(np.quantile(means, 0.975)),
                        "resamples": BOOTSTRAP_RESAMPLES,
                        "seed": config.STAGE1A1_BOOTSTRAP_SEED,
                    }
                )
    return output


def empirical_cdf(axis: plt.Axes, values: torch.Tensor, label: str, color: str) -> None:
    array = np.sort(values.detach().cpu().numpy().reshape(-1))
    probabilities = np.arange(1, len(array) + 1) / len(array)
    axis.plot(array, probabilities, label=label, color=color)


def plot_figures(
    mse_rows: list[dict[str, object]],
    gamma_cache: dict[tuple[str, int], torch.Tensor],
    margin_cache: dict[tuple[str, int, str], torch.Tensor],
    modulation_aggregates: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), sharex=True)
    for axis, component in zip(axes.flat, ("total", "diagonal", "offdiagonal", "noise")):
        for waveform in FIGURE_WAVEFORMS:
            means = [
                np.mean([float(row[component]) for row in mse_rows if row["waveform"] == waveform and row["snr_db"] == snr])
                for snr in SNRS
            ]
            axis.plot(SNRS, means, marker="o", color=COLORS[waveform], label=waveform)
        axis.set_title(component.title())
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
    axes[1, 0].set_xlabel(r"$E_s/N_0$ (dB)")
    axes[1, 1].set_xlabel(r"$E_s/N_0$ (dB)")
    axes[0, 0].set_ylabel("Mean MSE component")
    axes[1, 0].set_ylabel("Mean MSE component")
    axes[0, 1].legend()
    fig.suptitle("Frozen K=24 communication-MSE decomposition")
    fig.tight_layout()
    fig.savefig(OUT / "mse_decomposition_vs_snr.png", dpi=180)
    plt.close(fig)

    for snr_db in (12, 20):
        fig, axis = plt.subplots(figsize=(7.4, 4.9))
        for waveform in FIGURE_WAVEFORMS:
            gamma_db = 10 * torch.log10(gamma_cache[(waveform, snr_db)])
            empirical_cdf(axis, gamma_db, waveform, COLORS[waveform])
        axis.set_xlabel("Diagnostic effective SINR (dB)")
        axis.set_ylabel("Empirical CDF")
        axis.set_title(f"Per-stream effective-SINR distribution at {snr_db} dB")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"effective_sinr_cdf_{snr_db}dB.png", dpi=180)
        plt.close(fig)

    for modulation, filename in (("QPSK", "qpsk_decision_margin_cdf.png"), ("16QAM", "16qam_decision_margin_cdf.png")):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
        for axis, snr_db in zip(axes, (12, 20)):
            for waveform in FIGURE_WAVEFORMS:
                empirical_cdf(axis, margin_cache[(modulation, snr_db, waveform)], waveform, COLORS[waveform])
            axis.axvline(0, color="black", linewidth=1)
            axis.set_xlabel("Nearest-neighbor margin")
            axis.set_title(f"{snr_db} dB")
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("Empirical CDF")
        axes[1].legend()
        fig.suptitle(f"{modulation} decision-margin distribution")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)

    for modulation, filename in (("QPSK", "ber_vs_snr_qpsk.png"), ("16QAM", "ber_vs_snr_16qam.png")):
        fig, axis = plt.subplots(figsize=(7.4, 4.9))
        for waveform in FIGURE_WAVEFORMS:
            selected = [row for row in modulation_aggregates if row["modulation"] == modulation and row["waveform"] == waveform]
            axis.plot(SNRS, [row["ber"] for row in selected], marker="o", color=COLORS[waveform], label=waveform)
        axis.set_yscale("log")
        axis.set_xlabel(r"$E_s/N_0$ (dB)")
        axis.set_ylabel("Bit error rate")
        axis.set_title(f"Frozen K=24 {modulation} BER")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    waveforms, checkpoint = load_frozen_waveforms()
    before = waveforms["Learned"].clone()
    channels = sample_dd_v1(
        config.N_STAGE1A1_FORENSIC_CHANNELS,
        config.STAGE1A1_CHANNEL_SEED,
    ).matrices
    mse_rows, stream_rows, channel_rows, gamma_cache, decomposition_residual = analytic_forensics(
        channels, waveforms
    )
    modulation_rows, qpsk_margin_rows, constellation_margin_rows, margin_cache, _ = simulate_modulations(
        channels, waveforms
    )
    channel_rows, channel_comparisons = channel_comparisons_and_conditionals(
        channel_rows,
        modulation_rows,
        qpsk_margin_rows,
        constellation_margin_rows,
        margin_cache,
    )
    aggregates = aggregate_modulation_rows(modulation_rows)
    bootstrap = paired_bootstrap(modulation_rows)
    plot_figures(mse_rows, gamma_cache, margin_cache, aggregates)
    write_csv("mse_decomposition.csv", mse_rows)
    write_csv("stream_diagnostics.csv", stream_rows)
    write_csv("channel_diagnostics.csv", channel_rows)
    write_csv("channel_comparisons.csv", channel_comparisons)
    write_csv("qpsk_margin_statistics.csv", qpsk_margin_rows)
    write_csv("constellation_margin_statistics.csv", constellation_margin_rows)
    write_csv("modulation_channel_results.csv", modulation_rows)
    write_csv("modulation_results.csv", aggregates)
    write_csv("paired_bootstrap.csv", bootstrap)
    runtime = time.perf_counter() - started
    diagnostics = {
        "checkpoint": checkpoint,
        "protocol": {
            "channels": config.N_STAGE1A1_FORENSIC_CHANNELS,
            "snrs_db": SNRS,
            "k": K,
            "modulations": ["QPSK", "16QAM"],
            "qpsk_near_margin_epsilon": QPSK_EPSILON,
        },
        "seeds": {
            "channel": config.STAGE1A1_CHANNEL_SEED,
            "symbol": config.STAGE1A1_SYMBOL_SEED,
            "noise": config.STAGE1A1_NOISE_SEED,
            "bootstrap": config.STAGE1A1_BOOTSTRAP_SEED,
        },
        "maximum_mse_decomposition_residual": decomposition_residual,
        "frozen_checkpoint_unchanged": bool(torch.equal(before, waveforms["Learned"])),
        "channel_comparisons": channel_comparisons,
        "runtime_seconds": runtime,
    }
    (OUT / "stage1a1_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"Stage 1A.1 forensic run complete in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
