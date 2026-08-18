"""Corrective Stage 0E audit of classical delay-Doppler baselines."""

from __future__ import annotations

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
from src.audit import (
    canonical_afdm_c1,
    direct_afdm_idaft,
    expected_afdm_column_offset,
    otfs_reference_demodulate,
    otfs_reference_modulate,
    per_row_retained_energy,
    retained_energy,
    sample_classical_reference,
    sample_continuous_dd,
    sample_fractional_doppler,
    sample_grid_dd,
    serialize_columns,
)
from src.channel import (
    COMPLEX_DTYPE,
    assemble_dd_channel,
    complex_awgn,
    doppler_operator,
    fractional_delay_operator,
    make_generator,
    noise_variance,
    sample_dd_v1,
)
from src.metrics import bit_error_rate, normalized_mse, qpsk_decode, qpsk_encode, random_bits
from src.receiver import effective_operator, finite_budget_receiver, rowwise_topk
from src.waveforms import (
    afdm_canonical_waveform,
    afdm_k3_tuned_waveform,
    afdm_waveform,
    identity_waveform,
    ofdm_waveform,
    otfs_waveform,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage0e_audit"
K_VALUES = (1, 2, 3, 4, 6, 9, 12, 16, 24, 32, 48, 64)
FULL_SNRS = (0, 5, 10, 12, 15, 18, 20, 25, 30)
COLORS = {"IDENTITY": "0.45", "OFDM": "tab:blue", "OTFS": "tab:green", "AFDM": "tab:red"}
PRINCIPAL = ("OFDM", "OTFS", "AFDM")


def value(tensor: torch.Tensor | float) -> float:
    return float(torch.as_tensor(tensor).detach().cpu().item())


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = OUT / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def waveforms_64(afdm_c1: float = config.AFDM_C1) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=afdm_c1),
    }


def simulate(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    snrs_db: tuple[int, ...],
    k_values: tuple[int, ...],
    blocks_per_channel: int,
    seed: int,
    regime: str,
) -> list[dict[str, object]]:
    count, n, _ = channels.shape
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    generator = make_generator(seed)
    rows: list[dict[str, object]] = []
    for snr_db in snrs_db:
        sigma2 = noise_variance(snr_db)
        bits = random_bits((count, blocks_per_channel, n, 2), generator)
        symbols = qpsk_encode(bits)
        physical_noise = complex_awgn((count, blocks_per_channel, n), sigma2, generator)
        for name, waveform in waveforms.items():
            x = torch.einsum("ij,cbj->cbi", waveform, symbols)
            y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
            u = torch.einsum("ij,cbj->cbi", waveform.mH, y)
            for k in k_values:
                _, equalizer, analytic = finite_budget_receiver(effective[name], sigma2, k)
                estimates = torch.einsum("cij,cbj->cbi", equalizer, u)
                estimated_bits = qpsk_decode(estimates)
                errors = int(torch.count_nonzero(estimated_bits != bits).item())
                bit_count = bits.numel()
                rows.append(
                    {
                        "regime": regime,
                        "waveform": name,
                        "snr_db": snr_db,
                        "k": k,
                        "ber": errors / bit_count,
                        "bit_errors": errors,
                        "bit_count": bit_count,
                        "zero_error_95pct_upper": 3.0 / bit_count if errors == 0 else "",
                        "empirical_nmse": normalized_mse(symbols, estimates),
                        "mean_analytic_mse": value(torch.mean(analytic)),
                    }
                )
    return rows


def plot_ber_full_vs_k3(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True, sharey=True)
    for axis, k, title in zip(axes, (config.N, 3), ("Full-CSI LMMSE — K=N", "Limited representation — K=3")):
        for name in ("IDENTITY", *PRINCIPAL):
            selected = [row for row in rows if row["waveform"] == name and row["k"] == k]
            plotted = [max(float(row["ber"]), 0.5 / int(row["bit_count"])) for row in selected]
            axis.plot(
                [row["snr_db"] for row in selected], plotted, marker="o", color=COLORS[name],
                alpha=0.65 if name == "IDENTITY" else 1.0, linewidth=1.4 if name == "IDENTITY" else 2.0,
                label=name,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel(r"$E_s/N_0$ (dB)")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Bit error rate")
    axes[0].legend()
    fig.suptitle("DD-v1: full-channel knowledge versus top-3 mismatch")
    fig.tight_layout()
    fig.savefig(OUT / "ber_full_vs_k3.png", dpi=180)
    plt.close(fig)


def plot_budget(rows: list[dict[str, object]], snr_db: int) -> None:
    selected_snr = [row for row in rows if row["snr_db"] == snr_db]
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name in ("IDENTITY", *PRINCIPAL):
        selected = [row for row in selected_snr if row["waveform"] == name]
        axis.plot([row["k"] for row in selected], [row["ber"] for row in selected], marker="o", color=COLORS[name], label=name)
    axis.set_yscale("log")
    axis.set_xscale("log", base=2)
    axis.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    axis.set_xlabel("Retained interactions per row, K")
    axis.set_ylabel("Bit error rate")
    axis.set_title(f"DD-v1 receiver-budget sweep at {snr_db} dB")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"ber_vs_K_{snr_db}dB.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name in ("IDENTITY", *PRINCIPAL):
        selected = [row for row in selected_snr if row["waveform"] == name]
        x = [row["k"] for row in selected]
        color = COLORS[name]
        axis.plot(x, [row["mean_analytic_mse"] for row in selected], marker="o", color=color, label=f"{name} analytic")
        axis.plot(x, [row["empirical_nmse"] for row in selected], linestyle="--", color=color, alpha=0.65)
    axis.set_yscale("log")
    axis.set_xscale("log", base=2)
    axis.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    axis.set_xlabel("Retained interactions per row, K")
    axis.set_ylabel("Normalized MSE")
    axis.set_title(f"DD-v1 MSE versus receiver budget at {snr_db} dB\nsolid: analytic; dashed: empirical")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"mse_vs_K_{snr_db}dB.png", dpi=180)
    plt.close(fig)


def concentration_diagnostics(channels: torch.Tensor, waveforms: dict[str, torch.Tensor]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    rowwise_rows: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        effective = effective_operator(channels, waveform)
        for k in K_VALUES:
            eta, epsilon = retained_energy(effective, k)
            rows.append(
                {
                    "waveform": name,
                    "k": k,
                    "eta_mean": value(torch.mean(eta)),
                    "eta_median": value(torch.median(eta)),
                    "eta_p10": value(torch.quantile(eta, 0.10)),
                    "eta_p90": value(torch.quantile(eta, 0.90)),
                    "epsilon_mean": value(torch.mean(epsilon)),
                    "partition_max_abs": value(torch.max(torch.abs(eta + epsilon - 1))),
                }
            )
        per_row = per_row_retained_energy(effective, 3).reshape(-1)
        rowwise_rows.append(
            {
                "waveform": name,
                "k": 3,
                "mean": value(torch.mean(per_row)),
                "median": value(torch.median(per_row)),
                "p05": value(torch.quantile(per_row, 0.05)),
                "p10": value(torch.quantile(per_row, 0.10)),
                "minimum": value(torch.min(per_row)),
            }
        )
    return rows, rowwise_rows


def plot_concentration(rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name in ("IDENTITY", *PRINCIPAL):
        selected = [row for row in rows if row["waveform"] == name]
        x = np.asarray([row["k"] for row in selected], dtype=float)
        mean = np.asarray([row["eta_mean"] for row in selected], dtype=float)
        p10 = np.asarray([row["eta_p10"] for row in selected], dtype=float)
        p90 = np.asarray([row["eta_p90"] for row in selected], dtype=float)
        axis.plot(x, mean, marker="o", color=COLORS[name], label=name)
        axis.fill_between(x, p10, p90, color=COLORS[name], alpha=0.10)
    axis.set_xscale("log", base=2)
    axis.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Retained interactions per row, K")
    axis.set_ylabel(r"Retained energy $\eta_K$")
    axis.set_title("DD-v1 effective-operator concentration\nmean with 10th–90th channel percentiles")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "retained_energy_vs_K.png", dpi=180)
    plt.close(fig)


def mismatch_diagnostics(channels: torch.Tensor, waveforms: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    for name, waveform in waveforms.items():
        effective = effective_operator(channels, waveform)
        truncated = rowwise_topk(effective, 3)
        omitted = torch.sum(torch.abs(effective - truncated) ** 2, dim=(-2, -1)) / torch.sum(
            torch.abs(effective) ** 2, dim=(-2, -1)
        )
        singular_values = torch.linalg.svdvals(truncated)
        condition = singular_values[..., 0] / singular_values[..., -1]
        pinv = torch.linalg.pinv(truncated, rtol=1e-12)
        limiting = torch.sum(torch.abs(pinv @ effective - identity) ** 2, dim=(-2, -1)) / config.N
        finite = condition[torch.isfinite(condition)]
        rows.append(
            {
                "waveform": name,
                "omitted_energy_mean": value(torch.mean(omitted)),
                "omitted_energy_median": value(torch.median(omitted)),
                "condition_median": value(torch.median(finite)) if finite.numel() else math.inf,
                "condition_p90": value(torch.quantile(finite, 0.90)) if finite.numel() else math.inf,
                "condition_max": value(torch.max(finite)) if finite.numel() else math.inf,
                "singular_fraction": value(torch.mean((singular_values[..., -1] < 1e-12).double())),
                "limiting_distortion_mean": value(torch.mean(limiting)),
                "limiting_distortion_median": value(torch.median(limiting)),
                "limiting_distortion_p90": value(torch.quantile(limiting, 0.90)),
            }
        )
    return rows


def heatmap(matrix: torch.Tensor, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.3, 4.5))
    image = axis.imshow(torch.abs(matrix).cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axis.set_xlabel("Input column")
    axis.set_ylabel("Output row")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Magnitude")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def otfs_audit() -> tuple[dict[str, float], list[dict[str, object]]]:
    generator = make_generator(config.GLOBAL_SEED + 300)
    dd = (
        torch.randn((8, 8), dtype=torch.float64, generator=generator)
        + 1j * torch.randn((8, 8), dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    direct = otfs_reference_modulate(dd)
    matrix = otfs_waveform() @ serialize_columns(dd)
    recovered = otfs_reference_demodulate(direct, 8, 8)
    equivalence = {
        "two_stage_vs_kronecker_max_abs": value(torch.max(torch.abs(direct - matrix))),
        "clean_reconstruction_max_abs": value(torch.max(torch.abs(recovered - dd))),
    }
    cases = {
        "identity": torch.eye(config.N, dtype=COMPLEX_DTYPE),
        "integer_delay_2": fractional_delay_operator(2),
        "integer_doppler_2": doppler_operator(2),
        "integer_delay_2_doppler_1": doppler_operator(1) @ fractional_delay_operator(2),
        "three_on_grid_paths": assemble_dd_channel(
            torch.tensor([0.0, 2.0, 4.0]),
            torch.tensor([-2.0, 0.0, 2.0]),
            torch.tensor([0.6 + 0.1j, -0.2 + 0.5j, 0.3 - 0.4j], dtype=COMPLEX_DTYPE),
        ),
    }
    rows: list[dict[str, object]] = []
    waveform = otfs_waveform()
    for name, channel in cases.items():
        effective = effective_operator(channel, waveform)
        magnitudes = torch.abs(effective)
        significant = torch.sum(magnitudes > 1e-10, dim=-1).double()
        eta1, _ = retained_energy(effective, 1)
        eta3, _ = retained_energy(effective, 3)
        dominant = torch.argmax(magnitudes, dim=-1)
        offsets = (dominant - torch.arange(config.N)) % config.N
        rows.append(
            {
                "case": name,
                "significant_per_row_min": value(torch.min(significant)),
                "significant_per_row_mean": value(torch.mean(significant)),
                "significant_per_row_max": value(torch.max(significant)),
                "top1_energy": value(eta1),
                "top3_energy": value(eta3),
                "dominant_offset_count": int(torch.unique(offsets).numel()),
                "dominant_offsets": ";".join(map(str, torch.unique(offsets).tolist())),
            }
        )
        heatmap(effective, f"OTFS sanity: {name}", OUT / "otfs_sanity" / f"{name}.png")
    return equivalence, rows


def afdm_audit() -> tuple[dict[str, float], list[dict[str, object]]]:
    generator = make_generator(config.GLOBAL_SEED + 301)
    symbols = (
        torch.randn(config.N, dtype=torch.float64, generator=generator)
        + 1j * torch.randn(config.N, dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    direct = direct_afdm_idaft(symbols, config.AFDM_C1, config.AFDM_C2)
    matrix = afdm_waveform() @ symbols
    direct_error = value(torch.max(torch.abs(direct - matrix)))
    canonical_c1 = canonical_afdm_c1(config.N, alpha_max=2)
    canonical = afdm_waveform(c1=canonical_c1)
    rows: list[dict[str, object]] = []
    for delay in (0, 1, 2, 4):
        for doppler in (-2, -1, 0, 1, 2):
            effective = effective_operator(doppler_operator(doppler) @ fractional_delay_operator(delay), canonical)
            dominant = torch.argmax(torch.abs(effective), dim=-1)
            observed = (dominant - torch.arange(config.N)) % config.N
            expected = expected_afdm_column_offset(delay, doppler, config.N, canonical_c1)
            rows.append(
                {
                    "delay": delay,
                    "doppler": doppler,
                    "expected_offset": expected,
                    "observed_offsets": ";".join(map(str, torch.unique(observed).tolist())),
                    "all_rows_match": bool(torch.all(observed == expected)),
                    "minimum_dominant_magnitude": value(torch.min(torch.max(torch.abs(effective), dim=-1).values)),
                }
            )
            if (delay, doppler) in ((0, -2), (1, 0), (2, 2), (4, -1)):
                heatmap(
                    effective,
                    f"AFDM canonical path: delay={delay}, Doppler={doppler}",
                    OUT / "afdm_sanity" / f"delay_{delay}_doppler_{doppler}.png",
                )
    return {
        "direct_idaft_vs_matrix_max_abs": direct_error,
        "k3_tuned_c1": config.AFDM_C1,
        "canonical_grid_c1": canonical_c1,
        "dd_v1_guarded_c1": canonical_afdm_c1(config.N, alpha_max=1, leakage_guard=1),
        "c2": config.AFDM_C2,
    }, rows


def summarize_eta3(channels: torch.Tensor, waveforms: dict[str, torch.Tensor], regime: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        eta, _ = retained_energy(effective_operator(channels, waveform), 3)
        rows.append(
            {
                "regime": regime,
                "waveform": name,
                "eta3_mean": value(torch.mean(eta)),
                "eta3_median": value(torch.median(eta)),
                "eta3_p10": value(torch.quantile(eta, 0.10)),
                "eta3_p90": value(torch.quantile(eta, 0.90)),
            }
        )
    return rows


def plot_regime_heatmaps(regimes: dict[str, torch.Tensor], waveforms: dict[str, torch.Tensor]) -> None:
    for regime, channels in regimes.items():
        for name in PRINCIPAL:
            effective = effective_operator(channels[0], waveforms[name])
            heatmap(effective, f"{regime}: {name}", OUT / "channel_regimes" / f"{regime}_{name}.png")


def plot_reference(rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name in PRINCIPAL:
        selected = [row for row in rows if row["waveform"] == name]
        plotted = [max(float(row["ber"]), 0.5 / int(row["bit_count"])) for row in selected]
        axis.plot([row["snr_db"] for row in selected], plotted, marker="o", color=COLORS[name], label=name)
    axis.set_yscale("log")
    axis.set_xlabel(r"$E_s/N_0$ (dB)")
    axis.set_ylabel("Bit error rate")
    axis.set_title("N=256 classical reference — full exact LMMSE")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "classical_reference_full_lmmse.png", dpi=180)
    plt.close(fig)


def plot_doppler_severity(sim_rows: list[dict[str, object]], eta_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for axis, snr_db in zip(axes, (15, 20)):
        for name in PRINCIPAL:
            selected = [row for row in sim_rows if row["waveform"] == name and row["snr_db"] == snr_db and row["k"] == 3]
            axis.plot([row["nu_max"] for row in selected], [row["ber"] for row in selected], marker="o", color=COLORS[name], label=name)
        axis.set_yscale("log")
        axis.set_xlabel(r"Audit Doppler support $\nu_{max}$")
        axis.set_title(f"K=3 at {snr_db} dB")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Bit error rate")
    axes[0].legend()
    fig.suptitle("Continuous-delay Doppler-severity audit")
    fig.tight_layout()
    fig.savefig(OUT / "doppler_severity_ber_k3.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for name in PRINCIPAL:
        selected = [row for row in eta_rows if row["waveform"] == name]
        axis.plot([row["nu_max"] for row in selected], [row["eta3_mean"] for row in selected], marker="o", color=COLORS[name], label=name)
    axis.set_xlabel(r"Audit Doppler support $\nu_{max}$")
    axis.set_ylabel(r"Mean retained top-3 energy $\eta_3$")
    axis.set_ylim(0, 1.02)
    axis.set_title("Continuous-delay representation concentration")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "doppler_severity_retained_energy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    dd_v1 = sample_dd_v1(config.N_STAGE0_EVAL_CHANNELS, config.STAGE0_EVAL_SEED).matrices
    waveforms = waveforms_64()

    full_k3 = simulate(dd_v1, waveforms, FULL_SNRS, (3, config.N), 16, config.STAGE0_EVAL_SEED + 500, "DD-v1")
    write_csv("ber_full_vs_k3.csv", full_k3)
    plot_ber_full_vs_k3(full_k3)

    budget = simulate(dd_v1, waveforms, (15, 20), K_VALUES, 4, config.STAGE0_EVAL_SEED + 501, "DD-v1")
    write_csv("receiver_budget_sweep.csv", budget)
    plot_budget(budget, 15)
    plot_budget(budget, 20)

    concentration, rowwise = concentration_diagnostics(dd_v1, waveforms)
    mismatch = mismatch_diagnostics(dd_v1, waveforms)
    write_csv("retained_energy.csv", concentration)
    write_csv("rowwise_top3_concentration.csv", rowwise)
    write_csv("high_snr_mismatch.csv", mismatch)
    plot_concentration(concentration)

    otfs_equivalence, otfs_rows = otfs_audit()
    afdm_equivalence, afdm_rows = afdm_audit()
    write_csv("otfs_structural_sanity.csv", otfs_rows)
    write_csv("afdm_path_locations.csv", afdm_rows)

    regime_batches = {
        "GRID-DD": sample_grid_dd(256, config.STAGE0_EVAL_SEED + 600).matrices,
        "FRAC-DOPPLER": sample_fractional_doppler(256, config.STAGE0_EVAL_SEED + 601).matrices,
        "DD-v1": dd_v1[:256],
    }
    compatibility_sim: list[dict[str, object]] = []
    compatibility_eta: list[dict[str, object]] = []
    for index, (regime, channels) in enumerate(regime_batches.items()):
        compatibility_sim.extend(simulate(channels, waveforms, (15, 20), (3, config.N), 4, config.STAGE0_EVAL_SEED + 610 + index, regime))
        compatibility_eta.extend(summarize_eta3(channels, waveforms, regime))
    write_csv("channel_compatibility_metrics.csv", compatibility_sim)
    write_csv("channel_compatibility_retained_energy.csv", compatibility_eta)
    plot_regime_heatmaps(regime_batches, waveforms)

    afdm_variants = {
        "AFDM_K3_TUNED": afdm_k3_tuned_waveform(),
        "AFDM_CANONICAL_GUARDED": afdm_canonical_waveform(
            config.N,
            alpha_max=config.AFDM_DDV1_CANONICAL_ALPHA_MAX,
            leakage_guard=config.AFDM_DDV1_LEAKAGE_GUARD,
        ),
    }
    afdm_variant_sim = simulate(dd_v1, afdm_variants, (15, 20), (3, config.N), 8, config.STAGE0_EVAL_SEED + 620, "DD-v1")
    afdm_variant_eta: list[dict[str, object]] = []
    for name, waveform in afdm_variants.items():
        effective = effective_operator(dd_v1, waveform)
        for k in K_VALUES:
            eta, _ = retained_energy(effective, k)
            truncated = rowwise_topk(effective, k)
            singular = torch.linalg.svdvals(truncated)
            finite_condition = singular[..., 0] / singular[..., -1]
            afdm_variant_eta.append(
                {
                    "waveform": name,
                    "k": k,
                    "eta_mean": value(torch.mean(eta)),
                    "eta_median": value(torch.median(eta)),
                    "condition_median": value(torch.median(finite_condition[torch.isfinite(finite_condition)])),
                }
            )
    write_csv("afdm_variant_metrics.csv", afdm_variant_sim)
    write_csv("afdm_variant_retained_energy.csv", afdm_variant_eta)

    doppler_sim: list[dict[str, object]] = []
    doppler_eta: list[dict[str, object]] = []
    for index, nu_max in enumerate((0.25, 0.75, 1.5, 2.0, 4.0)):
        channels = sample_continuous_dd(256, config.STAGE0_EVAL_SEED + 700 + index, nu_max).matrices
        current = simulate(channels, {name: waveforms[name] for name in PRINCIPAL}, (15, 20), (3, config.N), 4, config.STAGE0_EVAL_SEED + 710 + index, f"nu_max={nu_max}")
        for row in current:
            row["nu_max"] = nu_max
        doppler_sim.extend(current)
        eta_current = summarize_eta3(channels, {name: waveforms[name] for name in PRINCIPAL}, f"nu_max={nu_max}")
        for row in eta_current:
            row["nu_max"] = nu_max
        doppler_eta.extend(eta_current)
    write_csv("doppler_severity_metrics.csv", doppler_sim)
    write_csv("doppler_severity_retained_energy.csv", doppler_eta)
    plot_doppler_severity(doppler_sim, doppler_eta)

    reference_channels = sample_classical_reference(64, config.STAGE0_EVAL_SEED + 800).matrices
    reference_waveforms = {
        "OFDM": ofdm_waveform(256),
        "OTFS": otfs_waveform(16, 16),
        "AFDM": afdm_canonical_waveform(256, alpha_max=2, c2=math.sqrt(2) / (100 * 256)),
    }
    reference = simulate(reference_channels, reference_waveforms, (0, 5, 10, 15, 20, 25, 30), (256,), 32, config.STAGE0_EVAL_SEED + 801, "N256-reference")
    write_csv("classical_reference_full_lmmse.csv", reference)
    plot_reference(reference)

    metadata = {
        "runtime_seconds": time.perf_counter() - start,
        "full_k3_bits_per_waveform_snr": 512 * 16 * 64 * 2,
        "reference_bits_per_waveform_snr": 64 * 32 * 256 * 2,
        "otfs": otfs_equivalence,
        "afdm": afdm_equivalence,
        "configuration": {
            "dd_v1_channels": 512,
            "compatibility_channels_per_regime": 256,
            "doppler_channels_per_range": 256,
            "reference_channels": 64,
            "reference_n": 256,
        },
    }
    (OUT / "audit_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
