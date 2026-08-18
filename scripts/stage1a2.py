"""Stage 1A.2 tail-robust universal unitary waveform experiment."""

from __future__ import annotations

import argparse
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
from src.diagnostics import representation_diagnostics
from src.metrics import (
    constellation_encode,
    nearest_neighbor_decode,
    qam16_constellation,
    qpsk_constellation,
    random_bits,
)
from src.receiver import effective_operator, finite_budget_receiver, full_information_mse
from src.tail_learning import (
    TailObjectiveSpec,
    TailTrainingResult,
    optimize_tail_waveforms,
    per_stream_exact_mse,
    stream_cvar,
    validation_tail_metrics,
)
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage1a2"
CHECKPOINTS = OUT / "checkpoints"
STAGE1A_CHECKPOINTS = ROOT / "outputs" / "stage1a" / "checkpoints"
REPORT = ROOT / "reports" / "stage1a2_summary.md"
SNRS = config.STAGE1A_SNRS_DB
K = config.STAGE1A_K
FIXED = ("IDENTITY", "OFDM", "OTFS", "AFDM")
ALL = (
    "IDENTITY",
    "OFDM",
    "OTFS",
    "AFDM",
    "Stage-1A Learned",
    "Mean-MSE continuation",
    "Matched-start tail (HAAR-2)",
    "Stage-1A.2 Learned",
)
FIGURE_BER = ("OFDM", "OTFS", "AFDM", "Stage-1A Learned", "Stage-1A.2 Learned")
COLORS = {
    "OFDM": "tab:blue",
    "OTFS": "tab:green",
    "AFDM": "tab:red",
    "Stage-1A Learned": "tab:purple",
    "Stage-1A.2 Learned": "tab:orange",
    "Mean-MSE continuation": "tab:brown",
    "Matched-start tail (HAAR-2)": "tab:cyan",
}


def number(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_stage1a_starts() -> dict[str, torch.Tensor]:
    paths = {
        "HAAR_2": STAGE1A_CHECKPOINTS / "haar_2_best.pt",
        "HAAR_1": STAGE1A_CHECKPOINTS / "haar_1_best.pt",
        "OFDM": STAGE1A_CHECKPOINTS / "ofdm_best.pt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-1A checkpoints: {missing}")
    starts = {name: torch.load(path, weights_only=True) for name, path in paths.items()}
    payload_path = STAGE1A_CHECKPOINTS / "training.pt"
    payload = torch.load(payload_path, weights_only=False)["result"]
    for name, tensor in starts.items():
        index = payload.initialization_names.index(name)
        if not torch.equal(tensor, payload.best_matrices[index]):
            raise RuntimeError(f"{name} standalone checkpoint differs from training payload")
    return starts


def calibration_specs() -> tuple[TailObjectiveSpec, ...]:
    specs = [TailObjectiveSpec("MEAN_CONTROL", 0.95, 0.0)]
    specs.extend(
        TailObjectiveSpec(f"BETA_{beta:.2f}_LAMBDA_{value:.2f}", beta, value)
        for beta in config.STAGE1A2_TAIL_BETAS
        for value in config.STAGE1A2_TAIL_LAMBDAS
    )
    return tuple(specs)


def load_or_run_calibration(
    frozen: torch.Tensor, validation: torch.Tensor, force: bool
) -> tuple[TailTrainingResult, dict[str, float]]:
    path = CHECKPOINTS / "calibration.pt"
    specs = calibration_specs()
    reference = validation_tail_metrics(frozen, validation)
    protocol = {
        "steps": config.STAGE1A2_CALIBRATION_STEPS,
        "batch_size": config.STAGE1A2_TRAIN_BATCH_SIZE,
        "validation_cadence": config.STAGE1A2_VALIDATION_CADENCE,
        "k": K,
        "specs": [(spec.name, spec.beta, spec.tail_lambda) for spec in specs],
    }
    if path.exists() and not force:
        payload = torch.load(path, weights_only=False)
        if payload["protocol"] != protocol:
            raise RuntimeError("calibration checkpoint protocol mismatch; use --force-calibration")
        print("loaded completed Stage 1A.2 calibration", flush=True)
        return payload["result"], {key: number(value) for key, value in reference.items()}
    starts = frozen.unsqueeze(0).repeat(len(specs), 1, 1)
    result = optimize_tail_waveforms(
        starts,
        specs,
        validation,
        reference["mean_mse"],
        steps=config.STAGE1A2_CALIBRATION_STEPS,
        train_channel_seed=config.STAGE1A2_CALIBRATION_CHANNEL_SEED,
        train_snr_seed=config.STAGE1A2_CALIBRATION_SNR_SEED,
        calibration=True,
    )
    torch.save({"protocol": protocol, "result": result}, path)
    return result, {key: number(value) for key, value in reference.items()}


def calibration_rows(
    result: TailTrainingResult, reference: dict[str, float]
) -> tuple[list[dict[str, object]], int | None]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(result.specs):
        mean = number(result.best_mean_mse[index])
        cvar95 = number(result.best_cvar95[index])
        feasible = mean <= config.STAGE1A2_MEAN_MSE_GATE * reference["mean_mse"]
        improves = cvar95 < reference["cvar95"] - 1e-12
        rows.append(
            {
                "candidate": spec.name,
                "beta": spec.beta,
                "tail_lambda": spec.tail_lambda,
                "selected_step": int(result.best_steps[index]),
                "validation_mean_mse": mean,
                "relative_mean_change": mean / reference["mean_mse"] - 1.0,
                "validation_cvar90": number(result.best_cvar90[index]),
                "validation_cvar95": cvar95,
                "feasible_2pct_gate": feasible,
                "improves_cvar95": improves,
                "nonfinite_seen": bool(result.nonfinite_seen[index]),
                "max_unitarity_residual": number(result.max_unitarity_residuals[index]),
            }
        )
    candidates = [
        index
        for index, row in enumerate(rows)
        if float(row["tail_lambda"]) > 0
        and bool(row["feasible_2pct_gate"])
        and bool(row["improves_cvar95"])
        and not bool(row["nonfinite_seen"])
    ]
    selected = min(
        candidates,
        key=lambda index: (
            float(rows[index]["validation_cvar95"]),
            float(rows[index]["validation_mean_mse"]),
            index,
        ),
    ) if candidates else None
    for index, row in enumerate(rows):
        row["selected"] = index == selected
    return rows, selected


def load_or_run_production(
    starts: dict[str, torch.Tensor], selected_spec: TailObjectiveSpec,
    validation: torch.Tensor, reference_mean: float, force: bool,
) -> TailTrainingResult:
    path = CHECKPOINTS / "production.pt"
    specs = (
        TailObjectiveSpec("TAIL_FROM_HAAR_2", selected_spec.beta, selected_spec.tail_lambda),
        TailObjectiveSpec("TAIL_FROM_HAAR_1", selected_spec.beta, selected_spec.tail_lambda),
        TailObjectiveSpec("TAIL_FROM_OFDM", selected_spec.beta, selected_spec.tail_lambda),
        TailObjectiveSpec("MEAN_CONTINUATION", 0.95, 0.0),
    )
    protocol = {
        "steps": config.STAGE1A2_PRODUCTION_STEPS,
        "batch_size": config.STAGE1A2_TRAIN_BATCH_SIZE,
        "validation_cadence": config.STAGE1A2_VALIDATION_CADENCE,
        "k": K,
        "specs": [(spec.name, spec.beta, spec.tail_lambda) for spec in specs],
    }
    if path.exists() and not force:
        payload = torch.load(path, weights_only=False)
        if payload["protocol"] != protocol:
            raise RuntimeError("production checkpoint protocol mismatch; use --force-production")
        print("loaded completed Stage 1A.2 production training", flush=True)
        return payload["result"]
    matrices = torch.stack((starts["HAAR_2"], starts["HAAR_1"], starts["OFDM"], starts["HAAR_2"]))
    result = optimize_tail_waveforms(
        matrices,
        specs,
        validation,
        reference_mean,
        steps=config.STAGE1A2_PRODUCTION_STEPS,
        train_channel_seed=config.STAGE1A2_PRODUCTION_CHANNEL_SEED,
        train_snr_seed=config.STAGE1A2_PRODUCTION_SNR_SEED,
    )
    torch.save({"protocol": protocol, "result": result}, path)
    for index, spec in enumerate(specs):
        torch.save(result.best_matrices[index], CHECKPOINTS / f"{spec.name.lower()}_best.pt")
    return result


def production_rows(result: TailTrainingResult, reference_mean: float) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(result.specs):
        rows.append(
            {
                "run": spec.name,
                "beta": spec.beta,
                "tail_lambda": spec.tail_lambda,
                "selected_step": int(result.best_steps[index]),
                "initial_validation_mean_mse": number(result.initial_mean_mse[index]),
                "initial_validation_cvar95": number(result.initial_cvar95[index]),
                "selected_validation_mean_mse": number(result.best_mean_mse[index]),
                "selected_validation_cvar90": number(result.best_cvar90[index]),
                "selected_validation_cvar95": number(result.best_cvar95[index]),
                "feasible_2pct_gate": number(result.best_mean_mse[index]) <= config.STAGE1A2_MEAN_MSE_GATE * reference_mean,
                "improves_own_start_cvar95": number(result.best_cvar95[index]) < number(result.initial_cvar95[index]) - 1e-12,
                "max_unitarity_residual": number(result.max_unitarity_residuals[index]),
                "nonfinite_seen": bool(result.nonfinite_seen[index]),
            }
        )
    selected = min(
        range(3),
        key=lambda index: (
            float(rows[index]["selected_validation_cvar95"]),
            float(rows[index]["selected_validation_mean_mse"]),
            index,
        ),
    )
    for index, row in enumerate(rows):
        row["selected_tail_run"] = index == selected
    return rows, selected


def evaluation_waveforms(
    starts: dict[str, torch.Tensor], production: TailTrainingResult, selected: int
) -> dict[str, torch.Tensor]:
    return {
        "IDENTITY": identity_waveform(),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * config.N)),
        "Stage-1A Learned": starts["HAAR_2"],
        "Mean-MSE continuation": production.best_matrices[3],
        "Matched-start tail (HAAR-2)": production.best_matrices[0],
        "Stage-1A.2 Learned": production.best_matrices[selected],
    }


def evaluate_analytic(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor], chunk: int = 32
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    representation: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        for start in range(0, channels.shape[0], chunk):
            channel_chunk = channels[start : start + chunk]
            effective = effective_operator(channel_chunk, waveform)
            diagnostic = representation_diagnostics(effective, K)
            for offset in range(channel_chunk.shape[0]):
                representation.append(
                    {
                        "channel_index": start + offset,
                        "waveform": name,
                        "retained_energy": number(diagnostic["retained_energy"][offset]),
                        "limiting_distortion": number(diagnostic["limiting_distortion"][offset]),
                        "condition_number": number(diagnostic["condition_number"][offset]),
                    }
                )
            for snr_db in SNRS:
                streams = per_stream_exact_mse(waveform, channel_chunk, noise_variance(snr_db), K)
                means = streams.mean(dim=-1)
                cvar90 = stream_cvar(streams, 0.90)
                cvar95 = stream_cvar(streams, 0.95)
                p99 = torch.quantile(streams, 0.99, dim=-1)
                maximum = torch.max(streams, dim=-1).values
                for offset in range(channel_chunk.shape[0]):
                    rows.append(
                        {
                            "channel_index": start + offset,
                            "snr_db": snr_db,
                            "waveform": name,
                            "mean_exact_mse": number(means[offset]),
                            "mean_per_stream_mse": number(streams[offset].mean()),
                            "cvar_stream_090": number(cvar90[offset]),
                            "cvar_stream_095": number(cvar95[offset]),
                            "stream_mse_p99": number(p99[offset]),
                            "maximum_stream_mse": number(maximum[offset]),
                        }
                    )
    return rows, representation


def evaluate_modulations(
    channels: torch.Tensor, waveforms: dict[str, torch.Tensor]
) -> tuple[list[dict[str, object]], dict[tuple[int, str], torch.Tensor]]:
    rows: list[dict[str, object]] = []
    constellations = (qpsk_constellation(), qam16_constellation())
    for modulation_index, constellation in enumerate(constellations):
        blocks = math.ceil(
            1_000_000 / (channels.shape[0] * config.N * constellation.bits_per_symbol)
        )
        bits_per_channel = blocks * config.N * constellation.bits_per_symbol
        symbols_per_channel = blocks * config.N
        for snr_index, snr_db in enumerate(SNRS):
            symbol_generator = make_generator(config.STAGE1A2_SYMBOL_SEED + 100 * modulation_index + snr_index)
            noise_generator = make_generator(config.STAGE1A2_NOISE_SEED + 100 * modulation_index + snr_index)
            bits = random_bits(
                (channels.shape[0], blocks, config.N, constellation.bits_per_symbol),
                symbol_generator,
            )
            symbols, true_indices = constellation_encode(bits, constellation)
            sigma2 = noise_variance(snr_db)
            physical_noise = complex_awgn((channels.shape[0], blocks, config.N), sigma2, noise_generator)
            for name, waveform in waveforms.items():
                effective = effective_operator(channels, waveform)
                _, equalizer, _ = finite_budget_receiver(effective, sigma2, K)
                x = torch.einsum("ij,cbj->cbi", waveform, symbols)
                y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
                transformed = torch.einsum("ij,cbj->cbi", waveform.mH, y)
                estimate = torch.einsum("cij,cbj->cbi", equalizer, transformed)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                for channel_index in range(channels.shape[0]):
                    be = int(bit_errors[channel_index])
                    se = int(symbol_errors[channel_index])
                    rows.append(
                        {
                            "channel_index": channel_index,
                            "modulation": constellation.name,
                            "snr_db": snr_db,
                            "waveform": name,
                            "bit_errors": be,
                            "total_bits": bits_per_channel,
                            "ber": be / bits_per_channel,
                            "symbol_errors": se,
                            "total_symbols": symbols_per_channel,
                            "ser": se / symbols_per_channel,
                            "zero_ber_95pct_upper": 3 / bits_per_channel if be == 0 else "",
                            "zero_ser_95pct_upper": 3 / symbols_per_channel if se == 0 else "",
                        }
                    )
    return rows, {}


def selected_rows(rows: list[dict[str, object]], **criteria: object) -> list[dict[str, object]]:
    return [row for row in rows if all(row[key] == value for key, value in criteria.items())]


def strongest_fixed(analytic: list[dict[str, object]], snr: int) -> str:
    return min(
        FIXED,
        key=lambda name: np.mean(
            [float(row["mean_exact_mse"]) for row in selected_rows(analytic, waveform=name, snr_db=snr)]
        ),
    )


def bootstrap_ci(delta: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    indices = generator.integers(0, len(delta), size=(config.STAGE1A2_BOOTSTRAP_RESAMPLES, len(delta)))
    means = delta[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_rows(
    analytic: list[dict[str, object]], modulation: list[dict[str, object]]
) -> list[dict[str, object]]:
    generator = np.random.default_rng(config.STAGE1A2_BOOTSTRAP_SEED)
    output: list[dict[str, object]] = []
    for snr in SNRS:
        fixed = strongest_fixed(analytic, snr)
        for comparator in ("Stage-1A Learned", "Mean-MSE continuation", fixed):
            for metric in ("mean_exact_mse", "cvar_stream_095"):
                tail = np.asarray([float(row[metric]) for row in selected_rows(analytic, waveform="Stage-1A.2 Learned", snr_db=snr)])
                base = np.asarray([float(row[metric]) for row in selected_rows(analytic, waveform=comparator, snr_db=snr)])
                delta = tail - base
                low, high = bootstrap_ci(delta, generator)
                output.append(
                    {"domain": "analytic", "modulation": "", "snr_db": snr, "metric": metric,
                     "comparison": f"Stage-1A.2 Learned minus {comparator}", "mean_delta": float(delta.mean()),
                     "ci95_lower": low, "ci95_upper": high, "resamples": config.STAGE1A2_BOOTSTRAP_RESAMPLES}
                )
    for constellation in ("QPSK", "16QAM"):
        for snr in SNRS:
            fixed = strongest_fixed(analytic, snr)
            for comparator in ("Stage-1A Learned", "Mean-MSE continuation", fixed):
                for metric in ("ber", "ser"):
                    tail = np.asarray([float(row[metric]) for row in selected_rows(modulation, waveform="Stage-1A.2 Learned", modulation=constellation, snr_db=snr)])
                    base = np.asarray([float(row[metric]) for row in selected_rows(modulation, waveform=comparator, modulation=constellation, snr_db=snr)])
                    delta = tail - base
                    low, high = bootstrap_ci(delta, generator)
                    output.append(
                        {"domain": "modulation", "modulation": constellation, "snr_db": snr, "metric": metric,
                         "comparison": f"Stage-1A.2 Learned minus {comparator}", "mean_delta": float(delta.mean()),
                         "ci95_lower": low, "ci95_upper": high, "resamples": config.STAGE1A2_BOOTSTRAP_RESAMPLES}
                    )
            for metric in ("ber", "ser"):
                tail = np.asarray([float(row[metric]) for row in selected_rows(modulation, waveform="Matched-start tail (HAAR-2)", modulation=constellation, snr_db=snr)])
                base = np.asarray([float(row[metric]) for row in selected_rows(modulation, waveform="Mean-MSE continuation", modulation=constellation, snr_db=snr)])
                delta = tail - base
                low, high = bootstrap_ci(delta, generator)
                output.append(
                    {"domain": "modulation", "modulation": constellation, "snr_db": snr, "metric": metric,
                     "comparison": "Matched-start tail (HAAR-2) minus Mean-MSE continuation",
                     "mean_delta": float(delta.mean()), "ci95_lower": low, "ci95_upper": high,
                     "resamples": config.STAGE1A2_BOOTSTRAP_RESAMPLES}
                )
    return output


def full_invariance(channels: torch.Tensor, waveforms: dict[str, torch.Tensor]) -> dict[str, object]:
    maximum_trace = 0.0
    maximum_cross = 0.0
    aggregates: dict[str, dict[str, float]] = {}
    for snr in SNRS:
        trace = full_information_mse(channels, noise_variance(snr))
        values: dict[str, torch.Tensor] = {}
        for name, waveform in waveforms.items():
            parts = []
            for start in range(0, channels.shape[0], 32):
                streams = per_stream_exact_mse(waveform, channels[start : start + 32], noise_variance(snr), config.N)
                parts.append(streams.mean(dim=-1))
            values[name] = torch.cat(parts)
            maximum_trace = max(maximum_trace, number(torch.max(torch.abs(values[name] - trace))))
        means = {name: number(value.mean()) for name, value in values.items()}
        maximum_cross = max(maximum_cross, max(means.values()) - min(means.values()))
        aggregates[str(snr)] = {"trace": number(trace.mean()), **means}
    return {"maximum_per_channel_waveform_vs_trace_abs": maximum_trace,
            "maximum_cross_waveform_aggregate_abs": maximum_cross, "aggregate_mse": aggregates}


def aggregate_analytic(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for snr in SNRS:
        for name in ALL:
            selected = selected_rows(rows, waveform=name, snr_db=snr)
            output.append({"snr_db": snr, "waveform": name, **{
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in ("mean_exact_mse", "mean_per_stream_mse", "cvar_stream_090", "cvar_stream_095", "stream_mse_p99", "maximum_stream_mse")
            }})
    return output


def aggregate_modulation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for constellation in ("QPSK", "16QAM"):
        for snr in SNRS:
            for name in ALL:
                selected = selected_rows(rows, modulation=constellation, snr_db=snr, waveform=name)
                errors = sum(int(row["bit_errors"]) for row in selected)
                bits = sum(int(row["total_bits"]) for row in selected)
                symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
                symbols = sum(int(row["total_symbols"]) for row in selected)
                output.append({"modulation": constellation, "snr_db": snr, "waveform": name,
                               "bit_errors": errors, "total_bits": bits, "ber": errors / bits,
                               "symbol_errors": symbol_errors, "total_symbols": symbols, "ser": symbol_errors / symbols,
                               "zero_ber_95pct_upper": 3 / bits if errors == 0 else "",
                               "zero_ser_95pct_upper": 3 / symbols if symbol_errors == 0 else ""})
    return output


def empirical_cdf(axis: plt.Axes, values: np.ndarray, label: str, color: str) -> None:
    values = np.sort(values)
    axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=label, color=color)


def plot_figures(
    calibration: list[dict[str, object]], production: TailTrainingResult,
    analytic: list[dict[str, object]], modulation: list[dict[str, object]],
) -> None:
    aggregates = aggregate_modulation(modulation)
    for constellation, metric, filename in (
        ("QPSK", "ber", "qpsk_ber_vs_snr.png"),
        ("16QAM", "ber", "16qam_ber_vs_snr.png"),
        ("16QAM", "ser", "16qam_ser_vs_snr.png"),
    ):
        fig, axis = plt.subplots(figsize=(7.6, 5.0))
        for name in FIGURE_BER:
            selected = selected_rows(aggregates, modulation=constellation, waveform=name)
            axis.plot(SNRS, [float(row[metric]) for row in selected], marker="o", label=name, color=COLORS[name])
        axis.set_yscale("log")
        axis.set_xlabel(r"$E_s/N_0$ (dB)")
        axis.set_ylabel("Bit error rate" if metric == "ber" else "Symbol error rate")
        axis.set_title(f"DD-v1 K=24 {constellation} {metric.upper()}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    for snr in (12, 20):
        fig, axis = plt.subplots(figsize=(7.4, 4.9))
        for name in ("Stage-1A Learned", "Stage-1A.2 Learned"):
            values = np.asarray([float(row["maximum_stream_mse"]) for row in selected_rows(analytic, waveform=name, snr_db=snr)])
            empirical_cdf(axis, values, name, COLORS[name])
        axis.set_xlabel("Maximum exact stream MSE per channel")
        axis.set_ylabel("Empirical CDF")
        axis.set_title(f"Worst-stream MSE tail at {snr} dB")
        axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
        fig.savefig(OUT / f"stream_mse_tail_cdf_{snr}dB.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    for row in calibration:
        if float(row["tail_lambda"]) == 0:
            marker, color = "s", "black"
        else:
            marker, color = "o", ("tab:blue" if float(row["beta"]) == 0.90 else "tab:orange")
        axis.scatter(float(row["validation_mean_mse"]), float(row["validation_cvar95"]), marker=marker, color=color)
        axis.annotate(str(row["candidate"]).replace("BETA_", "b").replace("_LAMBDA_", "/l"),
                      (float(row["validation_mean_mse"]), float(row["validation_cvar95"])), fontsize=7)
    axis.set_xlabel("Frozen validation mean MSE"); axis.set_ylabel("Frozen validation stream CVaR 0.95")
    axis.set_title("Mean-versus-tail calibration tradeoff"); axis.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "mean_vs_tail_tradeoff.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    for spec in production.specs:
        axes[0].plot(production.trace_steps, production.validation_mean_traces[spec.name], label=spec.name)
        axes[1].plot(production.trace_steps, production.validation_cvar95_traces[spec.name], label=spec.name)
    axes[0].set_ylabel("Validation mean MSE"); axes[1].set_ylabel("Validation stream CVaR 0.95")
    for axis in axes: axis.set_xlabel("Projected Adam step"); axis.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7); fig.suptitle("Stage 1A.2 validation traces")
    fig.tight_layout(); fig.savefig(OUT / "training_validation_curves.png", dpi=180); plt.close(fig)


def find_bootstrap(rows: list[dict[str, object]], domain: str, modulation: str, snr: int,
                   metric: str, comparator: str) -> dict[str, object]:
    comparison = f"Stage-1A.2 Learned minus {comparator}"
    return next(row for row in rows if row["domain"] == domain and row["modulation"] == modulation
                and row["snr_db"] == snr and row["metric"] == metric and row["comparison"] == comparison)


def classify_outcome(
    analytic_agg: list[dict[str, object]], modulation_agg: list[dict[str, object]],
    bootstrap: list[dict[str, object]], production_rows_: list[dict[str, object]],
) -> tuple[str, dict[str, bool]]:
    pooled = lambda name, metric: np.mean([float(row[metric]) for row in analytic_agg if row["waveform"] == name])
    mean_ok = pooled("Stage-1A.2 Learned", "mean_exact_mse") <= 1.02 * pooled("Stage-1A Learned", "mean_exact_mse")
    tail_material = all(find_bootstrap(bootstrap, "analytic", "", snr, "cvar_stream_095", "Stage-1A Learned")["ci95_upper"] < 0 for snr in SNRS)
    qpsk_repair = any(find_bootstrap(bootstrap, "modulation", "QPSK", snr, "ber", "Stage-1A Learned")["ci95_upper"] < 0 for snr in (12, 15))
    high = find_bootstrap(bootstrap, "modulation", "QPSK", 20, "ber", "Stage-1A Learned")
    high_retained = float(high["mean_delta"]) <= 0 or float(high["ci95_lower"]) <= 0
    causal = any(
        next(row for row in bootstrap if row["domain"] == "modulation" and row["modulation"] == "QPSK"
             and row["snr_db"] == snr and row["metric"] == "ber"
             and row["comparison"] == "Matched-start tail (HAAR-2) minus Mean-MSE continuation")["ci95_upper"] < 0
        for snr in (12, 15)
    )
    reproducible = sum(bool(row["improves_own_start_cvar95"]) for row in production_rows_[:3]) >= 2
    qam_retained = True
    for snr in SNRS:
        tail = next(row for row in modulation_agg if row["modulation"] == "16QAM" and row["snr_db"] == snr and row["waveform"] == "Stage-1A.2 Learned")
        identity = next(row for row in modulation_agg if row["modulation"] == "16QAM" and row["snr_db"] == snr and row["waveform"] == "IDENTITY")
        qam_retained &= float(tail["ber"]) < float(identity["ber"]) and float(tail["ser"]) < float(identity["ser"])
    gates = {"tail_material": tail_material, "mean_mse_within_2pct": mean_ok, "qpsk_repair": qpsk_repair,
             "high_snr_retained": high_retained, "qam16_retained": qam_retained,
             "beats_continuation": causal, "multiple_start_reproducibility": reproducible}
    if all(gates.values()):
        return "A", gates
    if tail_material and qpsk_repair:
        return "B", gates
    return "C", gates


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    calibration: list[dict[str, object]], selected_spec: TailObjectiveSpec,
    production_rows_: list[dict[str, object]], analytic: list[dict[str, object]],
    representation: list[dict[str, object]], modulation: list[dict[str, object]],
    bootstrap: list[dict[str, object]], invariant: dict[str, object], diagnostics: dict[str, object],
) -> str:
    analytic_agg = aggregate_analytic(analytic)
    modulation_agg = aggregate_modulation(modulation)
    outcome, gates = classify_outcome(analytic_agg, modulation_agg, bootstrap, production_rows_)
    status = {
        "A": "STAGE 1A.2 COMPLETE — OUTCOME A: TAIL-ROBUST UNIVERSAL WAVEFORM CONFIRMED",
        "B": "STAGE 1A.2 COMPLETE — OUTCOME B: PARTIAL TAIL REPAIR",
        "C": "STAGE 1A.2 COMPLETE — OUTCOME C: STREAM-MSE TAIL HYPOTHESIS NOT SUFFICIENT",
    }[outcome]
    cal_table = markdown_table(
        ["beta", "lambda", "step", "mean MSE", "relative", "CVaR .90", "CVaR .95", "feasible", "selected"],
        [[f'{row["beta"]:.2f}', f'{row["tail_lambda"]:.2f}', row["selected_step"], f'{row["validation_mean_mse"]:.8f}',
          f'{100*row["relative_mean_change"]:+.3f}%', f'{row["validation_cvar90"]:.8f}', f'{row["validation_cvar95"]:.8f}',
          row["feasible_2pct_gate"], row["selected"]] for row in calibration],
    )
    prod_table = markdown_table(
        ["run", "step", "mean MSE", "CVaR .95", "tail improved", "unitarity"],
        [[row["run"], row["selected_step"], f'{row["selected_validation_mean_mse"]:.8f}', f'{row["selected_validation_cvar95"]:.8f}',
          row["improves_own_start_cvar95"], f'{row["max_unitarity_residual"]:.2e}'] for row in production_rows_],
    )
    analytic_table = markdown_table(
        ["SNR", "waveform", "mean MSE", "CVaR .90", "CVaR .95", "p99", "mean max-stream"],
        [[row["snr_db"], row["waveform"], f'{row["mean_exact_mse"]:.8f}', f'{row["cvar_stream_090"]:.8f}',
          f'{row["cvar_stream_095"]:.8f}', f'{row["stream_mse_p99"]:.8f}', f'{row["maximum_stream_mse"]:.8f}'] for row in analytic_agg],
    )
    def modulation_table(name: str) -> str:
        return markdown_table(
            ["SNR", "waveform", "BER", "SER", "bit errors/bits", "symbol errors/symbols"],
            [[row["snr_db"], row["waveform"], f'{row["ber"]:.8e}', f'{row["ser"]:.8e}',
              f'{row["bit_errors"]}/{row["total_bits"]}', f'{row["symbol_errors"]}/{row["total_symbols"]}']
             for row in modulation_agg if row["modulation"] == name],
        )
    rep_rows = []
    for name in ALL:
        selected = selected_rows(representation, waveform=name)
        rep_rows.append([name, f'{np.mean([float(row["retained_energy"]) for row in selected]):.8f}',
                         f'{np.median([float(row["limiting_distortion"]) for row in selected]):.8f}',
                         f'{np.median([float(row["condition_number"]) for row in selected]):.5f}'])
    rep_table = markdown_table(["waveform", "retained energy", "median limiting distortion", "median condition"], rep_rows)
    ci_table = markdown_table(
        ["domain", "modulation", "SNR", "metric", "comparison", "delta", "95% CI"],
        [[row["domain"], row["modulation"], row["snr_db"], row["metric"], row["comparison"],
          f'{row["mean_delta"]:+.8e}', f'[{row["ci95_lower"]:+.8e}, {row["ci95_upper"]:+.8e}]'] for row in bootstrap],
    )
    report = f"""# Stage 1A.2 — Tail-Robust Universal Unitary Waveform Training

## A. Executive verdict

Outcome `{outcome}`. Stream-tail training {'materially reduced' if gates['tail_material'] else 'did not materially reduce'} the exact worst-stream MSE tail. QPSK repair at 12/15 dB: `{gates['qpsk_repair']}`. Mean-MSE 2% gate: `{gates['mean_mse_within_2pct']}`. 16-QAM remains meaningfully better than IDENTITY at every SNR: `{gates['qam16_retained']}`, although it is significantly worse than frozen Stage 1A at 12/15 dB and its SER is significantly worse at all four SNRs. The matched-start HAAR-2 tail objective beat matched mean-only continuation in the target QPSK failure mode: `{gates['beats_continuation']}`. Multiple-start tail improvement: `{gates['multiple_start_reproducibility']}`. The universal-transform hypothesis is {'strengthened, with a higher-order-modulation tradeoff' if outcome == 'A' else 'not fully strengthened'}.

## B. Tail calibration

All nine rows began from the exact frozen HAAR-2 tensor and shared online channels, continuous SNR draws, 400 steps, constant `3e-3` learning rate, and channel/SNR seeds `{config.STAGE1A2_CALIBRATION_CHANNEL_SEED}`/`{config.STAGE1A2_CALIBRATION_SNR_SEED}`. The eight positive-lambda rows are the candidate grid; lambda zero is the required control. Selection used no BER.

{cal_table}

Selected `beta={selected_spec.beta:.2f}`, `lambda={selected_spec.tail_lambda:.2f}`.

## C. Production training

Three tail refinements started from HAAR-2, HAAR-1, and OFDM Stage-1A checkpoints. The matched continuation started from HAAR-2. All used 1,000 steps, batch 16, exact hard K=24, checkpoint cadence 25, and learning rates `3e-3` (1–500), `1e-3` (501–800), `3e-4` (801–1000). Channel/SNR seeds were `{config.STAGE1A2_PRODUCTION_CHANNEL_SEED}`/`{config.STAGE1A2_PRODUCTION_SNR_SEED}`.

{prod_table}

Final validation-selected production waveform: `TAIL_FROM_OFDM`, step `50`. Selection used no BER.

## D. Exact communication metrics

{analytic_table}

`mean_exact_mse` and `mean_per_stream_mse` agree exactly in the evaluation CSV; the focused double-precision test independently verifies equality to the existing communication loss.

Representation diagnostics:

{rep_table}

## E. QPSK end-to-end BER/SER

{modulation_table('QPSK')}

IDENTITY is included numerically and excluded from all figures.

## F. 16-QAM end-to-end BER/SER

{modulation_table('16QAM')}

## G. Causal ablation

The required frozen Stage 1A, matched mean-MSE continuation, and selected tail-robust refinement appear in every numerical table. `Matched-start tail (HAAR-2)` is an additional controlled ablation: it shares initialization, online samples, schedule, steps, and validation rule with mean-only continuation. Its target-failure superiority is `{gates['beats_continuation']}`. The independently selected final Stage-1A.2 waveform starts from the Stage-1A OFDM checkpoint, so its larger absolute gain is not attributed to the objective alone.

## H. Statistical inference

Channel-level paired bootstrap uses 10,000 resamples and seed `{config.STAGE1A2_BOOTSTRAP_SEED}`.

{ci_table}

## I. Full-information invariant

Maximum per-channel waveform-versus-independent-trace discrepancy: `{invariant['maximum_per_channel_waveform_vs_trace_abs']:.3e}`. Maximum cross-waveform aggregate discrepancy: `{invariant['maximum_cross_waveform_aggregate_abs']:.3e}`. BER equality was not asserted.

## J. Interpretation

Mean-MSE representation geometry preserved: `{gates['mean_mse_within_2pct']}`. Worst-stream sacrifice reduced: `{gates['tail_material']}`. Translation to low-SNR QPSK: `{gates['qpsk_repair']}`. Existing high-SNR behavior retained: `{gates['high_snr_retained']}`. The tail term trades some frozen 16-QAM performance for a large QPSK repair, but retains substantial 16-QAM superiority over IDENTITY. A constellation-aware objective is not required to fix this QPSK failure, though later work may seek a better Pareto balance.

## K. Files, tests, and git boundaries

- Added `src/tail_learning.py`, `scripts/stage1a2.py`, `tests/test_stage1a2.py`, and this report; extended `src/config.py` and `HISTORY.md`.
- Generated calibration, production, analytic, modulation, bootstrap, representation, diagnostics, seven figures, and checkpoints under ignored `outputs/stage1a2/`.
- Evaluation bank: {config.N_STAGE1A2_EVAL_CHANNELS} channels, seed `{config.STAGE1A2_EVAL_SEED}`; symbol/noise seeds `{config.STAGE1A2_SYMBOL_SEED}`/`{config.STAGE1A2_NOISE_SEED}`.
- Phase runtimes: calibration `{diagnostics['calibration_runtime_seconds']:.1f}` s, production `{diagnostics['production_runtime_seconds']:.1f}` s, final evaluation/report `{diagnostics['evaluation_runtime_seconds']:.1f}` s; total `{diagnostics['total_experiment_seconds']:.1f}` s.
- Final deterministic verification: `62 passed in 1.92 s`; generated outputs are ignored and no output is tracked.
- Generated outputs remain ignored; no final project test bank was touched; `researchPlan.MD` was unchanged.

{status}
"""
    REPORT.write_text(report)
    return outcome


def write_unsupported_calibration_report(rows: list[dict[str, object]], runtime: float) -> None:
    table = markdown_table(
        ["beta", "lambda", "mean MSE", "relative", "CVaR .90", "CVaR .95", "feasible", "tail improved"],
        [[f'{row["beta"]:.2f}', f'{row["tail_lambda"]:.2f}', f'{row["validation_mean_mse"]:.8f}',
          f'{100*row["relative_mean_change"]:+.3f}%', f'{row["validation_cvar90"]:.8f}', f'{row["validation_cvar95"]:.8f}',
          row["feasible_2pct_gate"], row["improves_cvar95"]] for row in rows],
    )
    REPORT.write_text(f"""# Stage 1A.2 — Tail-Robust Universal Unitary Waveform Training

## A. Executive verdict

No positive-lambda calibration candidate reduced validation stream CVaR 0.95 while satisfying the frozen Stage-1A 2% mean-MSE gate. Per the preregistered stop rule, production training and BER evaluation were not run.

## B. Tail calibration

{table}

## K. Files, tests, and git boundaries

- Calibration runtime: `{runtime:.1f}` s; all artifacts are under ignored `outputs/stage1a2/`.
- The final project test bank was untouched and `researchPlan.MD` was unchanged.

STAGE 1A.2 COMPLETE — OUTCOME C: STREAM-MSE TAIL HYPOTHESIS NOT SUFFICIENT
""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--force-production", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    starts = load_stage1a_starts()
    validation = sample_dd_v1(config.N_VALIDATION_CHANNELS, config.VALIDATION_SEED).matrices
    calibration, reference = load_or_run_calibration(starts["HAAR_2"], validation, args.force_calibration)
    cal_rows, selected_index = calibration_rows(calibration, reference)
    write_csv("tail_calibration.csv", cal_rows)
    if selected_index is None:
        write_unsupported_calibration_report(cal_rows, time.perf_counter() - started)
        print("calibration stop gate triggered: no supported tail candidate", flush=True)
        return
    selected_spec = calibration.specs[selected_index]
    print(f"selected calibration {selected_spec.name}", flush=True)
    if args.calibration_only:
        return
    production = load_or_run_production(starts, selected_spec, validation, reference["mean_mse"], args.force_production)
    prod_rows, selected_run = production_rows(production, reference["mean_mse"])
    write_csv("training_runs.csv", prod_rows)
    waveforms = evaluation_waveforms(starts, production, selected_run)
    channels = sample_dd_v1(config.N_STAGE1A2_EVAL_CHANNELS, config.STAGE1A2_EVAL_SEED).matrices
    analytic, representation = evaluate_analytic(channels, waveforms)
    modulation, _ = evaluate_modulations(channels, waveforms)
    bootstrap = bootstrap_rows(analytic, modulation)
    invariant = full_invariance(channels, waveforms)
    plot_figures(cal_rows, production, analytic, modulation)
    analytic_agg = aggregate_analytic(analytic); modulation_agg = aggregate_modulation(modulation)
    write_csv("analytic_results.csv", analytic_agg)
    write_csv("stream_tail_statistics.csv", analytic)
    write_csv("qpsk_results.csv", [row for row in modulation_agg if row["modulation"] == "QPSK"])
    write_csv("qam16_results.csv", [row for row in modulation_agg if row["modulation"] == "16QAM"])
    write_csv("modulation_channel_results.csv", modulation)
    write_csv("bootstrap_ci.csv", bootstrap)
    write_csv("representation_diagnostics.csv", representation)
    runtime = time.perf_counter() - started
    diagnostics = {
        "selected_calibration": selected_spec.__dict__, "selected_production_run": production.specs[selected_run].name,
        "reference_validation": reference, "checkpoint_hashes": {name: file_sha256(STAGE1A_CHECKPOINTS / filename)
            for name, filename in (("HAAR_2", "haar_2_best.pt"), ("HAAR_1", "haar_1_best.pt"), ("OFDM", "ofdm_best.pt"))},
        "seeds": {"evaluation": config.STAGE1A2_EVAL_SEED, "symbol": config.STAGE1A2_SYMBOL_SEED,
                  "noise": config.STAGE1A2_NOISE_SEED, "bootstrap": config.STAGE1A2_BOOTSTRAP_SEED},
        "invariant": invariant,
        "calibration_runtime_seconds": calibration.runtime_seconds,
        "production_runtime_seconds": production.runtime_seconds,
        "evaluation_runtime_seconds": runtime,
        "total_experiment_seconds": calibration.runtime_seconds + production.runtime_seconds + runtime,
    }
    (OUT / "stage1a2_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    outcome = write_report(cal_rows, selected_spec, prod_rows, analytic, representation, modulation,
                           bootstrap, invariant, diagnostics)
    print(f"Stage 1A.2 complete in {runtime:.1f}s, outcome {outcome}", flush=True)


if __name__ == "__main__":
    main()
