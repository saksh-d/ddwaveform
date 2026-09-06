"""Resumable Pub-1E frozen-waveform qualification runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import noise_variance, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.pub1e.core import (
    aggregate, communication_rows, controlled_fractionality_2d, fixed_energy_support,
    paired_channel_interval, residual_statistics, safe_log10_ber_ratio,
    select_regularization, shifted_prior_channels, support_metrics,
)
from src.receiver import analytic_communication_mse, effective_operator, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform, otfs_waveform


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage_pub1e.json"
OUT = ROOT / "outputs/stage_pub1e"
DATA = OUT / "data"
FIG = OUT / "figures"
PROVENANCE = OUT / "provenance"
FROZEN = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
COLORS = {"Learned": "#d97706", "OTFS": "#15803d", "AFDM": "#dc2626", "OFDM": "#2563eb", "IDENTITY": "#6b7280"}
MARKERS = {"Learned": "o", "OTFS": "s", "AFDM": "^", "OFDM": "D", "IDENTITY": "x"}
CONDITIONS = (("integer", 0.0, 0.0), ("joint_0.5", 0.5, 0.5), ("nominal_dd_v1", None, None))


def protocol() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def ensure_dirs() -> None:
    for path in (DATA, FIG, PROVENANCE):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _matches(stored: object, expected: object) -> bool:
    try:
        return abs(float(stored) - float(expected)) < 1e-12
    except (TypeError, ValueError):
        return str(stored) == str(expected)


def _select_series(
    rows: list[dict[str, object]], x_key: str, expected_x: tuple[object, ...], **criteria: object,
) -> list[dict[str, object]]:
    """Select one fully keyed plotting row per expected x value."""
    selected = [row for row in rows if all(key in row and _matches(row[key], value) for key, value in criteria.items())]
    result = []
    for value in expected_x:
        matches = [row for row in selected if x_key in row and _matches(row[x_key], value)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one plotting row for {criteria}, {x_key}={value}; got {len(matches)}")
        result.append(matches[0])
    extras = [row for row in selected if not any(_matches(row[x_key], value) for value in expected_x)]
    if extras:
        raise RuntimeError(f"unexpected plotting rows for {criteria}: {len(extras)} extra {x_key} values")
    return result


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def sample_condition(name: str, count: int, seed: int) -> torch.Tensor:
    if name == "nominal_dd_v1":
        return sample_dd_v1(count, seed).matrices
    selected = next(item for item in CONDITIONS if item[0] == name)
    return controlled_fractionality_2d(count, seed, float(selected[1]), float(selected[2])).matrices


def selected_afdm() -> int:
    path = DATA / "afdm_selection.json"
    if not path.exists():
        raise RuntimeError("run --phase validate before evaluation")
    return int(json.loads(path.read_text())["selected_multiplier_2Nc1"])


def load_waveforms(*, controls: bool = False, afdm_variants: bool = False) -> dict[str, torch.Tensor]:
    learned, _ = load_frozen_waveform(FROZEN)
    values = {
        "Learned": learned, "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=selected_afdm() / (2 * config.N), c2=config.AFDM_C2),
        "OFDM": ofdm_waveform(),
    }
    if controls:
        values["IDENTITY"] = identity_waveform()
    if afdm_variants:
        values.update({
            "AFDM historical": afdm_waveform(c1=1 / (2 * config.N), c2=config.AFDM_C2),
            "AFDM canonical": afdm_waveform(c1=3 / (2 * config.N), c2=config.AFDM_C2),
            "AFDM guard": afdm_waveform(c1=5 / (2 * config.N), c2=config.AFDM_C2),
        })
    return values


def phase_provenance(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol()
    digest = file_sha256(FROZEN)
    if digest != cfg["checkpoint"]["sha256"]:
        raise RuntimeError(f"frozen checkpoint mismatch: {digest}")
    source = Path(cfg["checkpoint"]["producer_worktree"]) / cfg["checkpoint"]["relative_path"]
    source_digest = file_sha256(source)
    if source_digest != digest:
        raise RuntimeError("producer and local checkpoint differ")
    write_json(PROVENANCE / "inputs.json", {
        "run_kind": "smoke" if smoke else "production", "learned_training_steps": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target_ref": "origin/pub", "checkpoint_sha256": digest,
        "producer_path": str(source), "producer_commit": cfg["checkpoint"]["producer_commit"],
        "producer_checkpoint_sha256": source_digest, "protocol_sha256": file_sha256(CONFIG_PATH),
        "precision": "complex128/float64", "pub1d_available_in_checkout": (ROOT / "src/pub1d/core.py").exists(),
    })
    (PROVENANCE / "protocol_snapshot.json").write_text(CONFIG_PATH.read_text())


def phase_validate(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol(); count = 8 if smoke else int(cfg["banks"]["afdm_validation"]["channels"])
    channels = sample_dd_v1(count, int(cfg["banks"]["afdm_validation"]["seed"])).matrices
    candidates = tuple(int(v) for v in cfg["afdm"]["validation_candidates_2Nc1"])
    ks = tuple(int(v) for v in cfg["k_values"] if int(v) < config.N)
    snrs = tuple(int(v) for v in cfg["snr_db"])
    rows = []
    for multiplier in candidates:
        waveform = afdm_waveform(c1=multiplier / (2 * config.N), c2=config.AFDM_C2)
        actual = effective_operator(channels, waveform)
        values = []
        for k in ks:
            restricted = rowwise_topk(actual, k)
            for snr in snrs:
                from src.pub1e.core import historical_equalizer
                equalizer = historical_equalizer(restricted, noise_variance(snr))
                mse = analytic_communication_mse(actual, equalizer, noise_variance(snr))
                values.extend(float(value) for value in mse)
        rows.append({"multiplier_2Nc1": multiplier, "c1": multiplier / (2 * config.N), "mean_analytic_mse": float(np.mean(values)), "samples": len(values)})
    selected = min(rows, key=lambda row: (row["mean_analytic_mse"], row["multiplier_2Nc1"]))
    write_json(DATA / "afdm_selection.json", {
        "run_kind": "smoke" if smoke else "production", "selected_multiplier_2Nc1": selected["multiplier_2Nc1"],
        "metric": cfg["afdm"]["metric"], "tie_break": cfg["afdm"]["tie_break"], "rows": rows,
        "named_curves": {"historical": 1, "canonical": 3, "fractional_guard": 5},
        "admissibility": "odd 2Nc1 gives a unitary AFDM transform; candidates are at or above the derived alpha_max=1 canonical slope",
    })
    waveforms = load_waveforms(controls=True)
    checks = {name: float(torch.linalg.matrix_norm(value.mH @ value - torch.eye(config.N, dtype=value.dtype))) for name, value in waveforms.items()}
    write_json(DATA / "numerical_validation.json", {"unitarity_residuals": checks, "max_residual": max(checks.values()), "finite": all(np.isfinite(v) for v in checks.values())})


def _sizes(smoke: bool) -> tuple[int, int, int, int]:
    cfg = protocol()
    if smoke:
        return 8, 12, 8, 256
    return 512, int(cfg["banks"]["residual_statistics"]["channels"]), int(cfg["banks"]["receiver_validation"]["channels"]), int(cfg["bits_per_channel"])


def phase_e1(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol(); waveforms = load_waveforms(); test_n, stats_n, val_n, bits = _sizes(smoke)
    ks = tuple(int(v) for v in cfg["k_values"]); snrs = tuple(int(v) for v in cfg["snr_db"])
    all_rows: list[dict[str, object]] = []; selection_rows = []; start = time.perf_counter()
    for condition_index, (condition, _, _) in enumerate(CONDITIONS):
        test = sample_condition(condition, test_n, int(cfg["banks"]["confirmatory"]["seed"]) + 10 * condition_index)
        validation = sample_condition(condition, val_n, int(cfg["banks"]["receiver_validation"]["seed"]) + 10 * condition_index)
        stats_bank = sample_condition(condition, stats_n, int(cfg["banks"]["residual_statistics"]["seed"]) + 10 * condition_index)
        for k in ks:
            regularization = {}; statistics = {}
            for name, waveform in waveforms.items():
                scale, scores = select_regularization(validation, waveform, k, snrs, tuple(float(v) for v in cfg["receiver_ladder"]["tuned_regularization_candidates"]))
                regularization[name] = scale; statistics[name] = residual_statistics(stats_bank, waveform, k)
                selection_rows.extend({"condition": condition, "k": k, "waveform": name, "selected": scale, **row} for row in scores)
            eval_snrs = snrs if k == int(cfg["reference_k"]) else (18,)
            for family in ("historical", "tuned", "residual_aware"):
                rows = communication_rows(test, waveforms, (qam16_constellation(), qpsk_constellation()), eval_snrs, k, family, bits, 17800 + 10 * condition_index, 17900 + 10 * condition_index, regularization=regularization, statistics=statistics)
                all_rows.extend({"condition": condition, **row} for row in rows)
        rows = communication_rows(test, waveforms, (qam16_constellation(), qpsk_constellation()), snrs, config.N, "full", bits, 17800 + 10 * condition_index, 17900 + 10 * condition_index)
        all_rows.extend({"condition": condition, **row} for row in rows)
    keys = ("condition", "modulation", "snr_db", "waveform", "receiver_family", "k")
    write_csv(DATA / "e1_per_channel.csv", all_rows); write_csv(DATA / "e1_aggregate.csv", aggregate(all_rows, keys)); write_csv(DATA / "e1_receiver_validation.csv", selection_rows)
    intervals = []; seed = 18200
    for condition, _, _ in CONDITIONS:
        for modulation in ("16QAM", "QPSK"):
            for family in ("historical", "tuned", "residual_aware"):
                for k in ks:
                    learned = [r for r in all_rows if r["condition"] == condition and r["modulation"] == modulation and r["snr_db"] == 18 and r["receiver_family"] == family and r["k"] == k and r["waveform"] == "Learned"]
                    for baseline in ("AFDM", "OTFS", "OFDM"):
                        classical = [r for r in all_rows if r["condition"] == condition and r["modulation"] == modulation and r["snr_db"] == 18 and r["receiver_family"] == family and r["k"] == k and r["waveform"] == baseline]
                        intervals.append({"condition": condition, "modulation": modulation, "receiver_family": family, "k": k, "snr_db": 18, "contrast": f"Learned-{baseline}", **paired_channel_interval(learned, classical, seed, 200 if smoke else 10000)}); seed += 1
    write_csv(DATA / "e1_paired_intervals.csv", intervals)
    write_json(DATA / "e1_execution.json", {"run_kind": "smoke" if smoke else "production", "seconds": time.perf_counter() - start, "channels": test_n, "statistics_channels": stats_n, "validation_channels": val_n, "bits_per_channel": bits})
    plot_e1(read_csv(DATA / "e1_aggregate.csv")); plot_e1_intervals(read_csv(DATA / "e1_paired_intervals.csv"))


def plot_e1(rows: list[dict[str, str]]) -> None:
    cfg = protocol(); ks = tuple(int(value) for value in cfg["k_values"]); snrs = tuple(int(value) for value in cfg["snr_db"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex="col")
    for column, condition in enumerate(name for name, _, _ in CONDITIONS):
        for row_index, modulation in enumerate(("16QAM", "QPSK")):
            axis = axes[row_index, column]
            for waveform in ("Learned", "OTFS", "AFDM", "OFDM"):
                selected = _select_series(rows, "k", ks, condition=condition, modulation=modulation, snr_db=18, receiver_family="historical", waveform=waveform)
                axis.plot([int(r["k"]) / 64 for r in selected], [float(r["ber"]) for r in selected], color=COLORS[waveform], marker=MARKERS[waveform], label=waveform if column == 0 and row_index == 0 else None)
            axis.set_yscale("log"); axis.grid(alpha=.25); axis.set_title(f"{condition} — {modulation}"); axis.set_xlabel("K/N"); axis.set_ylabel("BER")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.01))
    save_figure(fig, "E1_receiver_sensitivity_budget")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for axis, (condition, _, _) in zip(axes, CONDITIONS):
        for family, linestyle in (("historical", "-"), ("tuned", "--"), ("residual_aware", ":"), ("full", "-.")):
            selected = _select_series(rows, "snr_db", snrs, condition=condition, modulation="16QAM", waveform="Learned", receiver_family=family, k=64 if family == "full" else 24)
            axis.plot([float(r["snr_db"]) for r in selected], [float(r["ber"]) for r in selected], linestyle=linestyle, marker="o", label=family)
        axis.set_yscale("log"); axis.grid(alpha=.25); axis.set_title(condition); axis.set_xlabel("Es/N0 (dB)")
    axes[0].set_ylabel("16-QAM BER"); axes[-1].legend(frameon=False, fontsize=8)
    save_figure(fig, "E1_learned_receiver_snr")

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True, sharey="row")
    for row_index, condition in enumerate(("joint_0.5", "nominal_dd_v1")):
        for column, family in enumerate(("historical", "tuned", "residual_aware", "full")):
            axis = axes[row_index, column]
            for waveform in ("Learned", "OTFS", "AFDM", "OFDM"):
                selected = _select_series(rows, "snr_db", snrs, condition=condition, modulation="16QAM", waveform=waveform, receiver_family=family, k=64 if family == "full" else 24)
                axis.plot([float(r["snr_db"]) for r in selected], [float(r["ber"]) for r in selected], color=COLORS[waveform], marker=MARKERS[waveform], label=waveform)
            axis.set_yscale("log"); axis.grid(alpha=.25); axis.set_title(f"{condition}\n{family}", fontsize=9); axis.set_xlabel("Es/N0 (dB)")
        axes[row_index, 0].set_ylabel("16-QAM BER")
    axes[0, -1].legend(frameon=False, fontsize=7)
    save_figure(fig, "E1_all_waveforms_receiver_ladder")


def plot_e1_intervals(rows: list[dict[str, str]]) -> None:
    for modulation in ("16QAM", "QPSK"):
        fig, axes = plt.subplots(3, 3, figsize=(9.2, 7.2), sharex=True)
        for row_index, (condition, _, _) in enumerate(CONDITIONS):
            for column, baseline in enumerate(("AFDM", "OTFS", "OFDM")):
                axis = axes[row_index, column]
                for family, marker in (("historical", "o"), ("tuned", "s"), ("residual_aware", "^")):
                    selected = _select_series(rows, "k", tuple(int(value) for value in protocol()["k_values"]), condition=condition, modulation=modulation, snr_db=18, receiver_family=family, contrast=f"Learned-{baseline}")
                    x = np.asarray([int(r["k"]) / 64 for r in selected]); y = np.asarray([float(r["mean_difference"]) for r in selected])
                    low = np.asarray([float(r["ci95_low"]) for r in selected]); high = np.asarray([float(r["ci95_high"]) for r in selected])
                    axis.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker=marker, linewidth=1, capsize=2, label=family)
                axis.axhline(0, color="black", linewidth=.8); axis.grid(alpha=.2); axis.set_title(f"{condition}: L - {baseline}"); axis.set_xlabel("K/N"); axis.set_ylabel("paired BER difference")
        axes[0, -1].legend(frameon=False, fontsize=7); save_figure(fig, f"E1_{modulation}_paired_receiver_uncertainty")


def phase_e2(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol(); waveforms = load_waveforms(); count = 8 if smoke else int(cfg["banks"]["e2_exploratory"]["channels"]); bits = 256 if smoke else int(cfg["bits_per_channel"])
    epsilons = tuple(float(v) for v in cfg["controlled_fractionality"]["epsilon_tau"]); rows = []; start = time.perf_counter()
    for tau in epsilons:
        for nu in epsilons:
            channels = controlled_fractionality_2d(count, int(cfg["banks"]["e2_exploratory"]["seed"]), tau, nu).matrices
            current = communication_rows(channels, waveforms, (qam16_constellation(), qpsk_constellation()), (18,), 24, "historical", bits, 17840, 17940)
            rows.extend({"epsilon_tau": tau, "epsilon_nu": nu, **row} for row in current)
    keys = ("epsilon_tau", "epsilon_nu", "modulation", "waveform", "receiver_family", "k", "snr_db")
    summary = aggregate(rows, keys); write_csv(DATA / "e2_per_channel.csv", rows); write_csv(DATA / "e2_aggregate.csv", summary)
    intervals = []; seed = int(cfg["banks"]["bootstrap"]["seed"])
    for tau in epsilons:
        for nu in epsilons:
            for modulation in ("16QAM", "QPSK"):
                learned = [r for r in rows if r["epsilon_tau"] == tau and r["epsilon_nu"] == nu and r["modulation"] == modulation and r["waveform"] == "Learned"]
                for baseline in ("AFDM", "OTFS"):
                    classical = [r for r in rows if r["epsilon_tau"] == tau and r["epsilon_nu"] == nu and r["modulation"] == modulation and r["waveform"] == baseline]
                    intervals.append({"epsilon_tau": tau, "epsilon_nu": nu, "modulation": modulation, "contrast": f"Learned-{baseline}", **paired_channel_interval(learned, classical, seed, 200 if smoke else 10000)})
                    seed += 1
    write_csv(DATA / "e2_pointwise_intervals.csv", intervals); write_json(DATA / "e2_execution.json", {"run_kind": "smoke" if smoke else "production", "seconds": time.perf_counter() - start, "channels_per_cell": count, "bits_per_channel": bits, "intervals": "pointwise"})
    plot_e2(summary, epsilons); plot_e2_absolute(summary, epsilons)


def plot_e2(rows: list[dict[str, object]], epsilons: tuple[float, ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True); images = []
    for row_index, modulation in enumerate(("16QAM", "QPSK")):
        for column, baseline in enumerate(("AFDM", "OTFS")):
            matrix = np.zeros((len(epsilons), len(epsilons))); marks = np.zeros_like(matrix, dtype=bool)
            for i, tau in enumerate(epsilons):
                for j, nu in enumerate(epsilons):
                    selected = {waveform: _select_series(rows, "snr_db", (18,), epsilon_tau=tau, epsilon_nu=nu, modulation=modulation, waveform=waveform, receiver_family="historical", k=24)[0] for waveform in ("Learned", "OTFS", "AFDM", "OFDM")}
                    matrix[i, j], marks[i, j] = safe_log10_ber_ratio(int(selected["Learned"]["bit_errors"]), int(selected["Learned"]["total_bits"]), int(selected[baseline]["bit_errors"]), int(selected[baseline]["total_bits"]))
            image = axes[row_index, column].imshow(matrix, origin="lower", extent=(-.05, .55, -.05, .55), vmin=-1, vmax=1, cmap="coolwarm")
            images.append(image); y, x = np.where(marks); axes[row_index, column].scatter(np.asarray(epsilons)[x], np.asarray(epsilons)[y], marker="x", color="black", s=16)
            axes[row_index, column].set_title(f"{modulation}: Learned / {baseline}"); axes[row_index, column].set_xlabel("Doppler offset $\\epsilon_\\nu$"); axes[row_index, column].set_ylabel("Delay offset $\\epsilon_\\tau$")
    fig.tight_layout(rect=(0, 0, .85, 1)); color_axis = fig.add_axes((.88, .15, .025, .72)); fig.colorbar(images[0], cax=color_axis, label="log10 BER ratio")
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"E2_joint_fractionality_map.{suffix}", dpi=220)
    plt.close(fig)


def plot_e2_absolute(rows: list[dict[str, object]], epsilons: tuple[float, ...]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.0), sharex=True, sharey=True)
    row_images = []
    for row_index, modulation in enumerate(("16QAM", "QPSK")):
        values = [float(r["ber"]) for r in rows if r["modulation"] == modulation]
        vmin = np.log10(max(min(value for value in values if value > 0), 1e-7)); vmax = np.log10(max(values))
        for column, waveform in enumerate(("Learned", "OTFS", "AFDM", "OFDM")):
            matrix = np.zeros((len(epsilons), len(epsilons)))
            for i, tau in enumerate(epsilons):
                for j, nu in enumerate(epsilons):
                    selected = _select_series(rows, "snr_db", (18,), epsilon_tau=tau, epsilon_nu=nu, modulation=modulation, waveform=waveform, receiver_family="historical", k=24)[0]
                    matrix[i, j] = np.log10(max(float(selected["ber"]), 1e-7))
            image = axes[row_index, column].imshow(matrix, origin="lower", extent=(-.05, .55, -.05, .55), vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row_index, column].set_title(f"{modulation}: {waveform}"); axes[row_index, column].set_xlabel("$\\epsilon_\\nu$"); axes[row_index, column].set_ylabel("$\\epsilon_\\tau$")
        row_images.append(image)
    fig.tight_layout(rect=(0, 0, .90, 1))
    for image, bounds in zip(row_images, ((.92, .56, .018, .34), (.92, .10, .018, .34))):
        color_axis = fig.add_axes(bounds); fig.colorbar(image, cax=color_axis, label="log10 absolute BER")
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"E2_absolute_ber_maps.{suffix}", dpi=220)
    plt.close(fig)


def phase_e3(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol(); waveforms = load_waveforms(); test_n, _, _, bits = _sizes(smoke); support_n = 12 if smoke else int(cfg["banks"]["nominal_support"]["channels"])
    nominal_bank = sample_dd_v1(support_n, int(cfg["banks"]["nominal_support"]["seed"])).matrices
    nominal = {name: fixed_energy_support(nominal_bank, waveform, 24) for name, waveform in waveforms.items()}
    definitions = (("delay", (1.0, 1.5, 2.0)), ("doppler", (1.0, 1.5, 2.0)), ("power", (0.0, .75, 1.5)))
    rows = []; start = time.perf_counter()
    for family_index, (family, severities) in enumerate(definitions):
        for severity_index, severity in enumerate(severities):
            test = shifted_prior_channels(test_n, 17700 + 100 * family_index, family, severity).matrices
            support_bank = shifted_prior_channels(support_n, 17600 + 100 * family_index, family, severity).matrices
            recalibrated = {name: fixed_energy_support(support_bank, waveform, 24) for name, waveform in waveforms.items()}
            for policy, supports in (("nominal_fixed", nominal), ("recalibrated_fixed", recalibrated), ("oracle_row_topk", None)):
                current = communication_rows(test, waveforms, (qam16_constellation(), qpsk_constellation()), (18,), 24, "historical", bits, 17880 + family_index, 17980 + family_index, supports=supports)
                recalls = {}
                for name, waveform in waveforms.items():
                    if supports is None:
                        recalls[name] = torch.ones(test_n, dtype=torch.float64)
                    else:
                        actual = effective_operator(test, waveform)
                        _, recalls[name] = support_metrics(actual, supports[name], 24)
                for row in current:
                    name = str(row["waveform"])
                    rows.append({"shift_family": family, "severity": severity, "support_policy": policy, "support_recall": float(recalls[name][int(row["channel_index"])]), **row})
    keys = ("shift_family", "severity", "support_policy", "modulation", "waveform", "receiver_family", "k", "snr_db")
    summary = aggregate(rows, keys)
    for output in summary:
        selected = [row["support_recall"] for row in rows if all(str(row[key]) == str(output[key]) for key in ("shift_family", "severity", "support_policy", "modulation", "waveform"))]
        output["mean_support_recall"] = float(np.mean(selected))
    intervals = []; seed = 18400
    for family, severities in definitions:
        for severity in severities:
            for modulation in ("16QAM", "QPSK"):
                for waveform in waveforms:
                    oracle = [r for r in rows if r["shift_family"] == family and r["severity"] == severity and r["modulation"] == modulation and r["waveform"] == waveform and r["support_policy"] == "oracle_row_topk"]
                    for policy in ("nominal_fixed", "recalibrated_fixed"):
                        fixed = [r for r in rows if r["shift_family"] == family and r["severity"] == severity and r["modulation"] == modulation and r["waveform"] == waveform and r["support_policy"] == policy]
                        intervals.append({"shift_family": family, "severity": severity, "modulation": modulation, "waveform": waveform, "contrast": f"{policy}-oracle", **paired_channel_interval(fixed, oracle, seed, 200 if smoke else 10000)}); seed += 1
    write_csv(DATA / "e3_per_channel.csv", rows); write_csv(DATA / "e3_aggregate.csv", summary); write_csv(DATA / "e3_paired_intervals.csv", intervals); write_json(DATA / "e3_execution.json", {"run_kind": "smoke" if smoke else "production", "seconds": time.perf_counter() - start, "channels": test_n, "support_channels": support_n, "bits_per_channel": bits, "severity_pairing": "shared base random variables within shift family"})
    plot_e3(summary)


def plot_e3(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    for column, family in enumerate(("delay", "doppler", "power")):
        for policy, linestyle in (("nominal_fixed", "--"), ("recalibrated_fixed", "-"), ("oracle_row_topk", ":")):
            severities = (0.0, .75, 1.5) if family == "power" else (1.0, 1.5, 2.0)
            selected = _select_series(rows, "severity", severities, shift_family=family, modulation="16QAM", waveform="Learned", support_policy=policy, receiver_family="historical", k=24, snr_db=18)
            axes[0, column].plot([float(r["severity"]) for r in selected], [float(r["ber"]) for r in selected], marker="o", linestyle=linestyle, label=policy)
            axes[1, column].plot([float(r["severity"]) for r in selected], [float(r["mean_retained_energy"]) for r in selected], marker="o", linestyle=linestyle, label=policy)
        axes[0, column].set_yscale("log"); axes[0, column].set_title(family); axes[0, column].set_ylabel("16-QAM BER"); axes[1, column].set_ylabel("Retained energy"); axes[1, column].set_xlabel("Mismatch severity"); axes[0, column].grid(alpha=.25); axes[1, column].grid(alpha=.25)
    axes[0, -1].legend(frameon=False, fontsize=7); save_figure(fig, "E3_support_prior_robustness")


def phase_e4(smoke: bool) -> None:
    ensure_dirs(); cfg = protocol(); waveforms = load_waveforms(afdm_variants=True); test_n, _, _, bits = _sizes(smoke); snrs = tuple(int(v) for v in cfg["snr_db"]); rows = []; start = time.perf_counter()
    definitions = (("favorable_joint_0.5", .5, .5), ("boundary_joint_0.2", .2, .2), ("unfavorable_integer", 0., 0.))
    for index, (label, tau, nu) in enumerate(definitions):
        channels = controlled_fractionality_2d(test_n, int(cfg["banks"]["confirmatory"]["seed"]) + 500, tau, nu).matrices
        current = communication_rows(channels, waveforms, (qam16_constellation(), qpsk_constellation()), snrs, 24, "historical", bits, 17920, 18020)
        rows.extend({"condition": label, **row} for row in current)
    channels = sample_dd_v1(test_n, int(cfg["banks"]["confirmatory"]["seed"]) + 600).matrices
    current = communication_rows(channels, waveforms, (qam16_constellation(), qpsk_constellation()), snrs, 64, "full", bits, 17930, 18030)
    rows.extend({"condition": "full_information_nominal", **row} for row in current)
    keys = ("condition", "modulation", "snr_db", "waveform", "receiver_family", "k")
    summary = aggregate(rows, keys); write_csv(DATA / "e4_per_channel.csv", rows); write_csv(DATA / "e4_aggregate.csv", summary)
    intervals = []; seed = 18500
    for condition in {str(row["condition"]) for row in rows}:
        for modulation in ("16QAM", "QPSK"):
            for snr in snrs:
                learned = [r for r in rows if r["condition"] == condition and r["modulation"] == modulation and r["snr_db"] == snr and r["waveform"] == "Learned"]
                for baseline in ("AFDM", "OTFS", "OFDM"):
                    classical = [r for r in rows if r["condition"] == condition and r["modulation"] == modulation and r["snr_db"] == snr and r["waveform"] == baseline]
                    intervals.append({"condition": condition, "modulation": modulation, "snr_db": snr, "contrast": f"Learned-{baseline}", **paired_channel_interval(learned, classical, seed, 200 if smoke else 10000)}); seed += 1
    write_csv(DATA / "e4_paired_intervals.csv", intervals); write_json(DATA / "e4_execution.json", {"run_kind": "smoke" if smoke else "production", "seconds": time.perf_counter() - start, "channels": test_n, "bits_per_channel": bits})
    plot_e4(summary)


def plot_e4(rows: list[dict[str, object]]) -> None:
    conditions = ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal")
    snrs = tuple(int(value) for value in protocol()["snr_db"])
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True)
    for row_index, modulation in enumerate(("16QAM", "QPSK")):
        for column, condition in enumerate(conditions):
            axis = axes[row_index, column]
            for waveform in ("Learned", "OTFS", "AFDM", "OFDM"):
                selected = _select_series(rows, "snr_db", snrs, condition=condition, modulation=modulation, waveform=waveform, receiver_family="full" if condition == "full_information_nominal" else "historical", k=64 if condition == "full_information_nominal" else 24)
                axis.plot([float(r["snr_db"]) for r in selected], [float(r["ber"]) for r in selected], color=COLORS[waveform], marker=MARKERS[waveform], label=waveform)
            axis.set_yscale("log"); axis.grid(alpha=.25); axis.set_title(condition.replace("_", " "), fontsize=9); axis.set_xlabel("Es/N0 (dB)"); axis.set_ylabel(f"{modulation} BER")
    axes[0, 0].legend(frameon=False, fontsize=8); save_figure(fig, "E4_qualified_operating_envelope")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharex=True, sharey=True)
    styles = (("Learned", "-"), ("OFDM", "-"), ("AFDM", "-"), ("AFDM historical", "--"), ("AFDM guard", ":"))
    for axis, condition in zip(axes, conditions):
        family = "full" if condition == "full_information_nominal" else "historical"; k = 64 if family == "full" else 24
        for waveform, linestyle in styles:
            selected = _select_series(rows, "snr_db", snrs, condition=condition, modulation="16QAM", waveform=waveform, receiver_family=family, k=k)
            color = COLORS.get(waveform, "#7c3aed" if waveform == "AFDM historical" else "#a21caf")
            axis.plot([float(r["snr_db"]) for r in selected], [float(r["ber"]) for r in selected], color=color, linestyle=linestyle, marker="o", markersize=3, label=waveform)
        axis.set_yscale("log"); axis.grid(alpha=.25); axis.set_title(condition.replace("boundary", "probe").replace("_", " "), fontsize=8); axis.set_xlabel("Es/N0 (dB)")
    axes[0].set_ylabel("16-QAM BER"); axes[-1].legend(frameon=False, fontsize=6)
    save_figure(fig, "E4_named_afdm_posthoc")


def _lookup(rows: list[dict[str, str]], **values: object) -> dict[str, str]:
    selected = [row for row in rows if all(_matches(row[key], value) for key, value in values.items())]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {values}, got {len(selected)}")
    return selected[0]


def derive_e4_slope1_intervals() -> list[dict[str, object]]:
    """Post hoc paired uncertainty from frozen E4 rows; performs no simulation."""
    source = read_csv(DATA / "e4_per_channel.csv"); rows = []; seed = 18600
    for condition in ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal"):
        family = "full" if condition == "full_information_nominal" else "historical"; k = 64 if family == "full" else 24
        for modulation in ("16QAM", "QPSK"):
            for snr in tuple(int(value) for value in protocol()["snr_db"]):
                common = {"condition": condition, "modulation": modulation, "snr_db": snr, "receiver_family": family, "k": k}
                learned = [row for row in source if all(_matches(row[key], value) for key, value in common.items()) and row["waveform"] == "Learned"]
                slope1 = [row for row in source if all(_matches(row[key], value) for key, value in common.items()) and row["waveform"] == "AFDM historical"]
                rows.append({**common, "contrast": "Learned-AFDM historical", **paired_channel_interval(learned, slope1, seed, 10000)})
                seed += 1
    write_csv(DATA / "e4_posthoc_slope1_intervals.csv", rows)
    return rows


def phase_report() -> None:
    execution = json.loads((DATA / "e4_execution.json").read_text())
    if execution["run_kind"] != "production":
        (DATA / "pub1f_handoff.json").unlink(missing_ok=True)
        (ROOT / "reports/stage_pub1e_summary.md").write_text("# Pub-1E — Frozen-waveform qualification and operating-regime mapping\n\n## Status: implementation validated; production pending\n\nThe complete reduced-shape smoke path ran successfully. Smoke values are integration diagnostics only and are not scientific evidence. Run production phases E1–E4, then regenerate this report and manifest.\n")
        (ROOT / "reports/stage_pub1f_handoff.md").write_text("# Pub-1F handoff\n\nPending Pub-1E production. Do not use smoke artifacts as a Pub-1F scientific handoff.\n")
        return
    e4 = read_csv(DATA / "e4_aggregate.csv"); e4_ci = read_csv(DATA / "e4_paired_intervals.csv"); slope1_ci = derive_e4_slope1_intervals(); e1 = read_csv(DATA / "e1_aggregate.csv"); e1_ci = read_csv(DATA / "e1_paired_intervals.csv"); e2 = read_csv(DATA / "e2_aggregate.csv"); e3 = read_csv(DATA / "e3_aggregate.csv")
    observations = []
    for condition in ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal"):
        values = {name: float(_lookup(e4, condition=condition, modulation="16QAM", snr_db=18, waveform=name)["ber"]) for name in ("Learned", "OTFS", "AFDM", "OFDM")}
        observations.append((condition, values))
    favorable = observations[0][1]; full = observations[3][1]
    wins = lambda values: values["Learned"] < min(values[name] for name in ("OTFS", "AFDM", "OFDM"))
    learned_favorable = wins(favorable)
    full_classical = min(full[name] for name in ("OTFS", "AFDM", "OFDM"))
    recommendation = "Proceed with a narrower claim" if learned_favorable else "Revise the scientific objective before new training"
    def ci(rows, **query):
        row = _lookup(rows, **query); return f"[{float(row['ci95_low']):.6g}, {float(row['ci95_high']):.6g}]"
    validation = read_csv(DATA / "e1_receiver_validation.csv")
    selected_scales = sorted({float(row["selected"]) for row in validation})
    map_counts = {}
    for modulation in ("16QAM", "QPSK"):
        for baseline in ("AFDM", "OTFS", "OFDM"):
            count = 0
            for tau in (0., .1, .2, .3, .4, .5):
                for nu in (0., .1, .2, .3, .4, .5):
                    learned = float(_lookup(e2, epsilon_tau=tau, epsilon_nu=nu, modulation=modulation, waveform="Learned")["ber"])
                    classical = float(_lookup(e2, epsilon_tau=tau, epsilon_nu=nu, modulation=modulation, waveform=baseline)["ber"])
                    count += learned < classical
            map_counts[(modulation, baseline)] = count
    text = [
        "# Pub-1E — Frozen-waveform qualification and operating-regime mapping", "",
        f"## Recommendation: {recommendation}", "",
        "Classification: under the historical K=24 receiver, the limited-representation claim is supported for joint/off-grid and nominal DD-v1 conditions, but not on the integer grid or with full information. Stronger receivers repair classical baselines and reduce the advantage. Proceed only with this qualified claim; do not retrain or start Pub-1F automatically.", "",
        "All numbers below are production results from 512-channel confirmatory banks unless explicitly labeled exploratory. Learned was frozen; training steps were zero.", "",
        "## E1 — Receiver sensitivity", "",
        "At K=24 and 18 dB:", "",
        "| condition | modulation | receiver | Learned | AFDM | OFDM | Learned-AFDM paired 95% CI |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for condition in ("integer", "joint_0.5", "nominal_dd_v1"):
        for modulation in ("16QAM", "QPSK"):
            for family in ("historical", "tuned", "residual_aware"):
                values = {name: float(_lookup(e1, condition=condition, modulation=modulation, snr_db=18, waveform=name, receiver_family=family, k=24)["ber"]) for name in ("Learned", "AFDM", "OFDM")}
                interval = ci(e1_ci, condition=condition, modulation=modulation, receiver_family=family, k=24, snr_db=18, contrast="Learned-AFDM")
                text.append(f"| {condition} | {modulation} | {family} | {values['Learned']:.6g} | {values['AFDM']:.6g} | {values['OFDM']:.6g} | {interval} |")
    text += ["", f"Validation-selected scales vary across waveform/condition/K in `{selected_scales}`. The all-waveform companion figure shows that receiver improvement is not a Learned-only intervention.", "", "Residual covariance materially repairs classical baselines. In nominal DD-v1/QPSK, residual-aware Learned-AFDM has a CI containing zero: inconclusive, not evidence of equality. Joint-fractionality gains survive but shrink; integer-grid losses persist. Full-information comparisons use a different declared reference condition.", "", "Residual statistics are waveform/condition/K-specific means and dense receive covariances estimated on an independent bank using the same oracle row-top-K rule later used by E1. They omit conditioning on realized retained values/support and remain moment-matched approximations. E3 fixed-support policies were evaluated only with the historical receiver, so no residual-statistics/support mismatch is present. Equal K does not imply equal stored information or compute: residual-aware adds two dense 64x64 complex matrices per bank entry and a direct dense solve.", "", "## E2 — Exploratory two-dimensional fractionality map", "", f"Descriptive lower-BER cell counts only: 16-QAM Learned is lower in {map_counts[('16QAM','AFDM')]}/36 cells versus AFDM, {map_counts[('16QAM','OTFS')]}/36 versus OTFS, and {map_counts[('16QAM','OFDM')]}/36 versus OFDM; QPSK counts are {map_counts[('QPSK','AFDM')]}/36, {map_counts[('QPSK','OTFS')]}/36, and {map_counts[('QPSK','OFDM')]}/36.", "", "The map suggests joint fractionality is favorable, while integer and isolated-axis losses remain. It does not estimate a decision boundary. Pointwise intervals do not support a map-wide significance claim. Predeclared K=16/32 companion maps were not run and must not be inferred from K=24.", "", "## E3 — Fixed-support prior mismatch", "", "For Learned 16-QAM at maximum delay shift, nominal/recalibrated/oracle BER is 0.04168/0.03833/0.03316; at maximum Doppler shift it is 0.03698/0.03383/0.02916. Recalibration recovers part, not all, of the loss. At delay factor 2, nominal support retains 99.13% energy yet BER is 25.7% worse than oracle. Retained energy is therefore not a sufficient receiver-quality statistic.", "", "The power-profile shift was benign only for the declared one-factor experiment. These results do not establish robustness to joint shifts.", "", "## E4 — Locked confirmatory envelope", "", "The label `boundary_joint_0.2` is retained as a historical data key; scientifically it is the predeclared intermediate/probe point, not an empirically estimated boundary.", "", "| condition | modulation | Learned | OTFS | AFDM | OFDM | Learned-AFDM paired 95% CI |", "|---|---|---:|---:|---:|---:|---:|" ]
    for condition in ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal"):
        for modulation in ("16QAM", "QPSK"):
            values = {name: float(_lookup(e4, condition=condition, modulation=modulation, snr_db=18, waveform=name)["ber"]) for name in ("Learned", "OTFS", "AFDM", "OFDM")}
            interval = ci(e4_ci, condition=condition, modulation=modulation, snr_db=18, contrast="Learned-AFDM")
            text.append(f"| {condition} | {modulation} | {values['Learned']:.6g} | {values['OTFS']:.6g} | {values['AFDM']:.6g} | {values['OFDM']:.6g} | {interval} |")
    text += ["", "Named AFDM 16-QAM results at 18 dB:", "", "| condition | selected | historical slope 1 | canonical slope 3 | guard slope 5 |", "|---|---:|---:|---:|---:|"]
    for condition in ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal"):
        values = {name: float(_lookup(e4, condition=condition, modulation="16QAM", snr_db=18, waveform=name)["ber"]) for name in ("AFDM", "AFDM historical", "AFDM canonical", "AFDM guard")}
        text.append(f"| {condition} | {values['AFDM']:.6g} | {values['AFDM historical']:.6g} | {values['AFDM canonical']:.6g} | {values['AFDM guard']:.6g} |")
    for condition in ("favorable_joint_0.5", "boundary_joint_0.2", "unfavorable_integer", "full_information_nominal"):
        row = _lookup(slope1_ci, condition=condition, modulation="16QAM", snr_db=18, contrast="Learned-AFDM historical")
        text.append(f"Post hoc paired Learned-slope1 CI at {condition}: [{float(row['ci95_low']):.6g}, {float(row['ci95_high']):.6g}].")
    text += ["", f"The favorable and intermediate/probe results use the historical receiver; integer reverses and full-information nominal favors classical baselines. Nominal full-information Learned/classical-best 16-QAM BER ratio is {full['Learned'] / max(full_classical, 1e-300):.3f}. Aggregate LMMSE MSE is unitary-invariant under full information, but finite-constellation BER need not be.", "", "AFDM guard selection searched odd slopes 3/5/7/9; slope 1 was a named historical comparator, not a candidate. Odd slopes are mathematically unitary here; the lower bound was an engineering guard against delay-Doppler overlap, not a unitarity requirement. The frozen selection remains slope 3 with no test-set reselection, although slope 1 is stronger in some E4 regimes. Slope1 plus the improved receiver ladder was not tested. OFDM remains a primary baseline and often sits between Learned and the delay-Doppler baselines in the favorable historical-receiver regime.", "", "## Established, unsupported, and limitations", "", "Established: historical-receiver joint-fractionality gains; narrower nominal-DD-v1 evidence; receiver-driven shrinkage; fixed-support degradation and partial recalibration recovery; integer/full-information reversals.", "", "Unsupported: universal superiority, empirical fractionality boundary, specialized classical receivers, equal information/storage/compute, joint prior shifts, coding/MIMO/pulse-shaped/RF behavior, or a new-training implication.", "", "## Reproducibility and preservation", "", "Waveforms share channels/bits/noise within each evaluated row set, and each waveform constructs its own support. E1 K=24 at 18 dB consumes SNR index 2 while other K values evaluate only 18 dB at index 0; cross-K curves are therefore not common-random-number paired. The E1 K=64 historical and full rows also must not be treated as paired duplicates. Some numeric seeds recur across different generator roles (bootstrap seed 18000 and E4 physical seed 18000); that is harmless cross-domain reuse, not leakage, but it invalidates a blanket claim that every numeric seed is unique. Exact derivations are in the protocol addendum.", "", "No simulations were rerun for this correction. The active plots, post hoc slope-1 intervals, report, and handoffs were regenerated only from saved Pub-1E tables. The old mixed-waveform E1 budget rendering is retained under `outputs/stage_pub1e/superseded/`.", "", "## Pub-1F boundary", "", "Pub-1F may begin only after its own protocol is frozen, fresh independent banks are assigned by generator role, and training reproducibility/objective ablation/regime replication/compute qualification are specified. It must include per-waveform constructions, OFDM, broader AFDM validation including slope 1, and explicit support/residual rules. Pub-1F was not executed.", ""]
    (ROOT / "reports/stage_pub1e_summary.md").write_text("\n".join(text))
    raw_paths = [DATA / f"e{index}_per_channel.csv" for index in range(1, 5)]
    handoff = {
        "stage": "Pub-1F handoff", "source_stage": "Pub-1E", "checkpoint_sha256": protocol()["checkpoint"]["sha256"], "config_sha256": file_sha256(CONFIG_PATH),
        "raw_data_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in raw_paths}, "protocol": protocol(),
        "active_figures": sorted(str(path.relative_to(ROOT)) for path in FIG.glob("*")), "superseded_figures": sorted(str(path.relative_to(ROOT)) for path in (OUT / "superseded").glob("*")),
        "receiver_definitions": protocol()["receiver_ladder"], "receiver_scope": "residual-aware only in E1 with waveform/condition/K-specific oracle-row-top-K statistics; E3 fixed supports use historical receiver only",
        "afdm_selection": json.loads((DATA / "afdm_selection.json").read_text()), "afdm_limitation": "slope 1 excluded from guarded 3/5/7/9 selection but valid and sometimes stronger; slope1 plus improved receivers untested",
        "conditions": dict(protocol()["confirmatory_conditions"]), "seed_derivations": {
            "E1": "test=17500+10*c; validation=17110+10*c; residual=17300+10*c; symbols=17800+10*c; noise=17900+10*c",
            "E2": "physical=17400 for every cell; symbols=17840; noise=17940",
            "E3": "test=17700+100*f; support=17600+100*f; nominal support=17200; symbols=17880+f; noise=17980+f",
            "E4": "anchored physical=18000, symbols=17920, noise=18020; nominal full physical=18100, symbols=17930, noise=18030",
        },
        "fresh_bank_requirement": "new role-separated validation/support/statistics/test/symbol/noise/bootstrap banks; numeric seeds need not be globally unique across generator domains but roles and streams must be explicit",
        "pub1f_primary_work": ["training reproducibility", "objective ablation", "qualified-regime replication", "information/storage/compute qualification"],
        "required_controls": ["per-waveform support/statistics construction", "OFDM", "historical and broader AFDM including slope 1", "explicit residual conditioning approximation"],
        "dense_cost_warning": "residual-aware stores dense complex 64x64 mean and covariance per waveform/condition/K and uses direct dense solve; equal K is not equal information/storage/compute",
        "limitations": ["residual receiver is approximate", "E2 cell counts are descriptive and intervals pointwise", "K16/K32 E2 maps not run", "slope1 improved-receiver ladder not run", "no equal-budget conclusion"],
    }
    write_json(DATA / "pub1f_handoff.json", handoff)
    (ROOT / "reports/stage_pub1f_handoff.md").write_text("# Pub-1F handoff\n\nMachine-readable authority: `outputs/stage_pub1e/data/pub1f_handoff.json`. Pub-1F has not begun.\n\nBefore starting: freeze a new protocol and fresh role-separated banks; preserve the exact Pub-1E checkpoint/config/raw hashes; reproduce training; ablate the objective; replicate the qualified historical-receiver regimes; and qualify information, dense storage, and solve cost separately. Retain per-waveform support/statistics construction, OFDM, and AFDM slope 1 plus the guarded family. Explicitly test the currently missing slope1/improved-receiver ladder. Do not call `(0.2,0.2)` a learned boundary or reinterpret equal K as equal information/compute.\n")


def phase_plots() -> None:
    """Replot every figure from saved tables without communication simulation."""
    plot_e1(read_csv(DATA / "e1_aggregate.csv")); plot_e1_intervals(read_csv(DATA / "e1_paired_intervals.csv"))
    cfg = protocol(); e2 = read_csv(DATA / "e2_aggregate.csv"); eps = tuple(float(v) for v in cfg["controlled_fractionality"]["epsilon_tau"]); plot_e2(e2, eps); plot_e2_absolute(e2, eps)
    plot_e3(read_csv(DATA / "e3_aggregate.csv")); plot_e4(read_csv(DATA / "e4_aggregate.csv"))


def phase_manifest() -> None:
    protected = {
        "checkpoint": {"before": protocol()["checkpoint"]["sha256"], "after": file_sha256(FROZEN)},
        "config": {"before": "2ab9c3ec86b81ef38f10d754e3f864723461ae191b93c0c6f058202eeb7d3488", "after": file_sha256(CONFIG_PATH)},
        "raw_tables": {str(path.relative_to(ROOT)): file_sha256(path) for path in (DATA / "e1_per_channel.csv", DATA / "e2_per_channel.csv", DATA / "e3_per_channel.csv", DATA / "e4_per_channel.csv")},
    }
    reports = [ROOT / "reports/stage_pub1e_summary.md", ROOT / "reports/stage_pub1e_protocol.md", ROOT / "reports/stage_pub1e_protocol_addendum.md", ROOT / "reports/stage_pub1e_correction_log.md", ROOT / "reports/stage_pub1f_handoff.md"]
    write_json(OUT / "correction_manifest.json", {
        "correction": "saved-data-only Pub-1E plot and interpretation correction", "simulations_rerun": False,
        "protected": protected, "prior_report_hashes": {
            "reports/stage_pub1e_summary.md": "25b50ece20cb667c25a62a397cb3bed623350f9f78edb776400c60ceaab12ee6",
            "reports/stage_pub1e_protocol.md": "92fb08b9269c802cbec8058d19e01160561a36abdd6c1a8bf703c2feeb79a0f3",
            "reports/stage_pub1f_handoff.md": "35355d8a160a282344977044d6257c180e2b4c82e868fb43aeb421c8fbeb9982",
        },
        "current_reports": {str(path.relative_to(ROOT)): file_sha256(path) for path in reports},
        "active_figures": {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(FIG.glob("*"))},
        "superseded_figures": {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted((OUT / "superseded").glob("*"))},
    })
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    records = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); records.append({"path": str(path.relative_to(ROOT)), "sha256": digest, "bytes": path.stat().st_size})
    write_json(OUT / "artifact_manifest.json", {"files": records, "count": len(records), "checkpoint_sha256_after": file_sha256(FROZEN)})


def phase_all(smoke: bool) -> None:
    phase_provenance(smoke); phase_validate(smoke); phase_e1(smoke); phase_e2(smoke); phase_e3(smoke); phase_e4(smoke); phase_report(); phase_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("provenance", "validate", "e1", "e2", "e3", "e4", "plots", "report", "manifest", "all"), default="all"); parser.add_argument("--smoke", action="store_true"); return parser.parse_args()


def main() -> None:
    args = parse_args(); functions = {"provenance": phase_provenance, "validate": phase_validate, "e1": phase_e1, "e2": phase_e2, "e3": phase_e3, "e4": phase_e4}
    if args.phase == "all": phase_all(args.smoke)
    elif args.phase == "plots": phase_plots()
    elif args.phase == "report": phase_report()
    elif args.phase == "manifest": phase_manifest()
    else: functions[args.phase](args.smoke)


if __name__ == "__main__":
    main()
