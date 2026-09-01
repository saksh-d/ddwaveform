"""Resumable Pub-1C Phase-1 publication completion runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.metrics import constellation_encode, nearest_neighbor_decode, qam16_constellation, qpsk_constellation, random_bits
from src.pub1a.core import REGIME_BY_KEY, aggregate_rows, waveform_set
from src.pub1c.core import (
    EPSILONS,
    FRACTIONAL_AXES,
    K_VALUES,
    PRIMARY_K,
    PRIMARY_RHO,
    balanced_symbol_indices,
    controlled_fractionality_channels,
    graph_diagnostics,
    median_full_information_channel,
    paired_bootstrap_difference,
    retained_energy_rows,
)
from src.pub1b.core import sample_scaled_dd
from src.receiver import effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2b import community_summary, ensemble_operator_statistics, spectral_centroids, spectral_overlap_matrix
from src.stage2e import communication_rows, sample_regime


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage_pub1c"
DATA = OUT / "data"
FIG = OUT / "figures"
FROZEN = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
PUB1A = ROOT / "outputs/stage_pub1a"
STAGE2B_ENERGY = ROOT / "outputs/stage2b/ensemble_operator_energy.npz"
TRAINED = {k: ROOT / f"outputs/stage2c/checkpoints/k{k}_ofdm_best.pt" for k in (16, 24, 32)}
HASHES = {
    "authoritative": "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc",
    "trained_k16": "a91d4fb2066d35a7b8e13ffbf869801544e63e8328a4318b04f5dab7ad49a115",
    "trained_k24": "73a401b6034c04863628684ae20be9a90c3ee866e1204445c205ab70027a8fa6",
    "trained_k32": "d8a04b5ce658e488021ef3c2f007556cea094a92320c31b4b7d05d0e23cd4e90",
    "stage2b_energy": "56767db8ba46460fb434d5a5b1257822d8dddd482b8b619be35e253628eba7d7",
}
SEEDS = {
    "fractionality": 14100, "budget": 14200, "symbols": 14300,
    "noise": 14400, "constellation": 14500,
    "constellation15_bank": 14700, "constellation15_symbols": 14800,
    "constellation15_noise": 14900, "contrast_symbols": 15000,
    "contrast_noise": 15100, "structure": config.STAGE1A2_EVAL_SEED,
}
COLORS = {"Learned": "#d97706", "OTFS": "#15803d", "AFDM": "#dc2626"}
MARKERS = {"Learned": "o", "OTFS": "s", "AFDM": "^"}
SCALING_DIMENSIONS = (128, 256)
SCALING_RHO_VALUES = (.125, .1875, .25, .3125, .375, .4375, .5, .625, .75, 1.0)
SCALING_STRUCTURE_CHANNELS = 1024
SCALING_STRUCTURE_SEED = 16100


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, name: str, *, w_pad: float | None = None) -> None:
    if w_pad is None:
        fig.tight_layout()
    else:
        fig.tight_layout(w_pad=w_pad)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def load_waveforms() -> dict[str, torch.Tensor]:
    learned, _ = load_frozen_waveform(FROZEN)
    audit = json.loads((PUB1A / "data/afdm_audit.json").read_text())
    if not audit["hard_gate"] or int(audit["selected_multiplier_2Nc1"]) != 3:
        raise RuntimeError("expected preserved Pub-1A validation-selected AFDM multiplier 3")
    values = waveform_set(learned, 3)
    return {name: values[name] for name in ("Learned", "OTFS", "AFDM")}


def phase_provenance(smoke: bool = False) -> None:
    actual = {
        "authoritative": file_sha256(FROZEN),
        **{f"trained_k{k}": file_sha256(path) for k, path in TRAINED.items()},
        "stage2b_energy": file_sha256(STAGE2B_ENERGY),
    }
    if actual != HASHES:
        raise RuntimeError(f"frozen input hash mismatch: {actual}")
    write_json(DATA / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target_ref": "origin/pub",
        "learned_training_steps": 0,
        "primary_receiver": "each waveform/realization retains its own strongest K entries per effective-channel row",
        "primary_k": PRIMARY_K,
        "primary_rho": PRIMARY_RHO,
        "k_values": K_VALUES,
        "epsilons": EPSILONS,
        "fractional_axes": FRACTIONAL_AXES,
        "snr_db": 18,
        "precision": "complex128/float64",
        "run_kind": "smoke" if smoke else "production",
        "hashes": actual,
        "reuse": {
            "Pub-1A": "frozen checkpoint, validation-selected AFDM multiplier 3, existing outputs preserved",
            "Stage-2B": "exact 1024-channel authoritative Learned ensemble-energy map reused for threshold study",
            "Stage-2C": "validation-selected OFDM-start K=16/24/32 checkpoint tensors reused for separate training-K study",
        },
        "excluded": ["training", "fixed-Q as primary", "pulse shaping", "oversampling audit", "channel estimation", "neural receiver", "AIU"],
    })


def _energy_summary(rows: list[dict[str, object]], extra: tuple[str, ...] = ()) -> list[dict[str, object]]:
    keys = extra + ("waveform", "k", "rho")
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(float(row["retained_energy"]))
    return [{**dict(zip(keys, key)), "mean_retained_energy": float(np.mean(values)), "p10_retained_energy": float(np.quantile(values, .1)), "n_channels": len(values)} for key, values in groups.items()]


def _paired_fractionality_intervals(
    communication: list[dict[str, object]], energy: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ber_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    seed = 14600
    for axis in FRACTIONAL_AXES:
        for epsilon in EPSILONS:
            for modulation in ("QPSK", "16QAM"):
                learned = sorted((row for row in communication if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["modulation"] == modulation and row["waveform"] == "Learned"), key=lambda row: int(row["channel_index"]))
                left = np.asarray([int(row["bit_errors"]) / int(row["total_bits"]) for row in learned])
                for baseline in ("AFDM", "OTFS"):
                    classical = sorted((row for row in communication if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["modulation"] == modulation and row["waveform"] == baseline), key=lambda row: int(row["channel_index"]))
                    right = np.asarray([int(row["bit_errors"]) / int(row["total_bits"]) for row in classical])
                    ber_rows.append({"fractional_axis": axis, "epsilon": epsilon, "modulation": modulation, "contrast": f"Learned-{baseline}", **paired_bootstrap_difference(left, right, seed=seed)})
                    seed += 1
            learned_energy = sorted((row for row in energy if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["waveform"] == "Learned"), key=lambda row: int(row["channel_index"]))
            left_energy = np.asarray([float(row["retained_energy"]) for row in learned_energy])
            for baseline in ("AFDM", "OTFS"):
                classical_energy = sorted((row for row in energy if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["waveform"] == baseline), key=lambda row: int(row["channel_index"]))
                right_energy = np.asarray([float(row["retained_energy"]) for row in classical_energy])
                energy_rows.append({"fractional_axis": axis, "epsilon": epsilon, "contrast": f"Learned-{baseline}", **paired_bootstrap_difference(left_energy, right_energy, seed=seed)})
                seed += 1
    return ber_rows, energy_rows


def phase_fractionality(smoke: bool) -> None:
    waveforms = load_waveforms()
    channels = 8 if smoke else 64
    bits = 32768 if smoke else 262144
    communication: list[dict[str, object]] = []
    energy: list[dict[str, object]] = []
    modes = {name: {f"oracle{PRIMARY_K}": None} for name in waveforms}
    for axis in FRACTIONAL_AXES:
        for epsilon in EPSILONS:
            bank = controlled_fractionality_channels(channels, SEEDS["fractionality"], epsilon, axis)
            rows = communication_rows(bank.matrices, waveforms, modes, (qpsk_constellation(), qam16_constellation()), (18,), bits, SEEDS["symbols"], SEEDS["noise"])
            communication.extend({"fractional_axis": axis, "epsilon": epsilon, **row} for row in rows)
            values = retained_energy_rows(bank.matrices, waveforms, (PRIMARY_K,))
            energy.extend({"fractional_axis": axis, "epsilon": epsilon, **row} for row in values)
    communication_summary = aggregate_rows(communication, ("fractional_axis", "epsilon"))
    energy_summary = _energy_summary(energy, ("fractional_axis", "epsilon"))
    ber_intervals, energy_intervals = _paired_fractionality_intervals(communication, energy)
    write_csv(DATA / "fractionality_per_channel.csv", communication)
    write_csv(DATA / "fractionality_summary.csv", communication_summary)
    write_csv(DATA / "fractionality_energy_per_channel.csv", energy)
    write_csv(DATA / "fractionality_energy_summary.csv", energy_summary)
    write_csv(DATA / "fractionality_paired_ber_intervals.csv", ber_intervals)
    write_csv(DATA / "fractionality_paired_energy_intervals.csv", energy_intervals)
    plot_fractionality(communication_summary, energy_summary)


def plot_fractionality(communication: list[dict[str, object]], energy: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5), sharex=True)
    for row, axis_name in enumerate(FRACTIONAL_AXES):
        for column, modulation in enumerate(("QPSK", "16QAM")):
            axis = axes[row, column]
            for name in COLORS:
                selected = sorted((v for v in communication if v["fractional_axis"] == axis_name and v["modulation"] == modulation and v["waveform"] == name), key=lambda v: float(v["epsilon"]))
                axis.semilogy([v["epsilon"] for v in selected], [max(float(v["ber"]), .5 / int(v["total_bits"])) for v in selected], color=COLORS[name], marker=MARKERS[name], label=name)
            axis.set_title(f"{axis_name}: {modulation} BER")
            axis.grid(alpha=.25)
        axis = axes[row, 2]
        for name in COLORS:
            selected = sorted((v for v in energy if v["fractional_axis"] == axis_name and v["waveform"] == name), key=lambda v: float(v["epsilon"]))
            axis.plot([v["epsilon"] for v in selected], [v["mean_retained_energy"] for v in selected], color=COLORS[name], marker=MARKERS[name], label=name)
        axis.set_title(f"{axis_name}: retained energy")
        axis.set_ylim(.90, 1.002)
        axis.grid(alpha=.25)
    for axis in axes[-1]:
        axis.set_xlabel("fractional severity epsilon")
    axes[1, 0].set_ylabel("BER at 18 dB")
    axes[1, 2].set_ylabel("oracle K=24 retained energy")
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, "P1_controlled_fractionality")


def phase_budget(smoke: bool) -> None:
    waveforms = load_waveforms()
    channels = 8 if smoke else 64
    bits = 65536 if smoke else 524288
    bank = sample_dd_v1(channels, SEEDS["budget"])
    modes = {name: {**{f"oracle{k}": None for k in K_VALUES if k < config.N}, "full": None} for name in waveforms}
    rows = communication_rows(bank.matrices, waveforms, modes, (qpsk_constellation(), qam16_constellation()), (18,), bits, SEEDS["symbols"] + 10, SEEDS["noise"] + 10)
    for row in rows:
        row["rho"] = int(row["k"]) / config.N
    summary = aggregate_rows(rows)
    for row in summary:
        row["rho"] = int(row["k"]) / config.N
    energy = retained_energy_rows(bank.matrices, waveforms)
    energy_summary = _energy_summary(energy)
    write_csv(DATA / "budget_per_channel.csv", rows)
    write_csv(DATA / "budget_summary.csv", summary)
    write_csv(DATA / "budget_energy_per_channel.csv", energy)
    write_csv(DATA / "budget_energy_summary.csv", energy_summary)
    plot_budget(summary, energy_summary)


def plot_budget(communication: list[dict[str, object]], energy: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for column, modulation in enumerate(("QPSK", "16QAM")):
        for name in COLORS:
            selected = sorted((v for v in communication if v["modulation"] == modulation and v["waveform"] == name), key=lambda v: float(v["rho"]))
            axes[column].semilogy([v["rho"] for v in selected], [max(float(v["ber"]), .5 / int(v["total_bits"])) for v in selected], color=COLORS[name], marker=MARKERS[name], label=name)
        axes[column].axvline(PRIMARY_RHO, color="black", ls="--", lw=.8)
        axes[column].set_title(f"{modulation} BER at 18 dB")
        axes[column].set_xlabel("interaction fraction rho=K/N")
        axes[column].grid(alpha=.25)
    for name in COLORS:
        selected = sorted((v for v in energy if v["waveform"] == name), key=lambda v: float(v["rho"]))
        axes[2].plot([v["rho"] for v in selected], [v["mean_retained_energy"] for v in selected], color=COLORS[name], marker=MARKERS[name], label=name)
    axes[2].axvline(PRIMARY_RHO, color="black", ls="--", lw=.8)
    axes[2].set_title("Retained effective-channel energy")
    axes[2].set_xlabel("interaction fraction rho=K/N")
    axes[2].set_ylim(.78, 1.01)
    axes[2].grid(alpha=.25)
    axes[0].set_ylabel("BER")
    axes[2].set_ylabel("mean retained energy")
    axes[0].legend(fontsize=8)
    save_figure(fig, "P2_budget_ber_and_energy")


def phase_constellations(smoke: bool) -> None:
    waveforms = load_waveforms()
    constellation = qam16_constellation()
    channels = 4 if smoke else 8
    blocks = 2 if smoke else 16
    samples: dict[str, np.ndarray] = {}
    summary: list[dict[str, object]] = []
    with torch.no_grad():
        for epsilon_index, epsilon in enumerate((0.0, 0.3, 0.5)):
            bank = controlled_fractionality_channels(channels, SEEDS["constellation"], epsilon, "joint")
            bits = random_bits((channels, blocks, config.N, constellation.bits_per_symbol), make_generator(SEEDS["symbols"] + 20))
            symbols, true_indices = constellation_encode(bits, constellation)
            noise = complex_awgn((channels, blocks, config.N), noise_variance(18), make_generator(SEEDS["noise"] + 20))
            samples[f"epsilon{epsilon:.1f}_truth"] = symbols.numpy().reshape(-1)
            for name, waveform in waveforms.items():
                transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
                physical = torch.einsum("cij,cbj->cbi", bank.matrices, transmitted) + noise
                received = torch.einsum("ij,cbj->cbi", waveform.mH, physical)
                operator = effective_operator(bank.matrices, waveform)
                equalizer = mismatched_lmmse_equalizer(rowwise_topk(operator, PRIMARY_K), noise_variance(18))
                estimate = torch.einsum("cij,cbj->cbi", equalizer, received)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                evm = torch.sum(torch.abs(estimate - symbols) ** 2) / torch.sum(torch.abs(symbols) ** 2)
                summary.append({
                    "epsilon": epsilon,
                    "waveform": name,
                    "k": PRIMARY_K,
                    "rho": PRIMARY_RHO,
                    "snr_db": 18,
                    "evm_rms": float(torch.sqrt(evm)),
                    "ber": float(torch.mean((decoded_bits != bits).to(torch.float64))),
                    "ser": float(torch.mean((decoded_indices != true_indices).to(torch.float64))),
                    "normalization": "none beyond common unit-power 16-QAM and project channel normalization",
                })
                samples[f"epsilon{epsilon:.1f}_{name}"] = estimate.numpy().reshape(-1)
    np.savez(DATA / "constellation_samples.npz", **samples)
    write_csv(DATA / "constellation_summary.csv", summary)
    plot_constellations(samples, summary)


def plot_constellations(samples: dict[str, np.ndarray], summary: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5), sharex=True, sharey=True)
    for row, epsilon in enumerate((0.0, 0.3, 0.5)):
        for column, name in enumerate(("Learned", "AFDM", "OTFS")):
            values = samples[f"epsilon{epsilon:.1f}_{name}"][:4000]
            record = next(v for v in summary if v["epsilon"] == epsilon and v["waveform"] == name)
            axes[row, column].scatter(values.real, values.imag, s=2, alpha=.20, color=COLORS[name], rasterized=True)
            truth = qam16_constellation().points.numpy()
            axes[row, column].scatter(truth.real, truth.imag, s=18, marker="x", color="black")
            axes[row, column].set_title(f"{name}, eps={epsilon:.1f}\nEVM={record['evm_rms']:.3f}, BER={record['ber']:.3g}", fontsize=9)
            axes[row, column].grid(alpha=.15)
    for axis in axes[-1]:
        axis.set_xlabel("in-phase")
    for axis in axes[:, 0]:
        axis.set_ylabel("quadrature")
    save_figure(fig, "P3_raw_16qam_constellations")


def phase_constellations_15db(smoke: bool) -> None:
    """Representative named-regime constellations; channel selection is waveform-invariant."""
    waveforms = load_waveforms()
    constellation = qam16_constellation()
    candidate_count = 16 if smoke else 256
    blocks = 4 if smoke else 128
    per_symbol = 4 if smoke else 200
    regime_keys = ("grid_dd", "fractional_doppler", "dd_v1")
    regime_labels = {
        "grid_dd": "GRID-DD",
        "fractional_doppler": "FRACTIONAL-DOPPLER",
        "dd_v1": "DD-v1 OFF-GRID",
    }
    protected_p3 = (
        DATA / "constellation_samples.npz",
        DATA / "constellation_summary.csv",
        FIG / "P3_raw_16qam_constellations.png",
        FIG / "P3_raw_16qam_constellations.pdf",
    )
    p3_before = {str(path.relative_to(OUT)): file_sha256(path) for path in protected_p3 if path.exists()}
    bits = random_bits(
        (blocks, config.N, constellation.bits_per_symbol),
        make_generator(SEEDS["constellation15_symbols"]),
    )
    symbols, true_indices = constellation_encode(bits, constellation)
    noise = complex_awgn(
        (blocks, config.N), noise_variance(15),
        make_generator(SEEDS["constellation15_noise"]),
    )
    plot_indices = balanced_symbol_indices(
        true_indices, symbol_count=len(constellation.points), per_symbol=per_symbol
    )
    arrays: dict[str, np.ndarray] = {}
    summary: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "snr_db": 15,
        "n": config.N,
        "k": PRIMARY_K,
        "rho": PRIMARY_RHO,
        "modulation": "16QAM",
        "candidate_channels_per_regime": candidate_count,
        "transmitted_symbols_per_panel": blocks * config.N,
        "display_samples_per_symbol": per_symbol,
        "display_samples_per_panel": per_symbol * len(constellation.points),
        "selection_rule": "closest channel to median full-information physical-channel MSE at 15 dB",
        "selection_uses_waveform": False,
        "selection_uses_finite_k": False,
        "selection_uses_ber": False,
        "receiver": "per-waveform realization-wise oracle row-top-K mismatched LMMSE",
        "post_receiver_alignment": "none",
        "normalization": "common unit-power 16-QAM and existing per-realization channel Frobenius normalization only",
        "symbols_seed": SEEDS["constellation15_symbols"],
        "noise_seed": SEEDS["constellation15_noise"],
        "run_kind": "smoke" if smoke else "production",
        "p3_hashes_before": p3_before,
        "regimes": {},
    }
    arrays["truth"] = symbols.detach().cpu().numpy().reshape(-1)
    arrays["true_indices"] = true_indices.detach().cpu().numpy().reshape(-1)
    arrays["plot_indices"] = plot_indices
    arrays["bits"] = bits.detach().cpu().numpy()
    arrays["physical_noise"] = noise.detach().cpu().numpy()
    with torch.no_grad():
        for regime_index, regime_key in enumerate(regime_keys):
            definition = REGIME_BY_KEY[regime_key]
            bank_seed = SEEDS["constellation15_bank"] + 100 * regime_index
            bank = sample_regime(definition.stage2e(), candidate_count, bank_seed)
            selected_index, full_mse = median_full_information_channel(bank.matrices, 15)
            channel = bank.matrices[selected_index]
            arrays[f"{regime_key}_channel"] = channel.detach().cpu().numpy()
            arrays[f"{regime_key}_candidate_full_mse"] = full_mse
            selected_gains = bank.pre_normalization_gains[selected_index]
            metadata["regimes"][regime_key] = {
                "label": regime_labels[regime_key],
                "bank_seed": bank_seed,
                "delay_kind": definition.delay_kind,
                "delay_max": definition.delay_max,
                "doppler_kind": definition.doppler_kind,
                "doppler_max": definition.doppler_max,
                "paths": definition.paths,
                "selected_channel_index": selected_index,
                "selected_full_information_mse": float(full_mse[selected_index]),
                "candidate_median_full_information_mse": float(np.median(full_mse)),
                "delays": bank.delays[selected_index].tolist(),
                "dopplers": bank.dopplers[selected_index].tolist(),
                "pre_normalization_gains": [
                    {"real": float(value.real), "imag": float(value.imag)} for value in selected_gains
                ],
                "normalization_scale": float(bank.normalization_scales[selected_index]),
            }
            for name in ("Learned", "OTFS", "AFDM"):
                waveform = waveforms[name]
                transmitted = torch.einsum("ij,bj->bi", waveform, symbols)
                physical = torch.einsum("ij,bj->bi", channel, transmitted) + noise
                received = torch.einsum("ij,bj->bi", waveform.mH, physical)
                operator = effective_operator(channel, waveform)
                truncated = rowwise_topk(operator, PRIMARY_K)
                support_counts = torch.count_nonzero(truncated, dim=-1)
                if not torch.all(support_counts == PRIMARY_K):
                    raise RuntimeError(f"oracle row budget failed for {regime_key}/{name}")
                equalizer = mismatched_lmmse_equalizer(truncated, noise_variance(15))
                estimate = torch.einsum("ij,bj->bi", equalizer, received)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                evm = torch.sum(torch.abs(estimate - symbols) ** 2) / torch.sum(torch.abs(symbols) ** 2)
                displayed = estimate.reshape(-1)[torch.as_tensor(plot_indices)]
                clipped = (torch.abs(displayed.real) > 2.25) | (torch.abs(displayed.imag) > 2.25)
                summary.append({
                    "regime": regime_key,
                    "regime_label": regime_labels[regime_key],
                    "selected_channel_index": selected_index,
                    "waveform": name,
                    "modulation": constellation.name,
                    "snr_db": 15,
                    "k": PRIMARY_K,
                    "rho": PRIMARY_RHO,
                    "evm_rms": float(torch.sqrt(evm)),
                    "ber": float(torch.mean((decoded_bits != bits).to(torch.float64))),
                    "ser": float(torch.mean((decoded_indices != true_indices).to(torch.float64))),
                    "total_bits": bits.numel(),
                    "total_symbols": symbols.numel(),
                    "support_nnz_per_row_min": int(support_counts.min()),
                    "support_nnz_per_row_max": int(support_counts.max()),
                    "display_clipped_fraction": float(clipped.to(torch.float64).mean()),
                    "alignment": "none",
                })
                arrays[f"{regime_key}_{name}"] = estimate.detach().cpu().numpy().reshape(-1)
    np.savez(DATA / "channel_type_constellation_samples_15db.npz", **arrays)
    write_csv(DATA / "channel_type_constellation_summary_15db.csv", summary)
    p3_after = {str(path.relative_to(OUT)): file_sha256(path) for path in protected_p3 if path.exists()}
    if p3_after != p3_before:
        raise RuntimeError("existing P3 constellation package changed during additive 15 dB phase")
    metadata["p3_hashes_after"] = p3_after
    metadata["p3_byte_identical"] = True
    write_json(DATA / "channel_type_constellation_channels_15db.json", metadata)
    plot_channel_type_constellations(arrays, summary, regime_keys, regime_labels)
    plot_integer_offgrid_constellations(arrays)


def plot_channel_type_constellations(
    arrays: dict[str, np.ndarray],
    summary: list[dict[str, object]],
    regime_keys: tuple[str, ...],
    regime_labels: dict[str, str],
) -> None:
    constellation = qam16_constellation()
    plot_indices = arrays["plot_indices"]
    boundaries = np.asarray((-2 / math.sqrt(10), 0.0, 2 / math.sqrt(10)))
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 10.2), sharex=True, sharey=True)
    for row, regime_key in enumerate(regime_keys):
        for column, name in enumerate(("Learned", "AFDM", "OTFS")):
            axis = axes[row, column]
            values = arrays[f"{regime_key}_{name}"]
            displayed = values[plot_indices]
            axis.scatter(
                displayed.real, displayed.imag,
                s=3, alpha=.24, color=COLORS[name], rasterized=True,
            )
            for boundary in boundaries:
                axis.axvline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
                axis.axhline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
            truth = constellation.points.detach().cpu().numpy()
            axis.scatter(
                truth.real, truth.imag, s=42, marker="x", color=COLORS[name],
                linewidths=1.6, alpha=1.0, zorder=3,
            )
            if row == 0:
                axis.set_title(name, fontsize=10)
            axis.set_xlim(-2.25, 2.25)
            axis.set_ylim(-2.25, 2.25)
            axis.set_aspect("equal")
            axis.grid(alpha=.10)
        axes[row, 0].set_ylabel(f"{regime_labels[regime_key]}\nquadrature")
    for axis in axes[-1]:
        axis.set_xlabel("in-phase")
    save_figure(fig, "P7_channel_types_15db_constellations")


def plot_integer_offgrid_constellations(arrays: dict[str, np.ndarray]) -> None:
    constellation = qam16_constellation()
    plot_indices = arrays["plot_indices"]
    boundaries = np.asarray((-2 / math.sqrt(10), 0.0, 2 / math.sqrt(10)))
    regimes = (("grid_dd", "Integer DD"), ("dd_v1", "Off-Grid DD"))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), sharex=True, sharey=True)
    for row, (regime_key, regime_label) in enumerate(regimes):
        for column, name in enumerate(("Learned", "AFDM", "OTFS")):
            axis = axes[row, column]
            values = arrays[f"{regime_key}_{name}"][plot_indices]
            axis.scatter(
                values.real, values.imag,
                s=3, alpha=.24, color=COLORS[name], rasterized=True,
            )
            for boundary in boundaries:
                axis.axvline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
                axis.axhline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
            truth = constellation.points.detach().cpu().numpy()
            axis.scatter(
                truth.real, truth.imag, s=42, marker="x", color="black",
                linewidths=1.6, alpha=1.0, zorder=3,
            )
            if row == 0:
                axis.set_title(name, fontsize=10)
            axis.set_xlim(-1.5, 1.5)
            axis.set_ylim(-1.5, 1.5)
            axis.set_aspect("equal")
            axis.grid(alpha=.10)
        axes[row, 0].set_ylabel(f"{regime_label}\nquadrature")
    for axis in axes[-1]:
        axis.set_xlabel("in-phase")
    save_figure(fig, "P8_integer_offgrid_15db_constellations", w_pad=2.0)


def phase_constellations_contrast(smoke: bool) -> None:
    """Paired integer/off-grid diagnostic with waveform-independent channel selection."""
    waveforms = load_waveforms()
    constellation = qam16_constellation()
    candidate_count = 8 if smoke else 64
    blocks = 4 if smoke else 128
    per_symbol = 4 if smoke else 200
    integer = controlled_fractionality_channels(
        candidate_count, SEEDS["fractionality"], 0.0, "joint"
    )
    offgrid = controlled_fractionality_channels(
        candidate_count, SEEDS["fractionality"], 0.5, "joint"
    )
    selected_index, full_mse = median_full_information_channel(offgrid.matrices, 18)
    bits = random_bits(
        (blocks, config.N, constellation.bits_per_symbol),
        make_generator(SEEDS["contrast_symbols"]),
    )
    symbols, true_indices = constellation_encode(bits, constellation)
    noise = complex_awgn(
        (blocks, config.N), noise_variance(18),
        make_generator(SEEDS["contrast_noise"]),
    )
    plot_indices = balanced_symbol_indices(
        true_indices, symbol_count=len(constellation.points), per_symbol=per_symbol
    )
    arrays: dict[str, np.ndarray] = {
        "truth": symbols.detach().cpu().numpy().reshape(-1),
        "true_indices": true_indices.detach().cpu().numpy().reshape(-1),
        "plot_indices": plot_indices,
        "bits": bits.detach().cpu().numpy(),
        "physical_noise": noise.detach().cpu().numpy(),
        "integer_dd_channel": integer.matrices[selected_index].detach().cpu().numpy(),
        "offgrid_dd_channel": offgrid.matrices[selected_index].detach().cpu().numpy(),
        "offgrid_candidate_full_mse": full_mse,
    }
    summary: list[dict[str, object]] = []
    with torch.no_grad():
        for regime_key, channel in (
            ("integer_dd", integer.matrices[selected_index]),
            ("offgrid_dd", offgrid.matrices[selected_index]),
        ):
            for name in ("Learned", "AFDM", "OTFS"):
                waveform = waveforms[name]
                transmitted = torch.einsum("ij,bj->bi", waveform, symbols)
                physical = torch.einsum("ij,bj->bi", channel, transmitted) + noise
                received = torch.einsum("ij,bj->bi", waveform.mH, physical)
                operator = effective_operator(channel, waveform)
                truncated = rowwise_topk(operator, PRIMARY_K)
                equalizer = mismatched_lmmse_equalizer(truncated, noise_variance(18))
                estimate = torch.einsum("ij,bj->bi", equalizer, received)
                decoded_bits, decoded_indices = nearest_neighbor_decode(estimate, constellation)
                error = torch.abs(estimate - symbols).reshape(-1)
                summary.append({
                    "regime": regime_key,
                    "waveform": name,
                    "snr_db": 18,
                    "k": PRIMARY_K,
                    "evm_rms": float(torch.sqrt(
                        torch.sum(error**2) / torch.sum(torch.abs(symbols) ** 2)
                    )),
                    "error_magnitude_p90": float(torch.quantile(error, .9)),
                    "ber": float(torch.mean((decoded_bits != bits).to(torch.float64))),
                    "ser": float(torch.mean((decoded_indices != true_indices).to(torch.float64))),
                    "total_bits": bits.numel(),
                    "total_symbols": symbols.numel(),
                    "alignment": "none",
                })
                arrays[f"{regime_key}_{name}"] = estimate.detach().cpu().numpy().reshape(-1)
    np.savez(DATA / "contrast_constellation_samples_18db.npz", **arrays)
    write_csv(DATA / "contrast_constellation_summary_18db.csv", summary)
    write_json(DATA / "contrast_constellation_selection_18db.json", {
        "snr_db": 18,
        "candidate_channels": candidate_count,
        "selected_channel_index": selected_index,
        "selected_offgrid_full_information_mse": float(full_mse[selected_index]),
        "candidate_median_offgrid_full_information_mse": float(np.median(full_mse)),
        "selection_rule": "closest off-grid channel to median full-information physical-channel MSE at 18 dB",
        "selection_uses_waveform": False,
        "selection_uses_finite_k": False,
        "selection_uses_ber_or_evm": False,
        "paired_integer_control_uses_same_channel_index_anchors_and_gains": True,
        "evidence_role": "single-channel illustration; P1 64-channel sweep remains population evidence",
        "epsilon": {"integer_dd": 0.0, "offgrid_dd": 0.5},
        "fractional_axis": "joint",
        "symbols_seed": SEEDS["contrast_symbols"],
        "noise_seed": SEEDS["contrast_noise"],
        "run_kind": "smoke" if smoke else "production",
    })
    plot_contrast_constellations(arrays)


def plot_contrast_constellations(arrays: dict[str, np.ndarray]) -> None:
    constellation = qam16_constellation()
    plot_indices = arrays["plot_indices"]
    boundaries = np.asarray((-2 / math.sqrt(10), 0.0, 2 / math.sqrt(10)))
    regimes = (("integer_dd", "Integer DD"), ("offgrid_dd", "Off-Grid DD"))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), sharex=True, sharey=True)
    for row, (regime_key, regime_label) in enumerate(regimes):
        for column, name in enumerate(("Learned", "AFDM", "OTFS")):
            axis = axes[row, column]
            values = arrays[f"{regime_key}_{name}"][plot_indices]
            axis.scatter(
                values.real, values.imag,
                s=3, alpha=.24, color=COLORS[name], rasterized=True,
            )
            for boundary in boundaries:
                axis.axvline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
                axis.axhline(boundary, color="#64748b", ls="--", lw=.55, alpha=.55)
            truth = constellation.points.detach().cpu().numpy()
            axis.scatter(
                truth.real, truth.imag, s=42, marker="x", color="black",
                linewidths=1.6, alpha=1.0, zorder=3,
            )
            if row == 0:
                axis.set_title(name, fontsize=10)
            axis.set_xlim(-1.5, 1.5)
            axis.set_ylim(-1.5, 1.5)
            axis.set_aspect("equal")
            axis.grid(alpha=.10)
        axes[row, 0].set_ylabel(f"{regime_label}\nquadrature")
    for axis in axes[-1]:
        axis.set_xlabel("in-phase")
    save_figure(fig, "P9_controlled_integer_offgrid_18db_constellations", w_pad=2.0)


def phase_structure(smoke: bool) -> None:
    reused = np.load(STAGE2B_ENERGY)["Learned"]
    graphs, rows = graph_diagnostics(reused)
    np.savez(DATA / "authoritative_threshold_graphs.npz", full_ensemble_energy=reused, **graphs)
    write_csv(DATA / "authoritative_threshold_graph_metrics.csv", rows)
    plot_threshold_graphs(graphs)
    plot_graph_metrics(rows)

    bank = sample_dd_v1(64 if smoke else config.N_STAGE1A2_EVAL_CHANNELS, SEEDS["structure"]).matrices
    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, object] = {}
    for k, path in TRAINED.items():
        waveform = torch.load(path, weights_only=True)
        energy, _ = ensemble_operator_statistics(bank, {"Learned": waveform}, k=k)
        full_energy = energy["Learned"]
        centroids = spectral_centroids(waveform)
        overlap = spectral_overlap_matrix(waveform)
        labels, summary = community_summary(full_energy, overlap, centroids)
        values = waveform.detach().resolve_conj().numpy()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(values, n=512, axis=0), axes=0)) ** 2
        spectrum /= np.maximum(spectrum.max(axis=0, keepdims=True), 1e-300)
        arrays.update({f"k{k}_energy": full_energy, f"k{k}_spectral_power": spectrum, f"k{k}_time_magnitude": np.abs(values), f"k{k}_labels": labels})
        summaries[str(k)] = {**summary, "training_k": k, "waveform_sha256": file_sha256(path), "full_operator_untruncated": True}
    np.savez(DATA / "training_k_full_structures.npz", **arrays)
    write_json(DATA / "training_k_community_summaries.json", summaries)
    plot_training_k_structures(arrays)


def plot_threshold_graphs(graphs: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.3), sharex=True, sharey=True)
    maximum = max(float(value.max()) for value in graphs.values())
    for axis, k in zip(axes.flat, K_VALUES):
        values = graphs[f"k{k}"]
        axis.imshow(np.maximum(10 * np.log10(np.maximum(values / maximum, 1e-12)), -50), origin="upper", cmap="magma", vmin=-50, vmax=0)
        axis.set_title(f"K={k}, rho={k/config.N:.3f}", fontsize=9)
    axes[-1, 0].set_xlabel("input symbol")
    axes[0, 0].set_ylabel("output symbol")
    save_figure(fig, "P4_same_waveform_threshold_graphs")


def plot_graph_metrics(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    rho = [row["rho"] for row in rows]
    axes[0].plot(rho, [row["community_count"] for row in rows], marker="o", label="spectral communities")
    axes[0].plot(rho, [row["connected_components"] for row in rows], marker="s", label="connected components")
    axes[1].plot(rho, [row["weighted_modularity"] for row in rows], marker="o")
    axes[2].plot(rho, [row["within_energy_fraction"] for row in rows], marker="o")
    axes[0].set_ylabel("count"); axes[1].set_ylabel("weighted modularity"); axes[2].set_ylabel("within-community energy")
    for axis in axes:
        axis.set_xlabel("rho=K/N"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    save_figure(fig, "P5_threshold_graph_diagnostics")


def scaling_ks(n: int) -> tuple[int, ...]:
    """Represent the P4 rho grid exactly at a scaled dimension."""
    values = tuple(round(rho * n) for rho in SCALING_RHO_VALUES)
    if len(values) != len(set(values)) or any(k < 1 or k > n for k in values):
        raise ValueError(f"P4 rho grid is not representable at N={n}: {values}")
    return values


def scaled_ensemble_energy(channels: torch.Tensor, waveform: torch.Tensor, *, chunk: int = 16) -> np.ndarray:
    """Mean untruncated |W^H H W|^2 for one fixed scaled waveform."""
    n = waveform.shape[-1]
    total = torch.zeros((n, n), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(channels), chunk):
            effective = effective_operator(channels[start : start + chunk], waveform)
            total += torch.sum(torch.abs(effective) ** 2, dim=0)
    return (total / len(channels)).numpy()


def plot_scaled_threshold_graphs(n: int, graphs: dict[str, np.ndarray], ks: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.3), sharex=True, sharey=True)
    maximum = max(float(value.max()) for value in graphs.values())
    for axis, k in zip(axes.flat, ks):
        values = graphs[f"k{k}"]
        image = np.maximum(10 * np.log10(np.maximum(values / maximum, 1e-12)), -50)
        axis.imshow(image, origin="upper", cmap="magma", vmin=-50, vmax=0)
        axis.set_title(f"K={k}, rho={k / n:.3f}", fontsize=9)
    axes[-1, 0].set_xlabel("input symbol")
    axes[0, 0].set_ylabel("output symbol")
    save_figure(fig, f"P4_N{n}_same_waveform_threshold_graphs")


def phase_scaling_structure(smoke: bool) -> None:
    """Extend P4 to preserved N=128/N=256 waveforms without retraining."""
    count = 64 if smoke else SCALING_STRUCTURE_CHANNELS
    for n in SCALING_DIMENSIONS:
        checkpoint = ROOT / "outputs/stage_pub1b/checkpoints" / f"N{n}" / f"learned_n{n}.pt"
        metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
        digest = file_sha256(checkpoint)
        if digest != metadata["checkpoint_sha256"]:
            raise RuntimeError(f"N={n} checkpoint hash mismatch")
        waveform = torch.load(checkpoint, weights_only=True)
        residual = float(torch.linalg.norm(waveform.mH @ waveform - torch.eye(n, dtype=waveform.dtype)))
        if residual > 1e-10:
            raise RuntimeError(f"N={n} waveform is not unitary: {residual}")
        full_energy = scaled_ensemble_energy(sample_scaled_dd(count, SCALING_STRUCTURE_SEED + n, n).matrices, waveform)
        ks = scaling_ks(n)
        graphs, rows = graph_diagnostics(full_energy, ks)
        for row in rows:
            row.update({"n": n, "channels": count, "checkpoint_sha256": digest})
        np.savez_compressed(DATA / f"N{n}_threshold_graphs.npz", full_ensemble_energy=full_energy, **graphs)
        write_csv(DATA / f"N{n}_threshold_graph_metrics.csv", rows)
        write_json(DATA / f"N{n}_threshold_graph_provenance.json", {
            "n": n, "channels": count, "seed": SCALING_STRUCTURE_SEED + n,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest,
            "unitarity_residual": residual, "rho_values": SCALING_RHO_VALUES,
            "k_values": ks, "training_steps": 0,
            "channel_sampler": "sample_scaled_dd under the frozen Pub-1B normalized scaling law",
            "interpretation": "one fixed waveform and full ensemble map per N; only the P4 threshold changes",
        })
        plot_scaled_threshold_graphs(n, graphs, ks)


def plot_training_k_structures(arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    maximum = max(float(arrays[f"k{k}_energy"].max()) for k in (16, 24, 32))
    for row, k in enumerate((16, 24, 32)):
        energy = arrays[f"k{k}_energy"]
        axes[row, 0].imshow(np.maximum(10 * np.log10(np.maximum(energy / maximum, 1e-12)), -50), origin="upper", cmap="magma", vmin=-50, vmax=0)
        axes[row, 1].imshow(arrays[f"k{k}_spectral_power"], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(0, config.N - 1, -.5, .5))
        axes[row, 2].imshow(arrays[f"k{k}_time_magnitude"], origin="upper", aspect="auto", cmap="cividis")
        axes[row, 0].set_ylabel(f"trained K={k}")
        axes[row, 1].set_ylabel("normalized frequency")
        axes[row, 2].set_ylabel("time sample")
    axes[0, 0].set_title("full ensemble energy")
    axes[0, 1].set_title("basis spectral power")
    axes[0, 2].set_title("basis time magnitude")
    for axis in axes[-1]:
        axis.set_xlabel("basis index")
    save_figure(fig, "P6_different_training_k_structures")


def phase_report() -> None:
    fractionality = read_csv(DATA / "fractionality_summary.csv")
    energy = read_csv(DATA / "fractionality_energy_summary.csv")
    budget = read_csv(DATA / "budget_summary.csv")
    graph = read_csv(DATA / "authoritative_threshold_graph_metrics.csv")
    ber_intervals = read_csv(DATA / "fractionality_paired_ber_intervals.csv")
    energy_intervals = read_csv(DATA / "fractionality_paired_energy_intervals.csv")
    trained = json.loads((DATA / "training_k_community_summaries.json").read_text())
    channel_type = read_csv(DATA / "channel_type_constellation_summary_15db.csv") if (DATA / "channel_type_constellation_summary_15db.csv").exists() else []
    scaling = {n: read_csv(DATA / f"N{n}_threshold_graph_metrics.csv") for n in SCALING_DIMENSIONS if (DATA / f"N{n}_threshold_graph_metrics.csv").exists()}

    def comm(axis: str, epsilon: str, modulation: str, waveform: str) -> dict[str, str]:
        return next(row for row in fractionality if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["modulation"] == modulation and row["waveform"] == waveform)

    def retained(axis: str, epsilon: str, waveform: str) -> float:
        return float(next(row for row in energy if row["fractional_axis"] == axis and row["epsilon"] == epsilon and row["waveform"] == waveform)["mean_retained_energy"])

    joint = {name: float(comm("joint", "0.5", "16QAM", name)["ber"]) for name in COLORS}
    ongrid = {name: float(comm("joint", "0.0", "16QAM", name)["ber"]) for name in COLORS}
    joint_energy = {name: retained("joint", "0.5", name) for name in COLORS}
    best_classical = min(joint["AFDM"], joint["OTFS"])
    selected_ber_intervals = [row for row in ber_intervals if row["fractional_axis"] == "joint" and row["epsilon"] == "0.5" and row["modulation"] == "16QAM"]
    selected_energy_intervals = [row for row in energy_intervals if row["fractional_axis"] == "joint" and row["epsilon"] == "0.5"]
    supported = all(float(row["ci95_high"]) < 0 for row in selected_ber_intervals) and all(float(row["ci95_low"]) > 0 for row in selected_energy_intervals)
    classification = "SUPPORTED" if supported else "NOT SUPPORTED"
    k24 = next(row for row in graph if row["k"] == "24")
    constellation_lines = []
    if channel_type:
        constellation_lines = [
            "",
            "## Representative 15 dB constellation geometry",
            "",
            "- P7 compares independently selected physical-median GRID-DD, FRACTIONAL-DOPPLER, and DD-v1 channels; selection uses waveform-invariant full-information MSE only.",
        ]
        for regime_key in ("grid_dd", "fractional_doppler", "dd_v1"):
            selected = {row["waveform"]: row for row in channel_type if row["regime"] == regime_key}
            label = selected["Learned"]["regime_label"]
            constellation_lines.append(
                f"- {label} raw EVM/BER: Learned {float(selected['Learned']['evm_rms']):.3f}/{float(selected['Learned']['ber']):.4g}, "
                f"OTFS {float(selected['OTFS']['evm_rms']):.3f}/{float(selected['OTFS']['ber']):.4g}, "
                f"AFDM {float(selected['AFDM']['evm_rms']):.3f}/{float(selected['AFDM']['ber']):.4g}."
            )
        constellation_lines.append("- These single-channel panels diagnose cluster geometry; the controlled multi-channel P1 sweep remains the population-level causal evidence.")
    lines = [
        "# Pub-1C Phase-1 publication completion",
        "",
        f"## Classification: {classification}",
        "",
        "Frozen claim tested: under fractional/off-grid DD channels, the authoritative Learned waveform remains more recoverable when every waveform receives its own oracle strongest interaction fraction. This is regime-specific, not a universal waveform or compute claim.",
        "",
        "## Controlled result",
        "",
        f"- Joint epsilon=0.5, 16-QAM, rho=0.375 BER: Learned {joint['Learned']:.6g}, AFDM {joint['AFDM']:.6g}, OTFS {joint['OTFS']:.6g}.",
        f"- Corresponding retained energy: Learned {joint_energy['Learned']:.6f}, AFDM {joint_energy['AFDM']:.6f}, OTFS {joint_energy['OTFS']:.6f}.",
        "- Paired 95% channel-bootstrap BER differences (Learned minus baseline): " + "; ".join(f"{row['contrast']} [{float(row['ci95_low']):.6g}, {float(row['ci95_high']):.6g}]" for row in selected_ber_intervals) + ".",
        "- Paired 95% retained-energy differences: " + "; ".join(f"{row['contrast']} [{float(row['ci95_low']):.6g}, {float(row['ci95_high']):.6g}]" for row in selected_energy_intervals) + ".",
        f"- On-grid 16-QAM BER: Learned {ongrid['Learned']:.6g}, AFDM {ongrid['AFDM']:.6g}, OTFS {ongrid['OTFS']:.6g}; classical behavior is reported rather than suppressed.",
        "- Budget curves include rho=1 and must be interpreted jointly with retained energy; no fixed K=24 scaling claim is made.",
        *constellation_lines,
        "",
        "## Structural interpretation",
        "",
        f"- The same authoritative full map at K=24 yields {k24['community_count']} spectral communities, modularity {float(k24['weighted_modularity']):.4f}, and within-community energy {float(k24['within_energy_fraction']):.4f}.",
        "- Counts change across thresholds; the figures test rather than assume an exact three-group law.",
        "- Separately trained K=16/24/32 tensors are compared only through full untruncated maps. Their detected group counts are " + ", ".join(f"K={k}: {trained[k]['group_count']}" for k in ("16", "24", "32")) + ".",
        *([f"- P4 N={n} extension reuses its preserved scaling checkpoint: at rho=0.375, {next(row for row in scaling[n] if row['rho'] == '0.375')['community_count']} communities retain {float(next(row for row in scaling[n] if row['rho'] == '0.375')['within_energy_fraction']):.4f} within-community energy." for n in SCALING_DIMENSIONS] if len(scaling) == len(SCALING_DIMENSIONS) else []),
        "",
        "## Method boundaries",
        "",
        "- Each waveform and realization uses its own row-top-K support. Learned support is never imposed on AFDM or OTFS.",
        "- Raw 16-QAM outputs use identical unit-power symbols/noise and no phase, gain, or constellation alignment.",
        "- Learned training steps: zero. Fixed-Q remains secondary. Phase 2 physical-front-end and imperfect-CSI work was not specified or executed.",
    ]
    (ROOT / "reports/stage_pub1c_summary.md").write_text("\n".join(lines) + "\n")
    write_json(DATA / "claim_classification.json", {"classification": classification, "criteria": {"both_paired_ber_ci95_upper_below_zero": all(float(row["ci95_high"]) < 0 for row in selected_ber_intervals), "both_paired_retained_energy_ci95_lower_above_zero": all(float(row["ci95_low"]) > 0 for row in selected_energy_intervals)}, "values": {"joint_16qam_ber": joint, "joint_retained_energy": joint_energy, "on_grid_16qam_ber": ongrid, "paired_ber_intervals": selected_ber_intervals, "paired_energy_intervals": selected_energy_intervals}})


def phase_manifest() -> None:
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "dataset_manifest.json")
    write_json(DATA / "dataset_manifest.json", {
        "files": [{"path": str(path.relative_to(OUT)), "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in files],
        "file_count": len(files),
    })


def phase_all(smoke: bool) -> None:
    phase_provenance(smoke)
    phase_fractionality(smoke)
    phase_budget(smoke)
    phase_constellations(smoke)
    phase_constellations_15db(smoke)
    phase_constellations_contrast(smoke)
    phase_structure(smoke)
    phase_scaling_structure(smoke)
    phase_report()
    phase_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("provenance", "fractionality", "budget", "constellations", "constellations-15db", "constellations-contrast", "structure", "scaling-structure", "report", "manifest", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.phase == "all":
        phase_all(args.smoke)
    else:
        function = globals()[f"phase_{args.phase.replace('-', '_')}"]
        function(*( [args.smoke] if args.phase in ("provenance", "fractionality", "budget", "constellations", "constellations-15db", "constellations-contrast", "structure", "scaling-structure") else []))
    print(f"PUB-1C {args.phase.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
