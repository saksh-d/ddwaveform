"""Resumable Stage Pub-1A frozen-waveform qualification runner."""

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
from src.channel import COMPLEX_DTYPE, assemble_dd_channel, make_generator
from src.metrics import constellation_encode, qam16_constellation, qam64_constellation, qpsk_constellation, random_bits
from src.pub1a.core import (
    CHECKPOINT_SHA256, CONSTELLATIONS, K_SWEEP, PRIMARY_WAVEFORMS, REGIME_BY_KEY, REGIMES,
    SAVED_WAVEFORMS, SELECTED_ATOMS, SNRS, afdm_structural_audit, aggregate_rows,
    analytic_mse_rows, configuration_record, fixed_statistical_masks,
    matched_fractionality_channels, practicality_data, waveform_set,
)
from src.receiver import effective_operator
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2b import path_resolved_operators
from src.stage2e import atom_spectral_time_metrics, communication_rows, sample_regime, stft_power


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage_pub1a"
DATA = OUT / "data"
FIG = OUT / "figures"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE2E = ROOT / "outputs" / "stage2e"
COLORS = {"Learned": "#d97706", "OTFS": "#15803d", "AFDM": "#dc2626", "OFDM": "#2563eb"}
MARKERS = {"Learned": "o", "OTFS": "s", "AFDM": "^", "OFDM": "D"}
SEEDS = {"validation": 12100, "support": 12200, "test": 12300, "symbols": 12400,
         "noise": 12500, "fractionality": 12600, "papr": 12700}


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, name: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def provenance() -> dict[str, object]:
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "target_ref": "origin/pub", "learned_checkpoint": str(FROZEN.relative_to(ROOT)),
        "learned_checkpoint_sha256": file_sha256(FROZEN), "learned_training_steps": 0,
        "seeds": SEEDS, "configuration": configuration_record(),
    }


def load_selected() -> tuple[torch.Tensor, int, dict[str, object]]:
    learned, checkpoint = load_frozen_waveform(FROZEN)
    audit_path = DATA / "afdm_audit.json"
    if not audit_path.exists():
        raise RuntimeError("run --phase afdm-audit first")
    audit = json.loads(audit_path.read_text())
    if not audit["hard_gate"]:
        raise RuntimeError("AFDM hard gate did not pass")
    return learned, int(audit["selected_multiplier_2Nc1"]), checkpoint


def phase_afdm_audit(smoke: bool) -> None:
    learned, checkpoint = load_frozen_waveform(FROZEN)
    validation = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 16 if smoke else 256, SEEDS["validation"]).matrices
    audit = afdm_structural_audit(validation)
    audit.update(checkpoint=checkpoint, learned_training_steps=0)
    write_json(DATA / "afdm_audit.json", audit)
    write_json(DATA / "provenance.json", provenance())
    write_json(DATA / "reuse_inventory.json", {
        "source_stage": "Stage-2E", "source_commit": "6d99f42aa4c6fd6d68b57f01f5e4d2285300cdb2",
        "source_directory": "outputs/stage2e", "files_preserved": len([p for p in STAGE2E.rglob("*") if p.is_file()]),
        "exact_reuse": ["DD-v1 QPSK/16-QAM Learned, OTFS, OFDM full K=64 and fixed statistical K=24 rows"],
        "regenerate": "all AFDM communication rows because final admissible AFDM differs from Stage-2E slope 1; all 64-QAM and new regime/K points were absent",
    })


def _run_rows(regime_key: str, waveforms: dict[str, torch.Tensor], modes, channels: int,
              target_bits: int, seed_offset: int, constellations=CONSTELLATIONS,
              snr_override: dict[str, tuple[int, ...]] | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    regime = REGIME_BY_KEY[regime_key]
    bank = sample_regime(regime.stage2e(), channels, SEEDS["test"] + seed_offset).matrices
    masks, _ = fixed_statistical_masks(waveforms, regime, tuple(sorted(set(modes))), 128 if channels < 32 else 1024, SEEDS["support"] + seed_offset)
    requested = {}
    for name in waveforms:
        requested[name] = {}
        for k in modes:
            requested[name]["full" if k == 64 else f"fixedStatK{k}"] = None if k == 64 else masks[name][f"fixedStatK{k}"]
    raw = []
    for modulation in constellations:
        snrs = (snr_override or SNRS)[modulation.name]
        values = communication_rows(bank, waveforms, requested, (modulation,), tuple(snrs), target_bits,
                                    SEEDS["symbols"] + seed_offset + 1000 * modulation.bits_per_symbol,
                                    SEEDS["noise"] + seed_offset + 1000 * modulation.bits_per_symbol)
        raw.extend({"regime": regime_key, **row} for row in values)
    analytic = analytic_mse_rows(bank, waveforms, requested, tuple(sorted({v for values in (snr_override or SNRS).values() for v in values})))
    return raw, [{"regime": regime_key, **row} for row in analytic]


def _reuse_stage2e_nominal() -> list[dict[str, object]]:
    rows = read_csv(STAGE2E / "nominal_communication_per_channel.csv")
    output = []
    for row in rows:
        if row["waveform"] not in ("Learned", "OTFS", "OFDM") or row["receiver_mode"] not in ("full", "fixedQK24"):
            continue
        output.append({"regime": "dd_v1", "channel_index": int(row["channel_index"]),
                       "modulation": row["modulation"], "snr_db": int(row["snr_db"]), "waveform": row["waveform"],
                       "receiver_mode": "fixedStatK24" if row["receiver_mode"] == "fixedQK24" else "full",
                       "k": int(row["k"]), "bit_errors": int(row["bit_errors"]), "total_bits": int(row["total_bits"]),
                       "symbol_errors": int(row["symbol_errors"]), "total_symbols": int(row["total_symbols"]),
                       "provenance": "reused Stage-2E exact DD-v1 configuration"})
    return output


def phase_communication(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    raw = []; analytic = []
    count = 8 if smoke else 64; bits = 8192 if smoke else 262144
    for index, regime in enumerate(("grid_dd", "fractional_doppler", "fractional_delay")):
        values, mse = _run_rows(regime, waveforms, (24, 64), count, bits, index * 10)
        raw.extend(values); analytic.extend(mse)
    if smoke:
        values, mse = _run_rows("dd_v1", waveforms, (24, 64), count, bits, 30)
        raw.extend(values); analytic.extend(mse)
    else:
        raw.extend(_reuse_stage2e_nominal())
        # Regenerate final AFDM with Stage-2E banks/seeds and add new 64-QAM for all waveforms.
        dd = sample_regime(REGIME_BY_KEY["dd_v1"].stage2e(), 512, 11101).matrices
        afdm_only = {"AFDM": waveforms["AFDM"]}
        masks, _ = fixed_statistical_masks(afdm_only, REGIME_BY_KEY["dd_v1"], (24, 64), 2048, 11102)
        modes = {"AFDM": {"fixedStatK24": masks["AFDM"]["fixedStatK24"], "full": None}}
        regen = communication_rows(dd, afdm_only, modes, (qpsk_constellation(), qam16_constellation()),
                                   (0, 5, 10, 12, 15, 18, 20, 25, 30), 1_048_576, 11103, 11104)
        raw.extend({"regime": "dd_v1", "provenance": "regenerated final AFDM on Stage-2E paired bank", **row} for row in regen)
        q64masks, _ = fixed_statistical_masks(waveforms, REGIME_BY_KEY["dd_v1"], (24, 64), 2048, 11102)
        q64modes = {name: {"fixedStatK24": q64masks[name]["fixedStatK24"], "full": None} for name in waveforms}
        q64 = communication_rows(dd, waveforms, q64modes, (qam64_constellation(),), SNRS["64QAM"], 524288, 11103 + 6000, 11104 + 6000)
        raw.extend({"regime": "dd_v1", "provenance": "Pub-1A new 64-QAM", **row} for row in q64)
        analytic.extend({"regime": "dd_v1", **row} for row in analytic_mse_rows(dd, waveforms, q64modes, tuple(sorted(set(SNRS["64QAM"] + SNRS["QPSK"])))))
    write_csv(DATA / "communication_per_channel.csv", raw)
    write_csv(DATA / "communication_summary.csv", aggregate_rows(raw, ("regime",)))
    write_csv(DATA / "communication_analytic_mse.csv", analytic)
    summary = aggregate_rows(raw, ("regime",))
    plot_communication(summary)
    plot_communication_capped(summary)


def phase_k_sweep(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    count = 8 if smoke else 64; bits = 8192 if smoke else 262144
    regime = REGIME_BY_KEY["dd_v1"]; bank = sample_regime(regime.stage2e(), count, SEEDS["test"] + 80).matrices
    masks, _ = fixed_statistical_masks(waveforms, regime, K_SWEEP, 128 if smoke else 2048, SEEDS["support"] + 80)
    raw = []
    for modulation, snr in ((qpsk_constellation(), 18), (qam16_constellation(), 18), (qam64_constellation(), 25)):
        rows = communication_rows(bank, waveforms, masks, (modulation,), (snr,), bits,
                                  SEEDS["symbols"] + 80 + modulation.bits_per_symbol,
                                  SEEDS["noise"] + 80 + modulation.bits_per_symbol)
        raw.extend(rows)
    summary = aggregate_rows(raw)
    write_csv(DATA / "k_sweep_per_channel.csv", raw); write_csv(DATA / "k_sweep_summary.csv", summary)
    plot_k_sweep(summary)
    comparison_masks = {name: {"fixedStatK24": masks[name]["fixedStatK24"], "oracle24": None, "full": None} for name in waveforms}
    compare = []
    for modulation, snr in ((qpsk_constellation(), 18), (qam16_constellation(), 18), (qam64_constellation(), 25)):
        compare.extend(communication_rows(bank, waveforms, comparison_masks, (modulation,), (snr,), bits,
                                          SEEDS["symbols"] + 180 + modulation.bits_per_symbol,
                                          SEEDS["noise"] + 180 + modulation.bits_per_symbol))
    compare_summary = aggregate_rows(compare)
    write_csv(DATA / "fixed_oracle_full_summary.csv", compare_summary); plot_receiver_modes(compare_summary)


def phase_stress(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    count = 8 if smoke else 64; bits = 8192 if smoke else 131072
    raw = []
    snrs = {"QPSK": (18,), "16QAM": (18,), "64QAM": (25,)}
    for index, key in enumerate(("dd_v1", "wider_doppler", "wider_delay", "more_paths")):
        values, _ = _run_rows(key, waveforms, (24,), count, bits, 220 + index * 10, snr_override=snrs)
        raw.extend(values)
    path_raw = []
    for index, paths in enumerate((1, 2, 3, 4, 5, 8)):
        temporary = REGIME_BY_KEY["dd_v1"].__class__(f"paths_{paths}", f"P={paths}", "continuous", 4.0, "continuous", .75, paths)
        bank = sample_regime(temporary.stage2e(), count, SEEDS["test"] + 300 + index).matrices
        masks, _ = fixed_statistical_masks(waveforms, temporary, (24,), 128 if smoke else 1024, SEEDS["support"] + 300 + index)
        modes = {name: {"fixedStatK24": masks[name]["fixedStatK24"]} for name in waveforms}
        for modulation, snr in ((qpsk_constellation(), 18), (qam16_constellation(), 18)):
            rows = communication_rows(bank, waveforms, modes, (modulation,), (snr,), bits,
                                      SEEDS["symbols"] + 300 + index, SEEDS["noise"] + 300 + index)
            path_raw.extend({"paths": paths, **row} for row in rows)
    gamma_raw = []
    support_regime = REGIME_BY_KEY["dd_v1"]
    masks, _ = fixed_statistical_masks(waveforms, support_regime, (24,), 128 if smoke else 1024, SEEDS["support"] + 400)
    modes = {name: {"fixedStatK24": masks[name]["fixedStatK24"]} for name in waveforms}
    for index, gamma in enumerate((0, .25, .5, .75, 1.0)):
        bank = matched_fractionality_channels(count, SEEDS["fractionality"], gamma).matrices
        rows = communication_rows(bank, waveforms, modes, (qpsk_constellation(),), (18,), bits,
                                  SEEDS["symbols"] + 400, SEEDS["noise"] + 400)
        gamma_raw.extend({"gamma": gamma, **row} for row in rows)
    stress_summary = aggregate_rows(raw, ("regime",)); path_summary = aggregate_rows(path_raw, ("paths",)); gamma_summary = aggregate_rows(gamma_raw, ("gamma",))
    write_csv(DATA / "stress_summary.csv", stress_summary); write_csv(DATA / "path_count_summary.csv", path_summary); write_csv(DATA / "fractionality_summary.csv", gamma_summary)
    plot_stress(stress_summary); plot_path_count(path_summary); plot_fractionality(gamma_summary)


def phase_operators(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    delays = torch.tensor([0., 1., 2., 3.]); grid_nu = torch.tensor([-1., 1., 0., -1.])
    frac_nu = torch.tensor([-.65, .55, .15, -.35]); frac_tau = torch.tensor([.25, 1.35, 2.45, 3.20])
    gains = torch.tensor([1., .75j, -.55, .4j], dtype=COMPLEX_DTYPE)
    cases = [("GRID-DD", delays, grid_nu), ("fractional Doppler", delays, frac_nu), ("full off-grid", frac_tau, frac_nu)]
    arrays = {}
    for case, tau, nu in cases:
        physical = assemble_dd_channel(tau, nu, gains)
        for name, waveform in waveforms.items():
            arrays[f"{case}_{name}"] = torch.abs(effective_operator(physical, waveform)).numpy()
    np.savez(DATA / "operator_arrays.npz", **arrays)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True, layout="constrained")
    vmax = max(arrays[f"GRID-DD_{name}"].max() for name in ("OFDM", "OTFS", "AFDM", "Learned"))
    for axis, name in zip(axes, ("OFDM", "OTFS", "AFDM", "Learned")):
        image = axis.imshow(arrays[f"GRID-DD_{name}"], origin="upper", vmin=0, vmax=vmax, aspect="equal"); axis.set_title(name); axis.set_xlabel("input j")
    axes[0].set_ylabel("output i (row 0 top)"); fig.colorbar(image, ax=axes, label="|A[i,j]|", shrink=.8); save_figure(fig, "A1_afdm_grid_audit", tight=False)
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True, layout="constrained")
    vmax = max(arrays[f"{case}_{name}"].max() for case, _, _ in cases for name in PRIMARY_WAVEFORMS)
    for row, (case, _, _) in enumerate(cases):
        for column, name in enumerate(("OTFS", "AFDM", "Learned")):
            image = axes[row, column].imshow(arrays[f"{case}_{name}"], origin="upper", vmin=0, vmax=vmax)
            if row == 0: axes[row, column].set_title(name)
            if column == 0: axes[row, column].set_ylabel(f"{case}\noutput i")
            if row == 2: axes[row, column].set_xlabel("input j")
    fig.colorbar(image, ax=axes, label="common-scale |A[i,j]|", shrink=.75); save_figure(fig, "A2_effective_channel_progression_common", tight=False)
    resolved = path_resolved_operators(waveforms["AFDM"], delays, grid_nu, gains)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True, layout="constrained")
    for index, axis in enumerate(axes):
        image = axis.imshow(torch.abs(resolved[index]).numpy(), origin="upper"); axis.set_title(f"path {index}"); axis.set_xlabel("input j")
    axes[0].set_ylabel("output i"); fig.colorbar(image, ax=axes, shrink=.8, label="|A_p[i,j]|"); save_figure(fig, "A3_afdm_path_resolved", tight=False)


def phase_waveform_figures(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    spectral = {}
    for name, waveform in waveforms.items():
        _, frequency, db = atom_spectral_time_metrics(waveform, 2048 if smoke else 16384); spectral[name] = db
    np.savez(DATA / "basis_spectra.npz", frequency=frequency, **spectral)
    plot_basis_spectra(frequency, spectral)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True, layout="constrained")
    for axis, name in zip(axes, ("OFDM", "OTFS", "AFDM", "Learned")):
        atom = waveforms[name][:, 16].detach().resolve_conj().numpy()
        power = stft_power(atom); image = axis.imshow(power, origin="lower", aspect="auto", extent=(0, 63, -.5, .5)); axis.set_title(name); axis.set_xlabel("time n")
    axes[0].set_ylabel("normalized frequency"); fig.colorbar(image, ax=axes, label="normalized STFT power", shrink=.8); save_figure(fig, "F3_time_frequency", tight=False)
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    n = np.arange(config.N)
    for row, name in enumerate(("OFDM", "OTFS", "AFDM", "Learned")):
        for atom in SELECTED_ATOMS:
            values = waveforms[name][:, atom].detach().resolve_conj().numpy(); axes[row, 0].plot(n, values.real, lw=.7, alpha=.75); axes[row, 1].plot(n, np.abs(values), lw=.7, alpha=.75)
        axes[row, 0].set_ylabel(name); axes[row, 0].grid(alpha=.2); axes[row, 1].grid(alpha=.2)
    axes[0, 0].set_title("Re{w_k[n]}"); axes[0, 1].set_title("|w_k[n]|"); axes[-1, 0].set_xlabel("sample n"); axes[-1, 1].set_xlabel("sample n"); save_figure(fig, "T1_selected_basis_atoms")
    generator = make_generator(12800); fig, axes = plt.subplots(4, 3, figsize=(11, 9), sharex=True)
    symbols_record = {}
    for modulation_index, constellation in enumerate((qpsk_constellation(), qam16_constellation())):
        bits = random_bits((config.N, constellation.bits_per_symbol), generator); symbols, _ = constellation_encode(bits, constellation); symbols_record[constellation.name] = symbols.numpy()
    symbols = symbols_record["16QAM"]
    for row, name in enumerate(("OFDM", "OTFS", "AFDM", "Learned")):
        x = waveforms[name].detach().resolve_conj().numpy() @ symbols
        for axis, values, title in zip(axes[row], (x.real, np.abs(x), np.abs(x) ** 2), ("Re{x[n]}", "|x[n]|", "|x[n]|²")):
            axis.plot(n, values, lw=.8); axis.grid(alpha=.2)
        axes[row, 0].set_ylabel(name)
    for axis, title in zip(axes[0], ("Re{x[n]}", "|x[n]|", "|x[n]|²")): axis.set_title(title)
    for axis in axes[-1]: axis.set_xlabel("sample n")
    save_figure(fig, "T2_common_16qam_block")


def waveform_peak_normalize_db(spectral: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Shift each waveform's complete spectrum array to one shared 0 dB peak."""
    return {name: values - np.max(values) for name, values in spectral.items()}


def plot_basis_spectra(frequency: np.ndarray, spectral: dict[str, np.ndarray]) -> None:
    """Render peak-normalized F1/F2 and Learned-only supplements."""
    order = ("OFDM", "OTFS", "AFDM", "Learned")
    normalized = waveform_peak_normalize_db(spectral)
    ylabel = "waveform-peak-normalized\nmagnitude (dB)"
    yticks = (-80, -60, -40, -20, 0)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)
    for axis, name in zip(axes, order):
        axis.plot(frequency, normalized[name], lw=.45, alpha=.22, color=COLORS[name])
        axis.set_title(name); axis.set_ylabel(ylabel); axis.set_ylim(-80, 1); axis.set_yticks(yticks); axis.grid(alpha=.2)
    axes[-1].set_xlabel("normalized frequency")
    save_figure(fig, "F1_all64_spectra")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)
    for axis, name in zip(axes, order):
        for atom in SELECTED_ATOMS:
            axis.plot(frequency, normalized[name][:, atom], lw=.8, alpha=.8, label=str(atom))
        axis.set_title(name); axis.set_ylabel(ylabel); axis.set_ylim(-80, 1); axis.set_yticks(yticks); axis.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=7, title="basis k", loc="lower right")
    axes[-1].set_xlabel("normalized frequency")
    save_figure(fig, "F2_selected_basis_spectra")

    learned = normalized["Learned"]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(frequency, learned, lw=.5, alpha=.24, color=COLORS["Learned"])
    axis.set_title("Learned — all 64 basis spectra"); axis.set_xlabel("normalized frequency")
    axis.set_ylabel(ylabel); axis.set_ylim(-80, 1); axis.set_yticks(yticks); axis.grid(alpha=.2)
    save_figure(fig, "F1L_learned_all64_spectra")

    fig, axis = plt.subplots(figsize=(10, 4))
    for atom in SELECTED_ATOMS:
        axis.plot(frequency, learned[:, atom], lw=.9, alpha=.85, label=str(atom))
    axis.set_title("Learned — selected basis spectra"); axis.set_xlabel("normalized frequency")
    axis.set_ylabel(ylabel); axis.set_ylim(-80, 1); axis.set_yticks(yticks); axis.grid(alpha=.2)
    axis.legend(ncol=4, fontsize=8, title="basis k", loc="lower right")
    save_figure(fig, "F2L_learned_selected_basis_spectra")


def phase_practicality(smoke: bool) -> None:
    learned, multiplier, _ = load_selected(); waveforms = waveform_set(learned, multiplier)
    result = practicality_data(waveforms, 64 if smoke else 2000, SEEDS["papr"]); write_json(DATA / "practicality.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        for name in PRIMARY_WAVEFORMS:
            row = next(value for value in result["papr"] if value["modulation"] == modulation and value["waveform"] == name)
            samples = np.sort(row["samples_db"]); ccdf = 1 - np.arange(1, len(samples) + 1) / len(samples)
            axis.semilogy(samples, np.maximum(ccdf, .5 / len(samples)), color=COLORS[name], label=name)
        axis.set_title(modulation); axis.set_xlabel("4x oversampled PAPR (dB)"); axis.grid(alpha=.25)
    axes[0].set_ylabel("CCDF"); axes[0].legend(); save_figure(fig, "P1_papr_ccdf")
    frequency = np.asarray(result["frequency"]); fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        for name in PRIMARY_WAVEFORMS:
            power = np.asarray(result["psd"][f"{modulation}_{name}"]); axis.plot(frequency, 10 * np.log10(np.maximum(power / power.max(), 1e-12)), color=COLORS[name], label=name)
        axis.axvline(-.4, color=".5", ls="--", lw=.7); axis.axvline(.4, color=".5", ls="--", lw=.7); axis.set_title(modulation); axis.set_xlabel("normalized frequency"); axis.grid(alpha=.25)
    axes[0].set_ylabel("average PSD (dB, peak-normalized)"); axes[0].legend(); save_figure(fig, "P2_average_psd")


def _plot_curves(axis, rows, xkey, selector):
    for name in PRIMARY_WAVEFORMS:
        selected = sorted((row for row in rows if row["waveform"] == name and selector(row)), key=lambda row: float(row[xkey]))
        if selected:
            axis.semilogy([float(row[xkey]) for row in selected], [max(float(row["ber"]), .5 / int(row["total_bits"])) for row in selected],
                          marker=MARKERS[name], color=COLORS[name], label=name, lw=1.4, ms=4)
    axis.grid(alpha=.25, which="both")


def plot_communication(rows):
    for number, (regime, label) in enumerate((("grid_dd", "GRID-DD"), ("fractional_doppler", "Fractional Doppler"), ("fractional_delay", "Fractional Delay"), ("dd_v1", "Full off-grid DD-v1")), 1):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
            _plot_curves(axis, rows, "snr_db", lambda r, m=modulation, g=regime: r["modulation"] == m and r["regime"] == g and r["receiver_mode"] == "fixedStatK24")
            axis.set_title(modulation); axis.set_xlabel("Es/N0 (dB)")
        axes[0].set_ylabel("BER"); axes[0].legend(); fig.suptitle(f"{label}: fixed statistical K-support, K=24"); save_figure(fig, f"C{number}_{regime}_ber")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        _plot_curves(axis, rows, "snr_db", lambda r, m=modulation: r["modulation"] == m and r["regime"] == "dd_v1" and r["receiver_mode"] == "full")
        axis.set_title(modulation); axis.set_xlabel("Es/N0 (dB)")
    axes[0].set_ylabel("BER"); axes[0].legend(); fig.suptitle("Full off-grid DD-v1: full representation K=64"); save_figure(fig, "C5_full_information")


def plot_communication_capped(rows, max_snr: int = 20) -> None:
    """Supplement C1-C5 with a measured-point-only common SNR window."""
    regimes = (("grid_dd", "GRID-DD"), ("fractional_doppler", "Fractional Doppler"),
               ("fractional_delay", "Fractional Delay"), ("dd_v1", "Full off-grid DD-v1"))
    for number, (regime, label) in enumerate(regimes, 1):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
            _plot_curves(
                axis, rows, "snr_db",
                lambda r, m=modulation, g=regime: r["modulation"] == m
                and r["regime"] == g and r["receiver_mode"] == "fixedStatK24"
                and float(r["snr_db"]) <= max_snr,
            )
            axis.set_xlim(0, max_snr); axis.set_xticks((0, 5, 10, 15, 20))
            axis.set_title(modulation); axis.set_xlabel("Es/N0 (dB)")
        axes[0].set_ylabel("BER"); axes[0].legend()
        fig.suptitle(f"{label}: fixed statistical K-support, K=24 (0–{max_snr} dB view)")
        save_figure(fig, f"C{number}_{regime}_ber_snr0_20")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        _plot_curves(
            axis, rows, "snr_db",
            lambda r, m=modulation: r["modulation"] == m and r["regime"] == "dd_v1"
            and r["receiver_mode"] == "full" and float(r["snr_db"]) <= max_snr,
        )
        axis.set_xlim(0, max_snr); axis.set_xticks((0, 5, 10, 15, 20))
        axis.set_title(modulation); axis.set_xlabel("Es/N0 (dB)")
    axes[0].set_ylabel("BER"); axes[0].legend()
    fig.suptitle(f"Full off-grid DD-v1: full representation K=64 (0–{max_snr} dB view)")
    save_figure(fig, "C5_full_information_snr0_20")


def plot_k_sweep(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        _plot_curves(axis, rows, "k", lambda r, m=modulation: r["modulation"] == m)
        axis.set_title(f"{modulation} ({18 if modulation != '64QAM' else 25} dB)"); axis.set_xlabel("retained coefficients per output row K")
    axes[0].set_ylabel("BER"); axes[0].legend(); save_figure(fig, "C6_receiver_budget")


def plot_receiver_modes(rows):
    labels = ("fixedStatK24", "oracle24", "full"); fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        x = np.arange(3); width = .24
        for index, name in enumerate(PRIMARY_WAVEFORMS):
            values = [next(float(r["ber"]) for r in rows if r["modulation"] == modulation and r["waveform"] == name and r["receiver_mode"] == mode) for mode in labels]
            axis.bar(x + (index - 1) * width, values, width, color=COLORS[name], label=name)
        axis.set_yscale("log"); axis.set_xticks(x, ("Fixed K=24", "Oracle K=24", "Full K=64"), rotation=15); axis.set_title(modulation); axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("BER"); axes[0].legend(); save_figure(fig, "C7_fixed_oracle_full")


def plot_stress(rows):
    regimes = ("dd_v1", "wider_doppler", "wider_delay", "more_paths"); fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for axis, modulation in zip(axes, ("QPSK", "16QAM", "64QAM")):
        x = np.arange(4); width=.24
        for index, name in enumerate(PRIMARY_WAVEFORMS):
            values=[next(float(r["ber"]) for r in rows if r["regime"]==regime and r["modulation"]==modulation and r["waveform"]==name) for regime in regimes]
            axis.bar(x+(index-1)*width, values, width, color=COLORS[name], label=name)
        axis.set_yscale("log"); axis.set_xticks(x, ("Nominal", "Wider ν", "Wider τ", "P=5"), rotation=20); axis.set_title(modulation); axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("BER"); axes[0].legend(); save_figure(fig, "C8_unseen_channel_stress")


def plot_path_count(rows):
    fig, axes=plt.subplots(1,2,figsize=(8.5,3.5),sharey=True)
    for axis, modulation in zip(axes,("QPSK","16QAM")):
        _plot_curves(axis, rows, "paths", lambda r,m=modulation:r["modulation"]==m); axis.set_title(modulation); axis.set_xlabel("path count P")
    axes[0].set_ylabel("BER");axes[0].legend();save_figure(fig,"C9_path_count")


def plot_fractionality(rows):
    fig,axis=plt.subplots(figsize=(5.5,3.5));_plot_curves(axis,rows,"gamma",lambda r:True);axis.set_xlabel("fractionality gamma");axis.set_ylabel("QPSK BER at 18 dB");axis.legend();save_figure(fig,"C10_fractionality")


def phase_figure_updates(smoke: bool = False) -> None:
    """Refresh spectral publication figures exclusively from saved arrays."""
    del smoke
    spectra_path = DATA / "basis_spectra.npz"
    if not spectra_path.exists():
        raise RuntimeError("figure-updates requires saved Pub-1A basis-spectrum data")
    with np.load(spectra_path) as saved:
        frequency = saved["frequency"]
        spectral = {name: saved[name] for name in ("OFDM", "OTFS", "AFDM", "Learned")}
    plot_basis_spectra(frequency, spectral)
    phase_report()


def phase_report() -> None:
    audit=json.loads((DATA/"afdm_audit.json").read_text()); provenance_data=provenance()
    provenance_data["final_afdm"]={"multiplier_2Nc1":audit["selected_multiplier_2Nc1"],"c1":audit["selected_c1"],"c2":audit["c2"]}
    dataset_files=[path for path in sorted(DATA.iterdir()) if path.suffix in (".csv",".json",".npz") and path.name!="dataset_manifest.json"]
    write_json(DATA/"dataset_manifest.json",{
        "schema":"ddwaveform.stage-pub1a.manifest.v1","provenance":provenance_data,
        "pairing":"common physical channels, symbol bits, and time-domain noise across waveforms within each setting",
        "receiver_modes":{"fixedStatK":"fixed channel-class support","oracleK":"realization-wise diagnostic","full":"K=64"},
        "datasets":[{"path":str(path.relative_to(OUT)),"sha256":file_sha256(path),"bytes":path.stat().st_size} for path in dataset_files],
    })
    comm=read_csv(DATA/"communication_summary.csv") if (DATA/"communication_summary.csv").exists() else []
    def winner(regime,modulation,mode,snr):
        candidates=[r for r in comm if r["regime"]==regime and r["modulation"]==modulation and r["receiver_mode"]==mode and int(r["snr_db"])==snr and r["waveform"] in PRIMARY_WAVEFORMS]
        return min(candidates,key=lambda r:float(r["ber"])) if candidates else None
    findings=[]
    for regime in ("grid_dd","fractional_doppler","fractional_delay","dd_v1"):
        for modulation,snr in (("QPSK",18),("16QAM",18),("64QAM",25)):
            row=winner(regime,modulation,"fixedStatK24",snr)
            if row: findings.append(f"- {regime}, {modulation}, {snr} dB, fixed K=24: **{row['waveform']}** lowest BER `{float(row['ber']):.4g}`.")
    sweep_counts=[]
    for regime in ("grid_dd","fractional_doppler","fractional_delay","dd_v1"):
        for modulation in ("QPSK","16QAM","64QAM"):
            settings=sorted({int(r["snr_db"]) for r in comm if r["regime"]==regime and r["modulation"]==modulation and r["receiver_mode"]=="fixedStatK24"})
            counts={name:0 for name in PRIMARY_WAVEFORMS}
            for snr in settings:
                row=winner(regime,modulation,"fixedStatK24",snr)
                if row: counts[row["waveform"]]+=1
            sweep_counts.append(f"- {regime}, {modulation}: lowest observed BER at {counts['Learned']}/{len(settings)} Learned, {counts['OTFS']}/{len(settings)} OTFS, {counts['AFDM']}/{len(settings)} AFDM SNR points.")
    k_rows=read_csv(DATA/"k_sweep_summary.csv")
    k_findings=[]
    for modulation in ("QPSK","16QAM","64QAM"):
        wins=[]
        for k in K_SWEEP:
            candidates=[r for r in k_rows if r["modulation"]==modulation and int(r["k"])==k and r["waveform"] in PRIMARY_WAVEFORMS]
            if candidates and min(candidates,key=lambda r:float(r["ber"]))["waveform"]=="Learned": wins.append(k)
        k_findings.append(f"- {modulation}: Learned is lowest at K={wins}; classical is lowest at the remaining tested K values.")
    mode_rows=read_csv(DATA/"fixed_oracle_full_summary.csv")
    mode_findings=[]
    for modulation in ("QPSK","16QAM","64QAM"):
        for mode in ("fixedStatK24","oracle24","full"):
            candidates=[r for r in mode_rows if r["modulation"]==modulation and r["receiver_mode"]==mode and r["waveform"] in PRIMARY_WAVEFORMS]
            row=min(candidates,key=lambda r:float(r["ber"]))
            mode_findings.append(f"- {modulation}, {mode}: {row['waveform']} lowest at `{float(row['ber']):.4g}`.")
    stress_rows=read_csv(DATA/"stress_summary.csv")
    stress_findings=[]
    for regime in ("dd_v1","wider_doppler","wider_delay","more_paths"):
        values=[]
        for modulation in ("QPSK","16QAM","64QAM"):
            row=next(r for r in stress_rows if r["regime"]==regime and r["modulation"]==modulation and r["waveform"]=="Learned")
            values.append(f"{modulation} {float(row['ber']):.4g}")
        stress_findings.append(f"- {regime}: Learned BER " + ", ".join(values) + "; Learned remained lowest among primary waveforms.")
    practical=json.loads((DATA/"practicality.json").read_text())
    papr_findings=[]
    for modulation in ("QPSK","16QAM","64QAM"):
        values=[next(r for r in practical["papr"] if r["modulation"]==modulation and r["waveform"]==name) for name in PRIMARY_WAVEFORMS]
        papr_findings.append(f"- {modulation} p99 PAPR: " + ", ".join(f"{r['waveform']} {r['p99_db']:.2f} dB" for r in values) + ".")
    dd_support=[winner("dd_v1",m,"fixedStatK24",s) for m,s in (("QPSK",18),("16QAM",18),("64QAM",25))]
    supported=sum(row is not None and row["waveform"]=="Learned" for row in dd_support)
    classification="SUPPORTED" if supported==3 else "PARTIALLY SUPPORTED" if supported else "NOT SUPPORTED"
    text=f"""# Stage Pub-1A — Frozen N=64 Publication Qualification

## A. Provenance

- Branch `{provenance_data['git_branch']}` at `{provenance_data['git_commit']}`; independent target `origin/pub`.
- Frozen Learned SHA-256 `{provenance_data['learned_checkpoint_sha256']}`; `learned_training_steps = 0`.
- Final AFDM: `2Nc1={audit['selected_multiplier_2Nc1']}`, `c1={audit['selected_c1']}`, `c2={audit['c2']}`.
- Stage-2E artifacts preserved. Exact nominal Learned/OTFS/OFDM rows reused; final AFDM and missing Pub-1A points regenerated.

## B. AFDM baseline

Repository convention is `H=sum h_p Delta(nu_p) Pi(tau_p)` and `A=W^H H W`. For integer paths the AFDM offset is `(2Nc1 tau + nu) mod N`. DD-v1 has nearest-integer Doppler support `alpha_max=1`; admissible integer and guarded slopes are 3 and 5. Both passed distinct four-path predicted/observed offsets. Selection used only mean validation-bank K=24 analytic MSE; BER, test data, and Learned performance were excluded.

Validation mean analytic MSE was `{audit['validation_summary'][0]['mean_analytic_mse']:.6f}` for slope 3 and `{audit['validation_summary'][1]['mean_analytic_mse']:.6f}` for slope 5. Slope 3 path offsets were predicted/observed `63/63, 4/4, 6/6, 8/8`, with nearest circular separation at least 2 bins.

## C. Channel definitions

- GRID-DD: integer `tau in {{0,...,4}}`, integer `nu in {{-1,0,1}}`.
- FRACTIONAL-DOPPLER: integer delay, continuous `nu ~ U[-0.75,0.75]`.
- FRACTIONAL-DELAY: continuous `tau ~ U[0,4]`, controlled continuous `nu ~ U[-0.20,0.20]`, unchanged from Stage-2E.
- FULL OFF-GRID / DD-v1: P=3, `tau ~ U[0,4]`, `nu ~ U[-0.75,0.75]`.
- Stress: doubled Doppler or delay support, and path count variation with common total normalization.

Separate disjoint statistics banks define each channel class's fixed statistical K-support. This is receiver class knowledge; none of these channels trained the waveform.

## D. Receiver representation budget

K counts retained effective-channel coefficients per output row. K=24 retains 37.5% of 64. Historically, K=16/24/32 left about 31.3%/20.4%/11.2% oracle headroom; K=24 was an intermediate bottleneck, not a theoretical constant. K is a representation budget only—no MAC/runtime claim is made.

## E. Communications findings

{chr(10).join(findings)}

Supplementary C1–C5 figures with the `_snr0_20` suffix show the same saved BER points on a common 0–20 dB Es/N0 window. They are plot-only views; the full-range figures and numerical results are unchanged.

Across complete BER-vs-SNR sweeps:

{chr(10).join(sweep_counts)}

Receiver-budget sweep:

{chr(10).join(k_findings)}

Fixed/oracle/full diagnostic:

{chr(10).join(mode_findings)}

Stress banks:

{chr(10).join(stress_findings)}

The matched fractionality sweep crosses from classical at gamma 0–0.25, AFDM at gamma 0.5, to Learned at gamma 0.75–1. Full K=64 results delimit the claim; QPSK and 16-QAM favor classical waveforms, while 64-QAM happens to favor Learned in this finite N=64 test. This is not a universal full-information claim. Raw error/bit and symbol-error/symbol counts are retained; sweep winner counts are descriptive, not confidence statements.

SNR axes use Es/N0. For cross-modulation conversion, `Eb/N0 = Es/N0 - 10 log10(bits/symbol)`: subtract 3.01, 6.02, and 7.78 dB for QPSK, 16-QAM, and 64-QAM.

## F. Physical waveform findings

F1/F2 apply one 0 dB peak normalization per waveform while preserving relative peak differences among its 64 atoms; the saved dense-DTFT arrays remain absolute and unchanged. Learned-only F1L/F2L provide uncluttered all-atom and selected-atom views under the same normalization. Dense DTFT exposes finite-block OFDM sinc-like responses; matched F3 STFT settings expose AFDM chirp behavior without assigning that label to Learned. PAPR uses matched data and 4x discrete-time oversampling. PSD/OOB uses a declared discrete-time `|f|>=0.4` convention; it is not ACLR or a continuous-time mask. Square 64-symbol/64-sample unitary transforms have matched nominal symbol density by construction.

{chr(10).join(papr_findings)}

All primary waveforms occupy essentially the full discrete-time Nyquist interval under the declared 99% metric; OOB fractions are near 0.20 because matched random symbols passed through square unitary transforms are spectrally white in expectation. Learned shows no obvious PAPR/PSD pathology, but this model cannot support an ACLR or spectral-efficiency claim.

## G. Publication claim

**{classification}.** Exact support comes from the listed fractional/off-grid modulation/SNR regimes where Learned has the lowest fixed-K BER. The classification is intentionally narrower than “Learned wins everywhere.”

## H. Limitations

- No universal superiority; grid-DD and full K=64 may favor classical waveforms.
- N=64 only; no runtime/MAC complexity result.
- No pulse shaping, continuous-time spectral mask, or ACLR claim.
- No new waveform-family law and no training.

## I. Pub-1B handoff — not implemented

- Sparse iterative LMMSE; BER versus MACs/runtime; N=128/256 scaling; possible AIU acceleration.

**STOP after Pub-1A.**
"""
    (OUT/"report.md").write_text(text)
    notebook={"cells":[{"cell_type":"markdown","id":"overview","metadata":{},"source":["# Pub-1A saved-result summary\n","This notebook loads saved Pub-1A outputs; it does not run Monte Carlo.\n"]},{"cell_type":"code","id":"load-results","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\n","import pandas as pd\n","ROOT = Path('..')\n","communication = pd.read_csv(ROOT/'outputs/stage_pub1a/data/communication_summary.csv')\n","k_sweep = pd.read_csv(ROOT/'outputs/stage_pub1a/data/k_sweep_summary.csv')\n","communication.head()\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
    write_json(ROOT/"notebooks"/"06_stage_pub1a_publication_qualification.ipynb",notebook)


def run(phase: str, smoke: bool) -> None:
    ensure_dirs()
    phases={"afdm-audit":phase_afdm_audit,"communication":phase_communication,"k-sweep":phase_k_sweep,
            "stress":phase_stress,"operators":phase_operators,"waveform-figures":phase_waveform_figures,
            "practicality":phase_practicality,"figure-updates":phase_figure_updates}
    if phase=="all":
        for name in ("afdm-audit","communication","k-sweep","stress","operators","waveform-figures","practicality"): phases[name](smoke)
        phase_report()
    elif phase=="report": phase_report()
    else: phases[phase](smoke)
    if file_sha256(FROZEN)!=CHECKPOINT_SHA256: raise RuntimeError("frozen Learned checkpoint changed")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("afdm-audit","communication","k-sweep","stress","operators","waveform-figures","practicality","figure-updates","report","all"),default="all");parser.add_argument("--smoke",action="store_true");args=parser.parse_args();run(args.phase,args.smoke)
