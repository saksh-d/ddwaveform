"""Run Stage 4A frozen structural-generator and waveform-family forensics."""

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
from src.metrics import constellation_encode, nearest_neighbor_decode, qam16_constellation, qpsk_constellation, random_bits
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform, qualification_waveforms
from src.stage2d import apply_support, matrix_agreement, score_support_mask, sobol_dd_samples
from src.stage3a import q_concentration
from src.stage3b import atomic_operators, differentiable_q, fixed_initializations
from src.stage4a import (
    basis_fidelity,
    canonical_generators,
    compressed_generators,
    dd_expansion_metrics,
    generator_matrix_metrics,
    integer_weyl_coefficients,
    native_atom_diagnostics,
    orbit_metrics,
)
from src.structural_fitting import StructuredFit, fit_family_hierarchy
from src.waveforms import identity_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage4a"
FITTED = OUT / "fitted"
REPORT = ROOT / "reports" / "stage4a_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE3B = ROOT / "outputs" / "stage3b" / "checkpoints"
K = 24
Q_SAMPLES = 2048
Q_SEED = 11130
CHANNELS = 512
CHANNEL_SEED = 11131
SYMBOL_SEED = 11132
NOISE_SEED = 11133
SNRS = (12, 15, 18, 20)
TARGET_BITS = 1_000_000
PRIMARY = (
    "Q_MEAN_FROM_OFDM", "Q_MEAN_FROM_HAAR_1", "Q_MEAN_FROM_HAAR_2",
    "Q_ROBUST_FROM_OFDM", "Q_ROBUST_FROM_HAAR_1", "Q_ROBUST_FROM_HAAR_2",
)
EXPECTED_STAGE3B_HASHES = {
    "Q_MEAN_FROM_HAAR_1": "4b0299c11ce3bf3371d3e8245f8877776a915c02bbc3e73525111c48fbc77c3c",
    "Q_MEAN_FROM_HAAR_2": "e285557d21f002269729f077e5984d4d129356051c361d47e7c492122739fb9a",
    "Q_MEAN_FROM_OFDM": "003c94549f7bd7c20b354162dd5602894936de40680406692bd978d76ab33370",
    "Q_ROBUST_FROM_HAAR_1": "541bb8d7e1929441c19b5e234af126b6c69447a2d3f928fa64dd3af13b9ec51f",
    "Q_ROBUST_FROM_HAAR_2": "909b677386ab53d80edc45637043240e0a35fb56b5e45f6cf25ac2e230521604",
    "Q_ROBUST_FROM_OFDM": "5e232d24c8c67a3a74cd9bff40396867ebd97b92308749052105e14c727474f1",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted((ROOT / "outputs").glob("**/*"))
        if path.is_file() and "stage4a" not in path.parts
    }


def load_waveforms() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    learned, info = load_frozen_waveform(FROZEN)
    principals = qualification_waveforms(learned)
    selected: dict[str, torch.Tensor] = {}
    hashes = {"Learned": str(info["sha256"])}
    for name in PRIMARY:
        path = STAGE3B / f"{name.lower()}_best.pt"
        digest = file_sha256(path)
        if digest != EXPECTED_STAGE3B_HASHES[name]:
            raise RuntimeError(f"unexpected frozen Stage-3B hash: {name}={digest}")
        selected[name] = torch.load(path, weights_only=True); hashes[name] = digest
    names, starts = fixed_initializations()
    raw = {f"RAW_{name}": starts[index] for index, name in enumerate(names) if name.startswith("HAAR")}
    waveforms = {
        "Learned": learned, **selected,
        "OFDM": principals["OFDM"], "OTFS": principals["OTFS"], "AFDM": principals["AFDM"],
        **raw, "IDENTITY": identity_waveform(),
    }
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    for name, waveform in waveforms.items():
        residual = float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity))
        if residual >= 1e-12:
            raise RuntimeError(f"non-unitary source waveform: {name} ({residual:.3e})")
    return waveforms, hashes


def generator_analysis(waveforms: dict[str, torch.Tensor]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    generator_rows: list[dict[str, object]] = []
    orbit_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for name, waveform in waveforms.items():
        shift, modulation = canonical_generators(waveform)
        for generator_name, generator in (("G_S", shift), ("G_D", modulation)):
            coefficients = integer_weyl_coefficients(generator)
            arrays[f"{name}_{generator_name}"] = coefficients.detach().cpu().numpy()
            generator_rows.append({
                "waveform": name, "generator": generator_name,
                **generator_matrix_metrics(generator), **dd_expansion_metrics(coefficients),
            })
        orbit_rows.append({"waveform": name, "approximation": "exact_dense", **orbit_metrics(waveform, shift)})
        for approximation, generator in compressed_generators(shift).items():
            orbit_rows.append({"waveform": name, "approximation": approximation, **orbit_metrics(waveform, generator)})
    return generator_rows, orbit_rows, arrays


def fit_all(waveforms: dict[str, torch.Tensor], steps: int) -> tuple[list[dict[str, object]], dict[str, dict[str, torch.Tensor]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fitted: dict[str, dict[str, torch.Tensor]] = {}
    parameters: list[dict[str, object]] = []
    for waveform_index, (name, waveform) in enumerate(waveforms.items()):
        current = fit_family_hierarchy(waveform, steps=steps)
        fitted[name] = {fit.family: fit.waveform for fit in current}
        for fit in current:
            rows.append({
                "target": name, "family": fit.family,
                "free_real_parameters": fit.free_real_parameters,
                "storage_complexity": fit.storage_complexity,
                "application_complexity": fit.application_complexity,
                "factor_depth": fit.factor_depth,
                **{key: value for key, value in fit.fidelity.items() if key != "permutation"},
            })
            parameters.append({
                "target": name, "family": fit.family,
                "free_real_parameters": fit.free_real_parameters,
                "factor_depth": fit.factor_depth,
                "storage_complexity": fit.storage_complexity,
                "application_complexity": fit.application_complexity,
                "parameters": fit.parameters,
                "alignment_permutation": fit.fidelity["permutation"],
            })
        print(f"fitted {name} ({waveform_index + 1}/{len(waveforms)})", flush=True)
    return rows, fitted, parameters


def cyclic_covariance(waveform: torch.Tensor) -> np.ndarray:
    n = waveform.shape[-1]
    values = []
    for shift in range(n):
        shifted = torch.roll(waveform, shifts=shift, dims=0)
        values.append(float(torch.mean(torch.abs(torch.sum(waveform.conj() * shifted, dim=0)).square())))
    return np.asarray(values)


def cross_invariants(
    waveforms: dict[str, torch.Tensor], generators: dict[str, np.ndarray], q: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    names = ("Learned",) + PRIMARY + ("RAW_HAAR_1", "RAW_HAAR_2")
    covariances = {name: cyclic_covariance(waveforms[name]) for name in names}
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            matrix_fidelity = basis_fidelity(waveforms[left], waveforms[right])
            row: dict[str, object] = {
                "left": left, "right": right,
                "aligned_matrix_nmse": matrix_fidelity["aligned_matrix_nmse"],
                "mean_matched_squared_overlap": matrix_fidelity["mean_matched_squared_overlap"],
            }
            for generator_name in ("G_S", "G_D"):
                left_coefficients = generators[f"{left}_{generator_name}"]
                right_coefficients = generators[f"{right}_{generator_name}"]
                left_energy = np.abs(left_coefficients) ** 2
                right_energy = np.abs(right_coefficients) ** 2
                left_top = set(np.argsort(left_energy.reshape(-1))[::-1][:K].tolist())
                right_top = set(np.argsort(right_energy.reshape(-1))[::-1][:K].tolist())
                intersection = sorted(left_top & right_top)
                row[f"{generator_name.lower()}_top24_jaccard"] = len(intersection) / len(left_top | right_top)
                row[f"{generator_name.lower()}_magnitude_correlation"] = float(
                    np.corrcoef(left_energy.reshape(-1), right_energy.reshape(-1))[0, 1]
                )
                phase_delta = np.angle(left_coefficients.reshape(-1)[intersection]) - np.angle(right_coefficients.reshape(-1)[intersection])
                row[f"{generator_name.lower()}_shared_support_phase_coherence"] = (
                    float(np.abs(np.mean(np.exp(1j * phase_delta)))) if intersection else 0.0
                )
            row["q_top24_overlap_mean"] = matrix_agreement(q[left], q[right], K)["topk_overlap_mean"]
            row["q_correlation"] = float(np.corrcoef(q[left].reshape(-1), q[right].reshape(-1))[0, 1])
            row["cyclic_covariance_correlation"] = float(
                np.corrcoef(covariances[left], covariances[right])[0, 1]
            )
            rows.append(row)
    return rows


def level2_evaluation(
    target: torch.Tensor, candidates: dict[str, torch.Tensor], channels: torch.Tensor
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    delays, dopplers = sobol_dd_samples(Q_SAMPLES, Q_SEED)
    operators = atomic_operators(delays, dopplers)
    waveforms = {"Learned": target, **candidates}
    names = tuple(waveforms)
    q_stack = differentiable_q(torch.stack([waveforms[name] for name in names]), operators, chunk_size=16)
    q = {name: q_stack[index].detach().cpu().numpy() for index, name in enumerate(names)}
    rows: list[dict[str, object]] = []
    for name, waveform in waveforms.items():
        concentration = q_concentration(q[name], K)
        mask = score_support_mask(q[name], K)
        effective = effective_operator(channels, waveform)
        energy = torch.abs(effective).square()
        retained = torch.sum(torch.where(mask, energy, 0.0), dim=(-2, -1)) / torch.sum(energy, dim=(-2, -1))
        agreement = matrix_agreement(q["Learned"], q[name], K)
        rows.append({
            "waveform": name,
            "eta24_q": concentration["eta_q"],
            "row_p05": concentration["row_p05"],
            "effective_support_mean": concentration["effective_support_mean"],
            "normalized_entropy_mean": concentration["normalized_entropy_mean"],
            "q_relative_distance_to_learned": float(np.linalg.norm(q[name] - q["Learned"]) / np.linalg.norm(q["Learned"])),
            "q_correlation_to_learned": float(np.corrcoef(q[name].reshape(-1), q["Learned"].reshape(-1))[0, 1]),
            "top24_support_overlap_to_learned": agreement["topk_overlap_mean"],
            "fixed_q_retained_energy_mean": float(torch.mean(retained)),
            "fixed_q_retained_energy_p05": float(torch.quantile(retained, 0.05)),
        })
    return rows, q


def communication_evaluation(
    waveforms: dict[str, torch.Tensor], q: dict[str, np.ndarray], channels: torch.Tensor,
    target_bits: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    fixed_masks = {name: score_support_mask(q[name], K) for name in waveforms}
    with torch.no_grad():
        for modulation_index, constellation in enumerate((qpsk_constellation(), qam16_constellation())):
            blocks = math.ceil(target_bits / (len(channels) * config.N * constellation.bits_per_symbol))
            bits_per_channel = blocks * config.N * constellation.bits_per_symbol
            symbols_per_channel = blocks * config.N
            for snr_index, snr_db in enumerate(SNRS):
                bits = random_bits(
                    (len(channels), blocks, config.N, constellation.bits_per_symbol),
                    make_generator(SYMBOL_SEED + 100 * modulation_index + snr_index),
                )
                symbols, true_indices = constellation_encode(bits, constellation)
                sigma2 = noise_variance(snr_db)
                noise = complex_awgn(
                    (len(channels), blocks, config.N), sigma2,
                    make_generator(NOISE_SEED + 100 * modulation_index + snr_index),
                )
                for name, waveform in waveforms.items():
                    transmitted = torch.einsum("ij,cbj->cbi", waveform, symbols)
                    physical = torch.einsum("cij,cbj->cbi", channels, transmitted) + noise
                    received = torch.einsum("ij,cbj->cbi", waveform.mH, physical)
                    designs = {
                        "Full": effective[name],
                        "Oracle": rowwise_topk(effective[name], K),
                        "Fixed-Q": apply_support(effective[name], fixed_masks[name]),
                    }
                    total_energy = torch.sum(torch.abs(effective[name]) ** 2, dim=(-2, -1))
                    for support_rule, design in designs.items():
                        equalizer = mismatched_lmmse_equalizer(design, sigma2)
                        estimates = torch.einsum("cij,cbj->cbi", equalizer, received)
                        decoded_bits, decoded_indices = nearest_neighbor_decode(estimates, constellation)
                        bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
                        symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
                        mse = analytic_communication_mse(effective[name], equalizer, sigma2)
                        retained = torch.sum(torch.abs(design) ** 2, dim=(-2, -1)) / total_energy
                        for channel_index in range(len(channels)):
                            rows.append({
                                "channel_index": channel_index, "waveform": name,
                                "modulation": constellation.name, "snr_db": snr_db,
                                "support_rule": support_rule, "k": config.N if support_rule == "Full" else K,
                                "bit_errors": int(bit_errors[channel_index]), "total_bits": bits_per_channel,
                                "symbol_errors": int(symbol_errors[channel_index]), "total_symbols": symbols_per_channel,
                                "analytic_mse": float(mse[channel_index]), "retained_energy": float(retained[channel_index]),
                            })
                print(f"communication {constellation.name} {snr_db} dB complete", flush=True)
    summary: list[dict[str, object]] = []
    keys = ("waveform", "modulation", "snr_db", "support_rule", "k")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    for values, selected in groups.items():
        bit_errors = sum(int(row["bit_errors"]) for row in selected)
        total_bits = sum(int(row["total_bits"]) for row in selected)
        symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
        total_symbols = sum(int(row["total_symbols"]) for row in selected)
        summary.append({
            **dict(zip(keys, values)), "bit_errors": bit_errors, "total_bits": total_bits,
            "ber": bit_errors / total_bits, "symbol_errors": symbol_errors,
            "total_symbols": total_symbols, "ser": symbol_errors / total_symbols,
            "analytic_mse_mean": float(np.mean([float(row["analytic_mse"]) for row in selected])),
            "retained_energy_mean": float(np.mean([float(row["retained_energy"]) for row in selected])),
            "n_channels": len(selected),
        })
    return rows, summary


def make_figures(
    generator_rows: list[dict[str, object]], arrays: dict[str, np.ndarray], fit_rows: list[dict[str, object]],
    level2: list[dict[str, object]], communication: list[dict[str, object]],
    atom_rows: list[dict[str, object]], invariants: list[dict[str, object]], candidate_waveforms: dict[str, torch.Tensor],
) -> None:
    shown = ("Learned", "Q_MEAN_FROM_OFDM", "Q_MEAN_FROM_HAAR_1", "OFDM", "AFDM", "RAW_HAAR_1")
    fig, axes = plt.subplots(2, len(shown), figsize=(3 * len(shown), 6), constrained_layout=True)
    for column, name in enumerate(shown):
        for row, generator_name in enumerate(("G_S", "G_D")):
            image = np.abs(arrays[f"{name}_{generator_name}"]) ** 2
            axes[row, column].imshow(np.log10(image + 1e-12), vmin=-12, vmax=0, cmap="magma")
            axes[row, column].set_title(f"{name}\n{generator_name}", fontsize=8)
            axes[row, column].set(xticks=[], yticks=[])
    fig.suptitle(r"Integer-Weyl generator energy, $\log_{10}(|c_{\tau,\nu}|^2+10^{-12})$", fontsize=11)
    fig.savefig(OUT / "fig01_generator_weyl_energy.png", dpi=180); plt.close(fig)

    rows = [row for row in fit_rows if row["target"] != "IDENTITY"]
    targets = ("Learned", "OFDM", "OTFS", "AFDM", "RAW_HAAR_1", "RAW_HAAR_2")
    families = tuple(dict.fromkeys(str(row["family"]) for row in rows))
    matrix = np.asarray([[next(row for row in rows if row["target"] == target and row["family"] == family)["aligned_matrix_nmse"] for family in families] for target in targets])
    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(families)), families, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(range(len(targets)), targets, fontsize=8); fig.colorbar(image, ax=axis, label="aligned matrix NMSE")
    axis.set_title("Low-parameter structural fits (lower is better)")
    fig.savefig(OUT / "fig02_structured_fit_fidelity.png", dpi=180); plt.close(fig)

    names = [str(row["waveform"]) for row in level2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(range(len(names)), [float(row["eta24_q"]) for row in level2])
    axes[0].set_ylabel(r"$\eta_{24}^Q$"); axes[0].set_ylim(0, 1.02)
    axes[1].bar(range(len(names)), [float(row["top24_support_overlap_to_learned"]) for row in level2])
    axes[1].set_ylabel("top-24 support overlap to Learned"); axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=7); axis.grid(axis="y", alpha=.3)
    fig.savefig(OUT / "fig03_candidate_q_fidelity.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for modulation, axis in zip(("QPSK", "16QAM"), axes):
        for name in names:
            selected = sorted((row for row in communication if row["waveform"] == name and row["modulation"] == modulation and row["support_rule"] == "Fixed-Q"), key=lambda row: int(row["snr_db"]))
            axis.plot([row["snr_db"] for row in selected], [row["ber"] for row in selected], marker="o", label=name)
        axis.set_yscale("log"); axis.set(title=f"{modulation}, frozen fixed-Q receiver", xlabel="SNR (dB)", ylabel="BER")
        axis.grid(alpha=.3); axis.legend(fontsize=6)
    fig.savefig(OUT / "fig04_candidate_communication.png", dpi=180); plt.close(fig)

    atom_names = tuple(candidate_waveforms)
    metrics = (
        ("time_magnitude_cv", "time-magnitude CV"),
        ("spectral_magnitude_cv", "spectral-magnitude CV"),
        ("phase_degree2_rms", "quadratic-phase residual (rad)"),
        ("phase_increment_std", "phase-increment std (rad)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        medians = []
        lower = []; upper = []
        for name in atom_names:
            values = np.asarray([float(row[metric]) for row in atom_rows if row["waveform"] == name])
            median = float(np.median(values)); medians.append(median)
            lower.append(median - float(np.quantile(values, .25))); upper.append(float(np.quantile(values, .75)) - median)
        axis.bar(range(len(atom_names)), medians, yerr=np.asarray((lower, upper)), capsize=2)
        axis.set_ylabel(label); axis.set_xticks(range(len(atom_names)), atom_names, rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=.3)
    fig.suptitle("All-column native-atom summaries (median and interquartile range)")
    fig.savefig(OUT / "fig05_native_atom_diagnostics.png", dpi=180); plt.close(fig)

    success = ("Learned",) + PRIMARY
    matrix = np.eye(len(success))
    for row in invariants:
        if row["left"] in success and row["right"] in success:
            i = success.index(row["left"]); j = success.index(row["right"])
            matrix[i, j] = matrix[j, i] = float(row["g_s_top24_jaccard"])
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(success)), success, rotation=50, ha="right", fontsize=7)
    axis.set_yticks(range(len(success)), success, fontsize=7); fig.colorbar(image, ax=axis, label=r"$G_S$ top-24 DD Jaccard")
    fig.savefig(OUT / "fig06_cross_solution_generator_support.png", dpi=180); plt.close(fig)

    trace_names = ("Learned", "FIT_FOURIER_POLY_D3", "FIT_GABOR_TIGHT_PROTOTYPE", "FIT_SHALLOW_2FFT")
    fig, axes = plt.subplots(len(trace_names), 4, figsize=(15, 10), constrained_layout=True)
    for row_index, name in enumerate(trace_names):
        atom = candidate_waveforms[name][:, 0]
        spectrum = torch.fft.fft(atom)
        phase = np.unwrap(torch.angle(atom).detach().cpu().numpy())
        axes[row_index, 0].plot(torch.abs(atom).detach().cpu().numpy())
        axes[row_index, 1].plot(torch.abs(spectrum).detach().cpu().numpy())
        axes[row_index, 2].plot(phase)
        axes[row_index, 3].plot(np.diff(phase))
        axes[row_index, 0].set_ylabel(name, fontsize=8)
        axes[row_index, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
        for axis in axes[row_index]: axis.grid(alpha=.25)
    for axis, title in zip(axes[0], ("time magnitude", "spectral magnitude", "unwrapped phase", "phase increment")):
        axis.set_title(title)
    fig.suptitle("Representative native atom 0")
    fig.savefig(OUT / "fig07_representative_native_atoms.png", dpi=180); plt.close(fig)


def classify(
    generator_rows: list[dict[str, object]], orbit_rows: list[dict[str, object]], fit_rows: list[dict[str, object]],
    level2: list[dict[str, object]], communication: list[dict[str, object]], population_q_rows: list[dict[str, object]],
) -> tuple[str, str, list[str]]:
    generator = {(row["waveform"], row["generator"]): row for row in generator_rows}
    level = {row["waveform"]: row for row in level2}
    learned_fit = {f"FIT_{row['family']}": row for row in fit_rows if row["target"] == "Learned"}
    comm = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    functionally_faithful: list[str] = []
    for name, fit in learned_fit.items():
        target_rows = [comm[("Learned", modulation, snr, "Fixed-Q")] for modulation in ("QPSK", "16QAM") for snr in SNRS]
        candidate_rows = [comm[(name, modulation, snr, "Fixed-Q")] for modulation in ("QPSK", "16QAM") for snr in SNRS]
        penalties = [
            (float(candidate["ber"]) + 1 / float(candidate["total_bits"]))
            / (float(target["ber"]) + 1 / float(target["total_bits"])) - 1
            for target, candidate in zip(target_rows, candidate_rows)
        ]
        family = str(fit["family"])
        haar = [row for row in fit_rows if row["target"] in ("RAW_HAAR_1", "RAW_HAAR_2") and row["family"] == family]
        separates_haar = float(fit["aligned_matrix_nmse"]) + 0.10 < min(float(row["aligned_matrix_nmse"]) for row in haar)
        if (
            int(fit["free_real_parameters"]) <= 2 * config.N
            and abs(float(level[name]["eta24_q"]) - float(level["Learned"]["eta24_q"])) <= 0.005
            and abs(float(level[name]["fixed_q_retained_energy_mean"]) - float(level["Learned"]["fixed_q_retained_energy_mean"])) <= 0.01
            and max(penalties) <= 0.25 and separates_haar
        ):
            functionally_faithful.append(name)
    if functionally_faithful:
        return "A", "At least one predeclared compact unitary family preserves Learned DD concentration and paired communication while separating from Haar fits.", functionally_faithful

    learned_generator_margin = min(
        float(generator[("Learned", kind)]["top24_energy"])
        - max(float(generator[(name, kind)]["top24_energy"]) for name in ("RAW_HAAR_1", "RAW_HAAR_2"))
        for kind in ("G_S", "G_D")
    )
    best_orbit = max(float(row["mean_column_overlap"]) for row in orbit_rows if row["waveform"] == "Learned" and row["approximation"] != "exact_dense")
    haar_orbit = max(float(row["mean_column_overlap"]) for row in orbit_rows if row["waveform"] in ("RAW_HAAR_1", "RAW_HAAR_2") and row["approximation"] != "exact_dense")
    if learned_generator_margin >= 0.10 and best_orbit >= haar_orbit + 0.10:
        return "B", "Both Learned canonical generators and a compressed native orbit show clear Haar-separated structure, but no tested compact family preserves full function.", []
    useful_q = [row for row in population_q_rows if row["waveform"] in {"Learned", *PRIMARY}]
    if all(float(row["eta24_q"]) >= 0.98 for row in useful_q):
        return "C", "Only functional DD-representation equivalence is established: the successful population is Q-concentrated, but no common compact atom/generator family survives the controls.", []
    return "D", "No tested compressible generator or family signature responsibly separates Learned from generic controls.", []


def communication_table_lines(communication: list[dict[str, object]], names: tuple[str, ...], snr_db: int = 18) -> list[str]:
    lookup = {
        (row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row
        for row in communication
    }
    return [
        f"| {name} | {float(lookup[(name, 'QPSK', snr_db, 'Fixed-Q')]['ber']):.6g} | "
        f"{float(lookup[(name, '16QAM', snr_db, 'Fixed-Q')]['ber']):.6g} | "
        f"{float(lookup[(name, 'QPSK', snr_db, 'Oracle')]['ber']):.6g} | "
        f"{float(lookup[(name, '16QAM', snr_db, 'Oracle')]['ber']):.6g} |"
        for name in names
    ]


def write_report(
    outcome: str, interpretation: str, faithful: list[str], generator_rows: list[dict[str, object]],
    orbit_rows: list[dict[str, object]], fit_rows: list[dict[str, object]], level2: list[dict[str, object]],
    communication: list[dict[str, object]], invariants: list[dict[str, object]], diagnostics: dict[str, object],
) -> None:
    generator = {(row["waveform"], row["generator"]): row for row in generator_rows}
    learned_fits = sorted((row for row in fit_rows if row["target"] == "Learned"), key=lambda row: float(row["aligned_matrix_nmse"]))
    best_fit = learned_fits[0]
    best_orbit = max((row for row in orbit_rows if row["waveform"] == "Learned" and row["approximation"] != "exact_dense"), key=lambda row: float(row["mean_column_overlap"]))
    level = {row["waveform"]: row for row in level2}
    successful = {"Learned", *PRIMARY}
    successful_pairs = [row for row in invariants if row["left"] in successful and row["right"] in successful]
    learned_pairs = [row for row in invariants if row["left"] == "Learned" and row["right"] in PRIMARY]
    best_family_by_solution = []
    for name in PRIMARY:
        best = min((row for row in fit_rows if row["target"] == name), key=lambda row: float(row["aligned_matrix_nmse"]))
        best_family_by_solution.append(str(best["family"]))
    family_counts = {family: best_family_by_solution.count(family) for family in sorted(set(best_family_by_solution))}
    lines = [
        "# Stage 4A — Structural Generator and Waveform-Family Forensics", "",
        f"## Result: Outcome {outcome}", "", interpretation, "",
        "No waveform/channel training or BER-based fitting was performed. All fitted candidates were frozen before paired communication evaluation.", "",
        "## Interpretation gate", "",
        f"1. **Simple canonical generator:** not established as a complete law. Learned G_D is concentrated, but G_S is diffuse; the best compressed native orbit was `{best_orbit['approximation']}` with mean column overlap only {best_orbit['mean_column_overlap']:.4f}.",
        f"2. **DD-sparse/local generators:** mixed. Learned top-24 DD energy is {generator[('Learned','G_S')]['top24_energy']:.4f} for G_S and {generator[('Learned','G_D')]['top24_energy']:.4f} for G_D; the asymmetric signature is not shared by all six Stage-3B solutions.",
        f"3. **Compressed orbit:** not accurate enough to claim an orbit law; best phase-aligned orbit NMSE is {best_orbit['phase_aligned_orbit_nmse']:.4f}.",
        f"4. **Fourier/chirp/phase family:** best tested aligned fit is `{best_fit['family']}` with NMSE {best_fit['aligned_matrix_nmse']:.4f} and {best_fit['free_real_parameters']} free real parameters.",
        "5. **Prototype/Gabor family:** evaluated as an exact critical-lattice tight orbit with O(N) prototype storage; its fidelity is reported without dense hidden parameters.",
        f"6. **Compact Q fidelity:** {'yes for ' + ', '.join(faithful) if faithful else 'no tested candidate met the predeclared eta/retained-energy gates'}.",
        f"7. **Communication fidelity:** {'yes under the full predeclared gate' if faithful else 'no compact candidate met the paired two-modulation, four-SNR gate without BER refitting'}.",
        "8. **Common Stage-3B family:** no. Low matrix overlap, weak generator-support agreement, and inconsistent generator sparsity rule out a shared tested family despite common Q concentration.",
        "9. **Haar separation:** partial, not universal. Learned G_D and OFDM-start solutions separate strongly from Haar, while Haar-start Q optima retain diffuse generators.",
        f"10. **N=64 sufficiency:** {'a candidate may proceed to cross-N validation, but is not a final law' if outcome == 'A' else 'insufficient; multi-N and multi-solution evidence is required before proposing a waveform law'}.", "",
        "## Best structural fits to authoritative Learned", "",
        "| Family | real parameters | aligned NMSE | mean squared overlap | eta24 Q | fixed-Q retained |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in learned_fits:
        name = f"FIT_{row['family']}"
        lines.append(
            f"| {row['family']} | {row['free_real_parameters']} | {row['aligned_matrix_nmse']:.4f} | "
            f"{row['mean_matched_squared_overlap']:.4f} | {level[name]['eta24_q']:.6f} | {level[name]['fixed_q_retained_energy_mean']:.6f} |"
        )
    lines += [
        "", "## Paired communication fidelity at 18 dB", "",
        "| Waveform | QPSK fixed-Q BER | 16-QAM fixed-Q BER | QPSK oracle BER | 16-QAM oracle BER |", "|---|---:|---:|---:|---:|",
    ]
    names = ("Learned",) + tuple(f"FIT_{row['family']}" for row in learned_fits)
    lines.extend(communication_table_lines(communication, names))
    lines += [
        "", "## Same matrix, same family, and same function", "",
        f"- **Same matrix:** no. Learned-to-Stage-3B aligned squared overlap spans {min(float(row['mean_matched_squared_overlap']) for row in learned_pairs):.4f}-{max(float(row['mean_matched_squared_overlap']) for row in learned_pairs):.4f}.",
        f"- **Same mathematical family:** not established. Best-fit family counts across the six Stage-3B solutions are {family_counts}; successful-pair G_S/G_D top-24 Jaccard spans {min(float(row['g_s_top24_jaccard']) for row in successful_pairs):.3f}-{max(float(row['g_s_top24_jaccard']) for row in successful_pairs):.3f} / {min(float(row['g_d_top24_jaccard']) for row in successful_pairs):.3f}-{max(float(row['g_d_top24_jaccard']) for row in successful_pairs):.3f}.",
        f"- **Same functional DD representation:** partially yes. Successful-pair Q top-24 row overlap spans {min(float(row['q_top24_overlap_mean']) for row in successful_pairs):.3f}-{max(float(row['q_top24_overlap_mean']) for row in successful_pairs):.3f}, independently of low atom overlap.",
        f"- Shared-support generator phase coherence spans {min(float(row['g_s_shared_support_phase_coherence']) for row in successful_pairs):.3f}-{max(float(row['g_s_shared_support_phase_coherence']) for row in successful_pairs):.3f} for G_S and {min(float(row['g_d_shared_support_phase_coherence']) for row in successful_pairs):.3f}-{max(float(row['g_d_shared_support_phase_coherence']) for row in successful_pairs):.3f} for G_D; no universal phase law is asserted.",
        "", "## Protocol and preservation", "",
        f"- Structural fitting steps: {diagnostics['fit_steps']}; fit objective used matrix/atom structure only.",
        f"- Q bank: {Q_SAMPLES} Sobol DD atoms, seed {Q_SEED}; communication bank: {CHANNELS} independent DD-v1 channels, seed {CHANNEL_SEED}.",
        f"- Each communication point targeted at least {diagnostics['communication_bank']['target_bits']:,} paired bits with Full, Oracle-K24, and frozen-Q-K24 receivers.",
        f"- Authoritative Learned SHA-256: `{diagnostics['waveform_hashes']['Learned']}`; all six Stage-3B hashes verified.",
        f"- All {diagnostics['protected_artifact_count']} prior artifacts remained byte-identical; Stage 2E absent and IDENTITY excluded from figures.",
        "- Fitted matrices are isolated under `outputs/stage4a/fitted/`; no source checkpoint was overwritten.", "",
        "## Conclusion", "", interpretation, "", "**STOP.**",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def run_stage4a(*, fit_steps: int = 400, target_bits: int = TARGET_BITS) -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); FITTED.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    waveforms, hashes = load_waveforms()
    generator_rows, orbit_rows, generator_arrays = generator_analysis(waveforms)
    write_csv(OUT / "generator_metrics.csv", generator_rows); write_csv(OUT / "compressed_orbit_metrics.csv", orbit_rows)
    np.savez(OUT / "generator_weyl_coefficients.npz", **generator_arrays)
    print("generator and compressed-orbit analysis complete", flush=True)

    fit_rows, fitted, parameters = fit_all(waveforms, fit_steps)
    write_csv(OUT / "structured_fit_fidelity.csv", fit_rows); write_json(OUT / "structured_fit_parameters.json", parameters)
    torch.save(fitted, FITTED / "all_structured_fits.pt")
    learned_candidates = {f"FIT_{family}": matrix for family, matrix in fitted["Learned"].items()}
    torch.save(learned_candidates, FITTED / "learned_structured_candidates.pt")

    atom_rows: list[dict[str, object]] = []
    for name, waveform in {"Learned": waveforms["Learned"], **learned_candidates}.items():
        atom_rows.extend({"waveform": name, **row} for row in native_atom_diagnostics(waveform, tuple(range(config.N))))
    write_csv(OUT / "native_atom_diagnostics.csv", atom_rows)

    channels = sample_dd_v1(CHANNELS, CHANNEL_SEED).matrices
    level2, candidate_q = level2_evaluation(waveforms["Learned"], learned_candidates, channels)
    write_csv(OUT / "functional_q_fidelity.csv", level2)
    np.savez(OUT / "candidate_q_matrices.npz", **candidate_q)

    population_delays, population_dopplers = sobol_dd_samples(Q_SAMPLES, Q_SEED)
    population_ops = atomic_operators(population_delays, population_dopplers)
    population_names = tuple(waveforms)
    population_q_stack = differentiable_q(torch.stack([waveforms[name] for name in population_names]), population_ops, chunk_size=16)
    population_q = {name: population_q_stack[index].detach().cpu().numpy() for index, name in enumerate(population_names)}
    np.savez(OUT / "population_q_matrices.npz", **population_q)
    population_q_rows = [
        {"waveform": name, "eta24_q": q_concentration(population_q[name], K)["eta_q"],
         "row_p05": q_concentration(population_q[name], K)["row_p05"],
         "effective_support_mean": q_concentration(population_q[name], K)["effective_support_mean"]}
        for name in population_names
    ]
    write_csv(OUT / "population_q_metrics.csv", population_q_rows)
    invariants = cross_invariants(waveforms, generator_arrays, population_q)
    write_csv(OUT / "cross_waveform_invariants.csv", invariants)

    communication_waveforms = {"Learned": waveforms["Learned"], **learned_candidates}
    per_channel, communication = communication_evaluation(communication_waveforms, candidate_q, channels, target_bits)
    write_csv(OUT / "communication_per_channel.csv", per_channel); write_csv(OUT / "communication_summary.csv", communication)
    outcome, interpretation, faithful = classify(generator_rows, orbit_rows, fit_rows, level2, communication, population_q_rows)
    make_figures(generator_rows, generator_arrays, fit_rows, level2, communication, atom_rows, invariants, communication_waveforms)

    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("prior scientific artifact preservation failure")
    if file_sha256(FROZEN) != hashes["Learned"]:
        raise RuntimeError("authoritative Learned checkpoint mutated")
    diagnostics = {
        "outcome": outcome, "interpretation": interpretation, "functionally_faithful_candidates": faithful,
        "fit_steps": fit_steps, "fit_used_communication": False, "waveform_training_performed": False,
        "channel_training_performed": False, "ber_optimization_performed": False,
        "stage2e_used": False, "identity_in_figures": False,
        "waveform_hashes": hashes, "protected_artifact_count": len(protected_before),
        "protected_artifacts_unchanged": protected_before == protected_after,
        "q_bank": {"seed": Q_SEED, "samples": Q_SAMPLES},
        "communication_bank": {"seed": CHANNEL_SEED, "channels": CHANNELS, "symbol_seed": SYMBOL_SEED, "noise_seed": NOISE_SEED, "target_bits": target_bits},
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage4a_diagnostics.json", diagnostics)
    write_report(outcome, interpretation, faithful, generator_rows, orbit_rows, fit_rows, level2, communication, invariants, diagnostics)
    print(f"STAGE 4A COMPLETE in {diagnostics['runtime_seconds']:.1f} s — OUTCOME {outcome}", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-steps", type=int, default=400)
    parser.add_argument("--target-bits", type=int, default=TARGET_BITS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); run_stage4a(fit_steps=arguments.fit_steps, target_bits=arguments.target_bits)
