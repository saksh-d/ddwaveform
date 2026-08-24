"""Run Stage 4D learned-basin geometry and relative-unitary decomposition."""

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
from src.channel import COMPLEX_DTYPE, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator
from src.receiver_geometry import apply_mask
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2d import score_support_mask, sobol_dd_samples
from src.stage3a import q_concentration
from src.stage3b import atomic_operators, differentiable_q
from src.stage3c import communication_rows
from src.stage4a import canonical_generators, dd_expansion_metrics, integer_weyl_coefficients
from src.stage4b import matrix_pair_metrics
from src.stage4c import operator_fidelity
from src.stage4d import (
    align_columns,
    banded_truncations,
    cayley_generator,
    entry_truncations,
    fit_physical_dictionary,
    fixed_structure_projections,
    generator_spectrum,
    karcher_mean,
    relative_unitary_metrics,
    spectral_truncations,
    tangent_analysis,
    unitary_distance,
    unitary_exponential,
    unitary_logarithm,
    unitary_logarithm_branch,
    weyl_truncations,
)
from src.waveforms import ofdm_waveform
from scripts.stage4b import load_population, summarize_communication
from scripts.stage4c import EXPECTED_LEARNED_HASHES, EXPECTED_STAGE3B_HASHES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage4d"
RECONSTRUCTIONS = OUT / "reconstructions"
REPORT = ROOT / "reports" / "stage4d_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE4B = ROOT / "outputs" / "stage4b" / "checkpoints"
K = 24
BRANCH_TOLERANCE = 0.05
BRANCHES = ("principal", "positive_cut", "negative_cut")
SPECTRAL_RANKS = (1, 2, 4, 8, 16, 32, 64)
WEYL_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
ENTRY_RANKS = WEYL_RANKS
BAND_WIDTHS = (0, 1, 2, 4, 8, 16, 32)
COMMUNICATION_LEVELS = {
    "spectral": (8, 16, 32, 64),
    "weyl": (16, 64, 256, 1024, 4096),
    "time_entry": (64, 256, 1024, 4096),
    "subcarrier_entry": (64, 256, 1024, 4096),
    "circular_band": (1, 4, 16, 32),
}
Q_SAMPLES = 2048
Q_SEED = 11230
OPERATOR_SAMPLES = 512
OPERATOR_SEED = 11231
CHANNELS = 512
CHANNEL_SEED = 11232
SYMBOL_SEED = 11233
NOISE_SEED = 11234
TARGET_BITS = 1_000_000
SNRS = (12, 15, 18, 20)
LEARNED = tuple(f"LEARNED_{index}" for index in range(5))
CONTROL_NAMES = (
    "Q_MEAN_FROM_OFDM",
    "Q_MEAN_FROM_HAAR_1",
    "AFDM",
    "OTFS",
    "RAW_HAAR_1",
    "RAW_HAAR_2",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted((ROOT / "outputs").glob("**/*"))
        if path.is_file() and "stage4d" not in path.parts
    }


def load_frozen_population() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    learned0, info = load_frozen_waveform(FROZEN)
    members = {"LEARNED_0": learned0}
    hashes = {"LEARNED_0": str(info["sha256"])}
    for index in range(1, 5):
        name = f"LEARNED_{index}"
        path = STAGE4B / f"learned_{index}.pt"
        members[name] = torch.load(path, weights_only=True)
        hashes[name] = file_sha256(path)
    if hashes != EXPECTED_LEARNED_HASHES:
        raise RuntimeError(f"Learned checkpoint hash mismatch: {hashes}")
    waveforms, verified = load_population(members)
    for name, expected in EXPECTED_STAGE3B_HASHES.items():
        if verified[name] != expected:
            raise RuntimeError(f"Stage-3B checkpoint hash mismatch: {name}")
    hashes.update({name: verified[name] for name in EXPECTED_STAGE3B_HASHES})
    return waveforms, hashes


def align_population(
    waveforms: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]]]:
    aligned: dict[str, torch.Tensor] = {"LEARNED_0": waveforms["LEARNED_0"]}
    metadata: dict[str, dict[str, object]] = {
        "LEARNED_0": {
            "reference": "LEARNED_0",
            "permutation": list(range(config.N)),
            "phases_real": [1.0] * config.N,
            "phases_imag": [0.0] * config.N,
            "mean_matched_squared_overlap": 1.0,
        }
    }
    for name in LEARNED[1:]:
        aligned[name], current = align_columns(waveforms["LEARNED_0"], waveforms[name])
        metadata[name] = {"reference": "LEARNED_0", **current}
    ofdm = waveforms["OFDM"]
    aligned["OFDM"] = ofdm
    for name in CONTROL_NAMES:
        aligned[name], current = align_columns(ofdm, waveforms[name])
        metadata[name] = {"reference": "OFDM", **current}
    return aligned, metadata


def center_geometry(
    waveforms: dict[str, torch.Tensor], aligned: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, object], list[dict[str, object]]]:
    native_center, native_diagnostics = karcher_mean(torch.stack([waveforms[name] for name in LEARNED]))
    center, aligned_diagnostics = karcher_mean(torch.stack([aligned[name] for name in LEARNED]))
    centers = {"NATIVE_CENTER": native_center, "ALIGNED_CENTER": center}
    rows: list[dict[str, object]] = []
    for convention, current_center, population in (
        ("native", native_center, waveforms),
        ("aligned", center, aligned),
    ):
        for name in (*LEARNED, *CONTROL_NAMES, "OFDM"):
            rows.append({"convention": convention, "waveform": name, **unitary_distance(current_center, population[name])})
    return centers, {"native": native_diagnostics, "aligned": aligned_diagnostics}, rows


def relative_analysis(
    waveforms: dict[str, torch.Tensor],
    aligned: dict[str, torch.Tensor],
    centers: dict[str, torch.Tensor],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, torch.Tensor]]:
    ofdm = waveforms["OFDM"]
    rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    generators: dict[str, torch.Tensor] = {}
    groups = (
        ("native", {**{name: waveforms[name] for name in (*LEARNED, *CONTROL_NAMES)}, "CENTER": centers["NATIVE_CENTER"]}),
        ("aligned", {**{name: aligned[name] for name in (*LEARNED, *CONTROL_NAMES)}, "CENTER": centers["ALIGNED_CENTER"]}),
    )
    for convention, population in groups:
        for name, waveform in population.items():
            relative = ofdm.mH @ waveform
            metrics = relative_unitary_metrics(relative)
            logarithm = unitary_logarithm(relative, branch_tolerance=BRANCH_TOLERANCE)
            key = f"{convention}:{name}"
            generators[key] = logarithm.generator
            rows.append({"convention": convention, "waveform": name, **metrics, **{f"generator_{k}": v for k, v in generator_spectrum(logarithm.generator).items() if not isinstance(v, list)}})
            phase_rows.extend(
                {"convention": convention, "waveform": name, "eigen_index": index, "eigenphase": float(phase), "branch_distance": float(np.pi - abs(phase))}
                for index, phase in enumerate(logarithm.eigenphases)
            )
    return rows, phase_rows, generators


def tangent_geometry(center: torch.Tensor, aligned: dict[str, torch.Tensor]) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    logs = [unitary_logarithm(center.mH @ aligned[name], branch_tolerance=BRANCH_TOLERANCE) for name in LEARNED]
    if not all(item.gate_passed for item in logs):
        raise RuntimeError("within-basin tangent logarithm failed its branch gate")
    tangents = torch.stack([item.generator for item in logs])
    metrics = tangent_analysis(tangents)
    center_coefficients = integer_weyl_coefficients(unitary_logarithm(ofdm_waveform().mH @ center).generator)
    center_order = torch.argsort(torch.abs(center_coefficients).square().reshape(-1), descending=True, stable=True)
    rows: list[dict[str, object]] = []
    coefficients = []
    for name, item in zip(LEARNED, logs):
        current = integer_weyl_coefficients(item.generator)
        coefficients.append(current.detach().cpu().numpy())
        current_order = torch.argsort(torch.abs(current).square().reshape(-1), descending=True, stable=True)
        row: dict[str, object] = {
            "waveform": name,
            "tangent_norm": float(torch.linalg.matrix_norm(item.generator)),
            "branch_margin": item.branch_margin,
        }
        for rank in (8, 16, 32, 64, 128):
            shared = len(set(center_order[:rank].tolist()) & set(current_order[:rank].tolist())) / rank
            row[f"center_top{rank}_weyl_overlap"] = shared
        rows.append(row)
    return metrics, rows, np.stack(coefficients)


def decompose_branches(
    relative: torch.Tensor,
    ofdm: torch.Tensor,
) -> tuple[
    list[dict[str, object]],
    dict[str, torch.Tensor],
    dict[str, object],
    dict[str, np.ndarray],
]:
    rows: list[dict[str, object]] = []
    candidates: dict[str, torch.Tensor] = {}
    metadata: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    branch_prefix = {"principal": "PR", "positive_cut": "PC", "negative_cut": "NC"}
    for branch in BRANCHES:
        result = unitary_logarithm_branch(relative, branch=branch)
        generator = result.generator
        branch_meta: dict[str, object] = {
            "branch_margin": result.branch_margin,
            "gate_passed": result.gate_passed,
            "reconstruction_residual": result.reconstruction_residual,
            "spectrum": generator_spectrum(generator),
        }
        spectral, spectral_rows = spectral_truncations(generator, SPECTRAL_RANKS)
        weyl, weyl_rows, weyl_meta = weyl_truncations(generator, WEYL_RANKS)
        time_entry, time_rows = entry_truncations(generator, ENTRY_RANKS, name="time_entry")
        subcarrier_entry, subcarrier_rows = entry_truncations(
            generator, ENTRY_RANKS, basis=ofdm, name="subcarrier_entry"
        )
        bands, band_rows = banded_truncations(generator, BAND_WIDTHS)
        physical, physical_row = fit_physical_dictionary(generator)
        for current in (*spectral_rows, *weyl_rows, *time_rows, *subcarrier_rows, *band_rows, *fixed_structure_projections(generator), physical_row):
            rows.append({"branch": branch, **current})
        branch_meta["weyl"] = weyl_meta
        branch_meta["physical_dictionary"] = physical_row
        metadata[branch] = branch_meta
        arrays[f"{branch}_generator"] = generator.detach().cpu().numpy()
        arrays[f"{branch}_weyl_coefficients"] = integer_weyl_coefficients(generator).detach().cpu().numpy()
        mappings = {
            "spectral": spectral,
            "weyl": weyl,
            "time_entry": time_entry,
            "subcarrier_entry": subcarrier_entry,
            "circular_band": bands,
        }
        for representation, levels in COMMUNICATION_LEVELS.items():
            for level in levels:
                approximation = mappings[representation][level]
                name = f"{branch_prefix[branch]}_{representation.upper()}_{level}"
                candidates[name] = ofdm @ unitary_exponential(approximation)
        candidates[f"{branch_prefix[branch]}_PHYSICAL_19"] = ofdm @ unitary_exponential(physical)
    cayley, cayley_metrics = cayley_generator(relative)
    cayley_spectrum = generator_spectrum(cayley)
    metadata["cayley"] = {**cayley_metrics, "spectrum": cayley_spectrum}
    arrays["cayley_generator"] = cayley.detach().cpu().numpy()
    return rows, candidates, metadata, arrays


def q_and_reconstruction_analysis(
    waveforms: dict[str, torch.Tensor],
    center: torch.Tensor,
    q_operators: torch.Tensor,
    evaluation_operators: torch.Tensor,
    channels: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor], list[dict[str, object]]]:
    names = tuple(waveforms)
    q_values: dict[str, np.ndarray] = {}
    for start in range(0, len(names), 8):
        current_names = names[start : start + 8]
        current = differentiable_q(torch.stack([waveforms[name] for name in current_names]), q_operators, chunk_size=16)
        for index, name in enumerate(current_names):
            q_values[name] = current[index].detach().cpu().numpy()
    masks = {name: score_support_mask(q_values[name], K) for name in names}
    rows: list[dict[str, object]] = []
    q_rows: list[dict[str, object]] = []
    center_q = q_values["ALIGNED_CENTER"]
    for index, name in enumerate(names):
        q = q_values[name]
        concentration = q_concentration(q, K)
        shift, modulation = canonical_generators(waveforms[name])
        shift_metrics = dd_expansion_metrics(integer_weyl_coefficients(shift))
        modulation_metrics = dd_expansion_metrics(integer_weyl_coefficients(modulation))
        effective = effective_operator(channels, waveforms[name])
        retained = apply_mask(effective, masks[name])
        fraction = torch.sum(torch.abs(retained).square(), dim=(-2, -1)) / torch.sum(torch.abs(effective).square(), dim=(-2, -1))
        q_rows.append({
            "waveform": name,
            "eta24_q": concentration["eta_q"],
            "one_minus_eta24_q": 1.0 - concentration["eta_q"],
            "row_p05": concentration["row_p05"],
            "q_relative_distance_to_center": float(np.linalg.norm(q - center_q) / np.linalg.norm(center_q)),
            "q_correlation_to_center": float(np.corrcoef(q.reshape(-1), center_q.reshape(-1))[0, 1]),
            "fixed_q_retained_energy_mean": float(fraction.mean()),
            "fixed_q_retained_energy_p05": float(torch.quantile(fraction, 0.05)),
            "g_s_top24_energy": shift_metrics["top24_energy"],
            "g_d_top24_energy": modulation_metrics["top24_energy"],
            "g_d_to_g_s_ratio": modulation_metrics["top24_energy"] / shift_metrics["top24_energy"],
        })
        if name not in {"ALIGNED_CENTER", *LEARNED, "OFDM", *CONTROL_NAMES}:
            matrix = matrix_pair_metrics(center, waveforms[name])
            matrix.pop("column_overlaps")
            operator, _ = operator_fidelity(center, waveforms[name], evaluation_operators)
            rows.append({"waveform": name, **matrix, **operator, **q_rows[-1]})
        if (index + 1) % 12 == 0:
            print(f"functional analysis {index + 1}/{len(names)}", flush=True)
    np.savez(OUT / "q_matrices.npz", **{name: value for name, value in q_values.items()})
    return rows, masks, q_rows


def communication_evaluation(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    target_bits: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = communication_rows(
        channels,
        waveforms,
        masks,
        (qpsk_constellation(), qam16_constellation()),
        SNRS,
        k=K,
        target_bits=target_bits,
        symbol_seed=SYMBOL_SEED,
        noise_seed=NOISE_SEED,
    )
    return rows, summarize_communication(rows)


def branch_comparison(
    approximation_rows: list[dict[str, object]], metadata: dict[str, object]
) -> dict[str, object]:
    rank95 = {branch: int(metadata[branch]["spectrum"]["rank_95"]) for branch in BRANCHES}
    weyl64 = {
        branch: next(float(row["relative_squared_residual"]) for row in approximation_rows if row["branch"] == branch and row["representation"] == "weyl" and int(row["complexity"]) == 64)
        for branch in BRANCHES
    }
    spread = max(weyl64.values()) - min(weyl64.values())
    material = max(rank95.values()) - min(rank95.values()) >= 4 or spread >= 0.10
    return {
        "rank95_by_branch": rank95,
        "weyl64_residual_by_branch": weyl64,
        "weyl64_residual_spread": spread,
        "material_structural_sensitivity": material,
        "cayley_condition_r_plus_i": metadata["cayley"]["condition_r_plus_i"],
        "cayley_condition_limit": 1.0 / BRANCH_TOLERANCE,
        "cayley_ill_conditioned": metadata["cayley"]["condition_r_plus_i"] >= 1.0 / BRANCH_TOLERANCE,
    }


def population_generator_comparison(
    waveforms: dict[str, torch.Tensor],
    center: torch.Tensor,
) -> list[dict[str, object]]:
    """Apply one fixed principal-generator compression pipeline to every population."""
    ofdm = waveforms["OFDM"]
    center_coefficients = integer_weyl_coefficients(unitary_logarithm(ofdm.mH @ center).generator)
    center_order = torch.argsort(torch.abs(center_coefficients).square().reshape(-1), descending=True, stable=True)
    rows: list[dict[str, object]] = []
    for name in (*LEARNED, *CONTROL_NAMES):
        relative = ofdm.mH @ waveforms[name]
        logarithm = unitary_logarithm(relative, branch_tolerance=BRANCH_TOLERANCE)
        spectrum = generator_spectrum(logarithm.generator)
        _, weyl_rows, weyl = weyl_truncations(logarithm.generator, (16, 64, 256, 1024, 4096))
        _, physical = fit_physical_dictionary(logarithm.generator)
        coefficients = integer_weyl_coefficients(logarithm.generator)
        order = torch.argsort(torch.abs(coefficients).square().reshape(-1), descending=True, stable=True)
        rows.append({
            "waveform": name,
            "population": "Learned" if name in LEARNED else ("Stage3B" if name.startswith("Q_") else ("Haar" if name.startswith("RAW_HAAR") else "Classical")),
            "branch_margin": logarithm.branch_margin,
            "principal_log_gate_passed": logarithm.gate_passed,
            "spectral_rank90": spectrum["rank_90"],
            "spectral_rank95": spectrum["rank_95"],
            "spectral_rank99": spectrum["rank_99"],
            "weyl_effective_support": weyl["effective_support"],
            "weyl_normalized_entropy": weyl["normalized_entropy"],
            **{f"weyl_rank{int(row['complexity'])}_residual": row["relative_squared_residual"] for row in weyl_rows},
            "center_top64_weyl_overlap": len(set(center_order[:64].tolist()) & set(order[:64].tolist())) / 64,
            "physical_dictionary_residual": physical["relative_squared_residual"],
        })
    return rows


def classify(
    relative_rows: list[dict[str, object]],
    center_rows: list[dict[str, object]],
    approximation_rows: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
    tangent: dict[str, object],
    branch: dict[str, object],
) -> tuple[str, dict[str, object]]:
    center_relative = next(row for row in relative_rows if row["convention"] == "aligned" and row["waveform"] == "CENTER")
    aligned_learned = [row for row in center_rows if row["convention"] == "aligned" and row["waveform"] in LEARNED]
    center_ofdm = next(row for row in center_rows if row["convention"] == "aligned" and row["waveform"] == "OFDM")
    basin_ratio = max(float(row["geodesic_distance"]) for row in aligned_learned) / float(center_ofdm["geodesic_distance"])
    principal_weyl = [row for row in approximation_rows if row["branch"] == "principal" and row["representation"] == "weyl"]
    low_weyl = min((int(row["complexity"]) for row in principal_weyl if float(row["relative_squared_residual"]) <= 0.05), default=math.inf)
    low_names = {
        "PR_SPECTRAL_8",
        "PR_SPECTRAL_16",
        "PR_WEYL_16",
        "PR_WEYL_64",
        "PR_TIME_ENTRY_64",
        "PR_SUBCARRIER_ENTRY_64",
        "PR_CIRCULAR_BAND_1",
        "PR_CIRCULAR_BAND_4",
        "PR_PHYSICAL_19",
    }
    low_function = [row for row in reconstruction_rows if row["waveform"] in low_names]
    functional = bool(low_function) and any(float(row["d_b"]) <= 0.20 and float(row["one_minus_eta24_q"]) <= 0.003 for row in low_function)
    gates = {
        "aligned_basin_radius_to_ofdm_displacement_ratio": basin_ratio,
        "stable_aligned_basin": basin_ratio < 0.25,
        "principal_log_branch_margin": float(center_relative["branch_margin"]),
        "principal_log_gate_passed": bool(center_relative["log_gate_passed"]),
        "branch_comparison": branch,
        "principal_weyl_rank_for_95pct_generator_fidelity": None if math.isinf(low_weyl) else low_weyl,
        "low_complexity_functional_reconstruction": functional,
        "observed_tangent_rank_upper_bound": tangent["centered_rank_upper_bound"],
    }
    if not gates["principal_log_gate_passed"] and (
        branch["material_structural_sensitivity"] or branch["cayley_ill_conditioned"]
    ):
        return "D", gates
    if low_weyl <= 64 and functional:
        return "A", gates
    if gates["stable_aligned_basin"]:
        return ("C" if int(tangent["centered_rank_upper_bound"]) <= 2 else "B"), gates
    return "E", gates


def make_figures(
    phase_rows: list[dict[str, object]],
    approximation_rows: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
    communication: list[dict[str, object]],
) -> None:
    plt.figure(figsize=(8.4, 4.6))
    for label, linestyle in (("CENTER", "-"), ("LEARNED_0", "--"), ("Q_MEAN_FROM_OFDM", ":")):
        selected = sorted(
            [row for row in phase_rows if row["convention"] == "aligned" and row["waveform"] == label],
            key=lambda row: int(row["eigen_index"]),
        )
        plt.plot([row["eigenphase"] for row in selected], label=label, linestyle=linestyle)
    plt.axhline(np.pi, color="black", linewidth=0.7)
    plt.axhline(-np.pi, color="black", linewidth=0.7)
    plt.xlabel("sorted eigenphase index")
    plt.ylabel("eigenphase [rad]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "fig01_relative_eigenphases.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.4, 5.0))
    styles = {"principal": "-", "positive_cut": "--", "negative_cut": ":"}
    colors = {
        "spectral": "tab:blue",
        "weyl": "tab:orange",
        "time_entry": "tab:green",
        "subcarrier_entry": "tab:red",
        "circular_band": "tab:purple",
    }
    for branch in BRANCHES:
        for representation in ("spectral", "weyl", "time_entry", "subcarrier_entry", "circular_band"):
            selected = [row for row in approximation_rows if row["branch"] == branch and row["representation"] == representation]
            plt.loglog(
                [max(1, int(row["complexity"])) for row in selected],
                [max(float(row["relative_squared_residual"]), 1e-16) for row in selected],
                linestyle=styles[branch],
                color=colors[representation],
                marker="o" if branch == "principal" else None,
                markersize=3,
                label=representation if branch == "principal" else None,
                alpha=0.75,
            )
    for branch in BRANCHES:
        plt.plot([], [], color="black", linestyle=styles[branch], label=branch.replace("_", " "))
    plt.xlabel("declared representation complexity")
    plt.ylabel("generator relative squared residual")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "fig02_generator_complexity.png", dpi=180)
    plt.close()

    selected = [row for row in reconstruction_rows if row["waveform"].startswith(("PR_WEYL", "PC_WEYL", "NC_WEYL"))]
    plt.figure(figsize=(8.4, 4.8))
    for prefix, label in (("PR", "principal"), ("PC", "positive cut"), ("NC", "negative cut")):
        current = sorted([row for row in selected if row["waveform"].startswith(prefix)], key=lambda row: int(row["waveform"].rsplit("_", 1)[-1]))
        complexity = [int(row["waveform"].rsplit("_", 1)[-1]) for row in current]
        plt.semilogx(complexity, [row["d_b"] for row in current], marker="o", label=f"d_B: {label}")
        plt.semilogx(complexity, [10 * row["one_minus_eta24_q"] for row in current], marker="x", linestyle="--", label=f"10(1-eta24): {label}")
    plt.xlabel("retained Weyl coefficients")
    plt.ylabel("functional discrepancy")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "fig03_weyl_functional_curves.png", dpi=180)
    plt.close()

    lookup = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): float(row["ber"]) for row in communication}
    display = ["OFDM", "ALIGNED_CENTER", "LEARNED_0", "Q_MEAN_FROM_OFDM", "PR_WEYL_64", "PR_WEYL_256", "PR_WEYL_1024", "PR_WEYL_4096"]
    x = np.arange(len(display)); width = 0.38
    qpsk = [lookup[(name, "QPSK", 18, "Q")] for name in display]
    qam = [lookup[(name, "16QAM", 18, "Q")] for name in display]
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(x - width / 2, qpsk, width, label="QPSK")
    plt.bar(x + width / 2, qam, width, label="16-QAM")
    plt.yscale("log")
    plt.xticks(x, display, rotation=28, ha="right")
    plt.ylabel("18 dB Fixed-Q BER")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "fig04_center_reconstruction_communication.png", dpi=180)
    plt.close()


def write_report(
    outcome: str,
    gates: dict[str, object],
    center_diagnostics: dict[str, object],
    center_rows: list[dict[str, object]],
    relative_rows: list[dict[str, object]],
    approximation_rows: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
    q_rows: list[dict[str, object]],
    communication: list[dict[str, object]],
    tangent: dict[str, object],
    tangent_rows: list[dict[str, object]],
    population_generators: list[dict[str, object]],
    diagnostics: dict[str, object],
) -> None:
    center_relative = next(row for row in relative_rows if row["convention"] == "aligned" and row["waveform"] == "CENTER")
    native_relative = next(row for row in relative_rows if row["convention"] == "native" and row["waveform"] == "CENTER")
    aligned_learned = [row for row in center_rows if row["convention"] == "aligned" and row["waveform"] in LEARNED]
    native_learned = [row for row in center_rows if row["convention"] == "native" and row["waveform"] in LEARNED]
    q_lookup = {row["waveform"]: row for row in q_rows}
    comm = {(row["waveform"], row["modulation"], int(row["snr_db"]), row["support_rule"]): row for row in communication}
    principal_weyl = [row for row in approximation_rows if row["branch"] == "principal" and row["representation"] == "weyl"]
    branch_info = diagnostics["branch_analysis"]
    branch_rank = branch_info["rank95_by_branch"]
    branch_weyl = branch_info["weyl64_residual_by_branch"]
    principal_spectrum = diagnostics["decomposition_metadata"]["principal"]["spectrum"]
    physical = diagnostics["decomposition_metadata"]["principal"]["physical_dictionary"]
    center_comm = comm[("ALIGNED_CENTER", "QPSK", 18, "Q")]["ber"], comm[("ALIGNED_CENTER", "16QAM", 18, "Q")]["ber"]
    learned_comm = [comm[(name, "QPSK", 18, "Q")]["ber"] for name in LEARNED], [comm[(name, "16QAM", 18, "Q")]["ber"] for name in LEARNED]
    center_relative_to_median = (
        abs(center_comm[0] / np.median(learned_comm[0]) - 1),
        abs(center_comm[1] / np.median(learned_comm[1]) - 1),
    )
    full_weyl = next(row for row in reconstruction_rows if row["waveform"] == "PR_WEYL_4096")
    low_weyl = next(row for row in reconstruction_rows if row["waveform"] == "PR_WEYL_64")
    learned_generators = [row for row in population_generators if row["population"] == "Learned"]
    stage3b_generators = [row for row in population_generators if row["population"] == "Stage3B"]
    haar_generators = [row for row in population_generators if row["population"] == "Haar"]
    lines = [
        "# Stage 4D — Learned-Basin Geometry and Relative-Unitary Decomposition",
        "",
        f"## Outcome {outcome}",
        "",
        ("**Outcome D — the exponential-generator coordinate is branch-unstable at the OFDM-to-Learned displacement.**" if outcome == "D" else f"**Outcome {outcome}.**"),
        "",
        "The five Learned solutions do form a tight unitary basin. However, the aligned center's OFDM-relative spectrum places eigenvalues immediately beside the logarithm branch cut. The principal logarithm reconstructs the matrix numerically, but it is not a unique stable structural coordinate. One-sided valid branches agree on the coarse noncompact conclusion, yet no branch is canonical and the Cayley alternative is severely ill-conditioned. Therefore Stage 4D does not promote `W = W_OFDM exp(K_theta)` as a credible compact law.",
        "",
        "## Frozen protocol and gauges",
        "",
        "- Native convention uses checkpoint column order and phases exactly.",
        "- Canonical Learned convention aligns every repetition once to `LEARNED_0` using only column permutation/phase equivalence. Controls align once to OFDM and are reported separately.",
        "- The center is an intrinsic Karcher mean initialized by the polar-projected Euclidean centroid; no BER, Q, or structural score enters alignment or centering.",
        f"- Aligned center converged in {center_diagnostics['aligned']['iterations']} iterations with unitarity residual {center_diagnostics['aligned']['unitarity_residual']:.3e}.",
        "",
        "## Basin geometry",
        "",
        f"Aligned member-to-center geodesic distance spans {min(row['geodesic_distance'] for row in aligned_learned):.5f}-{max(row['geodesic_distance'] for row in aligned_learned):.5f}; native distance spans {min(row['geodesic_distance'] for row in native_learned):.5f}-{max(row['geodesic_distance'] for row in native_learned):.5f}. The aligned basin radius is {gates['aligned_basin_radius_to_ofdm_displacement_ratio']:.3%} of the center-to-OFDM displacement.",
        f"Within-basin tangent norms span {min(row['tangent_norm'] for row in tangent_rows):.4f}-{max(row['tangent_norm'] for row in tangent_rows):.4f}. Centered observed rank is {tangent['centered_rank_upper_bound']} (necessarily at most four); this is descriptive and is not an intrinsic manifold-dimension claim.",
        "",
        "## Logarithm validity and branch sensitivity",
        "",
        f"The aligned-center principal branch margin is only {center_relative['branch_margin']:.6f} rad (native center {native_relative['branch_margin']:.6f} rad), below the predeclared {BRANCH_TOLERANCE:.2f}-rad gate. Skew residual is {center_relative['log_skew_residual']:.3e} and exponential reconstruction residual is {center_relative['log_reconstruction_residual']:.3e}; numerical exactness does not remove branch ambiguity.",
        f"Generator 95% spectral-energy ranks are principal/positive-cut/negative-cut = {branch_rank['principal']}/{branch_rank['positive_cut']}/{branch_rank['negative_cut']}. Their rank-64 Weyl residuals are {branch_weyl['principal']:.4f}/{branch_weyl['positive_cut']:.4f}/{branch_weyl['negative_cut']:.4f}; coarse noncompactness is robust to these choices.",
        f"The Cayley `(R+I)` condition number is {branch_info['cayley_condition_r_plus_i']:.3g}, versus the tolerance-derived admissible limit {branch_info['cayley_condition_limit']:.1f}; its near-minus-one amplification makes it unsuitable as a compact replacement here.",
        "",
        "## Principal-coordinate decomposition (diagnostic only)",
        "",
        f"The principal generator requires ranks {principal_spectrum['rank_90']}/{principal_spectrum['rank_95']}/{principal_spectrum['rank_99']}/{principal_spectrum['rank_99_5']} for 90/95/99/99.5% spectral energy. It is not low rank.",
        f"Principal Weyl rank-64 residual is {next(row['relative_squared_residual'] for row in principal_weyl if row['complexity']==64):.4f}; full rank-4096 returns the exact principal generator. Fixed physical-dictionary residual is {physical['relative_squared_residual']:.4f} with 19 real coefficients, and dominant normalized commutator mean/max are {physical['dominant_pair_commutator_mean']:.3f}/{physical['dominant_pair_commutator_maximum']:.3f}.",
        f"Rank-64 Weyl reconstruction has d_B={low_weyl['d_b']:.4f}, residual Q mass={low_weyl['one_minus_eta24_q']:.5f}, and G_D/G_S={low_weyl['g_d_to_g_s_ratio']:.3f}. Full Weyl reconstruction returns d_B={full_weyl['d_b']:.3e} and the center structurally, but stores the complete 4,096-coefficient operator and needs a dense exponential.",
        "",
        "## Learned versus control generators",
        "",
        f"Learned principal-log 95% spectral ranks span {min(row['spectral_rank95'] for row in learned_generators)}-{max(row['spectral_rank95'] for row in learned_generators)} and rank-64 Weyl residuals span {min(row['weyl_rank64_residual'] for row in learned_generators):.4f}-{max(row['weyl_rank64_residual'] for row in learned_generators):.4f}. Stage-3B values are rank {min(row['spectral_rank95'] for row in stage3b_generators)}-{max(row['spectral_rank95'] for row in stage3b_generators)} and residual {min(row['weyl_rank64_residual'] for row in stage3b_generators):.4f}-{max(row['weyl_rank64_residual'] for row in stage3b_generators):.4f}; Haar residual is {min(row['weyl_rank64_residual'] for row in haar_generators):.4f}-{max(row['weyl_rank64_residual'] for row in haar_generators):.4f}.",
        "Top-64 coefficient overlap, entropy/effective support, all nested Weyl residuals, and the same 19-term physical fit are reported for every Learned, Stage-3B, classical, and Haar member in `population_generator_comparison.csv`. No compact Learned-specific separation survives these controls.",
        "",
        "## Center and reconstruction communication",
        "",
        f"At 18 dB Fixed-Q, the center gives QPSK/16-QAM BER {center_comm[0]:.6g}/{center_comm[1]:.6g}; the five Learned members span {min(learned_comm[0]):.6g}-{max(learned_comm[0]):.6g} / {min(learned_comm[1]):.6g}-{max(learned_comm[1]):.6g}. Center deviations from the population medians are {center_relative_to_median[0]:.2%}/{center_relative_to_median[1]:.2%}, so it is communication-comparable without BER-based redefinition.",
        "All QPSK/16-QAM, 12/15/18/20 dB, Full/Oracle-K24/Fixed-Q-K24 results for every predeclared frozen reconstruction are in `communication_summary.csv`.",
        "",
        "## Interpretation gate",
        "",
        "1. **Yes under both reported conventions.** Native and canonical centers give similarly small population radii; the canonical convention is retained because the literal relative generator remains gauge-dependent.",
        "2. **No as a stable coordinate.** The principal log is numerically exact but fails the branch-margin gate.",
        f"3. **No.** Principal spectral 95% rank is {principal_spectrum['rank_95']} of 64.",
        "4. **No at genuinely low complexity.** DD/Weyl fidelity requires hundreds to thousands of terms.",
        "5. **No predeclared alternative is compact.** Time/subcarrier sparsity, bandedness, Toeplitz/circulant, 8x8 block/Kronecker, and the 19-term physical dictionary leave large residuals.",
        "6. **Locally yes.** Dominant within-basin directions and Weyl overlaps are reproducible descriptively, subject to only five samples.",
        f"7. **Yes.** Aligned basin radius/displacement ratio is {gates['aligned_basin_radius_to_ofdm_displacement_ratio']:.3f}.",
        "8. **Yes within each fixed branch.** Nested projections monotonically recover that branch's generator by construction.",
        "9. **Only at high complexity.** Low ranks do not enter the center's complex-B basin.",
        "10. **No monotone functional approach.** Residual Q mass worsens sharply at intermediate Weyl ranks and recovers only at the exact dense endpoint.",
        "11. **Approaches at intermediate-to-high complexity, nonmonotonically.** Full reception improves before exact matrix recovery, but supplies no compact structural law.",
        "12. **No monotone finite-K approach.** Oracle/Fixed-Q communication degrades badly at intermediate Weyl ranks and recovers only at the exact dense endpoint.",
        "13. **No robust learned-specific compact law.** The same nested compression pipeline on Stage-3B and Haar controls prevents interpreting generic dense-log behavior as Learned structure.",
        "14. **Reported without selection.** The center's full paired curves are frozen above and in the machine-readable table.",
        "15. **No.** `W_OFDM exp(K_theta)` remains branch-sensitive and noncompact at N=64.",
        "",
        "## Parameter and computation accounting",
        "",
        "A retained complex Weyl coefficient is counted as two real stored scalars before skew projection. Sparse coefficient storage is descriptive: every evaluated `exp(K_r)` was formed as a dense 64x64 exponential with O(N^3) offline cost, then stored/applied as a dense unitary with O(N^2) per vector. No tested truncation supplies a fast exponential/action implementation.",
        "",
        "## Preservation",
        "",
        f"- All five Learned and all six Stage-3B hashes verified; {diagnostics['protected_artifact_count']} prior artifacts remained byte-identical.",
        f"- Production banks: {diagnostics['q_samples']} Q atoms, {diagnostics['operator_samples']} complex-B atoms, {diagnostics['channels']} paired channels, at least {diagnostics['target_bits']} bits per waveform/point.",
        "- Stage 2E absent. No training, BER/Q/geometry fitting, truncation reselection, receiver change, or new waveform-family guess.",
        "- STOP. Proceed only after review; Stage 4D itself does not start multi-N work.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


def run_stage4d(
    *,
    q_samples: int = Q_SAMPLES,
    operator_samples: int = OPERATOR_SAMPLES,
    channels_count: int = CHANNELS,
    target_bits: int = TARGET_BITS,
) -> dict[str, object]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    RECONSTRUCTIONS.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    waveforms, hashes = load_frozen_population()
    aligned, alignment_metadata = align_population(waveforms)
    centers, center_diagnostics, center_rows = center_geometry(waveforms, aligned)
    write_json(OUT / "gauge_alignments.json", alignment_metadata)
    torch.save(centers["ALIGNED_CENTER"], OUT / "aligned_karcher_center.pt")
    torch.save(centers["NATIVE_CENTER"], OUT / "native_karcher_center.pt")
    write_csv(OUT / "center_distances.csv", center_rows)
    relative_rows, phase_rows, relative_generators = relative_analysis(waveforms, aligned, centers)
    write_csv(OUT / "relative_unitary_metrics.csv", relative_rows)
    write_csv(OUT / "relative_eigenphases.csv", phase_rows)
    np.savez(OUT / "relative_generators.npz", **{name.replace(":", "_"): value.detach().cpu().numpy() for name, value in relative_generators.items()})
    tangent, tangent_rows, tangent_coefficients = tangent_geometry(centers["ALIGNED_CENTER"], aligned)
    write_json(OUT / "tangent_space_summary.json", tangent)
    write_csv(OUT / "tangent_member_metrics.csv", tangent_rows)
    np.savez(OUT / "tangent_weyl_coefficients.npz", coefficients=tangent_coefficients)
    population_generators = population_generator_comparison({**aligned, "OFDM": waveforms["OFDM"]}, centers["ALIGNED_CENTER"])
    write_csv(OUT / "population_generator_comparison.csv", population_generators)
    relative = waveforms["OFDM"].mH @ centers["ALIGNED_CENTER"]
    approximation_rows, reconstructions, decomposition_metadata, arrays = decompose_branches(relative, waveforms["OFDM"])
    write_csv(OUT / "generator_approximation_curves.csv", approximation_rows)
    write_json(OUT / "coefficient_spectra_and_branch_diagnostics.json", decomposition_metadata)
    np.savez(OUT / "generator_coefficients.npz", **arrays)
    for name, waveform in reconstructions.items():
        torch.save(waveform, RECONSTRUCTIONS / f"{name.lower()}.pt")
    references = {
        "ALIGNED_CENTER": centers["ALIGNED_CENTER"],
        **{name: aligned[name] for name in LEARNED},
        "OFDM": waveforms["OFDM"],
        **{name: aligned[name] for name in CONTROL_NAMES},
    }
    evaluation_waveforms = {**references, **reconstructions}
    q_delays, q_dopplers = sobol_dd_samples(q_samples, Q_SEED)
    operator_delays, operator_dopplers = sobol_dd_samples(operator_samples, OPERATOR_SEED)
    q_operators = atomic_operators(q_delays, q_dopplers)
    evaluation_operators = atomic_operators(operator_delays, operator_dopplers)
    channels = sample_dd_v1(channels_count, CHANNEL_SEED).matrices
    reconstruction_rows, masks, q_rows = q_and_reconstruction_analysis(
        evaluation_waveforms, centers["ALIGNED_CENTER"], q_operators, evaluation_operators, channels
    )
    write_csv(OUT / "reconstruction_functional_metrics.csv", reconstruction_rows)
    write_csv(OUT / "waveform_q_generator_metrics.csv", q_rows)
    communication_channel, communication = communication_evaluation(channels, evaluation_waveforms, masks, target_bits)
    write_csv(OUT / "communication_per_channel.csv", communication_channel)
    write_csv(OUT / "communication_summary.csv", communication)
    branch = branch_comparison(approximation_rows, decomposition_metadata)
    outcome, gates = classify(relative_rows, center_rows, approximation_rows, reconstruction_rows, tangent, branch)
    make_figures(phase_rows, approximation_rows, reconstruction_rows, communication)
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage 4D changed a prior scientific artifact")
    diagnostics: dict[str, object] = {
        "outcome": outcome,
        "gates": gates,
        "gauge_conventions": {
            "native": "exact stored column order and phases",
            "aligned_learned": "one deterministic alignment to LEARNED_0 before all analysis",
            "aligned_controls": "one deterministic alignment to OFDM, reported separately",
        },
        "center_diagnostics": center_diagnostics,
        "branch_tolerance_radians": BRANCH_TOLERANCE,
        "branch_analysis": branch,
        "decomposition_metadata": decomposition_metadata,
        "truncation_protocol": {
            "spectral_ranks": SPECTRAL_RANKS,
            "weyl_ranks": WEYL_RANKS,
            "entry_ranks": ENTRY_RANKS,
            "band_half_widths": BAND_WIDTHS,
            "communication_levels": COMMUNICATION_LEVELS,
            "selection": "fully predeclared structural ordering; no BER/Q/operator/receiver result selected a level",
        },
        "checkpoint_hashes": hashes,
        "q_samples": q_samples,
        "q_seed": Q_SEED,
        "operator_samples": operator_samples,
        "operator_seed": OPERATOR_SEED,
        "channels": channels_count,
        "channel_seed": CHANNEL_SEED,
        "symbol_seed": SYMBOL_SEED,
        "noise_seed": NOISE_SEED,
        "target_bits": target_bits,
        "snrs_db": SNRS,
        "reconstruction_count": len(reconstructions),
        "protected_artifact_count": len(protected_before),
        "protected_artifacts_unchanged": protected_after == protected_before,
        "stage2e_used": False,
        "waveform_training_used": False,
        "ber_used_for_selection": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage4d_diagnostics.json", diagnostics)
    write_report(outcome, gates, center_diagnostics, center_rows, relative_rows, approximation_rows, reconstruction_rows, q_rows, communication, tangent, tangent_rows, population_generators, diagnostics)
    print(f"STAGE 4D COMPLETE in {diagnostics['runtime_seconds']:.1f}s — OUTCOME {outcome}; STOP", flush=True)
    return diagnostics


def _read_typed_csv(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            if value == "True":
                row[key] = True
            elif value == "False":
                row[key] = False
            elif value != "":
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def regenerate_stage4d_report() -> dict[str, object]:
    """Regenerate derived structural tables/report from frozen production outputs."""
    protected_before = protected_hashes()
    waveforms, _ = load_frozen_population()
    aligned, alignment_metadata = align_population(waveforms)
    centers, center_diagnostics, center_rows = center_geometry(waveforms, aligned)
    relative_rows, phase_rows, _ = relative_analysis(waveforms, aligned, centers)
    tangent, tangent_rows, _ = tangent_geometry(centers["ALIGNED_CENTER"], aligned)
    population_generators = population_generator_comparison({**aligned, "OFDM": waveforms["OFDM"]}, centers["ALIGNED_CENTER"])
    relative = waveforms["OFDM"].mH @ centers["ALIGNED_CENTER"]
    approximation_rows, _, decomposition_metadata, _ = decompose_branches(relative, waveforms["OFDM"])
    reconstruction_rows = _read_typed_csv(OUT / "reconstruction_functional_metrics.csv")
    q_rows = _read_typed_csv(OUT / "waveform_q_generator_metrics.csv")
    communication = _read_typed_csv(OUT / "communication_summary.csv")
    branch = branch_comparison(approximation_rows, decomposition_metadata)
    outcome, gates = classify(relative_rows, center_rows, approximation_rows, reconstruction_rows, tangent, branch)
    diagnostics = json.loads((OUT / "stage4d_diagnostics.json").read_text())
    diagnostics.update({
        "outcome": outcome,
        "gates": gates,
        "branch_analysis": branch,
        "decomposition_metadata": decomposition_metadata,
        "center_diagnostics": center_diagnostics,
        "report_regenerated_from_frozen_production_outputs": True,
    })
    write_json(OUT / "gauge_alignments.json", alignment_metadata)
    write_csv(OUT / "center_distances.csv", center_rows)
    write_csv(OUT / "relative_unitary_metrics.csv", relative_rows)
    write_csv(OUT / "relative_eigenphases.csv", phase_rows)
    write_csv(OUT / "tangent_member_metrics.csv", tangent_rows)
    write_csv(OUT / "population_generator_comparison.csv", population_generators)
    write_csv(OUT / "generator_approximation_curves.csv", approximation_rows)
    write_json(OUT / "coefficient_spectra_and_branch_diagnostics.json", decomposition_metadata)
    write_json(OUT / "stage4d_diagnostics.json", diagnostics)
    make_figures(phase_rows, approximation_rows, reconstruction_rows, communication)
    write_report(outcome, gates, center_diagnostics, center_rows, relative_rows, approximation_rows, reconstruction_rows, q_rows, communication, tangent, tangent_rows, population_generators, diagnostics)
    if protected_hashes() != protected_before:
        raise RuntimeError("Stage 4D report regeneration changed a prior artifact")
    print(f"STAGE 4D REPORT REGENERATED — OUTCOME {outcome}; frozen production evaluations unchanged", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--q-samples", type=int, default=Q_SAMPLES)
    parser.add_argument("--operator-samples", type=int, default=OPERATOR_SAMPLES)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--target-bits", type=int, default=TARGET_BITS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.report_only:
        regenerate_stage4d_report()
    else:
        run_stage4d(
            q_samples=arguments.q_samples,
            operator_samples=arguments.operator_samples,
            channels_count=arguments.channels,
            target_bits=arguments.target_bits,
        )
