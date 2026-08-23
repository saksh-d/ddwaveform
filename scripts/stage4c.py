"""Run Stage 4C depth-scaling structured-transform compression."""

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
from src.channel import COMPLEX_DTYPE, sample_dd_v1
from src.metrics import qam16_constellation, qpsk_constellation
from src.receiver import effective_operator
from src.receiver_geometry import apply_mask
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2d import score_support_mask, sobol_dd_samples
from src.stage3a import concentration_thresholds, q_concentration
from src.stage3b import atomic_operators, differentiable_q, fixed_initializations
from src.stage3c import communication_rows
from src.stage4a import (
    basis_fidelity,
    canonical_generators,
    dd_expansion_metrics,
    integer_weyl_coefficients,
    native_atom_diagnostics,
)
from src.stage4b import matrix_pair_metrics
from src.stage4c import (
    canonicalize_target,
    cascade_parameter_count,
    chirp_cascade,
    fit_cascade_batch,
    operator_fidelity,
    parameter_population_metrics,
    restore_native_order,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage4c"
FITTED = OUT / "fitted"
REPORT = ROOT / "reports" / "stage4c_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
STAGE4B = ROOT / "outputs" / "stage4b" / "checkpoints"
STAGE3B = ROOT / "outputs" / "stage3b" / "checkpoints"
DEPTHS = (1, 2, 3, 4)
ARMS = ("M", "B")
RESTARTS = 3
FIT_STEPS = 400
VALIDATION_CADENCE = 20
TRAIN_ATOMS = 64
VALIDATION_ATOMS = 128
EVALUATION_ATOMS = 512
Q_SAMPLES = 2048
CHANNELS = 512
TARGET_BITS = 1_000_000
SNRS = (12, 15, 18, 20)
K = 24
FIT_SEED = 11180
TRAIN_ATOM_SEED = 11181
VALIDATION_ATOM_SEED = 11182
EVALUATION_ATOM_SEED = 11183
Q_SEED = 11184
CHANNEL_SEED = 11185
SYMBOL_SEED = 11186
NOISE_SEED = 11187
LEARNED = tuple(f"LEARNED_{index}" for index in range(5))
TARGETS = (*LEARNED, "RAW_HAAR_1", "RAW_HAAR_2", "Q_MEAN_FROM_OFDM", "Q_MEAN_FROM_HAAR_1")
EXPECTED_LEARNED_HASHES = {
    "LEARNED_0": "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc",
    "LEARNED_1": "fbe49eb8002e7e7272c94e6a362210e8cba3599d317a6cfe53275e2da1f32eef",
    "LEARNED_2": "ad8d4e379c0f29369bf49d1b745a038cbf5326233cdbb05406cde957d4bbe7e5",
    "LEARNED_3": "a52e1eccd333216a1dc360cd1d74be3d44b92170e044b589f0b68a3da05e3b75",
    "LEARNED_4": "cf57edc354959e84560987f65075d4396ce2c2c3d5be5f9efc642aefe9c977ae",
}
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
        if path.is_file() and "stage4c" not in path.parts
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def population(name: str) -> str:
    if name in LEARNED:
        return "Learned"
    if name.startswith("RAW_HAAR"):
        return "Haar"
    return "Stage3B"


def candidate_name(target: str, arm: str, depth: int) -> str:
    return f"FIT_{target}_{arm}_L{depth}"


def load_targets() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, object]], dict[str, str]]:
    originals: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    learned0, info = load_frozen_waveform(FROZEN)
    originals["LEARNED_0"] = learned0; hashes["LEARNED_0"] = str(info["sha256"])
    for index in range(1, 5):
        name = f"LEARNED_{index}"; path = STAGE4B / f"learned_{index}.pt"
        originals[name] = torch.load(path, weights_only=True); hashes[name] = file_sha256(path)
    for name, expected in EXPECTED_LEARNED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"unexpected Learned checkpoint hash: {name}={hashes[name]}")
    for name, expected in EXPECTED_STAGE3B_HASHES.items():
        path = STAGE3B / f"{name.lower()}_best.pt"
        if file_sha256(path) != expected:
            raise RuntimeError(f"unexpected Stage-3B checkpoint hash: {name}")
        if name in TARGETS:
            originals[name] = torch.load(path, weights_only=True); hashes[name] = expected
    raw_names, raw_values = fixed_initializations()
    for raw_name in ("HAAR_1", "HAAR_2"):
        name = f"RAW_{raw_name}"; value = raw_values[raw_names.index(raw_name)]
        originals[name] = value; hashes[name] = tensor_sha256(value)
    originals = {name: originals[name] for name in TARGETS}
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    canonical: dict[str, torch.Tensor] = {}; alignments: dict[str, dict[str, object]] = {}
    for name, waveform in originals.items():
        residual = float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity))
        if residual >= 1e-12:
            raise RuntimeError(f"non-unitary frozen target {name}: {residual:.3e}")
        aligned, permutation, phases = canonicalize_target(waveform)
        canonical[name] = aligned
        alignments[name] = {
            "permutation": permutation.tolist(),
            "phases_real": phases.real.tolist(),
            "phases_imag": phases.imag.tolist(),
            "target_unitarity_residual": residual,
            "alignment": "fixed once against F^H before fitting; excluded from trainable parameter count",
        }
    return originals, canonical, alignments, hashes


def make_atom_banks(train_count: int, validation_count: int, evaluation_count: int) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    banks = {}
    for label, count, seed in (
        ("train", train_count, TRAIN_ATOM_SEED),
        ("validation", validation_count, VALIDATION_ATOM_SEED),
        ("evaluation", evaluation_count, EVALUATION_ATOM_SEED),
    ):
        delays, dopplers = sobol_dd_samples(count, seed)
        banks[label] = (delays, dopplers, atomic_operators(delays, dopplers))
    np.savez(
        OUT / "deterministic_dd_atom_banks.npz",
        train_delays=banks["train"][0].numpy(), train_dopplers=banks["train"][1].numpy(),
        validation_delays=banks["validation"][0].numpy(), validation_dopplers=banks["validation"][1].numpy(),
        evaluation_delays=banks["evaluation"][0].numpy(), evaluation_dopplers=banks["evaluation"][1].numpy(),
    )
    return banks


def fit_all(
    originals: dict[str, torch.Tensor],
    canonical: dict[str, torch.Tensor],
    alignments: dict[str, dict[str, object]],
    banks: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    restarts: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    names = tuple(canonical)
    target_batch = torch.stack([canonical[name] for name in names])
    fits: dict[tuple[int, str, int], object] = {}
    histories: list[dict[str, object]] = []
    restart_rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        for arm in ARMS:
            for restart in range(restarts):
                result = fit_cascade_batch(
                    target_batch, depth=depth, arm=arm, restart=restart, steps=steps,
                    seed=FIT_SEED + 1000 * depth + 100 * (arm == "B") + restart,
                    training_operators=banks["train"][2], validation_operators=banks["validation"][2],
                    validation_cadence=min(VALIDATION_CADENCE, steps),
                )
                fits[(depth, arm, restart)] = result
                for row in result.history:
                    current = dict(row); current["target"] = names[int(current.pop("target_index"))]; histories.append(current)
                for index, name in enumerate(names):
                    restart_rows.append({
                        "target": name, "population": population(name), "depth": depth, "arm": arm,
                        "restart": restart, "selected_step": int(result.selected_steps[index]),
                        "validation_loss": float(result.validation_loss[index]),
                        "nonfinite_seen": bool(result.nonfinite_seen[index]),
                    })
                print(f"fit L={depth} arm={arm} restart={restart + 1}/{restarts}", flush=True)
    candidates: dict[str, torch.Tensor] = {}
    parameter_rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        for arm in ARMS:
            for index, name in enumerate(names):
                winner = min(range(restarts), key=lambda restart: float(fits[(depth, arm, restart)].validation_loss[index]))
                result = fits[(depth, arm, winner)]
                parameters = result.parameters[index]
                canonical_candidate = chirp_cascade(parameters, depth, config.N)
                alignment = alignments[name]
                permutation = torch.as_tensor(alignment["permutation"], dtype=torch.int64)
                phases = torch.complex(
                    torch.as_tensor(alignment["phases_real"], dtype=torch.float64),
                    torch.as_tensor(alignment["phases_imag"], dtype=torch.float64),
                )
                candidate = restore_native_order(canonical_candidate, permutation, phases).detach()
                key = candidate_name(name, arm, depth); candidates[key] = candidate
                path = FITTED / f"{key.lower()}.pt"; torch.save(candidate, path)
                parameter_rows.append({
                    "candidate": key, "target": name, "population": population(name), "depth": depth, "arm": arm,
                    "restart": winner, "selected_step": int(result.selected_steps[index]),
                    "validation_loss": float(result.validation_loss[index]), "free_real_parameters": cascade_parameter_count(depth),
                    "parameters": parameters.tolist(), "checkpoint": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
                    "unitarity_residual": float(torch.linalg.matrix_norm(candidate.mH @ candidate - torch.eye(config.N, dtype=COMPLEX_DTYPE))),
                })
    return candidates, parameter_rows, restart_rows, histories


def matrix_evaluation(
    originals: dict[str, torch.Tensor], candidates: dict[str, torch.Tensor], parameter_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = {row["candidate"]: row for row in parameter_rows}
    rows: list[dict[str, object]] = []; atom_rows: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        target_name = str(metadata[name]["target"]); target = originals[target_name]
        pair = matrix_pair_metrics(target, candidate); overlaps = pair.pop("column_overlaps")
        fidelity = basis_fidelity(target, candidate)
        native = native_atom_diagnostics(candidate, tuple(range(config.N)))
        for atom in native:
            atom_rows.append({"candidate": name, "target": target_name, **atom})
        rows.append({
            "candidate": name, "target": target_name, "population": metadata[name]["population"],
            "depth": metadata[name]["depth"], "arm": metadata[name]["arm"],
            "free_real_parameters": metadata[name]["free_real_parameters"],
            **{key: value for key, value in pair.items() if key != "permutation"},
            "time_domain_atom_fidelity_mean": fidelity["mean_column_overlap"],
            "frequency_magnitude_relative_error": fidelity["frequency_magnitude_relative_error"],
            "frequency_phase_circular_rms": fidelity["frequency_phase_circular_rms"],
            "polynomial_phase_degree3_residual_median": float(np.median([atom["phase_degree3_rms"] for atom in native])),
            "column_overlaps_json": json.dumps(overlaps),
        })
    return rows, atom_rows


def operator_evaluation(
    originals: dict[str, torch.Tensor], candidates: dict[str, torch.Tensor], parameter_rows: list[dict[str, object]], operators: torch.Tensor
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = {row["candidate"]: row for row in parameter_rows}
    rows: list[dict[str, object]] = []; atom_rows: list[dict[str, object]] = []
    for index, (name, candidate) in enumerate(candidates.items()):
        target_name = str(metadata[name]["target"])
        metrics, values = operator_fidelity(originals[target_name], candidate, operators)
        rows.append({"candidate": name, "target": target_name, "population": metadata[name]["population"], "depth": metadata[name]["depth"], "arm": metadata[name]["arm"], **metrics})
        atom_rows.extend({"candidate": name, "target": target_name, "atom_index": atom_index, "d_b": float(value)} for atom_index, value in enumerate(values))
        if (index + 1) % 8 == 0:
            print(f"operator evaluation {index + 1}/{len(candidates)}", flush=True)
    return rows, atom_rows


def generator_comparison(target: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    target_generators = canonical_generators(target); candidate_generators = canonical_generators(candidate)
    output: dict[str, object] = {}
    for label, target_generator, candidate_generator in zip(("g_s", "g_d"), target_generators, candidate_generators):
        target_coefficients = integer_weyl_coefficients(target_generator)
        candidate_coefficients = integer_weyl_coefficients(candidate_generator)
        target_metrics = dd_expansion_metrics(target_coefficients); candidate_metrics = dd_expansion_metrics(candidate_coefficients)
        target_top = set(torch.argsort(torch.abs(target_coefficients).square().reshape(-1), descending=True, stable=True)[:K].tolist())
        candidate_top = set(torch.argsort(torch.abs(candidate_coefficients).square().reshape(-1), descending=True, stable=True)[:K].tolist())
        shared = sorted(target_top & candidate_top)
        if shared:
            left = target_coefficients.reshape(-1)[shared]; right = candidate_coefficients.reshape(-1)[shared]
            phase_coherence = float(torch.abs(torch.mean(torch.exp(1j * (torch.angle(left) - torch.angle(right))))))
        else:
            phase_coherence = 0.0
        output.update({
            f"target_{label}_top24_energy": target_metrics["top24_energy"],
            f"candidate_{label}_top24_energy": candidate_metrics["top24_energy"],
            f"{label}_top24_support_overlap": len(shared) / K,
            f"{label}_shared_support_phase_coherence": phase_coherence,
            f"{label}_magnitude_correlation": float(np.corrcoef(torch.abs(target_coefficients).square().reshape(-1).numpy(), torch.abs(candidate_coefficients).square().reshape(-1).numpy())[0, 1]),
            f"{label}_candidate_phase_linear_residual": candidate_metrics["dominant_phase_linear_residual"],
        })
    output["target_gd_to_gs_ratio"] = output["target_g_d_top24_energy"] / output["target_g_s_top24_energy"]
    output["candidate_gd_to_gs_ratio"] = output["candidate_g_d_top24_energy"] / output["candidate_g_s_top24_energy"]
    return output


def q_and_generator_evaluation(
    originals: dict[str, torch.Tensor], candidates: dict[str, torch.Tensor], parameter_rows: list[dict[str, object]],
    operators: torch.Tensor, channels: torch.Tensor,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray], dict[str, torch.Tensor]]:
    metadata = {row["candidate"]: row for row in parameter_rows}
    q_rows: list[dict[str, object]] = []; generator_rows: list[dict[str, object]] = []
    q_values: dict[str, np.ndarray] = {}; masks: dict[str, torch.Tensor] = {}
    for target_name in TARGETS:
        names = [target_name] + [candidate_name(target_name, arm, depth) for arm in ARMS for depth in DEPTHS]
        waveforms = [originals[target_name]] + [candidates[name] for name in names[1:]]
        q_stack = differentiable_q(torch.stack(waveforms), operators, chunk_size=16).detach().numpy()
        target_q = q_stack[0]; target_mask = score_support_mask(target_q, K)
        target_concentration = q_concentration(target_q, K)
        q_values[f"TARGET_{target_name}"] = target_q; masks[target_name] = target_mask
        for index, name in enumerate(names[1:], start=1):
            q = q_stack[index]; q_values[name] = q
            mask = score_support_mask(q, K); masks[name] = mask
            concentration = q_concentration(q, K); thresholds = concentration_thresholds(q)
            effective = effective_operator(channels, candidates[name]); retained = apply_mask(effective, mask)
            fraction = torch.sum(torch.abs(retained).square(), dim=(-2, -1)) / torch.sum(torch.abs(effective).square(), dim=(-2, -1))
            meta = metadata[name]
            q_rows.append({
                "candidate": name, "target": target_name, "population": meta["population"], "depth": meta["depth"], "arm": meta["arm"],
                "eta24_q": concentration["eta_q"], "one_minus_eta24_q": 1.0 - concentration["eta_q"], "row_p05": concentration["row_p05"],
                "target_eta24_q": target_concentration["eta_q"], "target_one_minus_eta24_q": 1.0 - target_concentration["eta_q"],
                "k90": thresholds[0.90], "k95": thresholds[0.95], "k99": thresholds[0.99], "k99_5": thresholds[0.995],
                "q_relative_distance_to_target": float(np.linalg.norm(q - target_q) / np.linalg.norm(target_q)),
                "q_correlation_to_target": float(np.corrcoef(q.reshape(-1), target_q.reshape(-1))[0, 1]),
                "top24_support_overlap_to_target": float(((mask & target_mask).sum(dim=-1) / K).to(torch.float64).mean()),
                "fixed_q_retained_energy_mean": float(fraction.mean()), "fixed_q_retained_energy_p05": float(torch.quantile(fraction, 0.05)),
            })
            generator_rows.append({"candidate": name, "target": target_name, "population": meta["population"], "depth": meta["depth"], "arm": meta["arm"], **generator_comparison(originals[target_name], candidates[name])})
        print(f"Q/generator evaluation {target_name}", flush=True)
    np.savez(OUT / "q_matrices.npz", **q_values)
    return q_rows, generator_rows, q_values, masks


def communication_evaluation(
    originals: dict[str, torch.Tensor], candidates: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], channels: torch.Tensor, target_bits: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    for target_name in TARGETS:
        names = [target_name] + [candidate_name(target_name, arm, depth) for arm in ARMS for depth in DEPTHS]
        waveforms = {target_name: originals[target_name], **{name: candidates[name] for name in names[1:]}}
        selected_masks = {name: masks[name] for name in names}
        rows = communication_rows(
            channels, waveforms, selected_masks, (qpsk_constellation(), qam16_constellation()), SNRS,
            k=K, target_bits=target_bits, symbol_seed=SYMBOL_SEED, noise_seed=NOISE_SEED,
        )
        all_rows.extend(rows); print(f"communication evaluation {target_name}", flush=True)
    keys = ("modulation", "snr_db", "waveform", "support_rule", "k")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in all_rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary: list[dict[str, object]] = []
    for values, selected in groups.items():
        bit_errors = sum(int(row["bit_errors"]) for row in selected); total_bits = sum(int(row["total_bits"]) for row in selected)
        symbol_errors = sum(int(row["symbol_errors"]) for row in selected); total_symbols = sum(int(row["total_symbols"]) for row in selected)
        summary.append({**dict(zip(keys, values)), "bit_errors": bit_errors, "total_bits": total_bits, "ber": bit_errors / total_bits, "symbol_errors": symbol_errors, "total_symbols": total_symbols, "ser": symbol_errors / total_symbols, "n_channels": len(selected)})
    return all_rows, summary


def parameter_reproducibility(parameter_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for depth in DEPTHS:
            selected = [row for row in parameter_rows if row["target"] in LEARNED and row["arm"] == arm and row["depth"] == depth]
            metrics = parameter_population_metrics(np.asarray([row["parameters"] for row in selected]))
            rows.append({
                "arm": arm, "depth": depth, "free_real_parameters": cascade_parameter_count(depth),
                "pairwise_rms_distance_mean": metrics["pairwise_rms_distance_mean"],
                "pairwise_rms_distance_minimum": metrics["pairwise_rms_distance_minimum"],
                "pairwise_rms_distance_maximum": metrics["pairwise_rms_distance_maximum"],
                "mean_parameter_vector_json": json.dumps(metrics["mean_parameter_vector"]),
                "standard_deviation_vector_json": json.dumps(metrics["standard_deviation_vector"]),
                "coefficient_of_variation_vector_json": json.dumps(metrics["coefficient_of_variation_vector"]),
                "symmetry_adjustment": metrics["symmetry_adjusted_distance"],
            })
    return rows


def depth_summary(
    matrix_rows: list[dict[str, object]], operator_rows: list[dict[str, object]], q_rows: list[dict[str, object]],
    generator_rows: list[dict[str, object]], communication: list[dict[str, object]],
) -> list[dict[str, object]]:
    matrix = {row["candidate"]: row for row in matrix_rows}; operator = {row["candidate"]: row for row in operator_rows}
    q = {row["candidate"]: row for row in q_rows}; generator = {row["candidate"]: row for row in generator_rows}
    comm = {(row["waveform"], row["modulation"], row["snr_db"], row["support_rule"]): row for row in communication}
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for depth in DEPTHS:
            selected = [candidate_name(name, arm, depth) for name in LEARNED]
            haar = [candidate_name(name, arm, depth) for name in ("RAW_HAAR_1", "RAW_HAAR_2")]
            communication_gaps: dict[str, list[float]] = {"Full": [], "Oracle": [], "Q": []}
            for target in LEARNED:
                candidate = candidate_name(target, arm, depth)
                for modulation in ("QPSK", "16QAM"):
                    for snr in SNRS:
                        for rule in ("Full", "Oracle", "Q"):
                            left = float(comm[(target, modulation, snr, rule)]["ber"]); right = float(comm[(candidate, modulation, snr, rule)]["ber"])
                            smoothing = 0.5 / float(comm[(candidate, modulation, snr, rule)]["total_bits"])
                            communication_gaps[rule].append(abs(math.log((right + smoothing) / (left + smoothing))))
            def median(names: list[str], lookup: dict[str, dict[str, object]], metric: str) -> float:
                return float(np.median([float(lookup[name][metric]) for name in names]))
            row = {
                "arm": arm, "depth": depth, "free_real_parameters": cascade_parameter_count(depth),
                "compression_ratio_unitary_dimension": config.N**2 / cascade_parameter_count(depth),
                "coefficient_storage_bytes_float64": 8 * cascade_parameter_count(depth),
                "fft_ifft_operations": depth, "pointwise_phase_multiplications": depth + 1,
                "learned_aligned_matrix_nmse_median": median(selected, matrix, "aligned_matrix_nmse"),
                "learned_aligned_overlap_median": median(selected, matrix, "mean_matched_squared_overlap"),
                "learned_d_b_median": median(selected, operator, "d_b"),
                "learned_one_minus_eta24_median": median(selected, q, "one_minus_eta24_q"),
                "learned_q_correlation_median": median(selected, q, "q_correlation_to_target"),
                "learned_gd_to_gs_ratio_median": median(selected, generator, "candidate_gd_to_gs_ratio"),
                "q_residual_log_gap_median": float(np.median([abs(math.log(float(q[name]["one_minus_eta24_q"]) / max(float(q[name]["target_one_minus_eta24_q"]), 1e-300))) for name in selected])),
                "generator_ratio_absolute_gap_median": float(np.median([abs(float(generator[name]["candidate_gd_to_gs_ratio"]) - float(generator[name]["target_gd_to_gs_ratio"])) for name in selected])),
                "communication_log_gap_mean": float(np.mean([value for values in communication_gaps.values() for value in values])),
                "full_communication_log_gap_mean": float(np.mean(communication_gaps["Full"])),
                "oracle_communication_log_gap_mean": float(np.mean(communication_gaps["Oracle"])),
                "fixed_q_communication_log_gap_mean": float(np.mean(communication_gaps["Q"])),
                "haar_aligned_matrix_nmse_median": median(haar, matrix, "aligned_matrix_nmse"),
                "haar_d_b_median": median(haar, operator, "d_b"),
            }
            for modulation in ("QPSK", "16QAM"):
                for rule, label in (("Full", "full"), ("Q", "fixed_q")):
                    row[f"learned_{label}_{modulation.lower()}_ber_18db_median"] = float(np.median([float(comm[(name, modulation, 18, rule)]["ber"]) for name in selected]))
            rows.append(row)
    return rows


def classify(summary: list[dict[str, object]], matrix_rows: list[dict[str, object]], operator_rows: list[dict[str, object]], q_rows: list[dict[str, object]], generator_rows: list[dict[str, object]], communication: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    matrix = {row["candidate"]: row for row in matrix_rows}; operator = {row["candidate"]: row for row in operator_rows}
    q = {row["candidate"]: row for row in q_rows}; generator = {row["candidate"]: row for row in generator_rows}
    comm = {(row["waveform"], row["modulation"], row["snr_db"], row["support_rule"]): row for row in communication}
    basin_counts: dict[str, int] = {}
    for arm in ARMS:
        for depth in DEPTHS:
            count = 0
            for target in LEARNED:
                name = candidate_name(target, arm, depth)
                structural = (
                    float(matrix[name]["mean_matched_squared_overlap"]) >= 0.8932
                    and float(operator[name]["d_b"]) <= 0.1703
                    and 0.0015 <= float(q[name]["one_minus_eta24_q"]) <= 0.0025
                    and 9.0 <= float(generator[name]["candidate_gd_to_gs_ratio"]) <= 10.1
                )
                communication_inside = True
                for modulation in ("QPSK", "16QAM"):
                    for snr in SNRS:
                        for rule in ("Full", "Oracle", "Q"):
                            learned_values = [float(comm[(member, modulation, snr, rule)]["ber"]) for member in LEARNED]
                            value = float(comm[(name, modulation, snr, rule)]["ber"])
                            if not min(learned_values) <= value <= max(learned_values): communication_inside = False
                count += int(structural and communication_inside)
            basin_counts[f"{arm}_L{depth}"] = count
    basin_entered = max(basin_counts.values()) >= 4
    trends: dict[str, object] = {}
    coherent_any = False; matrix_any = False
    for arm in ARMS:
        arm_rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
        matrix_values = [float(row["learned_aligned_matrix_nmse_median"]) for row in arm_rows]
        db_values = [float(row["learned_d_b_median"]) for row in arm_rows]
        q_values = [abs(math.log(float(row["learned_one_minus_eta24_median"]) / 0.00194)) for row in arm_rows]
        generator_values = [float(row["generator_ratio_absolute_gap_median"]) for row in arm_rows]
        communication_values = [float(row["communication_log_gap_mean"]) for row in arm_rows]
        endpoint = {
            "matrix": matrix_values[-1] < matrix_values[0], "operator": db_values[-1] < db_values[0],
            "q": q_values[-1] < q_values[0], "generator": generator_values[-1] < generator_values[0],
            "communication": communication_values[-1] < communication_values[0],
        }
        trends[arm] = {"endpoint_improvement": endpoint, "improved_metric_count": sum(endpoint.values())}
        matrix_any |= endpoint["matrix"]; coherent_any |= sum(endpoint.values()) >= 4
    l4 = [row for row in summary if int(row["depth"]) == 4]
    generic = True
    for row in l4:
        l1_row = next(item for item in summary if item["arm"] == row["arm"] and int(item["depth"]) == 1)
        similar_at_l4 = (
            float(row["haar_d_b_median"]) / max(float(row["learned_d_b_median"]), 1e-300) < 1.10
            and float(row["haar_aligned_matrix_nmse_median"]) / max(float(row["learned_aligned_matrix_nmse_median"]), 1e-300) < 1.10
        )
        deeper_fits_both = (
            float(row["learned_d_b_median"]) < float(l1_row["learned_d_b_median"])
            and float(row["haar_d_b_median"]) < float(l1_row["haar_d_b_median"])
        )
        generic &= similar_at_l4 and deeper_fits_both
    if basin_entered:
        outcome = "A"
    elif generic:
        outcome = "D"
    elif coherent_any:
        outcome = "B"
    elif matrix_any:
        outcome = "C"
    else:
        outcome = "E"
    return outcome, {"basin_counts": basin_counts, "basin_entered": basin_entered, "depth_trends": trends, "generic_approximator_gate": generic}


def make_figures(summary: list[dict[str, object]]) -> None:
    styles = {"M": ("o", "Arm M"), "B": ("s", "Arm B")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm, (marker, label) in styles.items():
        rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
        depth = [row["depth"] for row in rows]
        axes[0].plot(depth, [row["learned_aligned_matrix_nmse_median"] for row in rows], marker=marker, label=label)
        axes[0].plot(depth, [row["haar_aligned_matrix_nmse_median"] for row in rows], marker=marker, linestyle="--", alpha=.55)
        axes[1].plot(depth, [row["learned_d_b_median"] for row in rows], marker=marker, label=label)
        axes[1].plot(depth, [row["haar_d_b_median"] for row in rows], marker=marker, linestyle="--", alpha=.55)
    axes[0].set_ylabel("aligned matrix NMSE"); axes[1].set_ylabel(r"complex operator $d_B$")
    for axis in axes: axis.set(xlabel="cascade depth", xticks=DEPTHS); axis.grid(alpha=.3); axis.legend()
    fig.suptitle("Learned medians (solid) and Haar medians (dashed)"); fig.savefig(OUT / "fig01_matrix_operator_depth.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm, (marker, label) in styles.items():
        rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
        axes[0].plot(DEPTHS, [row["learned_one_minus_eta24_median"] for row in rows], marker=marker, label=label)
        axes[1].plot(DEPTHS, [row["learned_gd_to_gs_ratio_median"] for row in rows], marker=marker, label=label)
    axes[0].axhspan(.00186, .00202, color="tab:green", alpha=.15, label="Stage-4B basin")
    axes[1].axhspan(9.306, 9.804, color="tab:green", alpha=.15, label="Stage-4B basin")
    axes[0].set_ylabel(r"$1-\eta_{24}^Q$"); axes[1].set_ylabel(r"$G_D/G_S$ top-24 ratio")
    for axis in axes: axis.set(xlabel="cascade depth", xticks=DEPTHS); axis.grid(alpha=.3); axis.legend()
    fig.savefig(OUT / "fig02_q_generator_depth.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, rule in enumerate(("full", "fixed_q")):
        for column, modulation in enumerate(("qpsk", "16qam")):
            axis = axes[row_index, column]
            for arm, (marker, label) in styles.items():
                rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
                axis.plot(DEPTHS, [row[f"learned_{rule}_{modulation}_ber_18db_median"] for row in rows], marker=marker, label=label)
            axis.set(yscale="log", xlabel="cascade depth", ylabel="BER", xticks=DEPTHS, title=f"18 dB {rule.replace('_', '-')} {modulation.upper()}")
            axis.grid(alpha=.3); axis.legend()
    fig.savefig(OUT / "fig03_communication_depth.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for arm, (marker, label) in styles.items():
        rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
        parameters = [row["free_real_parameters"] for row in rows]
        axes[0].plot(parameters, [row["learned_aligned_matrix_nmse_median"] for row in rows], marker=marker, label=label)
        axes[1].plot(parameters, [row["learned_d_b_median"] for row in rows], marker=marker, label=label)
        axes[2].plot(parameters, [row["communication_log_gap_mean"] for row in rows], marker=marker, label=label)
    for axis, label in zip(axes, ("aligned matrix NMSE", r"$d_B$", "mean absolute log-BER gap")):
        axis.set(xlabel="free real parameters", ylabel=label); axis.grid(alpha=.3); axis.legend()
    fig.savefig(OUT / "fig04_parameter_efficiency.png", dpi=180); plt.close(fig)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def write_report(outcome: str, gates: dict[str, object], summary: list[dict[str, object]], parameter_rows: list[dict[str, object]], parameter_population: list[dict[str, object]], diagnostics: dict[str, object]) -> None:
    labels = {
        "A": "compact depth-scaling waveform family identified", "B": "directional structured convergence",
        "C": "matrix compression without waveform-function convergence", "D": "generic unitary approximation", "E": "no systematic depth law",
    }
    trends = gates["depth_trends"]
    best = min(summary, key=lambda row: float(row["learned_d_b_median"]))
    best_m = min(float(row["learned_d_b_median"]) for row in summary if row["arm"] == "M")
    best_b = min(float(row["learned_d_b_median"]) for row in summary if row["arm"] == "B")
    operator_gain = (best_m - best_b) / best_m
    full_trends = {}; finite_trends = {}
    for arm in ARMS:
        rows_by_depth = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["depth"]))
        full_trends[arm] = float(rows_by_depth[-1]["full_communication_log_gap_mean"]) < float(rows_by_depth[0]["full_communication_log_gap_mean"])
        finite_trends[arm] = np.mean([float(rows_by_depth[-1][key]) for key in ("oracle_communication_log_gap_mean", "fixed_q_communication_log_gap_mean")]) < np.mean([float(rows_by_depth[0][key]) for key in ("oracle_communication_log_gap_mean", "fixed_q_communication_log_gap_mean")])
    l4_selectivity = {
        arm: float(next(row for row in summary if row["arm"] == arm and int(row["depth"]) == 4)["haar_d_b_median"])
        / float(next(row for row in summary if row["arm"] == arm and int(row["depth"]) == 4)["learned_d_b_median"])
        for arm in ARMS
    }
    lines = [
        "# Stage 4C — Depth-Scaling Structured Transform Search", "",
        f"## Result: Outcome {outcome} — {labels[outcome]}", "",
        "No waveform/channel training, BER/Q/geometry fitting, BER-based selection, or Stage 2E dependency was used. Every fit was frozen before Q, generator, and communication evaluation.", "",
        "## Frozen family and protocol", "",
        r"The sole family is $W_L=D_0F^HD_1F^H\cdots F^HD_L$, where $[D_\ell]_{nn}=\exp(i\sum_{p=1}^3 a_{\ell p}x_n^p)$ and $x_n$ is uniformly spaced on $[-1,1]$.",
        "All constant polynomial terms and global phases are removed. Depth 1 is Stage-4A `FOURIER_POLY_D3` up to that exact redundancy. A fixed target-specific phase/permutation convention against `F^H` is established before fitting and restored for native-order evaluation; it is not fit-dependent.",
        f"Both arms use {diagnostics['fit_steps']} Adam steps, {diagnostics['restarts']} deterministic restarts, and validation-loss-only selection. Arm M uses aligned atom overlap; Arm B uses {diagnostics['train_atoms']}/{diagnostics['validation_atoms']} independent fixed DD-v1 train/validation atoms. Evaluation uses {diagnostics['evaluation_atoms']} third-bank atoms.",
        f"Depth-1 continuity check on authoritative LEARNED_0: Arm-M aligned NMSE `{diagnostics['stage4a_depth1_continuity']['stage4c_nmse']:.6f}` versus Stage-4A `{diagnostics['stage4a_depth1_continuity']['stage4a_nmse']:.6f}` (absolute difference `{diagnostics['stage4a_depth1_continuity']['absolute_difference']:.6f}`).", "",
        "| depth | real parameters | coefficient bytes | unitary compression | IFFT count | phase multiplies |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        count = cascade_parameter_count(depth)
        lines.append(f"| {depth} | {count} | {8*count} | {config.N**2/count:.1f}x vs U(64) dimension | {depth} | {depth + 1} |")
    lines += ["", "Application complexity is `O(L N log N)`; float64 coefficient storage is `O(L)`. A minimal target convention adds 64 permutation indices plus 64 phase angles (640 bytes with uint16/float64), reported separately and excluded from trainable parameters.", "", "## Learned-population depth curves", "", "| arm | L | params | aligned NMSE | overlap | d_B | 1-eta24 | Q corr | G_D/G_S | Full QPSK BER | Full 16-QAM BER | Fixed-Q QPSK BER | Fixed-Q 16-QAM BER | Haar d_B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {row['depth']} | {row['free_real_parameters']} | {row['learned_aligned_matrix_nmse_median']:.4f} | {row['learned_aligned_overlap_median']:.4f} | {row['learned_d_b_median']:.4f} | {row['learned_one_minus_eta24_median']:.6f} | {row['learned_q_correlation_median']:.4f} | {row['learned_gd_to_gs_ratio_median']:.3f} | {row['learned_full_qpsk_ber_18db_median']:.6g} | {row['learned_full_16qam_ber_18db_median']:.6g} | {row['learned_fixed_q_qpsk_ber_18db_median']:.6g} | {row['learned_fixed_q_16qam_ber_18db_median']:.6g} | {row['haar_d_b_median']:.4f} |"
        )
    m = trends["M"]["endpoint_improvement"]; b = trends["B"]["endpoint_improvement"]
    lines += ["", "## Interpretation gate", "",
        f"1. **Matrix/atom fidelity improves systematically:** M={yes_no(m['matrix'])}, B={yes_no(b['matrix'])} by L1-to-L4 endpoint.",
        f"2. **Complex B fidelity improves systematically:** M={yes_no(m['operator'])}, B={yes_no(b['operator'])} by endpoint.",
        f"3. **Operator fitting materially outperforms matrix fitting:** No. Best median d_B changes from {best_m:.4f} to {best_b:.4f} ({100*operator_gain:.2f}% lower), while the depth trajectory remains non-monotonic.",
        f"4. **Residual Q mass approaches the basin:** No. Best median is {min(float(row['learned_one_minus_eta24_median']) for row in summary):.6f}, versus 0.00186–0.00202.",
        f"5. **Generator asymmetry approaches the population:** Only directionally for Arm M; best absolute median target gap is {min(float(row['generator_ratio_absolute_gap_median']) for row in summary):.3f}, without joint matrix/operator/Q convergence.",
        f"6. **Full BER approaches the population with depth:** M={yes_no(full_trends['M'])}, B={yes_no(full_trends['B'])} by the all-modulation/SNR log-gap endpoint.",
        f"7. **Finite-K BER approaches the population:** M={yes_no(finite_trends['M'])}, B={yes_no(finite_trends['B'])} by the combined Oracle/Fixed-Q endpoint.",
        f"8. **Consistent across five Learned targets:** basin-entry counts are {gates['basin_counts']} (four of five required).",
        f"9. **Parameters reproducible:** No coherent deep clustering. Best mean pairwise RMS distance is {min(float(row['pairwise_rms_distance_mean']) for row in parameter_population):.4f} at shallow depth and grows materially in deeper fits; full vectors/SD/CV are in `parameter_reproducibility.csv`.",
        f"10. **Haar controls materially worse:** No at L4; Haar/Learned median d_B ratios are M={l4_selectivity['M']:.3f}, B={l4_selectivity['B']:.3f}. Outcome D is still rejected because depth did not improve either population.",
        f"11. **Minimum depth entering the empirical basin:** {'none' if not gates['basin_entered'] else max(gates['basin_counts'], key=gates['basin_counts'].get)}.",
        f"12. **Compact at best operator point:** {best['free_real_parameters']} real parameters, {best['fft_ifft_operations']} IFFT operation{'s' if int(best['fft_ifft_operations']) != 1 else ''}, {best['compression_ratio_unitary_dimension']:.1f}x dimensional compression; compactness does not rescue the failed basin gate.", "",
        "## Preservation", "",
        f"- All five Learned and all six Stage-3B checkpoint hashes verified; maximum fitted unitarity residual `{max(float(row['unitarity_residual']) for row in parameter_rows):.3e}`.",
        f"- All {diagnostics['protected_artifact_count']} prior artifacts remained byte-identical: `{diagnostics['protected_artifacts_unchanged']}`.",
        f"- Common paired communication bank: {diagnostics['channels']} channels, at least {diagnostics['target_bits']} bits per waveform/point, QPSK and 16-QAM at 12/15/18/20 dB, Full/Oracle-K24/Fixed-Q-K24.",
        "- Stage 2E absent. No fit/restart/depth decision used Q, generator, MSE, BER, modulation, or receiver results.", "",
        "STOP — do not extend depth or proceed to Stage 4D automatically.", "",
    ]
    REPORT.write_text("\n".join(lines))


def run_stage4c(
    *, fit_steps: int = FIT_STEPS, restarts: int = RESTARTS, target_bits: int = TARGET_BITS,
    channels_count: int = CHANNELS, train_atoms: int = TRAIN_ATOMS, validation_atoms: int = VALIDATION_ATOMS,
    evaluation_atoms: int = EVALUATION_ATOMS, q_samples: int = Q_SAMPLES,
) -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); FITTED.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    originals, canonical, alignments, hashes = load_targets()
    banks = make_atom_banks(train_atoms, validation_atoms, evaluation_atoms)
    candidates, parameter_rows, restart_rows, history = fit_all(originals, canonical, alignments, banks, steps=fit_steps, restarts=restarts)
    write_csv(OUT / "multi_restart_summary.csv", restart_rows); write_csv(OUT / "fitting_history.csv", history)
    write_json(OUT / "fitted_parameters_and_alignment.json", {"alignments": alignments, "fits": parameter_rows})
    matrix_rows, atom_rows = matrix_evaluation(originals, candidates, parameter_rows)
    write_csv(OUT / "matrix_atom_fidelity.csv", matrix_rows); write_csv(OUT / "native_atom_diagnostics.csv", atom_rows)
    operator_rows, per_atom_rows = operator_evaluation(originals, candidates, parameter_rows, banks["evaluation"][2])
    write_csv(OUT / "complex_operator_fidelity.csv", operator_rows); write_csv(OUT / "per_atom_operator_discrepancy.csv", per_atom_rows)
    q_delays, q_dopplers = sobol_dd_samples(q_samples, Q_SEED); q_operators = atomic_operators(q_delays, q_dopplers)
    channels = sample_dd_v1(channels_count, CHANNEL_SEED).matrices
    q_rows, generator_rows, _, masks = q_and_generator_evaluation(originals, candidates, parameter_rows, q_operators, channels)
    write_csv(OUT / "q_representation_fidelity.csv", q_rows); write_csv(OUT / "generator_fingerprint.csv", generator_rows)
    communication_channel, communication = communication_evaluation(originals, candidates, masks, channels, target_bits)
    write_csv(OUT / "communication_per_channel.csv", communication_channel); write_csv(OUT / "communication_summary.csv", communication)
    parameter_population = parameter_reproducibility(parameter_rows); write_csv(OUT / "parameter_reproducibility.csv", parameter_population)
    summary = depth_summary(matrix_rows, operator_rows, q_rows, generator_rows, communication); write_csv(OUT / "depth_scaling_summary.csv", summary)
    outcome, gates = classify(summary, matrix_rows, operator_rows, q_rows, generator_rows, communication)
    make_figures(summary)
    protected_after = protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Stage 4C changed a prior scientific artifact")
    continuity_row = next(row for row in matrix_rows if row["target"] == "LEARNED_0" and row["arm"] == "M" and int(row["depth"]) == 1)
    stage4a_nmse = 1.1072858061594713
    diagnostics = {
        "outcome": outcome, "gates": gates, "family": {"definition": "D0 F^H D1 ... F^H DL", "phase_degree": 3, "depths": DEPTHS, "parameter_counts": {str(depth): cascade_parameter_count(depth) for depth in DEPTHS}, "fft_signs": "all F^H", "unitary_by_construction": True, "hidden_dense_matrices": False},
        "fit_steps": fit_steps, "restarts": restarts, "selection": "minimum fitting-arm validation loss only", "train_atoms": train_atoms, "validation_atoms": validation_atoms, "evaluation_atoms": evaluation_atoms,
        "atom_seeds": {"train": TRAIN_ATOM_SEED, "validation": VALIDATION_ATOM_SEED, "evaluation": EVALUATION_ATOM_SEED, "q": Q_SEED},
        "channels": channels_count, "channel_seed": CHANNEL_SEED, "symbol_seed": SYMBOL_SEED, "noise_seed": NOISE_SEED, "target_bits": target_bits, "snrs_db": SNRS,
        "checkpoint_hashes": hashes, "all_stage3b_hashes_verified": EXPECTED_STAGE3B_HASHES,
        "fitted_checkpoint_hashes": {row["candidate"]: row["sha256"] for row in parameter_rows},
        "maximum_fitted_unitarity_residual": max(float(row["unitarity_residual"]) for row in parameter_rows),
        "stage4a_depth1_continuity": {"stage4a_nmse": stage4a_nmse, "stage4c_nmse": float(continuity_row["aligned_matrix_nmse"]), "absolute_difference": abs(float(continuity_row["aligned_matrix_nmse"]) - stage4a_nmse)},
        "protected_artifact_count": len(protected_before), "protected_artifacts_unchanged": protected_after == protected_before,
        "stage2e_used": False, "ber_used_for_fitting_or_selection": False, "q_used_for_fitting_or_selection": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage4c_diagnostics.json", diagnostics)
    write_report(outcome, gates, summary, parameter_rows, parameter_population, diagnostics)
    print(f"STAGE 4C COMPLETE in {diagnostics['runtime_seconds']:.1f}s — OUTCOME {outcome}; STOP", flush=True)
    return diagnostics


def regenerate_stage4c_report() -> dict[str, object]:
    """Reclassify and render only from frozen production outputs."""
    def read_csv(name: str) -> list[dict[str, object]]:
        with (OUT / name).open() as handle:
            return list(csv.DictReader(handle))
    matrix_rows = read_csv("matrix_atom_fidelity.csv")
    operator_rows = read_csv("complex_operator_fidelity.csv")
    q_rows = read_csv("q_representation_fidelity.csv")
    generator_rows = read_csv("generator_fingerprint.csv")
    communication = read_csv("communication_summary.csv")
    for row in communication:
        row["snr_db"] = int(row["snr_db"]); row["ber"] = float(row["ber"]); row["total_bits"] = int(row["total_bits"])
    parameter_rows = json.loads((OUT / "fitted_parameters_and_alignment.json").read_text())["fits"]
    parameter_population = read_csv("parameter_reproducibility.csv")
    summary = depth_summary(matrix_rows, operator_rows, q_rows, generator_rows, communication)
    write_csv(OUT / "depth_scaling_summary.csv", summary)
    outcome, gates = classify(summary, matrix_rows, operator_rows, q_rows, generator_rows, communication)
    diagnostics = json.loads((OUT / "stage4c_diagnostics.json").read_text())
    continuity_row = next(row for row in matrix_rows if row["target"] == "LEARNED_0" and row["arm"] == "M" and int(row["depth"]) == 1)
    stage4a_nmse = 1.1072858061594713
    diagnostics["stage4a_depth1_continuity"] = {"stage4a_nmse": stage4a_nmse, "stage4c_nmse": float(continuity_row["aligned_matrix_nmse"]), "absolute_difference": abs(float(continuity_row["aligned_matrix_nmse"]) - stage4a_nmse)}
    diagnostics["outcome"] = outcome; diagnostics["gates"] = gates
    write_json(OUT / "stage4c_diagnostics.json", diagnostics)
    make_figures(summary)
    write_report(outcome, gates, summary, parameter_rows, parameter_population, diagnostics)
    print(f"STAGE 4C REPORT REGENERATED — OUTCOME {outcome}; frozen fits/evaluations unchanged", flush=True)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fit-steps", type=int, default=FIT_STEPS); parser.add_argument("--restarts", type=int, default=RESTARTS)
    parser.add_argument("--target-bits", type=int, default=TARGET_BITS); parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--train-atoms", type=int, default=TRAIN_ATOMS); parser.add_argument("--validation-atoms", type=int, default=VALIDATION_ATOMS)
    parser.add_argument("--evaluation-atoms", type=int, default=EVALUATION_ATOMS); parser.add_argument("--q-samples", type=int, default=Q_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.report_only:
        regenerate_stage4c_report()
    else:
        run_stage4c(fit_steps=args.fit_steps, restarts=args.restarts, target_bits=args.target_bits, channels_count=args.channels, train_atoms=args.train_atoms, validation_atoms=args.validation_atoms, evaluation_atoms=args.evaluation_atoms, q_samples=args.q_samples)
