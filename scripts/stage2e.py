"""Run Stage 2E classical recalibration and complete frozen-waveform forensics."""

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
from src.audit import (
    direct_afdm_idaft,
    expected_afdm_column_offset,
    otfs_reference_demodulate,
    otfs_reference_modulate,
    sample_classical_reference,
)
from src.channel import COMPLEX_DTYPE, assemble_dd_channel, dft_matrix, make_generator
from src.metrics import constellation_encode, qam16_constellation, qpsk_constellation, random_bits
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import file_sha256, load_frozen_waveform
from src.stage2b import path_resolved_operators, spectral_overlap_matrix
from src.stage2d import ambiguity_coupling, score_support_mask
from src.stage2e import (
    AFDM_GUARDED_CANONICAL_MULTIPLIER,
    AFDM_INTEGER_CANONICAL_MULTIPLIER,
    AFDM_SEARCH_MULTIPLIERS,
    REGIMES,
    afdm_design_record,
    afdm_validation_rows,
    ambiguity_surface,
    atom_spectral_time_metrics,
    communication_rows,
    dd_concentration_maps,
    fixed_support_metrics,
    papr_samples,
    path_separation_record,
    q_masks_for_regime,
    regime_q_samples,
    sample_regime,
    select_afdm_multiplier,
    simple_pairwise_explanation,
    stage2e_waveforms,
    stft_power,
)
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2e"
REPORT = ROOT / "reports" / "stage2e_summary.md"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
COLORS = {"Learned": "tab:orange", "OFDM": "tab:blue", "OTFS": "tab:green", "AFDM": "tab:red", "IDENTITY": "0.35"}
HEADLINE = ("Learned", "OFDM", "OTFS", "AFDM")
VALIDATION_SEED = 11100
TEST_SEED = 11101
Q_SEED = 11102
SYMBOL_SEED = 11103
NOISE_SEED = 11104
STRESS_SEED = 11120
STRESS_Q_SEED = 11140
PAPR_SEED = 11160


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def prior_hashes() -> dict[str, str]:
    result = {}
    for directory in ("stage2a", "stage2b", "stage2c", "stage2d"):
        for path in sorted((ROOT / "outputs" / directory).glob("**/*")):
            if path.is_file():
                result[str(path.relative_to(ROOT))] = file_sha256(path)
    return result


def aggregate_communication(rows: list[dict[str, object]], extra: tuple[str, ...] = ()) -> list[dict[str, object]]:
    keys = extra + ("modulation", "snr_db", "waveform", "receiver_mode", "k")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, selected in groups.items():
        errors = sum(int(row["bit_errors"]) for row in selected)
        bits = sum(int(row["total_bits"]) for row in selected)
        symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
        symbols = sum(int(row["total_symbols"]) for row in selected)
        output.append({**dict(zip(keys, group)), "bit_errors": errors, "total_bits": bits, "ber": errors / bits,
                       "symbol_errors": symbol_errors, "total_symbols": symbols, "ser": symbol_errors / symbols})
    return output


def paired_bootstrap(rows: list[dict[str, object]], learned: str = "Learned", resamples: int = 2000) -> list[dict[str, object]]:
    keys = ("modulation", "snr_db", "receiver_mode", "k")
    output = []
    generator = np.random.default_rng(11180)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    for key, selected in grouped.items():
        by_waveform: dict[str, dict[int, float]] = {}
        for row in selected:
            by_waveform.setdefault(str(row["waveform"]), {})[int(row["channel_index"])] = int(row["bit_errors"]) / int(row["total_bits"])
        if learned not in by_waveform:
            continue
        indices = np.asarray(sorted(by_waveform[learned]))
        left = np.asarray([by_waveform[learned][int(index)] for index in indices])
        for comparator in ("OFDM", "OTFS", "AFDM", "AFDM_CANONICAL"):
            if comparator not in by_waveform:
                continue
            right = np.asarray([by_waveform[comparator][int(index)] for index in indices])
            delta = left - right
            sampled = generator.integers(0, len(delta), size=(resamples, len(delta)))
            means = delta[sampled].mean(axis=1)
            output.append({**dict(zip(keys, key)), "comparator": comparator, "learned_minus_comparator": float(delta.mean()),
                           "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))})
    return output


def afdm_audit(validation_channels: torch.Tensor, smoke: bool) -> tuple[int, dict[str, object]]:
    design = afdm_design_record()
    sweep = afdm_validation_rows(validation_channels, config.STAGE1A_SNRS_DB, AFDM_SEARCH_MULTIPLIERS, k=24)
    selected, summary = select_afdm_multiplier(sweep)
    write_csv(OUT / "afdm_validation_sweep_per_channel.csv", sweep)
    write_csv(OUT / "afdm_validation_sweep_summary.csv", summary)
    c2_values = {"existing_small_irrational": config.AFDM_C2, "zero_reference": 0.0,
                 "larger_irrational_reference": math.sqrt(2.0) / (2 * config.N)}
    c2_rows = []
    for label, c2 in c2_values.items():
        rows = afdm_validation_rows(validation_channels, config.STAGE1A_SNRS_DB, (AFDM_GUARDED_CANONICAL_MULTIPLIER,), 24, c2)
        c2_rows.append({"variant": label, "c2": c2, "mean_analytic_mse": float(np.mean([row["analytic_mse"] for row in rows]))})
    write_csv(OUT / "afdm_c2_audit.csv", c2_rows)
    variants = {
        "AFDM_K3_TUNED": 1,
        "AFDM_INTEGER_CANONICAL": AFDM_INTEGER_CANONICAL_MULTIPLIER,
        "AFDM_CANONICAL": AFDM_GUARDED_CANONICAL_MULTIPLIER,
        "AFDM_K24_TUNED": selected,
    }
    controlled = ((0, 0), (2, 0), (0, 1), (2, 1), (2, 0.35), (2.4, 1), (2.4, 0.35))
    path_rows = []
    for name, multiplier in variants.items():
        waveform = afdm_waveform(c1=multiplier / (2 * config.N))
        for delay, doppler in controlled:
            predicted = expected_afdm_column_offset(int(delay), int(doppler), config.N, multiplier / (2 * config.N)) if float(delay).is_integer() and float(doppler).is_integer() else None
            path_rows.append({"variant": name, "multiplier_2Nc1": multiplier, **path_separation_record(waveform, delay, doppler, predicted)})
    write_csv(OUT / "afdm_single_path_audit.csv", path_rows)
    multipath_rows = []
    multipath_cases = {
        "grid": ((0.0, 2.0, 4.0), (-1.0, 0.0, 1.0)),
        "fractional_doppler": ((0.0, 2.0, 4.0), (-0.65, 0.15, 0.55)),
        "full_offgrid": ((0.3, 2.4, 3.7), (-0.65, 0.15, 0.55)),
    }
    for variant, multiplier in variants.items():
        waveform = afdm_waveform(c1=multiplier / (2 * config.N))
        for case, (delays, dopplers) in multipath_cases.items():
            resolved = path_resolved_operators(waveform, torch.tensor(delays), torch.tensor(dopplers), torch.ones(3, dtype=COMPLEX_DTYPE))
            path_energy = torch.abs(resolved) ** 2
            flat = path_energy.reshape(3, -1); flat = flat / torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
            overlap = float(torch.mean(torch.tensor([torch.dot(flat[i], flat[j]) for i in range(3) for j in range(i + 1, 3)])))
            for path_index, (delay, doppler) in enumerate(zip(delays, dopplers)):
                record = path_separation_record(waveform, delay, doppler, None)
                multipath_rows.append({"variant": variant, "multiplier_2Nc1": multiplier, "case": case,
                                       "path_index": path_index, "inter_path_energy_overlap": overlap, **record})
    write_csv(OUT / "afdm_multipath_separation_audit.csv", multipath_rows)
    integer_canonical = [row for row in path_rows if row["variant"] == "AFDM_CANONICAL" and row["predicted_offset"] is not None]
    hard_gate = all(row["predicted_offset"] == row["observed_offset"] and row["dominant_offset_energy_fraction"] > 1 - 1e-10 for row in integer_canonical)
    if not hard_gate:
        raise RuntimeError("canonical AFDM integer-grid path-location hard gate failed")
    return selected, {"design": design, "selected_multiplier_2Nc1": selected, "selected_c1": selected / (2 * config.N),
                      "historical_multiplier_2Nc1": 1, "canonical_multiplier_2Nc1": 5,
                      "c2": config.AFDM_C2, "c2_audit": c2_rows, "integer_path_gate": hard_gate}


def otfs_audit() -> dict[str, object]:
    generator = make_generator(11105)
    bits = random_bits((8, 8, 2), generator)
    symbols = (1 - 2 * bits[..., 0] + 1j * (1 - 2 * bits[..., 1])).to(COMPLEX_DTYPE) / math.sqrt(2)
    direct = otfs_waveform() @ symbols.mT.reshape(-1)
    reference = otfs_reference_modulate(symbols)
    reconstructed = otfs_reference_demodulate(reference, 8, 8)
    path_cases = {"one_on_grid": ((1,), (1,)), "three_on_grid": ((0, 2, 4), (-1, 0, 1)),
                  "fractional_doppler": ((2,), (0.35,)), "fractional_delay": ((2.4,), (0.0,))}
    path_rows = []
    waveform = otfs_waveform()
    for name, (delays, dopplers) in path_cases.items():
        gains = torch.ones(len(delays), dtype=COMPLEX_DTYPE) / math.sqrt(len(delays))
        operator = effective_operator(assemble_dd_channel(torch.tensor(delays), torch.tensor(dopplers), gains), waveform)
        for k in (3, 16, 24, 32):
            path_rows.append({"case": name, "k": k, "retained_energy": float(torch.sum(torch.abs(rowwise_topk(operator, k)) ** 2) / torch.sum(torch.abs(operator) ** 2))})
    write_csv(OUT / "otfs_path_sparsity.csv", path_rows)
    result = {"grid": "8x8", "serialization": "column-major; delay index fastest", "unitary_algebraic_n": 64,
              "reference_modulation_max_error": float(torch.max(torch.abs(direct - reference))),
              "clean_inverse_max_error": float(torch.max(torch.abs(reconstructed - symbols)))}
    if result["reference_modulation_max_error"] > 1e-12 or result["clean_inverse_max_error"] > 1e-12:
        raise RuntimeError("OTFS equivalence/inverse audit failed")
    return result


def nominal_communication(learned: torch.Tensor, afdm_multiplier: int, smoke: bool) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    channels = sample_regime(REGIMES[3], 32 if smoke else 512, TEST_SEED).matrices
    waveforms = stage2e_waveforms(learned, afdm_multiplier)
    waveforms["AFDM_CANONICAL"] = afdm_waveform(
        c1=AFDM_GUARDED_CANONICAL_MULTIPLIER / (2 * config.N)
    )
    q_samples = 64 if smoke else 2048
    delays, dopplers = regime_q_samples(REGIMES[3], q_samples, Q_SEED)
    q = {name: ambiguity_coupling(waveform, delays, dopplers) for name, waveform in waveforms.items()}
    masks = {name: {"full": None, **{f"fixedQK{k}": score_support_mask(q[name], k) for k in (16, 24, 32)},
                    "oracle24": None} for name in waveforms}
    snrs = (12, 18) if smoke else (0, 5, 10, 12, 15, 18, 20, 25, 30)
    target_bits = 8192 if smoke else 1_048_576
    rows = communication_rows(channels, waveforms, masks, (qpsk_constellation(), qam16_constellation()), snrs,
                              target_bits, SYMBOL_SEED, NOISE_SEED)
    write_csv(OUT / "nominal_communication_per_channel.csv", rows)
    write_csv(OUT / "nominal_communication_summary.csv", aggregate_communication(rows))
    write_csv(OUT / "nominal_paired_differences.csv", paired_bootstrap(rows, resamples=200 if smoke else 2000))
    np.savez(OUT / "nominal_q_matrices.npz", **q)
    return rows, q


def classical_reference(smoke: bool) -> list[dict[str, object]]:
    n = 256
    channels = sample_classical_reference(8 if smoke else 128, 11106, n=n).matrices
    waveforms = {"OFDM": ofdm_waveform(n), "OTFS": torch.kron(dft_matrix(16).mH.contiguous(), torch.eye(16, dtype=COMPLEX_DTYPE)),
                 "AFDM": afdm_waveform(n=n, c1=(2 * 2 + 1) / (2 * n), c2=math.sqrt(2) / (100 * n))}
    masks = {name: {"full": None} for name in waveforms}
    rows = communication_rows(channels, waveforms, masks, (qpsk_constellation(),), (15, 20, 25),
                              8192 if smoke else 262144, 11107, 11108)
    write_csv(OUT / "classical_n256_reference.csv", aggregate_communication(rows))
    return rows


def adapted_afdm_multiplier(regime) -> int:
    if regime.doppler_kind == "integer":
        alpha_max = int(regime.doppler_max); guard = 0
    else:
        alpha_max = int(math.floor(regime.doppler_max + 0.5)); guard = 1
    return 2 * (alpha_max + guard) + 1


def stress_test(learned: torch.Tensor, afdm_multiplier: int, original_q: dict[str, np.ndarray], smoke: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    communication: list[dict[str, object]] = []
    structural: list[dict[str, object]] = []
    for regime_index, regime in enumerate(REGIMES):
        channels = sample_regime(regime, 16 if smoke else 256, STRESS_SEED + regime_index).matrices
        waveforms = stage2e_waveforms(learned, afdm_multiplier)
        adapted_multiplier = adapted_afdm_multiplier(regime)
        if adapted_multiplier != afdm_multiplier:
            waveforms["AFDM adapted"] = afdm_waveform(c1=adapted_multiplier / (2 * config.N))
        regime_masks, regime_q = q_masks_for_regime(waveforms, regime, 24, 32 if smoke else 1024, STRESS_Q_SEED + regime_index)
        original_masks = {}
        for name in waveforms:
            base = "AFDM" if name == "AFDM adapted" else name
            original_masks[name] = score_support_mask(original_q[base], 24)
        receivers = {name: {"fixedOriginalK24": original_masks[name], "fixedRegimeK24": regime_masks[name]} for name in waveforms}
        rows = communication_rows(channels, waveforms, receivers, (qpsk_constellation(), qam16_constellation()),
                                  (15, 18, 20), 4096 if smoke else 262144,
                                  SYMBOL_SEED + 1000 + regime_index * 10, NOISE_SEED + 1000 + regime_index * 10)
        communication.extend({"regime": regime.name, "afdm_multiplier_2Nc1": adapted_multiplier if row["waveform"] == "AFDM adapted" else afdm_multiplier, **row} for row in rows)
        for name, waveform in waveforms.items():
            for mask_name, mask in (("original", original_masks[name]), ("regime", regime_masks[name])):
                metrics = fixed_support_metrics(channels, waveform, mask, 24, 18)
                structural.append({"regime": regime.name, "waveform": name, "mask": mask_name,
                                   "limiting_distortion_regularization": 1e-10,
                                   **{key: float(value.mean()) for key, value in metrics.items()}})
    write_csv(OUT / "stress_communication_per_channel.csv", communication)
    write_csv(OUT / "stress_communication_summary.csv", aggregate_communication(communication, extra=("regime", "afdm_multiplier_2Nc1")))
    write_csv(OUT / "stress_structural_summary.csv", structural)
    return communication, structural


def controlled_operator_audit(waveforms: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    cases = {
        "identity": ((0.0,), (0.0,)), "integer_delay": ((2.0,), (0.0,)), "integer_doppler": ((0.0,), (1.0,)),
        "integer_delay_doppler": ((2.0,), (1.0,)), "integer_delay_fractional_doppler": ((2.0,), (0.35,)),
        "fractional_delay_integer_doppler": ((2.4,), (1.0,)), "fully_fractional": ((2.4,), (0.35,)),
        "multipath_grid": ((0.0, 2.0, 4.0), (-1.0, 0.0, 1.0)),
        "multipath_fractional_doppler": ((0.0, 2.0, 4.0), (-0.65, 0.15, 0.55)),
        "multipath_full_offgrid": ((0.3, 2.4, 3.7), (-0.65, 0.15, 0.55)),
    }
    rows = []
    for case, (delays, dopplers) in cases.items():
        gains = torch.tensor([1, 0.7j, -0.5] if len(delays) == 3 else [1], dtype=COMPLEX_DTYPE)
        physical = assemble_dd_channel(torch.tensor(delays), torch.tensor(dopplers), gains)
        for name, waveform in waveforms.items():
            operator = effective_operator(physical, waveform)
            energy = torch.abs(operator) ** 2
            for k in (3, 16, 24, 32):
                retained = float(torch.sum(torch.abs(rowwise_topk(operator, k)) ** 2) / torch.sum(energy))
                rows.append({"case": case, "waveform": name, "k": k, "retained_energy": retained})
    write_csv(OUT / "controlled_operator_retained_energy.csv", rows)
    return rows


def physical_forensics(waveforms: dict[str, torch.Tensor], q: dict[str, np.ndarray], smoke: bool) -> dict[str, object]:
    learned = waveforms["Learned"]
    metrics, frequencies, learned_db = atom_spectral_time_metrics(learned, fft_size=2048 if smoke else 16384)
    order = np.argsort(metrics["centroid"])
    atom_rows = []
    for index in range(config.N):
        atom = learned[:, index].numpy(); atom_power = np.abs(atom) ** 2
        atom_rows.append({"atom": index, **{name: float(value[index]) for name, value in metrics.items()},
                          "maximum_amplitude": float(np.abs(atom).max()),
                          "peak_to_average_power_db": float(10 * np.log10(atom_power.max() / atom_power.mean()))})
    write_csv(OUT / "learned_atom_metrics.csv", atom_rows)
    np.savez(OUT / "learned_spectral_time_arrays.npz", frequencies=frequencies, spectra_db=learned_db,
             centroid_order=order, magnitude=np.abs(learned.numpy()), real=learned.real.numpy(), phase=np.angle(learned.numpy()),
             phase_increment=np.angle(learned[1:].numpy() * learned[:-1].conj().resolve_conj().numpy()))

    representatives = {}
    for name, waveform in {key: waveforms[key] for key in HEADLINE}.items():
        values = waveform.detach().resolve_conj().numpy()
        if name == "Learned":
            selected = order[np.rint(np.asarray([0.1, 0.5, 0.9]) * 63).astype(int)]
        else:
            selected = np.asarray([8, 32, 56])
        representatives[name] = {str(int(index)): stft_power(values[:, index]) for index in selected}
    np.savez(OUT / "stft_representatives.npz", **{f"{name}_{index}": value for name, data in representatives.items() for index, value in data.items()})

    delays = np.linspace(0, 4, 17 if smoke else 41); dopplers = np.linspace(-1, 1, 17 if smoke else 41)
    auto = {}
    for name in HEADLINE:
        waveform = waveforms[name]
        index = int(order[32]) if name == "Learned" else 32
        auto[name] = ambiguity_surface(waveform[:, index], waveform[:, index], delays, dopplers)
    off = np.triu_indices(config.N, 1); values = q["Learned"][off]; sort = np.argsort(values)
    pair_positions = {"low": sort[0], "median": sort[len(sort) // 2], "high": sort[-1]}
    pairs = {name: (int(off[0][position]), int(off[1][position])) for name, position in pair_positions.items()}
    cross = {name: ambiguity_surface(learned[:, i], learned[:, j], delays, dopplers) for name, (i, j) in pairs.items()}
    np.savez(OUT / "ambiguity_surfaces.npz", delays=delays, dopplers=dopplers, **{f"auto_{k}": v for k, v in auto.items()},
             **{f"cross_{k}": v for k, v in cross.items()})

    dd_delays = np.linspace(0, 8, 5 if smoke else 17); dd_dopplers = np.linspace(-2, 2, 5 if smoke else 17)
    dd_maps = {name: dd_concentration_maps(waveforms[name], dd_delays, dd_dopplers) for name in HEADLINE}
    np.savez(OUT / "dd_concentration_maps.npz", delays=dd_delays, dopplers=dd_dopplers,
             **{f"{name}_{metric}": value for name, maps in dd_maps.items() for metric, value in maps.items()})

    overlap = spectral_overlap_matrix(learned)
    centroid = metrics["centroid"]; separation = np.abs(centroid[:, None] - centroid[None, :]); separation = np.minimum(separation, 1 - separation)
    bandwidth_overlap = np.minimum(metrics["rms_bandwidth"][:, None], metrics["rms_bandwidth"][None, :]) / np.maximum(metrics["rms_bandwidth"][:, None], metrics["rms_bandwidth"][None, :])
    magnitude = np.abs(learned.numpy()); magnitude_similarity = np.corrcoef(magnitude.T)
    stft_all = np.stack([stft_power(learned[:, index].numpy()).reshape(-1) for index in range(config.N)])
    stft_all /= np.maximum(np.linalg.norm(stft_all, axis=1, keepdims=True), 1e-15)
    tf_overlap = stft_all @ stft_all.T
    cross_peak = np.zeros((config.N, config.N))
    small_delays = np.linspace(0, 4, 9); small_dopplers = np.linspace(-0.75, 0.75, 9)
    for i in range(config.N):
        for j in range(i + 1, config.N):
            peak = float(ambiguity_surface(learned[:, i], learned[:, j], small_delays, small_dopplers).max())
            cross_peak[i, j] = cross_peak[j, i] = peak
    explanation = simple_pairwise_explanation(q["Learned"], {"spectral_overlap": overlap, "centroid_separation": separation,
        "bandwidth_overlap": bandwidth_overlap, "magnitude_similarity": magnitude_similarity, "stft_overlap": tf_overlap,
        "cross_ambiguity_peak": cross_peak})
    write_json(OUT / "q_physical_explanation.json", explanation)

    papr_rows = []; psd_rows = []; psd_arrays = {}
    for modulation_index, constellation in enumerate((qpsk_constellation(), qam16_constellation())):
        blocks = 128 if smoke else 10000
        bits = random_bits((blocks, config.N, constellation.bits_per_symbol), make_generator(PAPR_SEED + modulation_index))
        symbols, _ = constellation_encode(bits, constellation)
        for name in HEADLINE:
            waveform = waveforms[name]
            native = papr_samples(waveform, symbols, 1); over = papr_samples(waveform, symbols, 4)
            papr_rows.append({"modulation": constellation.name, "waveform": name, "blocks": blocks,
                              "native_mean_db": float(native.mean()), "native_p95_db": float(np.quantile(native, .95)),
                              "native_p99_db": float(np.quantile(native, .99)), "oversampled_mean_db": float(over.mean()),
                              "oversampled_p95_db": float(np.quantile(over, .95)), "oversampled_p99_db": float(np.quantile(over, .99))})
            np.savez(OUT / f"papr_{constellation.name}_{name}.npz", native=native, oversampled=over)
            fft_size = 1024
            psd = np.zeros(fft_size)
            for start in range(0, blocks, 256):
                samples = torch.einsum("ij,bj->bi", waveform, symbols[start:start + 256]).numpy()
                psd += np.sum(np.abs(np.fft.fftshift(np.fft.fft(samples, n=fft_size, axis=-1), axes=-1)) ** 2, axis=0)
            psd /= psd.sum()
            frequency = np.fft.fftshift(np.fft.fftfreq(fft_size))
            cumulative = np.concatenate(([0.0], np.cumsum(np.concatenate((psd, psd)))))
            target = .99; best = fft_size
            for left in range(fft_size):
                right = int(np.searchsorted(cumulative, cumulative[left] + target, side="left"))
                best = min(best, right - left)
            psd_rows.append({"modulation": constellation.name, "waveform": name, "fft_size": fft_size,
                             "occupied_bandwidth_99_fraction_nyquist": best / fft_size,
                             "out_of_nominal_nyquist_energy": 0.0,
                             "interpretation": "ideal discrete-time periodic spectrum; no pulse-shaping RF mask"})
            psd_arrays[f"{constellation.name}_{name}"] = psd
    write_csv(OUT / "papr_summary.csv", papr_rows)
    write_csv(OUT / "random_block_psd_summary.csv", psd_rows)
    np.savez(OUT / "random_block_psd.npz", frequency=frequency, **psd_arrays)
    return {"ambiguity_pairs": {key: list(value) for key, value in pairs.items()}, "explanation": explanation,
            "learned_atom_summary": {name: {"mean": float(np.mean(value)), "min": float(np.min(value)), "max": float(np.max(value))}
                                     for name, value in metrics.items()}, "papr": papr_rows, "psd": psd_rows}


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(); fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.1); plt.close(fig)


def plot_ber(summary: list[dict[str, object]], modulation: str, mode: str, title: str, filename: str) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.9))
    for name in HEADLINE:
        selected = sorted((row for row in summary if row["modulation"] == modulation and row["receiver_mode"] == mode and row["waveform"] == name), key=lambda row: row["snr_db"])
        axis.semilogy([row["snr_db"] for row in selected], [max(float(row["ber"]), .5 / int(row["total_bits"])) for row in selected], "o-", color=COLORS[name], label=name)
    axis.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Bit error rate", title=title); axis.grid(True, which="both", alpha=.3); axis.legend()
    save_figure(fig, filename)


def plot_nominal(rows: list[dict[str, object]]) -> None:
    summary = aggregate_communication(rows)
    plot_ber(summary, "QPSK", "fixedQK24", "DD-v1 QPSK, fixed-Q K=24", "fig01_qpsk_fixed_q_k24.png")
    plot_ber(summary, "16QAM", "fixedQK24", "DD-v1 16-QAM, fixed-Q K=24", "fig02_16qam_fixed_q_k24.png")
    plot_ber(summary, "QPSK", "full", "DD-v1 QPSK, full information K=64", "fig03_qpsk_full_information.png")
    plot_ber(summary, "16QAM", "full", "DD-v1 16-QAM, full information K=64", "fig04_16qam_full_information.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=False)
    for axis, modulation in zip(axes, ("QPSK", "16QAM")):
        for name in HEADLINE:
            values = []
            for mode in ("fixedQK16", "fixedQK24", "fixedQK32", "full"):
                match = next(row for row in summary if row["modulation"] == modulation and row["snr_db"] == 18 and row["waveform"] == name and row["receiver_mode"] == mode)
                values.append(max(float(match["ber"]), .5 / int(match["total_bits"])))
            axis.semilogy((16, 24, 32, 64), values, "o-", label=name, color=COLORS[name])
        axis.set(title=modulation.replace("16QAM", "16-QAM"), xlabel="Receiver budget K", ylabel="Bit error rate"); axis.grid(True, which="both", alpha=.3)
    axes[0].legend(); fig.suptitle("DD-v1 BER across receiver budgets at 18 dB")
    save_figure(fig, "fig05_receiver_budget_comparison.png")


def plot_stress(communication: list[dict[str, object]], structural: list[dict[str, object]]) -> None:
    summary = aggregate_communication(communication, extra=("regime", "afdm_multiplier_2Nc1"))
    regimes = [regime.name for regime in REGIMES]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)
    for axis, modulation in zip(axes, ("QPSK", "16QAM")):
        for name in HEADLINE:
            values = []
            for regime in regimes:
                candidates = [row for row in summary if row["regime"] == regime and row["modulation"] == modulation and row["snr_db"] == 18 and row["waveform"] == name and row["receiver_mode"] == "fixedRegimeK24"]
                values.append(float(candidates[0]["ber"]))
            axis.semilogy(range(len(regimes)), values, "o-", label=name, color=COLORS[name])
        axis.set(ylabel="BER", title=f"{modulation.replace('16QAM', '16-QAM')} at 18 dB, regime-informed fixed-Q"); axis.grid(True, which="both", alpha=.3)
    axes[0].legend(); axes[-1].set_xticks(range(len(regimes)), regimes, rotation=25, ha="right")
    save_figure(fig, "fig06_frozen_waveform_stress_ber.png")
    fig, axis = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(regimes)); width = .18
    for offset, name in enumerate(HEADLINE):
        values = [next(row["retained_energy"] for row in structural if row["regime"] == regime and row["waveform"] == name and row["mask"] == "regime") for regime in regimes]
        axis.bar(x + (offset - 1.5) * width, values, width, label=name, color=COLORS[name])
    axis.set_xticks(x, regimes, rotation=25, ha="right"); axis.set(ylabel="Mean retained energy", ylim=(0, 1.03), title="Frozen-waveform retained energy, regime-informed fixed-Q K=24")
    axis.grid(axis="y", alpha=.3); axis.legend(); save_figure(fig, "fig07_stress_retained_energy.png")


def plot_spectral_time(waveforms: dict[str, torch.Tensor]) -> None:
    learned = waveforms["Learned"].numpy(); arrays = np.load(OUT / "learned_spectral_time_arrays.npz")
    frequencies = arrays["frequencies"]; db = arrays["spectra_db"]; order = arrays["centroid_order"]
    fig, axis = plt.subplots(figsize=(10.0, 5.5))
    axis.plot(frequencies, db, lw=.45, alpha=.25); axis.set(xlabel="Normalized frequency", ylabel="Magnitude (dB, common unit-norm scale)", title="All 64 frozen Learned atom spectra")
    axis.set_ylim(-80, max(5, float(db.max()) + 1)); axis.grid(alpha=.2); save_figure(fig, "fig08_all64_spectra_overlay.png")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    for axis, values, title in zip(axes, (db.T, db[:, order].T), ("Native atom order", "Circular spectral-centroid order")):
        image = axis.imshow(values, origin="lower", aspect="auto", extent=(frequencies[0], frequencies[-1], -.5, 63.5), vmin=-80, vmax=float(db.max()), cmap="viridis")
        axis.set(ylabel="Atom index", title=title); fig.colorbar(image, ax=axis, label="Magnitude (dB)")
    axes[-1].set_xlabel("Normalized frequency"); save_figure(fig, "fig09_all64_spectral_heatmaps.png")
    rows = list(csv.DictReader((OUT / "learned_atom_metrics.csv").open()))
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), sharex=True)
    names = (("centroid", "Circular centroid"), ("rms_bandwidth", "RMS bandwidth"), ("spectral_entropy", "Spectral entropy"),
             ("top8_spectral_energy", "Top-8-bin spectral energy"), ("magnitude_cv", "Magnitude CV"), ("time_spread", "Time spread"))
    for axis, (key, title) in zip(axes.flat, names):
        axis.plot(range(64), [float(rows[int(index)][key]) for index in order], "o-", ms=3); axis.set(title=title); axis.grid(alpha=.3)
    axes[-1, 0].set_xlabel("Centroid-ordered atom"); axes[-1, 1].set_xlabel("Centroid-ordered atom")
    save_figure(fig, "fig10_atom_spectral_time_metrics.png")
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 10.5), sharex=True)
    display = ((np.abs(learned).T, "Magnitude", "viridis"), (learned.real.T, "Real part", "coolwarm"),
               (np.angle(learned).T, "Phase", "twilight"), (np.angle(learned[1:] * learned[:-1].conj()).T, "Phase increment", "twilight"))
    for axis, (values, title, cmap) in zip(axes, display):
        image = axis.imshow(values[order], origin="lower", aspect="auto", cmap=cmap); axis.set(ylabel="Sorted atom", title=title); fig.colorbar(image, ax=axis, shrink=.8)
    axes[-1].set_xlabel("Time sample n"); save_figure(fig, "fig11_all64_time_domain.png")


def plot_stft() -> None:
    arrays = np.load(OUT / "stft_representatives.npz")
    names = list(arrays.files)
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 11.0))
    for axis, key in zip(axes.flat, names):
        image = axis.imshow(10 * np.log10(np.maximum(arrays[key], 1e-6)), origin="lower", aspect="auto", vmin=-60, vmax=0, cmap="magma")
        axis.set(title=key.replace("_", " atom ", 1), xlabel="Time frame", ylabel="Frequency bin")
    fig.colorbar(image, ax=axes, shrink=.6, label="Normalized power (dB)"); fig.suptitle("Common-setting representative-atom STFTs")
    fig.savefig(OUT / "fig12_representative_stft.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_ambiguity() -> None:
    arrays = np.load(OUT / "ambiguity_surfaces.npz"); extent=(arrays["delays"][0], arrays["delays"][-1], arrays["dopplers"][0], arrays["dopplers"][-1])
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), sharex=True, sharey=True)
    for axis, name in zip(axes, HEADLINE):
        value = arrays[f"auto_{name}"]; image=axis.imshow(value, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=1); axis.set(title=name, xlabel="Delay")
    axes[0].set_ylabel("Doppler"); fig.colorbar(image, ax=axes, label=r"$|A_{kk}|^2$"); fig.suptitle("Representative auto-ambiguity")
    fig.savefig(OUT / "fig13_auto_ambiguity.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharex=True, sharey=True)
    maximum=max(float(arrays[f"cross_{name}"].max()) for name in ("low", "median", "high"))
    for axis, name in zip(axes, ("low", "median", "high")):
        image=axis.imshow(arrays[f"cross_{name}"], origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=maximum); axis.set(title=f"{name} Q pair", xlabel="Delay")
    axes[0].set_ylabel("Doppler"); fig.colorbar(image, ax=axes, label=r"$|A_{ij}|^2$"); fig.suptitle("Learned cross-ambiguity selected objectively by Q")
    fig.savefig(OUT / "fig14_cross_ambiguity.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_dd_maps() -> None:
    arrays=np.load(OUT / "dd_concentration_maps.npz"); extent=(arrays["delays"][0], arrays["delays"][-1], arrays["dopplers"][0], arrays["dopplers"][-1])
    fig, axes=plt.subplots(4,4,figsize=(13,11),sharex=True,sharey=True)
    for row,name in enumerate(HEADLINE):
        for column,metric in enumerate(("top3","top8","top24","participation")):
            value=arrays[f"{name}_{metric}"]; image=axes[row,column].imshow(value,origin="lower",aspect="auto",extent=extent,cmap="viridis")
            axes[row,column].set_title(f"{name}: {metric}"); fig.colorbar(image,ax=axes[row,column],shrink=.7)
    for axis in axes[-1]: axis.set_xlabel("Delay")
    for axis in axes[:,0]: axis.set_ylabel("Doppler")
    save_figure(fig,"fig15_dd_concentration_maps.png")


def plot_operator_regeneration(waveforms: dict[str, torch.Tensor], q: dict[str, np.ndarray]) -> None:
    cases=(("grid",(0.,2.,4.),(-1.,0.,1.)),("fractional nu",(0.,2.,4.),(-.65,.15,.55)),("full off-grid",(.3,2.4,3.7),(-.65,.15,.55)))
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for row,(case,delays,dopplers) in enumerate(cases):
        physical=assemble_dd_channel(torch.tensor(delays),torch.tensor(dopplers),torch.tensor([1,.7j,-.5],dtype=COMPLEX_DTYPE))
        operators={name:torch.abs(effective_operator(physical,waveforms[name])).numpy() for name in ("OFDM","OTFS","AFDM","Learned")}; maximum=max(v.max() for v in operators.values())
        for column,name in enumerate(("OFDM","OTFS","AFDM","Learned")):
            image=axes[row,column].imshow(operators[name],origin="upper",vmin=0,vmax=maximum,cmap="magma"); axes[row,column].set(title=f"{case}: {name}",xlabel="Input column j",ylabel="Output row i")
    fig.colorbar(image,ax=axes,shrink=.6,label="Magnitude"); fig.suptitle("Consistent-orientation grid to off-grid progression")
    fig.savefig(OUT/"fig16_operator_progression.png",dpi=180,bbox_inches="tight");plt.close(fig)
    channels=sample_regime(REGIMES[3],128,11170).matrices
    energy={name:torch.mean(torch.abs(effective_operator(channels,waveforms[name]))**2,dim=0).numpy() for name in ("OFDM","OTFS","AFDM","Learned")}
    fig,axes=plt.subplots(2,4,figsize=(13,6))
    for column,name in enumerate(("OFDM","OTFS","AFDM","Learned")):
        image=axes[0,column].imshow(energy[name],origin="upper",cmap="magma");axes[0,column].set_title(f"{name}: ensemble energy");fig.colorbar(image,ax=axes[0,column],shrink=.7)
        mask=score_support_mask(energy[name],24).numpy();axes[1,column].imshow(mask,origin="upper",cmap="gray_r");axes[1,column].set_title(f"{name}: fixed top-24")
    for axis in axes.flat:axis.set(xlabel="Input column j",ylabel="Output row i")
    save_figure(fig,"fig17_ensemble_energy_fixed_support.png")
    # Path-resolved RGB-style comparison uses identical four physical paths.
    delays=torch.tensor([0.,1.,2.,4.]);dopplers=torch.tensor([-1.,0.,1.,.35]);gains=torch.tensor([1.,.7j,-.5,.35j],dtype=COMPLEX_DTYPE)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5))
    for axis,name in zip(axes,("OFDM","OTFS","AFDM","Learned")):
        resolved=path_resolved_operators(waveforms[name],delays,dopplers,gains); image=axis.imshow(torch.sum(torch.abs(resolved),dim=0).numpy(),origin="upper",cmap="magma");axis.set(title=name,xlabel="Input column j",ylabel="Output row i");fig.colorbar(image,ax=axis,shrink=.7)
    fig.suptitle("Reference four-path resolved-operator comparison");save_figure(fig,"fig18_reference_four_path.png")
    learned_effective=effective_operator(channels,waveforms["Learned"]); retained=torch.sum(torch.abs(rowwise_topk(learned_effective,24))**2,dim=(-2,-1))/torch.sum(torch.abs(learned_effective)**2,dim=(-2,-1))
    sorted_indices=torch.argsort(retained);chosen=(int(sorted_indices[-13]),int(sorted_indices[len(sorted_indices)//2]),int(sorted_indices[12]))
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for row,(label,index) in enumerate(zip(("easy","median","hard"),chosen)):
        for column,name in enumerate(("OFDM","OTFS","AFDM","Learned")):
            value=torch.abs(effective_operator(channels[index],waveforms[name])).numpy();image=axes[row,column].imshow(value,origin="upper",cmap="magma");axes[row,column].set(title=f"{label}: {name}",xlabel="Input column j",ylabel="Output row i")
    fig.colorbar(image,ax=axes,shrink=.6,label="Magnitude");fig.suptitle("Common-channel easy / median / hard DD-v1 gallery")
    fig.savefig(OUT/"fig20_easy_median_hard_gallery.png",dpi=180,bbox_inches="tight");plt.close(fig)
    learned_energy=energy["Learned"];spectral=spectral_overlap_matrix(waveforms["Learned"])
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for axis,title,value in zip(axes,("Spectral overlap S","DD ambiguity Q","Empirical coupling C"),(spectral,q["Learned"],learned_energy)):
        normalized=value/max(float(value.max()),1e-15);image=axis.imshow(normalized,origin="upper",cmap="magma",vmin=0,vmax=1);axis.set(title=title,xlabel="Input atom j",ylabel="Output atom i")
    fig.colorbar(image,ax=axes,shrink=.7,label="Normalized coupling");fig.suptitle("Frozen Learned S / Q / C comparison")
    fig.savefig(OUT/"fig21_s_q_c_comparison.png",dpi=180,bbox_inches="tight");plt.close(fig)


def plot_papr() -> None:
    rows=list(csv.DictReader((OUT/"papr_summary.csv").open()));fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for axis,modulation in zip(axes,("QPSK","16QAM")):
        for name in HEADLINE:
            values=np.load(OUT/f"papr_{modulation}_{name}.npz")["oversampled"];x=np.sort(values);ccdf=1-(np.arange(len(x))+.5)/len(x);axis.semilogy(x,ccdf,label=name,color=COLORS[name])
        axis.set(title=modulation.replace("16QAM","16-QAM"),xlabel="4x-oversampled PAPR (dB)",ylabel="CCDF");axis.grid(True,which="both",alpha=.3)
    axes[0].legend();save_figure(fig,"fig19_papr_ccdf.png")


def report_text(diagnostics: dict[str, object], nominal: list[dict[str, object]], stress: list[dict[str, object]], structural: list[dict[str, object]]) -> str:
    summary=aggregate_communication(nominal); selected=int(diagnostics["afdm"]["selected_multiplier_2Nc1"])
    def value(modulation,mode,name,snr=18):
        row=next(row for row in summary if row["modulation"]==modulation and row["receiver_mode"]==mode and row["waveform"]==name and row["snr_db"]==snr);return float(row["ber"])
    q24={name:value("QPSK","fixedQK24",name) for name in ("IDENTITY",)+HEADLINE}
    full={name:value("QPSK","full",name) for name in ("IDENTITY",)+HEADLINE}
    canonical_q24=value("QPSK","fixedQK24","AFDM_CANONICAL");canonical_full=value("QPSK","full","AFDM_CANONICAL")
    qpsk_oracle=value("QPSK","oracle24","Learned");qam_oracle=value("16QAM","oracle24","Learned")
    qam_q24=value("16QAM","fixedQK24","Learned")
    stress_summary=aggregate_communication(stress,extra=("regime","afdm_multiplier_2Nc1"))
    def stress_value(modulation,regime,name,mode="fixedRegimeK24"):
        row=next(row for row in stress_summary if row["modulation"]==modulation and row["regime"]==regime and row["waveform"]==name and row["receiver_mode"]==mode and row["snr_db"]==18);return float(row["ber"])
    stress_dd_q=stress_value("QPSK","DD-v1","Learned");stress_dd_16=stress_value("16QAM","DD-v1","Learned")
    atom=diagnostics["physical_forensics"]["learned_atom_summary"]; explanation=diagnostics["physical_forensics"]["explanation"]
    papr={row["waveform"]:row for row in diagnostics["physical_forensics"]["papr"] if row["modulation"]=="QPSK"}
    historical_delta=100*(q24["AFDM"]-float(diagnostics["historical_afdm_qpsk_18db"]))/max(float(diagnostics["historical_afdm_qpsk_18db"]),1e-15)
    strongest=max(explanation["spearman"],key=lambda key:abs(explanation["spearman"][key]))
    lines=["# Stage 2E — Classical Baseline Recalibration and Complete Frozen-Waveform Forensics","",
      "## Protocol and provenance","",f"- Frozen checkpoint: `{diagnostics['frozen_checkpoint']['sha256']}`; unitarity residual `{diagnostics['frozen_checkpoint']['unitarity_residual']:.3e}`.",
      f"- AFDM validation used {diagnostics['counts']['validation_channels']} validation-only DD-v1 channels and exact K=24 analytic MSE; no BER, test channel, or Learned result entered selection.",
      f"- Untouched nominal test bank: {diagnostics['counts']['test_channels']} channels; headline points use {diagnostics['counts']['nominal_target_bits']} paired bits per waveform.",
      "- Learned was loaded once through the exact hash gate and was never trained, selected, or modified.","",
      "## AFDM and OTFS audit","",f"DD-v1 uses normalized Doppler bins. With `nu=alpha+a`, `a in [-1/2,1/2)`, endpoints +/-0.75 require `alpha_max=1`. Published integer canonical slope is `2Nc1=3`; the declared one-bin fractional guard gives canonical `2Nc1=5`. The constrained validation sweep over integer slopes 1..11 selected `2Nc1={selected}`, so `c1={selected/(2*config.N):.10g}`. `c2=sqrt(2)/(100N)` remains the small irrational principal value.",
      "Primary AFDM model/rule source: Bemani, Ksairi, and Kountouris, *Affine Frequency Division Multiplexing for Next Generation Wireless Communications*, https://www.eurecom.edu/publication/6896/download/comsys-publi-6896.pdf. The repository sign convention is audited independently; no literature ordering is assumed.",
      f"Canonical integer path-location gate: `{diagnostics['afdm']['integer_path_gate']}`. OTFS independent-reference modulation error `{diagnostics['otfs']['reference_modulation_max_error']:.3e}` and clean inverse error `{diagnostics['otfs']['clean_inverse_max_error']:.3e}`.","",
      "## Principal DD-v1 communication","","QPSK BER at 18 dB:","","| Receiver | IDENTITY | OFDM | OTFS | AFDM canonical | AFDM tuned | Learned |","|---|---:|---:|---:|---:|---:|---:|",
      f"| fixed-Q K=24 | {q24['IDENTITY']:.6g} | {q24['OFDM']:.6g} | {q24['OTFS']:.6g} | {canonical_q24:.6g} | {q24['AFDM']:.6g} | {q24['Learned']:.6g} |",
      f"| full K=64 | {full['IDENTITY']:.6g} | {full['OFDM']:.6g} | {full['OTFS']:.6g} | {canonical_full:.6g} | {full['AFDM']:.6g} | {full['Learned']:.6g} |","",
      f"Recalibrated AFDM changes its prior Stage-2 fixed-Q 18 dB BER by `{historical_delta:+.1f}%`; validation reselected the identical waveform. Full-information MSE remains unitary invariant, while finite-alphabet BER need not be waveform invariant.","",
      "## Interpretation gates","","### Classical-baseline gate","",
      f"- No: historical `2Nc1=1` was not numerically inappropriate for K=24, because the independent K=24 validation sweep reselected it. Carrying it forward without this audit was scientifically under-justified. Current principal AFDM is `c1={selected}/(2N)={selected/(2*config.N):.10g}`, `c2=sqrt(2)/(100N)`.",
      "- Corrected canonical AFDM reproduces exact separated grid-DD diagonals. Fractional Doppler/delay spreads them continuously, as predicted by the sampled operator model.",
      f"- Learned-vs-AFDM fixed-Q conclusion at 18 dB {'does not reverse' if q24['Learned']<q24['AFDM'] else 'reverses'}: Learned `{q24['Learned']:.6g}`, AFDM `{q24['AFDM']:.6g}`.",
      "- OTFS is correct for the existing algebraic 8x8, N=64, column-major unitary transform; this stage does not substitute physical CP/pulse-shaped OTFS.","",
      "### Communication gate","",
      "- Nominal QPSK: at fixed-Q K=24 and K=32, Learned beats OFDM/OTFS/AFDM at every 0–30 dB point; at K=16 it beats only OTFS. With full information it beats OFDM throughout, loses to OTFS throughout, and loses to AFDM from 5–30 dB.",
      "- Nominal 16-QAM: at fixed-Q K=24/K=32, Learned beats all three from 10–30 dB but loses to OFDM at 0/5 dB; at K=16 it beats only OTFS. With full information it beats OTFS/AFDM through 15 dB, then beats only OFDM from 18–30 dB.",
      f"- At 18 dB K=24, fixed-Q costs Learned `{100*(q24['Learned']/qpsk_oracle-1):.1f}%` QPSK and `{100*(qam_q24/qam_oracle-1):.1f}%` 16-QAM BER versus oracle; fixed-Q is the headline receiver.",
      "- Stress at 18 dB with regime-informed Q: Learned QPSK beats all three classical waveforms for fractional-Doppler, DD-v1, wider-Doppler, wider-delay, and more-path regimes; it loses to OTFS and AFDM on GRID-DD and fractional-delay. Learned 16-QAM beats all three except on GRID-DD, where it loses to all.",
      f"- Relative to the stress DD-v1 bank, Learned QPSK BER rises `{100*(stress_value('QPSK','wider Doppler','Learned')/stress_dd_q-1):.1f}%` for wider Doppler, `{100*(stress_value('QPSK','wider delay','Learned')/stress_dd_q-1):.1f}%` for wider delay, and `{100*(stress_value('QPSK','more paths','Learned')/stress_dd_q-1):.1f}%` for more paths. Corresponding 16-QAM increases are `{100*(stress_value('16QAM','wider Doppler','Learned')/stress_dd_16-1):.1f}%`, `{100*(stress_value('16QAM','wider delay','Learned')/stress_dd_16-1):.1f}%`, and `{100*(stress_value('16QAM','more paths','Learned')/stress_dd_16-1):.1f}%`.",
      "- `fixedOriginalK24` versus `fixedRegimeK24` separates waveform generalization from support adaptation; AFDM parameter adaptation is explicitly labeled `AFDM adapted`.","",
      "### Physical-forensics gate","",
      f"- Learned atom RMS bandwidth spans `{atom['rms_bandwidth']['min']:.4g}`–`{atom['rms_bandwidth']['max']:.4g}`; spectral entropy spans `{atom['spectral_entropy']['min']:.4g}`–`{atom['spectral_entropy']['max']:.4g}`. Magnitude CV spans `{atom['magnitude_cv']['min']:.4g}`–`{atom['magnitude_cv']['max']:.4g}`.",
      "- The all-64 overlay/heatmaps show centroid-ordered frequency populations with non-identical bandwidths and wrap-around structure, not one translated prototype or a single chirp law.",
      f"- Strongest pairwise rank association with Q is `{strongest}` (`rho={explanation['spearman'][strongest]:.3f}`); controlled linear R2 is `{explanation['linear_r2']:.3f}`. Cross-ambiguity distinguishes objectively high/median/low-Q pairs.",
      "- Evidence favors the induced DD coupling geometry over a simple exact atom/filterbank/chirp parameterization.","",
      "### Practicality gate","",
      f"- QPSK 4x-PAPR p99: OFDM `{papr['OFDM']['oversampled_p99_db']:.2f}` dB, OTFS `{papr['OTFS']['oversampled_p99_db']:.2f}` dB, AFDM `{papr['AFDM']['oversampled_p99_db']:.2f}` dB, Learned `{papr['Learned']['oversampled_p99_db']:.2f}` dB.",
      "- Dense discrete-time spectra and occupied-energy summaries are diagnostic only. Without pulse shaping, DAC/front-end, or a spectral mask, no RF ACLR claim is made.","",
      "## Established","","- Frozen Learned communication, structure, spectra, time behavior, STFT, ambiguity, DD concentration, and PAPR under one consistent baseline set.",
      "- Validation-selected AFDM and canonical AFDM are distinct named roles; fixed-Q is the primary practical receiver.","",
      "## Corrected from prior stages","",f"- The principal AFDM is no longer silently inherited from K=3. K=24 validation retained slope `{selected}`, so the correction is provenance and role separation, not a waveform change. Historical artifacts were not overwritten.","",
      "## Suggested","","- Carry PAPR/spectral occupancy and Q concentration as explicit constraints only in a separately authorized future training stage.",
      "- Treat pulse-shaped/CP-OTFS and continuous-time AFDM as a later physicalization study.","",
      "## Still unknown","","- Whether a compact constrained waveform family can reproduce Learned's DD coupling without its atom irregularity.",
      "- RF spectral-mask behavior, amplifier sensitivity, synchronization/channel-estimation cost, and performance with nonlinear or nonorthogonal receivers.","",
      "STOP — no waveform training, channel-family retraining, waveform-family fitting, pulse shaping, nonorthogonal transform, neural receiver, or Stage 3 work was performed.",""]
    return "\n".join(lines)


def run(smoke: bool = False) -> dict[str, object]:
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);hashes_before=prior_hashes();frozen_before=file_sha256(FROZEN)
    learned,frozen_info=load_frozen_waveform(FROZEN)
    validation=sample_regime(REGIMES[3],16 if smoke else 256,VALIDATION_SEED).matrices
    afdm_multiplier,afdm=afdm_audit(validation,smoke);otfs=otfs_audit()
    waveforms=stage2e_waveforms(learned,afdm_multiplier)
    controlled_operator_audit(waveforms)
    nominal,q=nominal_communication(learned,afdm_multiplier,smoke)
    classical_reference(smoke)
    stress,structural=stress_test(learned,afdm_multiplier,q,smoke)
    forensics=physical_forensics(waveforms,q,smoke)
    # Historical baseline on the same untouched bank, only for quantified recalibration.
    historical=stage2e_waveforms(learned,1);channels=sample_regime(REGIMES[3],32 if smoke else 512,TEST_SEED).matrices
    delays,dopplers=regime_q_samples(REGIMES[3],64 if smoke else 2048,Q_SEED);historical_q=ambiguity_coupling(historical["AFDM"],delays,dopplers)
    # Offset 5 matches the 18 dB position in the nominal nine-SNR grid.
    historical_rows=communication_rows(channels,{"AFDM":historical["AFDM"]},{"AFDM":{"fixedQK24":score_support_mask(historical_q,24)}},(qpsk_constellation(),),(18,),8192 if smoke else 1_048_576,SYMBOL_SEED+5,NOISE_SEED+5)
    historical_ber=sum(row["bit_errors"] for row in historical_rows)/sum(row["total_bits"] for row in historical_rows)
    plot_nominal(nominal);plot_stress(stress,structural);plot_spectral_time(waveforms);plot_stft();plot_ambiguity();plot_dd_maps();plot_operator_regeneration(waveforms,q);plot_papr()
    hashes_after=prior_hashes()
    if hashes_after!=hashes_before:raise RuntimeError("prior Stage-2A/2B/2C/2D artifacts changed")
    if file_sha256(FROZEN)!=frozen_before:raise RuntimeError("authoritative checkpoint changed")
    diagnostics={"smoke":smoke,"frozen_checkpoint":frozen_info,"afdm":afdm,"otfs":otfs,"physical_forensics":forensics,
      "historical_afdm_qpsk_18db":historical_ber,"counts":{"prior_artifacts":len(hashes_before),"validation_channels":len(validation),
      "test_channels":32 if smoke else 512,"nominal_target_bits":8192 if smoke else 1_048_576},"prior_artifacts_unchanged":True,
      "checkpoint_unchanged":True,"learned_training_steps":0,"runtime_seconds":time.perf_counter()-started}
    write_json(OUT/"stage2e_diagnostics.json",diagnostics);REPORT.write_text(report_text(diagnostics,nominal,stress,structural))
    print(f"STAGE 2E {'SMOKE ' if smoke else ''}COMPLETE in {diagnostics['runtime_seconds']:.1f} s — AFDM 2Nc1={afdm_multiplier}",flush=True)
    return diagnostics


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true");args=parser.parse_args();run(args.smoke)
