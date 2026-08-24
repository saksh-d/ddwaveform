"""Reusable scientific core for Stage Pub-1A.

The module contains no optimizer or training entry point.  All support masks
are fixed from independent channel-class statistics before test realizations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from src import config
from src.audit import expected_afdm_column_offset, normalize_operators
from src.channel import COMPLEX_DTYPE, REAL_DTYPE, ChannelBatch, assemble_dd_channel, make_generator, noise_variance
from src.metrics import Constellation, qam16_constellation, qam64_constellation, qpsk_constellation
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from src.stage2b import operator_offset_metrics, path_resolved_operators
from src.stage2d import apply_support, score_support_mask
from src.stage2e import (
    Regime,
    afdm_design_record,
    afdm_validation_rows,
    atom_spectral_time_metrics,
    communication_rows,
    papr_samples,
    regime_q_samples,
    sample_regime,
    select_afdm_multiplier,
    stft_power,
)
from src.stage2d import ambiguity_coupling
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


CHECKPOINT_SHA256 = "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc"
AFDM_ADMISSIBLE_MULTIPLIERS = (3, 5)
K_SWEEP = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64)
SELECTED_ATOMS = (0, 8, 16, 24, 32, 40, 48, 56)
PRIMARY_WAVEFORMS = ("Learned", "OTFS", "AFDM")
SAVED_WAVEFORMS = ("Learned", "OTFS", "AFDM", "OFDM")
CONSTELLATIONS = (qpsk_constellation(), qam16_constellation(), qam64_constellation())
SNRS = {
    "QPSK": (0, 5, 10, 12, 15, 18, 20, 25, 30),
    "16QAM": (0, 5, 10, 12, 15, 18, 20, 25, 30),
    "64QAM": (5, 10, 15, 18, 20, 22, 25, 28, 30, 32, 35),
}


@dataclass(frozen=True)
class RegimeDefinition:
    key: str
    label: str
    delay_kind: str
    delay_max: float
    doppler_kind: str
    doppler_max: float
    paths: int = 3

    def stage2e(self) -> Regime:
        return Regime(self.label, self.delay_kind, self.delay_max, self.doppler_kind, self.doppler_max, self.paths)


REGIMES = (
    RegimeDefinition("grid_dd", "GRID-DD", "integer", 4.0, "integer", 1.0),
    RegimeDefinition("fractional_doppler", "FRACTIONAL-DOPPLER", "integer", 4.0, "continuous", 0.75),
    RegimeDefinition("fractional_delay", "FRACTIONAL-DELAY", "continuous", 4.0, "continuous", 0.20),
    RegimeDefinition("dd_v1", "FULL OFF-GRID / DD-v1", "continuous", 4.0, "continuous", 0.75),
    RegimeDefinition("wider_doppler", "WIDER-DOPPLER", "continuous", 4.0, "continuous", 1.50),
    RegimeDefinition("wider_delay", "WIDER-DELAY", "continuous", 8.0, "continuous", 0.75),
    RegimeDefinition("more_paths", "MORE-PATHS", "continuous", 4.0, "continuous", 0.75, 5),
)
REGIME_BY_KEY = {value.key: value for value in REGIMES}


def waveform_set(learned: torch.Tensor, afdm_multiplier: int) -> dict[str, torch.Tensor]:
    return {
        "Learned": learned,
        "OTFS": otfs_waveform(),
        "AFDM": afdm_waveform(c1=afdm_multiplier / (2 * config.N), c2=config.AFDM_C2),
        "OFDM": ofdm_waveform(),
    }


def fixed_statistical_masks(
    waveforms: dict[str, torch.Tensor], regime: RegimeDefinition, ks: tuple[int, ...], samples: int, seed: int
) -> tuple[dict[str, dict[str, torch.Tensor | None]], dict[str, np.ndarray]]:
    delays, dopplers = regime_q_samples(regime.stage2e(), samples, seed)
    scores = {name: ambiguity_coupling(waveform, delays, dopplers) for name, waveform in waveforms.items()}
    masks = {
        name: {f"fixedStatK{k}": score_support_mask(scores[name], k) for k in ks if k < config.N}
        for name in waveforms
    }
    for name in waveforms:
        if config.N in ks:
            masks[name]["full"] = None
    return masks, scores


def aggregate_rows(rows: list[dict[str, object]], extra: tuple[str, ...] = ()) -> list[dict[str, object]]:
    keys = extra + ("modulation", "snr_db", "waveform", "receiver_mode", "k")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for values, selected in grouped.items():
        bit_errors = sum(int(row["bit_errors"]) for row in selected)
        total_bits = sum(int(row["total_bits"]) for row in selected)
        symbol_errors = sum(int(row["symbol_errors"]) for row in selected)
        total_symbols = sum(int(row["total_symbols"]) for row in selected)
        result.append({
            **dict(zip(keys, values)),
            "bit_errors": bit_errors,
            "total_bits": total_bits,
            "ber": bit_errors / total_bits,
            "symbol_errors": symbol_errors,
            "total_symbols": total_symbols,
            "ser": symbol_errors / total_symbols,
            "n_channels": len(selected),
        })
    return result


def analytic_mse_rows(
    channels: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    masks: dict[str, dict[str, torch.Tensor | None]],
    snrs: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    with torch.no_grad():
        for name, waveform in waveforms.items():
            actual = effective_operator(channels, waveform)
            for mode, mask in masks[name].items():
                if mode == "full":
                    retained, k = actual, config.N
                elif mode.startswith("oracle"):
                    k = int(mode.removeprefix("oracle")); retained = rowwise_topk(actual, k)
                else:
                    k = int(mode.removeprefix("fixedStatK")); retained = apply_support(actual, mask)
                for snr in snrs:
                    equalizer = mismatched_lmmse_equalizer(retained, noise_variance(snr))
                    values = analytic_communication_mse(actual, equalizer, noise_variance(snr))
                    rows.append({"waveform": name, "receiver_mode": mode, "k": k, "snr_db": snr,
                                 "analytic_mse": float(values.mean()),
                                 "aggregate_error_check": float(torch.mean(torch.linalg.matrix_norm(equalizer @ actual - identity, dim=(-2, -1)) ** 2 / config.N))})
    return rows


def afdm_structural_audit(validation_channels: torch.Tensor) -> dict[str, object]:
    """Derive admissible slopes, verify four-path separation, then select by validation MSE."""
    design = afdm_design_record()
    validation = afdm_validation_rows(validation_channels, (12, 15, 18, 20), AFDM_ADMISSIBLE_MULTIPLIERS, 24, config.AFDM_C2)
    selected, summary = select_afdm_multiplier(validation)
    anchors_delay = (0, 1, 2, 3)
    anchors_doppler = (-1, 1, 0, -1)
    fractional_doppler = (-0.65, 0.55, 0.15, -0.35)
    fractional_delay = (0.25, 1.35, 2.45, 3.20)
    gains = torch.tensor([1.0, 0.75j, -0.55, 0.4j], dtype=COMPLEX_DTYPE)
    candidates = []
    for multiplier in AFDM_ADMISSIBLE_MULTIPLIERS:
        waveform = afdm_waveform(c1=multiplier / (2 * config.N), c2=config.AFDM_C2)
        resolved = path_resolved_operators(waveform, torch.tensor(anchors_delay), torch.tensor(anchors_doppler), gains)
        records = []
        observed = []
        for path_index, (delay, doppler) in enumerate(zip(anchors_delay, anchors_doppler)):
            metrics = operator_offset_metrics(resolved[path_index])
            predicted = expected_afdm_column_offset(delay, doppler, config.N, multiplier / (2 * config.N))
            observed.append(int(metrics["dominant_offset"]))
            records.append({"path": path_index, "delay": delay, "doppler": doppler,
                            "predicted_offset": int(predicted), "observed_offset": int(metrics["dominant_offset"]),
                            "nearest_separation": 0, "dominant_offset_energy_fraction": float(metrics["dominant_offset_energy_fraction"])})
        for index, value in enumerate(observed):
            distances = [min((value - other) % config.N, (other - value) % config.N) for j, other in enumerate(observed) if j != index]
            records[index]["nearest_separation"] = int(min(distances))
        grid_distinct = len(set(observed)) == len(observed)
        exact = all(row["predicted_offset"] == row["observed_offset"] for row in records)
        cases = {}
        for case, delays, dopplers in (
            ("grid", anchors_delay, anchors_doppler),
            ("fractional_doppler", anchors_delay, fractional_doppler),
            ("full_offgrid", fractional_delay, fractional_doppler),
        ):
            physical = assemble_dd_channel(torch.tensor(delays), torch.tensor(dopplers), gains)
            operator = effective_operator(physical, waveform)
            energy = torch.abs(operator) ** 2
            cases[case] = {"dominant_offset": int(operator_offset_metrics(operator)["dominant_offset"]),
                           "top4_energy_fraction": float(torch.sum(torch.abs(rowwise_topk(operator, 4)) ** 2) / torch.sum(energy)),
                           "top24_energy_fraction": float(torch.sum(torch.abs(rowwise_topk(operator, 24)) ** 2) / torch.sum(energy))}
        candidates.append({"multiplier_2Nc1": multiplier, "c1": multiplier / (2 * config.N),
                           "grid_distinct": grid_distinct, "predicted_equals_observed": exact,
                           "paths": records, "cases": cases})
    hard_gate = all(row["grid_distinct"] and row["predicted_equals_observed"] for row in candidates)
    if not hard_gate:
        raise RuntimeError("Pub-1A AFDM hard gate failed: admissible candidates did not produce distinct predicted grid paths")
    return {"design": design, "admissible_multipliers_2Nc1": list(AFDM_ADMISSIBLE_MULTIPLIERS),
            "validation_summary": summary, "selected_multiplier_2Nc1": selected,
            "selected_c1": selected / (2 * config.N), "c2": config.AFDM_C2,
            "hard_gate": hard_gate, "candidates": candidates,
            "selection_rule": "minimum mean K=24 analytic MSE on validation bank; no BER/test/Learned input"}


def matched_fractionality_channels(count: int, seed: int, gamma: float) -> ChannelBatch:
    generator = make_generator(seed)
    paths = config.P
    anchor_tau = torch.randint(0, 5, (count, paths), generator=generator).to(REAL_DTYPE)
    anchor_nu = torch.randint(-1, 2, (count, paths), generator=generator).to(REAL_DTYPE)
    delta_tau = (torch.rand((count, paths), dtype=REAL_DTYPE, generator=generator) - 0.5) * 0.9
    delta_nu = (torch.rand((count, paths), dtype=REAL_DTYPE, generator=generator) - 0.5) * 0.9
    tau = torch.clamp(anchor_tau + gamma * delta_tau, 0.0, 4.0)
    nu = torch.clamp(anchor_nu + gamma * delta_nu, -1.0, 1.0)
    scale = math.sqrt(1 / (2 * paths))
    gains = scale * (torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator) + 1j * torch.randn((count, paths), dtype=REAL_DTYPE, generator=generator))
    gains = gains.to(COMPLEX_DTYPE)
    raw = assemble_dd_channel(tau, nu, gains)
    matrices, scales = normalize_operators(raw, config.N)
    return ChannelBatch(matrices, tau, nu, gains, scales)


def practicality_data(waveforms: dict[str, torch.Tensor], blocks: int, seed: int) -> dict[str, object]:
    output: dict[str, object] = {"oversampling": 4, "psd_fft_size": 4096,
                                "oob_convention": "energy outside normalized frequency [-0.4,0.4)"}
    papr_rows = []
    psd: dict[str, np.ndarray] = {}
    for modulation_index, constellation in enumerate(CONSTELLATIONS):
        bits = torch.randint(0, 2, (blocks, config.N, constellation.bits_per_symbol), generator=make_generator(seed + modulation_index))
        from src.metrics import constellation_encode
        symbols, _ = constellation_encode(bits, constellation)
        for name, waveform in waveforms.items():
            values = papr_samples(waveform, symbols, oversample=4)
            papr_rows.append({"modulation": constellation.name, "waveform": name, "mean_db": float(values.mean()),
                              "median_db": float(np.median(values)), "p95_db": float(np.quantile(values, .95)),
                              "p99_db": float(np.quantile(values, .99)), "max_db": float(values.max()),
                              "samples_db": values.tolist()})
            transmitted = torch.einsum("ij,bj->bi", waveform, symbols).numpy()
            spectrum = np.fft.fftshift(np.fft.fft(transmitted, 4096, axis=-1), axes=-1)
            power = np.mean(np.abs(spectrum) ** 2, axis=0)
            power /= power.sum()
            frequency = np.fft.fftshift(np.fft.fftfreq(4096))
            key = f"{constellation.name}_{name}"
            psd[key] = power
    frequency = np.fft.fftshift(np.fft.fftfreq(4096))
    psd_rows = []
    for key, power in psd.items():
        modulation, name = key.split("_", 1)
        order = np.argsort(power)[::-1]
        chosen = order[: int(np.searchsorted(np.cumsum(power[order]), .99)) + 1]
        occupied = float(frequency[chosen].max() - frequency[chosen].min()) if len(chosen) else 0.0
        oob = float(power[np.abs(frequency) >= .4].sum())
        psd_rows.append({"modulation": modulation, "waveform": name, "occupied_bandwidth_99": occupied, "oob_energy_fraction": oob})
    output.update(papr=papr_rows, psd_summary=psd_rows, frequency=frequency.tolist(),
                  psd={key: value.tolist() for key, value in psd.items()})
    return output


def configuration_record() -> dict[str, object]:
    return {"n": config.N, "learned_training_steps": 0, "k_sweep": list(K_SWEEP),
            "selected_atoms": list(SELECTED_ATOMS), "snr_db": {key: list(value) for key, value in SNRS.items()},
            "regimes": [asdict(value) for value in REGIMES],
            "k24_history": {"retained_fraction": 24 / 64, "headroom_percent": {"K16": 31.3, "K24": 20.4, "K32": 11.2}}}
