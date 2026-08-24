"""Export every Stage 2E figure's underlying data as copyable JSON."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.channel import COMPLEX_DTYPE, assemble_dd_channel
from src.receiver import effective_operator, rowwise_topk
from src.stage2a import load_frozen_waveform
from src.stage2b import path_resolved_operators, spectral_overlap_matrix
from src.stage2d import score_support_mask
from src.stage2e import REGIMES, sample_regime, stage2e_waveforms


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2e"
DESTINATION = OUT / "machine_readable"
FROZEN = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
HEADLINE = ("Learned", "OFDM", "OTFS", "AFDM")


def _scalar(value: str) -> object:
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(name: str) -> list[dict[str, object]]:
    with (OUT / name).open(newline="") as handle:
        return [{key: _scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def array(value: np.ndarray | torch.Tensor) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value).tolist()


def base(figure: int, title: str, description: str) -> dict[str, object]:
    return {
        "schema": "ddwaveform.stage2e.figure-data.v1",
        "figure": figure,
        "title": title,
        "source_image": f"fig{figure:02d}_*.png",
        "description": description,
        "matrix_orientation": "matrix[row][column] uses output row top-to-bottom and input index left-to-right",
    }


def write_figure(figure: int, slug: str, payload: dict[str, object]) -> Path:
    path = DESTINATION / f"fig{figure:02d}_{slug}.json"
    with path.open("w") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return path


def communication_figures() -> list[tuple[int, str, dict[str, object]]]:
    rows = read_csv("nominal_communication_summary.csv")
    definitions = (
        (1, "qpsk_fixed_q_k24", "QPSK", "fixedQK24", "DD-v1 QPSK BER versus SNR, fixed-Q K=24"),
        (2, "16qam_fixed_q_k24", "16QAM", "fixedQK24", "DD-v1 16-QAM BER versus SNR, fixed-Q K=24"),
        (3, "qpsk_full_information", "QPSK", "full", "DD-v1 QPSK BER versus SNR, full information K=64"),
        (4, "16qam_full_information", "16QAM", "full", "DD-v1 16-QAM BER versus SNR, full information K=64"),
    )
    output = []
    for figure, slug, modulation, mode, title in definitions:
        selected = [row for row in rows if row["modulation"] == modulation and row["receiver_mode"] == mode and row["waveform"] in HEADLINE]
        output.append((figure, slug, {**base(figure, title, "Exact aggregate error counts and BER values used by the plotted curves."),
            "axes": {"x": {"name": "Es/N0", "unit": "dB"}, "y": {"name": "bit_error_rate", "scale": "log"}},
            "modulation": modulation, "receiver_mode": mode, "records": selected}))
    selected = [row for row in rows if row["modulation"] in ("QPSK", "16QAM") and row["snr_db"] == 18
                and row["receiver_mode"] in ("fixedQK16", "fixedQK24", "fixedQK32", "full") and row["waveform"] in HEADLINE]
    output.append((5, "receiver_budget_comparison", {**base(5, "DD-v1 BER across receiver budgets at 18 dB", "Exact BER values for both modulation panels."),
        "axes": {"x": {"name": "receiver_budget_K", "values": [16, 24, 32, 64]}, "y": {"name": "bit_error_rate", "scale": "log"}},
        "records": selected}))
    return output


def stress_figures() -> list[tuple[int, str, dict[str, object]]]:
    communication = read_csv("stress_communication_summary.csv")
    selected = [row for row in communication if row["snr_db"] == 18 and row["receiver_mode"] == "fixedRegimeK24" and row["waveform"] in HEADLINE]
    figure6 = {**base(6, "Frozen-waveform stress BER", "Exact 18 dB regime-informed fixed-Q BER values for QPSK and 16-QAM."),
        "axes": {"x": {"name": "channel_regime"}, "y": {"name": "bit_error_rate", "scale": "log"}}, "records": selected}
    structural = read_csv("stress_structural_summary.csv")
    selected_structural = [row for row in structural if row["mask"] == "regime" and row["waveform"] in HEADLINE]
    figure7 = {**base(7, "Frozen-waveform retained energy", "Mean structural metrics for regime-informed fixed-Q K=24."),
        "axes": {"x": {"name": "channel_regime"}, "y": {"name": "mean_retained_energy"}}, "records": selected_structural}
    return [(6, "frozen_waveform_stress_ber", figure6), (7, "stress_retained_energy", figure7)]


def learned_atom_figures() -> list[tuple[int, str, dict[str, object]]]:
    values = np.load(OUT / "learned_spectral_time_arrays.npz")
    frequency = values["frequencies"]
    spectra = values["spectra_db"]
    order = values["centroid_order"].astype(int)
    figure8 = {**base(8, "All 64 Learned atom spectra", "Exact dense-DTFT magnitude curves with the common unit-norm scale."),
        "axes": {"x": {"name": "normalized_frequency", "unit": "cycles/sample", "values": array(frequency)},
                 "y": {"name": "magnitude", "unit": "dB"}},
        "series": {f"atom_{index:02d}": array(spectra[:, index]) for index in range(64)}}
    figure9 = {**base(9, "All 64 Learned spectral heatmaps", "The same exact spectrum matrix in native and centroid-sorted atom order."),
        "axes": {"x": {"name": "normalized_frequency", "unit": "cycles/sample", "values": array(frequency)},
                 "y": {"name": "atom_index"}},
        "native_atom_order": {"atom_indices": list(range(64)), "magnitude_db": array(spectra.T)},
        "centroid_order": {"atom_indices": array(order), "magnitude_db": array(spectra[:, order].T)}}
    atom_rows = read_csv("learned_atom_metrics.csv")
    by_atom = {int(row["atom"]): row for row in atom_rows}
    figure10 = {**base(10, "Learned atom spectral and time metrics", "Per-atom metrics in circular spectral-centroid order."),
        "x_order_atom_indices": array(order), "records": [by_atom[int(index)] for index in order]}
    figure11 = {**base(11, "All 64 Learned time-domain atoms", "Exact time-domain matrices in centroid order."),
        "axes": {"x": {"name": "time_sample", "values": list(range(64))}, "y": {"name": "centroid_sorted_atom"}},
        "atom_indices": array(order), "magnitude": array(values["magnitude"].T[order]),
        "real_part": array(values["real"].T[order]), "phase_radians": array(values["phase"].T[order]),
        "phase_increment_radians": array(values["phase_increment"].T[order])}
    return [(8, "all64_spectra_overlay", figure8), (9, "all64_spectral_heatmaps", figure9),
            (10, "atom_spectral_time_metrics", figure10), (11, "all64_time_domain", figure11)]


def stft_and_ambiguity_figures() -> list[tuple[int, str, dict[str, object]]]:
    stft = np.load(OUT / "stft_representatives.npz")
    figure12 = {**base(12, "Representative-atom STFTs", "Normalized STFT power matrices; plotted values are 10*log10(max(power,1e-6))."),
        "analysis": {"window": "Hann", "window_length": 16, "hop": 2, "fft_size": 128, "boundary": "circular padding"},
        "normalized_power": {key: array(stft[key]) for key in stft.files}}
    ambiguity = np.load(OUT / "ambiguity_surfaces.npz")
    axes = {"delay": array(ambiguity["delays"]), "doppler": array(ambiguity["dopplers"])}
    figure13 = {**base(13, "Representative auto-ambiguity", "Squared ambiguity magnitude over the controlled delay-Doppler grid."),
        "axes": axes, "squared_magnitude": {key.removeprefix("auto_"): array(ambiguity[key]) for key in ambiguity.files if key.startswith("auto_")}}
    diagnostics = json.loads((OUT / "stage2e_diagnostics.json").read_text())
    figure14 = {**base(14, "Learned cross-ambiguity by Q", "Squared cross-ambiguity for objectively selected low, median, and high-Q pairs."),
        "axes": axes, "atom_pairs": diagnostics["physical_forensics"]["ambiguity_pairs"],
        "squared_magnitude": {key.removeprefix("cross_"): array(ambiguity[key]) for key in ambiguity.files if key.startswith("cross_")}}
    return [(12, "representative_stft", figure12), (13, "auto_ambiguity", figure13), (14, "cross_ambiguity", figure14)]


def dd_map_figure() -> tuple[int, str, dict[str, object]]:
    maps = np.load(OUT / "dd_concentration_maps.npz")
    payload = {**base(15, "Delay-Doppler concentration maps", "Exact top-K concentration and participation maps."),
        "axes": {"delay": array(maps["delays"]), "doppler": array(maps["dopplers"])},
        "maps": {key: array(maps[key]) for key in maps.files if key not in ("delays", "dopplers")}}
    return 15, "dd_concentration_maps", payload


def reconstructed_operator_figures() -> list[tuple[int, str, dict[str, object]]]:
    diagnostics = json.loads((OUT / "stage2e_diagnostics.json").read_text())
    learned, _ = load_frozen_waveform(FROZEN)
    waveforms = stage2e_waveforms(learned, int(diagnostics["afdm"]["selected_multiplier_2Nc1"]))
    cases = (("grid", (0.0, 2.0, 4.0), (-1.0, 0.0, 1.0)),
             ("fractional_doppler", (0.0, 2.0, 4.0), (-0.65, 0.15, 0.55)),
             ("full_off_grid", (0.3, 2.4, 3.7), (-0.65, 0.15, 0.55)))
    progression = {}
    for name, delays, dopplers in cases:
        physical = assemble_dd_channel(torch.tensor(delays), torch.tensor(dopplers),
                                       torch.tensor([1, 0.7j, -0.5], dtype=COMPLEX_DTYPE))
        progression[name] = {waveform: array(torch.abs(effective_operator(physical, waveforms[waveform]))) for waveform in ("OFDM", "OTFS", "AFDM", "Learned")}
    figure16 = {**base(16, "Grid to off-grid operator progression", "Deterministically reconstructed exact magnitude matrices."),
        "cases": progression, "gains": [{"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.7}, {"real": -0.5, "imag": 0.0}]}

    channels = sample_regime(REGIMES[3], 128, 11170).matrices
    energy = {name: torch.mean(torch.abs(effective_operator(channels, waveforms[name])) ** 2, dim=0).numpy()
              for name in ("OFDM", "OTFS", "AFDM", "Learned")}
    figure17 = {**base(17, "Ensemble energy and fixed support", "Exact 128-channel ensemble energy and deterministic top-24 masks."),
        "channel_regime": "DD-v1", "channel_count": 128, "seed": 11170,
        "waveforms": {name: {"ensemble_energy": array(value), "fixed_top24_mask": array(score_support_mask(value, 24).to(torch.int8))}
                      for name, value in energy.items()}}

    delays = torch.tensor([0.0, 1.0, 2.0, 4.0]); dopplers = torch.tensor([-1.0, 0.0, 1.0, 0.35])
    gains = torch.tensor([1.0, 0.7j, -0.5, 0.35j], dtype=COMPLEX_DTYPE)
    four_path = {}
    for name in ("OFDM", "OTFS", "AFDM", "Learned"):
        resolved = path_resolved_operators(waveforms[name], delays, dopplers, gains)
        four_path[name] = {"path_magnitude": array(torch.abs(resolved)), "summed_path_magnitude": array(torch.sum(torch.abs(resolved), dim=0))}
    figure18 = {**base(18, "Reference four-path operator comparison", "Exact per-path and summed magnitude matrices."),
        "paths": [{"delay": float(delay), "doppler": float(doppler), "gain_real": float(gain.real), "gain_imag": float(gain.imag)}
                  for delay, doppler, gain in zip(delays, dopplers, gains)], "waveforms": four_path}

    learned_effective = effective_operator(channels, waveforms["Learned"])
    retained = torch.sum(torch.abs(rowwise_topk(learned_effective, 24)) ** 2, dim=(-2, -1)) / torch.sum(torch.abs(learned_effective) ** 2, dim=(-2, -1))
    sorted_indices = torch.argsort(retained)
    chosen = {"easy": int(sorted_indices[-13]), "median": int(sorted_indices[len(sorted_indices) // 2]), "hard": int(sorted_indices[12])}
    gallery = {difficulty: {name: array(torch.abs(effective_operator(channels[index], waveforms[name])))
                            for name in ("OFDM", "OTFS", "AFDM", "Learned")} for difficulty, index in chosen.items()}
    figure20 = {**base(20, "Common-channel easy, median, hard gallery", "Exact operator magnitude matrices and deterministic selected channel indices."),
        "channel_seed": 11170, "selection_metric": "Learned top-24 retained energy", "channel_indices": chosen,
        "learned_retained_energy": {difficulty: float(retained[index]) for difficulty, index in chosen.items()}, "matrices": gallery}

    q = np.load(OUT / "nominal_q_matrices.npz")["Learned"]
    figure21 = {**base(21, "Frozen Learned S, Q, C comparison", "Exact matrices before the per-panel maximum normalization used for display."),
        "spectral_overlap_S": array(spectral_overlap_matrix(waveforms["Learned"])), "dd_ambiguity_Q": array(q),
        "empirical_coupling_C": array(energy["Learned"]), "channel_count_for_C": 128, "channel_seed_for_C": 11170}
    return [(16, "operator_progression", figure16), (17, "ensemble_energy_fixed_support", figure17),
            (18, "reference_four_path", figure18), (20, "easy_median_hard_gallery", figure20),
            (21, "s_q_c_comparison", figure21)]


def papr_figure() -> tuple[int, str, dict[str, object]]:
    curves = {}
    for modulation in ("QPSK", "16QAM"):
        curves[modulation] = {}
        for name in HEADLINE:
            values = np.load(OUT / f"papr_{modulation}_{name}.npz")["oversampled"]
            sorted_values = np.sort(values)
            ccdf = 1 - (np.arange(len(sorted_values)) + 0.5) / len(sorted_values)
            curves[modulation][name] = {"papr_db": array(sorted_values), "ccdf": array(ccdf)}
    payload = {**base(19, "PAPR complementary CDF", "Exact sorted 4x-oversampled PAPR samples and plotting-position CCDF values."),
        "oversampling_factor": 4, "curves": curves, "summary": read_csv("papr_summary.csv")}
    return 19, "papr_ccdf", payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_summary(values: list[Any] | np.ndarray) -> dict[str, object]:
    matrix = np.asarray(values, dtype=float)
    flat = np.abs(matrix).reshape(-1)
    energy = flat ** 2
    total = max(float(energy.sum()), 1e-300)
    ordered = np.sort(energy)[::-1]
    result: dict[str, object] = {
        "shape": list(matrix.shape), "minimum": float(matrix.min()), "maximum": float(matrix.max()),
        "mean": float(matrix.mean()), "standard_deviation": float(matrix.std()),
        "top_1pct_energy_fraction": float(ordered[: max(1, int(np.ceil(0.01 * len(ordered))))].sum() / total),
        "top_10pct_energy_fraction": float(ordered[: max(1, int(np.ceil(0.10 * len(ordered))))].sum() / total),
        "effective_participation": float(total ** 2 / max(float(np.sum(energy ** 2)), 1e-300)),
    }
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        n = matrix.shape[0]
        profile = np.asarray([np.sum(matrix[np.arange(n), (np.arange(n) + offset) % n] ** 2) for offset in range(n)])
        profile /= max(float(profile.sum()), 1e-300)
        result["dominant_column_minus_row_offset"] = int(np.argmax(profile))
        result["dominant_offset_energy_fraction"] = float(profile.max())
        result["top3_entry_energy_fraction"] = float(np.sum(np.sort(matrix ** 2, axis=1)[:, -3:]) / total)
        result["top24_entry_energy_fraction"] = float(np.sum(np.sort(matrix ** 2, axis=1)[:, -24:]) / total)
    return result


def surface_summary(values: list[Any], x: list[float], y: list[float]) -> dict[str, object]:
    matrix = np.asarray(values, dtype=float)
    weight = np.maximum(matrix, 0.0)
    total = max(float(weight.sum()), 1e-300)
    probability = weight / total
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    x_mean = float(np.sum(probability * x_values[None, :]))
    y_mean = float(np.sum(probability * y_values[:, None]))
    peak = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return {
        **matrix_summary(matrix),
        "peak": {"x": float(x_values[peak[1]]), "y": float(y_values[peak[0]]), "value": float(matrix[peak])},
        "weighted_centroid": {"x": x_mean, "y": y_mean},
        "weighted_rms_spread": {
            "x": float(np.sqrt(np.sum(probability * (x_values[None, :] - x_mean) ** 2))),
            "y": float(np.sqrt(np.sum(probability * (y_values[:, None] - y_mean) ** 2))),
        },
    }


def concise_export(definitions: list[tuple[int, str, dict[str, object]]]) -> dict[str, object]:
    by_figure = {figure: payload for figure, _, payload in definitions}
    atom_rows = read_csv("learned_atom_metrics.csv")
    concise: list[dict[str, object]] = []
    for figure in range(1, 22):
        source = by_figure[figure]
        item: dict[str, object] = {"figure": figure, "title": source["title"]}
        if figure <= 7:
            item["records"] = source["records"]
            item["reduction"] = "none; all plotted aggregate points retained"
        elif figure == 8:
            metrics = ("centroid", "rms_bandwidth", "spectral_entropy", "top4_spectral_energy",
                       "top8_spectral_energy", "top16_spectral_energy")
            item["per_atom_spectral_metrics"] = [{key: row[key] for key in ("atom",) + metrics} for row in atom_rows]
            item["global_ranges"] = {key: {"min": min(float(row[key]) for row in atom_rows),
                                                  "max": max(float(row[key]) for row in atom_rows),
                                                  "mean": float(np.mean([float(row[key]) for row in atom_rows]))} for key in metrics}
            item["reduction"] = "16,384-bin curves replaced by exact per-atom centroid, bandwidth, entropy, and top-bin energy metrics"
        elif figure == 9:
            item["native_atom_order"] = list(range(64))
            item["spectral_centroid_order"] = source["centroid_order"]["atom_indices"]
            item["per_atom_centroid"] = [{"atom": row["atom"], "centroid": row["centroid"]} for row in atom_rows]
            item["reduction"] = "dense heatmaps replaced by native/sorted orders and exact centroids; detailed metrics are in figure 8"
        elif figure == 10:
            item["centroid_order_atom_indices"] = source["x_order_atom_indices"]
            item["records"] = source["records"]
            item["reduction"] = "none; all 64 plotted per-atom metric records retained"
        elif figure == 11:
            keys = ("atom", "magnitude_cv", "peak_magnitude", "maximum_amplitude",
                    "peak_to_average_power_db", "time_centroid", "time_spread")
            item["per_atom_time_metrics"] = [{key: row[key] for key in keys} for row in atom_rows]
            item["centroid_order_atom_indices"] = source["atom_indices"]
            item["reduction"] = "four dense time-domain heatmaps replaced by exact per-atom amplitude, PAPR, centroid, and spread metrics"
        elif figure == 12:
            summaries = {}
            for key, values in source["normalized_power"].items():
                matrix = np.asarray(values, dtype=float)
                summaries[key] = {**matrix_summary(matrix),
                                  "frequency_marginal": array(matrix.sum(axis=1) / max(float(matrix.sum()), 1e-300)),
                                  "time_marginal": array(matrix.sum(axis=0) / max(float(matrix.sum()), 1e-300))}
            item["analysis"] = source["analysis"]
            item["representative_atom_summaries"] = summaries
            item["reduction"] = "STFT matrices replaced by concentration summaries plus exact frequency/time marginals"
        elif figure in (13, 14):
            axes = source["axes"]
            data_key = "squared_magnitude"
            item["atom_pairs"] = source.get("atom_pairs")
            item["surface_summaries"] = {key: surface_summary(values, axes["delay"], axes["doppler"])
                                         for key, values in source[data_key].items()}
            item["reduction"] = "41x41 ambiguity surfaces replaced by peaks, centroids, RMS spreads, and concentration statistics"
        elif figure == 15:
            item["map_summaries"] = {key: matrix_summary(values) for key, values in source["maps"].items()}
            item["delay_range"] = [min(source["axes"]["delay"]), max(source["axes"]["delay"])]
            item["doppler_range"] = [min(source["axes"]["doppler"]), max(source["axes"]["doppler"])]
            item["reduction"] = "17x17 DD maps replaced by per-waveform/metric range and concentration summaries"
        elif figure == 16:
            item["cases"] = {case: {name: matrix_summary(values) for name, values in matrices.items()}
                             for case, matrices in source["cases"].items()}
            item["reduction"] = "operator matrices replaced by dominant offsets and energy/concentration summaries"
        elif figure == 17:
            item["channel_regime"] = source["channel_regime"]
            item["channel_count"] = source["channel_count"]
            item["seed"] = source["seed"]
            item["waveforms"] = {name: {"ensemble_energy": matrix_summary(data["ensemble_energy"]),
                                                   "fixed_top24_density": float(np.mean(data["fixed_top24_mask"]))}
                                 for name, data in source["waveforms"].items()}
            item["reduction"] = "ensemble matrices/masks replaced by energy concentration, dominant offsets, and mask density"
        elif figure == 18:
            item["paths"] = source["paths"]
            item["waveforms"] = {}
            for name, data in source["waveforms"].items():
                paths = np.asarray(data["path_magnitude"], dtype=float)
                normalized = paths.reshape(len(paths), -1)
                normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-300)
                overlaps = [float(np.dot(normalized[i], normalized[j])) for i in range(len(paths)) for j in range(i + 1, len(paths))]
                item["waveforms"][name] = {"summed_operator": matrix_summary(data["summed_path_magnitude"]),
                                                     "mean_pairwise_path_overlap": float(np.mean(overlaps)),
                                                     "maximum_pairwise_path_overlap": float(np.max(overlaps))}
            item["reduction"] = "per-path matrices replaced by summed-operator concentration and inter-path overlaps"
        elif figure == 19:
            item["summary"] = source["summary"]
            item["reduction"] = "10,000-point CCDF curves replaced by exact mean, p95, and p99 PAPR summaries"
        elif figure == 20:
            item["channel_indices"] = source["channel_indices"]
            item["learned_retained_energy"] = source["learned_retained_energy"]
            item["matrices"] = {difficulty: {name: matrix_summary(values) for name, values in matrices.items()}
                                for difficulty, matrices in source["matrices"].items()}
            item["reduction"] = "12 operator matrices replaced by dominant offsets and concentration summaries"
        else:
            matrices = {"spectral_overlap_S": np.asarray(source["spectral_overlap_S"]),
                        "dd_ambiguity_Q": np.asarray(source["dd_ambiguity_Q"]),
                        "empirical_coupling_C": np.asarray(source["empirical_coupling_C"])}
            upper = np.triu_indices(64, 1)
            item["matrix_summaries"] = {key: matrix_summary(value) for key, value in matrices.items()}
            item["pairwise_pearson"] = {f"{left}_vs_{right}": float(np.corrcoef(matrices[left][upper], matrices[right][upper])[0, 1])
                                         for left, right in (("spectral_overlap_S", "dd_ambiguity_Q"),
                                                             ("spectral_overlap_S", "empirical_coupling_C"),
                                                             ("dd_ambiguity_Q", "empirical_coupling_C"))}
            item["physical_explanation"] = json.loads((OUT / "q_physical_explanation.json").read_text())
            item["reduction"] = "three 64x64 matrices replaced by concentration summaries, pairwise correlations, and explanatory regression"
        concise.append(item)
    diagnostics = json.loads((OUT / "stage2e_diagnostics.json").read_text())
    return {
        "schema": "ddwaveform.stage2e.concise-figure-summary.v1",
        "purpose": "Compact machine-readable scientific content of all Stage 2E figures",
        "figure_count": len(concise),
        "dense_data_policy": "Communication points retained exactly; dense visual matrices summarized with declared DSP statistics",
        "key_provenance": {"checkpoint_sha256": diagnostics["frozen_checkpoint"]["sha256"],
                           "afdm_selected_2Nc1": diagnostics["afdm"]["selected_multiplier_2Nc1"],
                           "learned_training_steps": diagnostics["learned_training_steps"]},
        "figures": concise,
    }


def export() -> dict[str, object]:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    definitions = communication_figures() + stress_figures() + learned_atom_figures()
    definitions += stft_and_ambiguity_figures() + [dd_map_figure()]
    definitions += reconstructed_operator_figures() + [papr_figure()]
    definitions.sort(key=lambda item: item[0])
    if [item[0] for item in definitions] != list(range(1, 22)):
        raise RuntimeError("export definitions must cover figures 1 through 21 exactly once")
    entries = []
    combined_figures = []
    for figure, slug, payload in definitions:
        path = write_figure(figure, slug, payload)
        combined_figures.append(payload)
        entries.append({"figure": figure, "file": path.name, "bytes": path.stat().st_size,
                        "sha256": sha256(path), "title": payload["title"]})
    combined = {
        "schema": "ddwaveform.stage2e.all-figure-data.v1",
        "format": "UTF-8 JSON; self-contained and directly parseable",
        "precision": "Exact stored float64/derived plotting values; no pixel digitization or downsampling",
        "figure_count": len(combined_figures),
        "figures": combined_figures,
    }
    combined_path = DESTINATION / "all_figures.json"
    with combined_path.open("w") as handle:
        json.dump(combined, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    concise = concise_export(definitions)
    concise_path = DESTINATION / "concise_all_figures.json"
    with concise_path.open("w") as handle:
        json.dump(concise, handle, indent=2, allow_nan=False)
        handle.write("\n")
    manifest = {
        "schema": "ddwaveform.stage2e.figure-manifest.v1",
        "format": "UTF-8 JSON; each file is self-contained and can be copied directly into a JSON parser",
        "precision": "Exact stored float64/derived plotting values serialized as JSON numbers; no pixel digitization or downsampling",
        "figure_count": len(entries),
        "combined_file": {"file": combined_path.name, "bytes": combined_path.stat().st_size,
                          "sha256": sha256(combined_path)},
        "concise_file": {"file": concise_path.name, "bytes": concise_path.stat().st_size,
                         "sha256": sha256(concise_path)},
        "figures": entries,
    }
    with (DESTINATION / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2); handle.write("\n")
    return manifest


if __name__ == "__main__":
    result = export()
    print(f"EXPORTED {result['figure_count']} FIGURES TO {DESTINATION}")
