"""Run Stage 2B frozen waveform and effective-operator forensics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from src import config
from src.channel import COMPLEX_DTYPE, assemble_dd_channel, sample_dd_v1
from src.receiver import effective_operator, finite_budget_receiver, rowwise_topk
from src.stage2a import FIGURE_WAVEFORMS, load_frozen_waveform, qualification_waveforms
from src.stage2b import (
    PATH_COLORS,
    afdm_path_audit,
    community_summary,
    coupling_matrix,
    dense_atom_spectra,
    difficulty_quantile_indices,
    ensemble_operator_statistics,
    path_resolved_operators,
    path_rgb,
    operator_offset_metrics,
    positive_shift_coordinates,
    representative_atom_indices,
    spectral_centroids,
    spectral_overlap_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2b"
CHECKPOINT = ROOT / "outputs" / "stage1a2" / "checkpoints" / "tail_from_ofdm_best.pt"
NAMES = ("OFDM", "OTFS", "AFDM", "Learned")
PERCENTILE_LABELS = ("p10", "p30", "p50", "p70", "p90")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def physical_channel(delays: np.ndarray, dopplers: np.ndarray, gains: np.ndarray) -> torch.Tensor:
    return assemble_dd_channel(
        torch.as_tensor(delays, dtype=torch.float64),
        torch.as_tensor(dopplers, dtype=torch.float64),
        torch.as_tensor(gains, dtype=COMPLEX_DTYPE),
    )


def operator_grid(channel: torch.Tensor, waveforms: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: torch.abs(effective_operator(channel, waveforms[name])).detach().cpu().numpy()
        for name in NAMES
    }


def orientation_audit() -> dict[str, object]:
    rows, columns = positive_shift_coordinates(config.N, 1)
    matrix = np.zeros((config.N, config.N))
    matrix[rows, columns] = 1.0
    fig, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.imshow(matrix, origin="upper", cmap="binary", vmin=0, vmax=1)
    axis.set_xlabel("Input column (increases right)"); axis.set_ylabel("Output row (increases down)")
    axis.set_title("Orientation sanity: positive circular shift by one")
    fig.tight_layout(); fig.savefig(OUT / "fig00_orientation_sanity.png", dpi=180); plt.close(fig)
    return {
        "imshow_origin": "upper",
        "matrix_modified_for_display": False,
        "positive_shift_rule": "row=(column+1) mod N",
        "visual_direction": "one pixel below the main diagonal, with wrap at top-right",
        "first_coordinate": [int(rows[0]), int(columns[0])],
        "wrap_coordinate": [int(rows[-1]), int(columns[-1])],
    }


def plot_atoms(waveforms: dict[str, torch.Tensor]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    selected: dict[str, np.ndarray] = {}
    rows = []
    maximum = max(float(torch.max(torch.abs(waveforms[name]))) for name in NAMES)
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 14.0), sharex=True)
    samples = np.arange(config.N)
    for row_index, name in enumerate(NAMES):
        indices, centroids = representative_atom_indices(waveforms[name])
        selected[name] = indices
        values = waveforms[name][:, indices].detach().resolve_conj().cpu().numpy()
        for local, (index, label) in enumerate(zip(indices, PERCENTILE_LABELS)):
            axes[row_index, 0].plot(samples, values[:, local].real, label=f"{label}: {int(index)}")
            axes[row_index, 1].plot(samples, np.abs(values[:, local]), label=f"{label}: {int(index)}")
            rows.append({"waveform": name, "percentile": label, "column": int(index), "spectral_centroid": float(centroids[index])})
        noun = "basis atoms" if name == "Learned" else "atoms"
        axes[row_index, 0].set_title(f"{name} {noun}: real part")
        axes[row_index, 1].set_title(f"{name} {noun}: magnitude")
        axes[row_index, 0].set_ylim(-1.05 * maximum, 1.05 * maximum)
        axes[row_index, 1].set_ylim(0, 1.05 * maximum)
        axes[row_index, 0].set_ylabel("Amplitude")
        axes[row_index, 1].legend(fontsize=7, ncol=2)
        for axis in axes[row_index]: axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Discrete sample index"); axes[-1, 1].set_xlabel("Discrete sample index")
    fig.suptitle("Deterministically selected transmitted basis waveforms", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig01_representative_time_atoms.png", dpi=180); plt.close(fig)
    return rows, selected


def plot_dense_spectra(waveforms: dict[str, torch.Tensor], selected: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 13.0), sharex=True, sharey=True)
    for axis, name in zip(axes, NAMES):
        frequencies, spectra = dense_atom_spectra(waveforms[name], selected[name])
        for local, (index, label) in enumerate(zip(selected[name], PERCENTILE_LABELS)):
            axis.plot(frequencies, spectra[:, local], label=f"{label}: {int(index)}")
        axis.set_title(f"{name} dense DTFT")
        axis.set_ylabel("Magnitude (dB)"); axis.set_ylim(-80, 2)
        axis.grid(True, alpha=0.25); axis.legend(fontsize=8, ncol=5)
    axes[-1].set_xlabel(r"Normalized frequency $f/F_s$")
    fig.suptitle("Dense spectra of representative finite-duration atoms", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig02_dense_atom_spectra.png", dpi=180); plt.close(fig)


def plot_operator_grid(
    arrays: dict[str, np.ndarray], filename: str, title: str, path_context: str = ""
) -> None:
    maximum = max(float(value.max()) for value in arrays.values())
    fig = plt.figure(figsize=(15.0, 4.1))
    grid = fig.add_gridspec(1, 5, width_ratios=(1, 1, 1, 1, 0.05), left=0.05, right=0.94, bottom=0.12, top=0.86, wspace=0.12)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    for axis, name in zip(axes, NAMES):
        image = axis.imshow(arrays[name], origin="upper", cmap="magma", vmin=0, vmax=maximum)
        axis.set_title(name); axis.set_xlabel("Input column")
    axes[0].set_ylabel("Output row")
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 4]), label="Magnitude")
    fig.suptitle(title + (f"\n{path_context}" if path_context else ""), y=1.02)
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_aggregate_and_paths(
    delays: torch.Tensor,
    dopplers: torch.Tensor,
    gains: torch.Tensor,
    waveforms: dict[str, torch.Tensor],
    filename: str,
    title: str,
) -> None:
    path_sets = {name: path_resolved_operators(waveforms[name], delays, dopplers, gains) for name in NAMES}
    aggregates = {name: torch.abs(torch.sum(path_sets[name], dim=0)).numpy() for name in NAMES}
    aggregate_max = max(float(value.max()) for value in aggregates.values())
    path_max = max(float(torch.max(torch.abs(value))) for value in path_sets.values())
    fig = plt.figure(figsize=(9.4, 14.2))
    grid = fig.add_gridspec(4, 3, width_ratios=(1, 1, 0.06), left=0.08, right=0.92, bottom=0.05, top=0.96, hspace=0.25, wspace=0.18)
    axes = np.asarray([[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(4)])
    for row, name in enumerate(NAMES):
        image = axes[row, 0].imshow(aggregates[name], origin="upper", cmap="magma", vmin=0, vmax=aggregate_max)
        axes[row, 1].imshow(path_rgb(path_sets[name], path_max), origin="upper")
        axes[row, 0].set_title(f"{name}: aggregate |A|")
        axes[row, 1].set_title(f"{name}: path-resolved")
        axes[row, 0].set_ylabel("Output row")
    axes[-1, 0].set_xlabel("Input column"); axes[-1, 1].set_xlabel("Input column")
    fig.colorbar(image, cax=fig.add_subplot(grid[:, 2]), label="Aggregate magnitude")
    handles = [plt.Line2D([0], [0], color=color, lw=5, label=f"path {index}") for index, color in enumerate(PATH_COLORS[: len(delays)])]
    axes[0, 1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle(title, y=0.995)
    fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def plot_single_path_gallery(waveforms: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    cases = ((0, 0), (2, 0), (0, 0.5), (2, 0.5), (2.5, 0), (2.5, 0.5))
    fig, axes = plt.subplots(len(cases), 4, figsize=(12.5, 17.0), sharex=True, sharey=True)
    rows = []
    for row, (delay, doppler) in enumerate(cases):
        arrays = operator_grid(physical_channel(np.asarray([delay]), np.asarray([doppler]), np.asarray([1 + 0j])), waveforms)
        maximum = max(float(value.max()) for value in arrays.values())
        for column, name in enumerate(NAMES):
            axes[row, column].imshow(arrays[name], origin="upper", cmap="magma", vmin=0, vmax=maximum)
            axes[row, column].set_title(f"{name}\n(tau,nu)=({delay},{doppler})", fontsize=9)
            rows.append({"delay": delay, "doppler": doppler, "waveform": name, "maximum_magnitude": float(arrays[name].max())})
    fig.suptitle("Controlled single-path operator gallery", y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.04, top=0.96, hspace=0.3, wspace=0.12)
    fig.savefig(OUT / "fig05_single_path_gallery.png", dpi=180); plt.close(fig)
    return rows


def plot_matrix_quartet(
    matrices: dict[str, np.ndarray], filename: str, title: str, cmap: str, vmin: float, vmax: float, label: str
) -> None:
    fig = plt.figure(figsize=(15.2, 4.1))
    grid = fig.add_gridspec(1, 5, width_ratios=(1, 1, 1, 1, 0.05), left=0.05, right=0.94, bottom=0.12, top=0.86, wspace=0.12)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    for axis, name in zip(axes, NAMES):
        image = axis.imshow(matrices[name], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name); axis.set_xlabel("Input column")
    axes[0].set_ylabel("Output row"); fig.colorbar(image, cax=fig.add_subplot(grid[0, 4]), label=label)
    fig.suptitle(title, y=0.99)
    fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def plot_learned_association(
    overlap: np.ndarray, energy: np.ndarray, support: np.ndarray, order: np.ndarray, labels: np.ndarray
) -> None:
    coupling = coupling_matrix(energy)
    matrices = (overlap, coupling, support)
    reordered = tuple(matrix[np.ix_(order, order)] for matrix in matrices)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.4), sharex=True, sharey=True)
    titles = ("Spectral overlap", "Normalized operator coupling", "Top-24 support probability")
    for column, title in enumerate(titles):
        axes[0, column].imshow(matrices[column], origin="upper", cmap="viridis", vmin=0, vmax=1)
        axes[1, column].imshow(reordered[column], origin="upper", cmap="viridis", vmin=0, vmax=1)
        axes[0, column].set_title(f"Original order: {title}")
        axes[1, column].set_title(f"Centroid order: {title}")
    ordered_labels = labels[order]
    boundaries = np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1
    for axis in axes[1]:
        for boundary in boundaries:
            axis.axhline(boundary - 0.5, color="white", lw=0.7)
            axis.axvline(boundary - 0.5, color="white", lw=0.7)
    axes[0, 0].set_ylabel("Original output row"); axes[1, 0].set_ylabel("Centroid-sorted output row")
    for axis in axes[-1]: axis.set_xlabel("Input column")
    fig.suptitle("Learned spectral grouping and DD-v1 operator association", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig11_learned_spectral_operator_association.png", dpi=180); plt.close(fig)


def run_stage2b() -> dict[str, object]:
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    learned, checkpoint = load_frozen_waveform(CHECKPOINT); before = learned.clone()
    all_waveforms = qualification_waveforms(learned)
    waveforms = {name: all_waveforms[name] for name in NAMES}
    bank = sample_dd_v1(config.N_STAGE1A2_EVAL_CHANNELS, config.STAGE1A2_EVAL_SEED)
    orientation = orientation_audit()

    # Hard AFDM gate before later interpretation.
    integer_paths = ((0, -1), (2, 0), (4, 1), (1, 2), (3, -2))
    afdm_rows = afdm_path_audit(waveforms["AFDM"], integer_paths, config.AFDM_C1)
    if not all(bool(row["matches"]) for row in afdm_rows):
        raise RuntimeError(f"AFDM integer path-center audit failed: {afdm_rows}")
    afdm_parameters = {
        "c1": config.AFDM_C1,
        "c2": config.AFDM_C2,
        "xi_equals_2N_c1": round(2 * config.N * config.AFDM_C1),
        "guard_parameter": 0,
        "c1_construction": "validation-selected K=3 baseline; no maximum delay/Doppler support assumption",
        "project_ddv1_delay_max": config.DELAY_MAX,
        "project_ddv1_doppler_max_abs": config.DOPPLER_MAX,
        "integer_support_law": "column_minus_row = (xi*delay + Doppler) mod N",
    }
    write_csv(OUT / "afdm_integer_path_audit.csv", afdm_rows)
    write_json(OUT / "afdm_parameters.json", afdm_parameters)
    print("AFDM integer-center gate passed", flush=True)

    atom_rows, selected = plot_atoms(waveforms); plot_dense_spectra(waveforms, selected)
    write_csv(OUT / "representative_atom_selection.csv", atom_rows)

    median_index, difficulty = difficulty_quantile_indices(bank.matrices, (0.50,))
    median_index = median_index[0]
    plot_operator_grid(operator_grid(bank.matrices[median_index], waveforms), "fig03_corrected_representative_operator.png", "Frozen DD-v1 median-control channel: corrected top-origin orientation")
    normalized_gains = bank.pre_normalization_gains[median_index] / bank.normalization_scales[median_index]
    plot_aggregate_and_paths(bank.delays[median_index], bank.dopplers[median_index], normalized_gains, waveforms, "fig04_representative_path_resolved.png", "Frozen DD-v1 median-control channel: aggregate and physical-path contributions")

    single_rows = plot_single_path_gallery(waveforms); write_csv(OUT / "single_path_gallery_metrics.csv", single_rows)

    gains3 = np.asarray((1.0 + 0j, 0.72 * np.exp(0.7j), 0.55 * np.exp(-1.1j)))
    progression = {
        "GRID-DD": (np.asarray((0.0, 2.0, 4.0)), np.asarray((-1.0, 0.0, 1.0))),
        "FRACTIONAL-DOPPLER": (np.asarray((0.0, 2.0, 4.0)), np.asarray((-0.65, 0.20, 0.72))),
        "FULL-OFFGRID": (np.asarray((0.35, 2.25, 3.70)), np.asarray((-0.65, 0.20, 0.72))),
    }
    progression_rows = []
    afdm_progression_rows = []
    for index, (regime, (delays, dopplers)) in enumerate(progression.items()):
        plot_aggregate_and_paths(torch.tensor(delays), torch.tensor(dopplers), torch.tensor(gains3), waveforms, f"fig06{chr(97+index)}_progression_{regime.lower().replace('-', '_')}.png", f"Controlled multipath progression: {regime}")
        for path, (delay, doppler, gain) in enumerate(zip(delays, dopplers, gains3)):
            progression_rows.append({"regime": regime, "path": path, "delay": delay, "doppler": doppler, "gain_real": gain.real, "gain_imag": gain.imag, "gain_magnitude": abs(gain)})
            metrics = operator_offset_metrics(path_resolved_operators(waveforms["AFDM"], torch.tensor([delay]), torch.tensor([doppler]), torch.tensor([gain]))[0])
            afdm_progression_rows.append({
                "regime": regime,
                "path": path,
                "delay": delay,
                "doppler": doppler,
                "continuous_predicted_offset": float((round(2 * config.N * config.AFDM_C1) * delay + doppler) % config.N),
                **metrics,
            })
    write_csv(OUT / "controlled_progression_paths.csv", progression_rows)
    write_csv(OUT / "afdm_progression_path_centers.csv", afdm_progression_rows)

    gains4 = np.asarray((1.0 + 0j, 0.82 * np.exp(0.4j), 0.64 * np.exp(-0.8j), 0.48 * np.exp(1.2j)))
    delays4 = np.asarray((0.0, 1.0, 3.0, 4.0)); grid4 = np.asarray((-2.0, 0.0, 1.0, 2.0)); frac4 = np.asarray((-1.6, 0.25, 1.35, 1.7))
    plot_aggregate_and_paths(torch.tensor(delays4), torch.tensor(grid4), torch.tensor(gains4), waveforms, "fig07a_reference_style_grid_four_path.png", "Reference-style four-path illustration: integer delay and Doppler")
    plot_aggregate_and_paths(torch.tensor(delays4), torch.tensor(frac4), torch.tensor(gains4), waveforms, "fig07b_reference_style_fractional_doppler_four_path.png", "Reference-style four-path illustration: same delays/gains, fractional Doppler")
    write_csv(OUT / "reference_style_four_path.csv", [{"path": i, "delay": delays4[i], "grid_doppler": grid4[i], "fractional_doppler": frac4[i], "gain_real": gains4[i].real, "gain_imag": gains4[i].imag, "gain_magnitude": abs(gains4[i])} for i in range(4)])

    gallery_indices, difficulty = difficulty_quantile_indices(bank.matrices)
    gallery_arrays = []
    gallery_rows = []
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 9.7), sharex=True, sharey=True)
    for row, (quantile, channel_index) in enumerate(zip((0.10, 0.50, 0.90), gallery_indices)):
        arrays = operator_grid(bank.matrices[channel_index], waveforms); gallery_arrays.append(arrays)
        maximum = max(float(value.max()) for value in arrays.values())
        for column, name in enumerate(NAMES):
            axes[row, column].imshow(arrays[name], origin="upper", cmap="magma", vmin=0, vmax=maximum)
            axes[row, column].set_title(f"{name}: q{int(100*quantile)}, channel {channel_index}", fontsize=9)
        normalized = bank.pre_normalization_gains[channel_index] / bank.normalization_scales[channel_index]
        for path in range(config.P):
            gallery_rows.append({"difficulty_quantile": quantile, "channel_index": channel_index, "identity_difficulty": difficulty[channel_index], "path": path, "delay": float(bank.delays[channel_index, path]), "doppler": float(bank.dopplers[channel_index, path]), "gain_magnitude": float(torch.abs(normalized[path]))})
    axes[1, 0].set_ylabel("Output row"); [axis.set_xlabel("Input column") for axis in axes[-1]]
    fig.suptitle("Frozen DD-v1 easy / median / hard gallery", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig08_ddv1_easy_median_hard.png", dpi=180); plt.close(fig)
    write_csv(OUT / "ddv1_easy_median_hard_channels.csv", gallery_rows)

    print("Computing full-bank ensemble statistics", flush=True)
    energy, support = ensemble_operator_statistics(bank.matrices, waveforms)
    np.savez(OUT / "ensemble_operator_energy.npz", **energy); np.savez(OUT / "top24_support_probability.npz", **support)
    global_max = max(float(value.max()) for value in energy.values())
    energy_db = {name: np.maximum(10 * np.log10(np.maximum(value / global_max, 1e-300)), -50) for name, value in energy.items()}
    plot_matrix_quartet(energy_db, "fig09_ensemble_average_operator_energy.png", "Full frozen bank: ensemble-average operator energy (common global normalization)", "magma", -50, 0, "Energy (dB relative to global maximum)")
    plot_matrix_quartet(support, "fig10_top24_support_probability.png", "Full frozen bank: row-wise top-24 support probability", "viridis", 0, 1, "Probability")

    overlap = spectral_overlap_matrix(learned)
    centroids = spectral_centroids(learned)
    labels, communities = community_summary(energy["Learned"], overlap, centroids)
    same_group = labels[:, None] == labels[None, :]
    learned_support = support["Learned"]
    support_distribution = learned_support / np.maximum(learned_support.sum(axis=1, keepdims=True), 1e-300)
    communities.update({
        "within_group_support_fraction": float(learned_support[same_group].sum() / learned_support.sum()),
        "within_group_support_probability_mean": float(learned_support[same_group].mean()),
        "cross_group_support_probability_mean": float(learned_support[~same_group].mean()),
        "mean_row_support_entropy_normalized": float(
            np.mean(-np.sum(support_distribution * np.log(support_distribution + 1e-300), axis=1) / np.log(config.N))
        ),
    })
    order = np.asarray(communities["centroid_order"])
    plot_learned_association(overlap, energy["Learned"], support["Learned"], order, labels)
    np.savez(OUT / "learned_association_matrices.npz", spectral_overlap=overlap, operator_energy=energy["Learned"], normalized_coupling=coupling_matrix(energy["Learned"]), support_probability=support["Learned"], community_labels=labels, centroid_order=order)
    write_json(OUT / "learned_community_summary.json", communities)

    if not torch.equal(before, learned): raise RuntimeError("frozen Learned tensor mutated")
    diagnostics = {
        "checkpoint": checkpoint,
        "orientation": orientation,
        "afdm": {"parameters": afdm_parameters, "integer_path_gate_passed": True, "audit_rows": afdm_rows},
        "afdm_progression_path_centers": afdm_progression_rows,
        "atom_selection": atom_rows,
        "representative_channel_index": median_index,
        "difficulty_gallery_indices": gallery_indices,
        "community": communities,
        "ensemble_total_energy": {name: float(value.sum()) for name, value in energy.items()},
        "mean_support_count_per_row": {name: float(value.sum(axis=1).mean()) for name, value in support.items()},
        "checkpoint_unchanged": True,
        "waveform_tensor_unchanged": True,
        "channel_count": len(bank.matrices),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage2b_diagnostics.json", diagnostics)
    print(f"STAGE 2B COMPLETE in {diagnostics['runtime_seconds']:.1f} s", flush=True)
    return diagnostics


if __name__ == "__main__":
    run_stage2b()
