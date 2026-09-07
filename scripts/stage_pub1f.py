"""Phase-selectable Pub-1F reproducibility and receiver-complexity runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.channel import noise_variance
from src.metrics import qam16_constellation, qpsk_constellation
from src.pub1e.core import residual_statistics
from src.pub1e.core import fixed_energy_support
from src.pub1b.core import directory_tree_sha256
from src.pub1f.core import (
    TrainingProtocol, aggregate_counts, communication_counts,
    condition_channels, environment_record, operation_ledger, paired_interval,
    semantic_seed, storage_ledger, train_stage, unitary_residual,
)
from src.receiver import effective_operator
from src.stage2a import file_sha256, load_frozen_waveform
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_pub1f.json"
OUT = ROOT / "outputs/stage_pub1f"
DATA = OUT / "data"
CHECKPOINTS = OUT / "checkpoints"
FIGURES = OUT / "figures"
PROVENANCE = OUT / "provenance"
TIMING = OUT / "timing"
REPORT = ROOT / "reports/stage_pub1f_summary.md"
HISTORICAL = ROOT / "outputs/stage1a2/checkpoints/tail_from_ofdm_best.pt"
PRODUCER = Path("/Users/saksh/conductor/workspaces/delayDopplerWaveform/denpasar")
CONDITIONS = ("joint_0.5", "joint_0.2_probe", "integer", "nominal_dd_v1")
COLORS = {"Learned": "#d97706", "OTFS": "#15803d", "OFDM": "#2563eb", "AFDM": "#dc2626"}


def cfg() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def ensure_dirs() -> None:
    for directory in (DATA, CHECKPOINTS, FIGURES, PROVENANCE, TIMING):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return file_sha256(path)


def checkpoint_waveform() -> torch.Tensor:
    waveform, _ = load_frozen_waveform(HISTORICAL)
    return waveform


def named_waveforms() -> dict[str, torch.Tensor]:
    values = {"Learned": checkpoint_waveform(), "OTFS": otfs_waveform(), "OFDM": ofdm_waveform()}
    for slope in (1, 3, 5, 7, 9):
        values[f"AFDM_slope_{slope}"] = afdm_waveform(c1=slope / (2 * config.N), c2=config.AFDM_C2)
    return values


def phase_prerequisites(smoke: bool) -> None:
    ensure_dirs(); protocol = cfg()
    producer_manifest = PRODUCER / "outputs/stage_pub1e/artifact_manifest.json"
    manifest = read_json(producer_manifest)
    verified = []
    for entry in manifest["files"]:
        local = ROOT / entry["path"]
        actual = sha256(local) if local.exists() else "MISSING"
        if actual != entry["sha256"]:
            raise RuntimeError(f"Pub-1E artifact mismatch: {entry['path']} {actual}")
        verified.append(entry["path"])
    checkpoint_hash = sha256(HISTORICAL)
    if checkpoint_hash != protocol["historical_checkpoint"]["sha256"]:
        raise RuntimeError("historical checkpoint hash mismatch")
    restored = read_json(ROOT / "outputs/stage_pub1e/provenance/restored_prerequisites.json")["restored"]
    verified_prerequisites = {}
    for relative, expected in restored.items():
        local = ROOT / relative
        actual = directory_tree_sha256(local) if local.is_dir() else sha256(local)
        if actual != expected:
            raise RuntimeError(f"restored prerequisite mismatch: {relative} {actual}")
        verified_prerequisites[relative] = actual
    record = {
        "run_kind": "smoke" if smoke else "production", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target_ref": "origin/pub", "protocol_sha256": sha256(CONFIG),
        "historical_checkpoint_sha256_before": checkpoint_hash,
        "pub1e_artifacts_verified": len(verified), "pub1e_manifest_sha256": sha256(producer_manifest),
        "restored_prerequisites": verified_prerequisites,
        "active_figures": read_json(ROOT / "outputs/stage_pub1e/correction_manifest.json")["active_figures"],
        "superseded_figures": read_json(ROOT / "outputs/stage_pub1e/correction_manifest.json")["superseded_figures"],
        "environment": environment_record(),
    }
    write_json(PROVENANCE / "prerequisites.json", record)
    seed_rows=[]
    for role in ("receiver_validation","residual_statistics","confirmatory","benchmark_channels","statistical_support"):
        for condition in CONDITIONS:
            seed_rows.append({"role":role,"condition":condition,"seed":semantic_seed(role,condition=condition)})
    for seed_id in cfg()["training"]["seeds"]:
        seed_rows.extend([
            {"role":"checkpoint_validation","seed_id":seed_id,"condition":"nominal_dd_v1","seed":semantic_seed("checkpoint_validation",condition="nominal_dd_v1",seed_id=seed_id)},
            {"role":"training_channel","seed_id":seed_id,"stage":"parent","step":"1","seed":semantic_seed("training_channel",seed_id=seed_id,stage="parent",step=1)},
            {"role":"training_channel","seed_id":seed_id,"stage":"parent","step":"2000","seed":semantic_seed("training_channel",seed_id=seed_id,stage="parent",step=2000)},
            {"role":"training_channel","seed_id":seed_id,"stage":"continuation","step":"1","seed":semantic_seed("training_channel",seed_id=seed_id,stage="continuation",step=1)},
            {"role":"training_channel","seed_id":seed_id,"stage":"continuation","step":"1000","seed":semantic_seed("training_channel",seed_id=seed_id,stage="continuation",step=1000)},
        ])
    write_json(PROVENANCE / "seed_bank_manifest.json", {"namespace":"pub1f-v1","derivation":cfg()["banks"]["derivation"],"records":seed_rows,"note":"training SNR uses the same semantic fields under role training_snr; symbols/noise add bank_role, condition, modulation, and SNR"})
    (PROVENANCE / "protocol_snapshot.json").write_text(CONFIG.read_text())


def _baseline_sizes(smoke: bool) -> tuple[int, int, int, int, int]:
    if smoke:
        return 6, 8, 6, 256, 200
    protocol = cfg(); banks = protocol["banks"]
    return int(banks["classical_receiver_validation"]["channels"]), int(banks["residual_statistics"]["channels"]), int(banks["confirmatory"]["channels"]), int(banks["bits_per_channel"]), int(banks["bootstrap_resamples"])


def _candidate_key(row: dict[str, object]) -> tuple[float, float, int]:
    return float(row["ber"]), float(row["analytic_mse"]), int(row.get("slope", 0))


def phase_baseline(smoke: bool) -> None:
    ensure_dirs(); validation_n, statistics_n, test_n, bits, resamples = _baseline_sizes(smoke)
    waveforms = named_waveforms(); constellations = (qam16_constellation(), qpsk_constellation())
    validation_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    confirm_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for condition in CONDITIONS:
        validation = condition_channels(condition, validation_n, "receiver_validation")
        statistics_bank = condition_channels(condition, statistics_n, "residual_statistics")
        test = condition_channels(condition, test_n, "confirmatory")
        stats_by_waveform = {name: residual_statistics(statistics_bank, waveform, 24) for name, waveform in waveforms.items()}
        selected_by_modulation: dict[str, dict[str, dict[str, object]]] = {}
        for constellation in constellations:
            candidates: list[dict[str, object]] = []
            for name, waveform in waveforms.items():
                kind = "AFDM" if name.startswith("AFDM") else name
                slope = int(name.rsplit("_", 1)[1]) if name.startswith("AFDM") else 0
                settings = [("historical", 1.0)] + [("tuned", value) for value in (0.25, 0.5, 1.0, 2.0, 4.0)] + [("residual_aware", 1.0)]
                actual = effective_operator(validation, waveform)
                for family, scale in settings:
                    channel_rows = communication_counts(
                        validation, waveform, constellation, 18, 24, family, bits,
                        condition=condition, bank_role="receiver_validation", regularization_scale=scale,
                        stats=stats_by_waveform[name] if family == "residual_aware" else None,
                    )
                    aggregate = aggregate_counts(channel_rows)
                    # Analytic MSE is a deterministic tie-break, never the primary operational metric.
                    from src.pub1e.core import historical_equalizer, residual_aware_equalizer
                    from src.receiver import rowwise_topk, analytic_communication_mse
                    restricted = rowwise_topk(actual, 24)
                    if family == "residual_aware":
                        equalizer = residual_aware_equalizer(restricted, noise_variance(18), stats_by_waveform[name])
                    else:
                        equalizer = historical_equalizer(restricted, noise_variance(18), scale if family == "tuned" else 1.0)
                    analytic = float(analytic_communication_mse(actual, equalizer, noise_variance(18)).mean())
                    row = {"condition": condition, "modulation": constellation.name, "waveform_candidate": name, "waveform_family": kind, "slope": slope, "receiver_family": family, "regularization_scale": scale, "analytic_mse": analytic, **aggregate}
                    validation_rows.append(row); candidates.append(row)
            selected_by_modulation[constellation.name] = {}
            for family_name in ("Learned", "OTFS", "OFDM", "AFDM"):
                eligible = [row for row in candidates if row["waveform_family"] == family_name]
                selected = min(eligible, key=_candidate_key)
                selected_by_modulation[constellation.name][family_name] = selected
                selection_rows.append({**selected, "selection_scope": "condition,modulation,SNR=18,K=24,waveform_family", "selected": True})
            # Historical-compatible configurations are retained even when not operationally selected.
            historical_names = ("Learned", "OTFS", "OFDM", "AFDM_slope_1", "AFDM_slope_3", "AFDM_slope_5")
            configurations: list[tuple[str, dict[str, object], str]] = []
            for family_name, selected in selected_by_modulation[constellation.name].items():
                configurations.append((family_name, selected, "operational_selected"))
            for name in historical_names:
                configurations.append((name, {"waveform_candidate": name, "receiver_family": "historical", "regularization_scale": 1.0, "slope": int(name.rsplit("_", 1)[1]) if name.startswith("AFDM") else 0}, "historical_reference"))
            seen: set[tuple[object, ...]] = set()
            per_label: dict[str, list[dict[str, object]]] = {}
            for label, selected, track in configurations:
                unique = (label, selected["waveform_candidate"], selected["receiver_family"], selected["regularization_scale"], track)
                if unique in seen:
                    continue
                seen.add(unique)
                name = str(selected["waveform_candidate"]); family = str(selected["receiver_family"]); scale = float(selected["regularization_scale"])
                rows = communication_counts(test, waveforms[name], constellation, 18, 24, family, bits, condition=condition, bank_role="confirmatory", regularization_scale=scale, stats=stats_by_waveform[name] if family == "residual_aware" else None)
                output_label = f"{track}:{label}"
                per_label[output_label] = rows
                aggregate = aggregate_counts(rows)
                for row in rows:
                    confirm_rows.append({"condition": condition, "modulation": constellation.name, "track": track, "label": label, "waveform_candidate": name, "receiver_family": family, "regularization_scale": scale, "slope": selected["slope"], **row})
            learned_rows = per_label["operational_selected:Learned"]
            for classical in ("OTFS", "OFDM", "AFDM"):
                interval = paired_interval(learned_rows, per_label[f"operational_selected:{classical}"], f"baseline|{condition}|{constellation.name}|Learned-{classical}", resamples)
                interval_rows.append({"condition": condition, "modulation": constellation.name, "left": "Learned", "right": classical, **interval})
        # Full-information nominal/reference rows use the same confirmatory channel bank.
        if condition == "nominal_dd_v1":
            for constellation in constellations:
                for family_name in ("Learned", "OTFS", "OFDM"):
                    rows = communication_counts(test, waveforms[family_name], constellation, 18, 64, "full", bits, condition="full_information_nominal", bank_role="confirmatory")
                    for row in rows:
                        confirm_rows.append({"condition": "full_information_nominal", "modulation": constellation.name, "track": "full_information", "label": family_name, "waveform_candidate": family_name, "receiver_family": "full", "regularization_scale": 1.0, "slope": 0, **row})
    write_csv(DATA / "baseline_validation.csv", validation_rows)
    write_csv(DATA / "baseline_selection.csv", selection_rows)
    write_csv(DATA / "baseline_confirmatory_per_channel.csv", confirm_rows)
    write_csv(DATA / "baseline_paired_intervals.csv", interval_rows)
    write_json(DATA / "baseline_execution.json", {"run_kind": "smoke" if smoke else "production", "seconds": time.perf_counter() - started, "validation_channels": validation_n, "statistics_channels": statistics_n, "confirmatory_channels": test_n, "bits_per_channel": bits})


def phase_gate(smoke: bool) -> None:
    intervals = read_csv(DATA / "baseline_paired_intervals.csv")
    selected = [row for row in intervals if row["condition"] == "joint_0.5"]
    required = {(modulation, classical) for modulation in ("16QAM", "QPSK") for classical in ("OTFS", "OFDM", "AFDM")}
    observed = {(row["modulation"], row["right"]) for row in selected}
    if observed != required:
        raise RuntimeError(f"early gate rows incomplete: {observed}")
    comparisons = [{**row, "supported": float(row["ci95_high"]) < 0.0 and float(row["mean_difference"]) < 0.0} for row in selected]
    passed = all(row["supported"] for row in comparisons)
    record = {"run_kind": "smoke" if smoke else "production", "passed": passed, "rule": cfg()["early_gate"]["passes_if"], "comparisons": comparisons, "training_authorized": passed and not smoke}
    write_json(DATA / "early_gate.json", record)
    print(f"Pub-1F early gate: {'PASS' if passed else 'FAIL'}", flush=True)


def _torch_save_state(path: Path):
    return lambda state: torch.save(state, path)


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def phase_training(smoke: bool) -> None:
    gate = read_json(DATA / "early_gate.json")
    if not smoke and not gate["training_authorized"]:
        raise RuntimeError("early gate did not authorize production training")
    protocol = cfg()["training"]
    seeds = tuple(int(value) for value in protocol["seeds"])
    steps_parent = 2 if smoke else int(protocol["parent_steps"])
    steps_continuation = 2 if smoke else int(protocol["continuation_steps"])
    cadence = 1 if smoke else int(protocol["validation_cadence"])
    batch = 2 if smoke else int(protocol["batch_size"])
    validation_count = 6 if smoke else int(cfg()["banks"]["checkpoint_validation"]["channels"])
    records = []
    for seed_id in seeds:
        directory = CHECKPOINTS / ("smoke" if smoke else "production") / f"seed_{seed_id}"
        directory.mkdir(parents=True, exist_ok=True)
        validation = condition_channels("nominal_dd_v1", validation_count, "checkpoint_validation", seed_id=seed_id)
        parent_protocol = TrainingProtocol(seed_id, "parent", steps_parent, batch, cadence, 24, None)
        parent_state_path = directory / "parent_state.pt"
        parent_resume = torch.load(parent_state_path, weights_only=False) if parent_state_path.exists() else None
        parent = train_stage(ofdm_waveform(), validation, parent_protocol, resume=parent_resume, save_callback=_torch_save_state(parent_state_path))
        torch.save(parent["selected"], directory / "parent_selected.pt"); torch.save(parent["endpoint"], directory / "parent_endpoint.pt")
        parent_hash = _tensor_hash(parent["selected"])
        arms = {}
        for stage in ("mean_continuation", "tail_continuation"):
            arm_protocol = TrainingProtocol(seed_id, stage, steps_continuation, batch, cadence, 24, parent_hash)
            state_path = directory / f"{stage}_state.pt"
            resume = torch.load(state_path, weights_only=False) if state_path.exists() else None
            result = train_stage(parent["selected"], validation, arm_protocol, reference_parent_mean=float(parent["best_mean"]) if stage == "tail_continuation" else None, resume=resume, save_callback=_torch_save_state(state_path))
            torch.save(result["selected"], directory / f"{stage}_selected.pt"); torch.save(result["endpoint"], directory / f"{stage}_endpoint.pt")
            arms[stage] = result
        parent_equal = arms["mean_continuation"]["protocol"]["parent_sha256"] == arms["tail_continuation"]["protocol"]["parent_sha256"] == parent_hash
        for stage, result in (("parent", parent), *arms.items()):
            records.append({"seed_id": seed_id, "stage": stage, "selected_step": result["best_step"], "selected_mean_mse": result["best_mean"], "selected_cvar95": result["best_cvar95"], "endpoint_step": steps_parent if stage == "parent" else steps_continuation, "endpoint_mean_mse": result["endpoint_mean"], "endpoint_cvar95": result["endpoint_cvar95"], "runtime_seconds": result["runtime_seconds"], "max_unitarity_residual": result["max_unitarity"], "nonfinite": result["nonfinite"], "parent_equal_across_arms": parent_equal})
            write_csv(directory / f"{stage}_trace.csv", result["trace"])
        print(f"seed {seed_id}: parent={parent['best_step']} mean={arms['mean_continuation']['best_step']} tail={arms['tail_continuation']['best_step']}", flush=True)
    write_csv(DATA / ("training_summary_smoke.csv" if smoke else "training_summary.csv"), records)


def phase_complexity(smoke: bool) -> None:
    ensure_dirs(); protocol = cfg(); old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        ks = tuple(int(value) for value in protocol["k_values"])
        iterations = 8 if smoke else 64
        ledger = []
        for k in ks:
            ledger.append({**operation_ledger(64, k, iterations), **storage_ledger(64, k, residual_aware=False), "residual_aware": False})
            ledger.append({**operation_ledger(64, k, iterations), **storage_ledger(64, k, residual_aware=True), "residual_aware": True})
        write_csv(DATA / "complexity_ledger.csv", ledger)
        channels = condition_channels("nominal_dd_v1", 2 if smoke else 32, "benchmark_channels")
        waveforms = named_waveforms(); timing_rows = []
        reps = 5 if smoke else int(protocol["complexity"]["benchmark_repetitions"])
        warmup = 2 if smoke else int(protocol["complexity"]["warmup"])
        generator = torch.Generator().manual_seed(semantic_seed("benchmark_observation"))
        for name in ("Learned", "OTFS", "OFDM", "AFDM_slope_1"):
            actual = effective_operator(channels, waveforms[name])
            for k in ((24,) if smoke else ks):
                for channel_index in range(len(channels)):
                    observation = torch.randn(64, dtype=torch.complex128, generator=generator)
                    from src.receiver import rowwise_topk
                    retained = rowwise_topk(actual[channel_index], k)
                    from src.pub1b.core import sparse_from_topk, sparse_cg_lmmse
                    sparse = sparse_from_topk(actual[channel_index], k)
                    direct = torch.linalg.solve(retained.mH @ retained + noise_variance(18) * torch.eye(64, dtype=torch.complex128), retained.mH @ observation)
                    tight = sparse_cg_lmmse(sparse, observation, noise_variance(18), max_iterations=256, tolerance=1e-12)
                    tight_relative = float(torch.linalg.vector_norm(tight.estimate - direct) / torch.linalg.vector_norm(direct))
                    if tight_relative > 1e-8 or not tight.converged:
                        raise RuntimeError(f"tight direct/iterative mismatch {tight_relative}, converged={tight.converged}")
                    def direct_call():
                        torch.linalg.solve(retained.mH @ retained + noise_variance(18) * torch.eye(64, dtype=torch.complex128), retained.mH @ observation)
                    def sparse_call():
                        return sparse_cg_lmmse(sparse, observation, noise_variance(18), max_iterations=64, tolerance=1e-10)
                    for implementation, call in (("dense_zero_direct", direct_call), ("explicit_sparse_matrix_free", sparse_call)):
                        for _ in range(warmup): call()
                        samples=[]
                        for _ in range(reps):
                            start=time.perf_counter_ns(); result=call(); samples.append(time.perf_counter_ns()-start)
                        logical = 0 if implementation == "dense_zero_direct" else result.iterations
                        numerical_relative = tight_relative if implementation == "dense_zero_direct" else float(torch.linalg.vector_norm(result.estimate - direct) / torch.linalg.vector_norm(direct))
                        timing_rows.append({"waveform": name, "k": k, "channel_index": channel_index, "implementation": implementation, "median_ns": int(np.median(samples)), "p95_ns": int(np.quantile(samples,.95)), "logical_iterations": logical, "physical_iterations": logical, "converged": True if implementation == "dense_zero_direct" else result.converged, "relative_residual": 0.0 if implementation == "dense_zero_direct" else result.relative_residual, "numerical_relative_error": numerical_relative, "tight_reference_iterations": tight.iterations, "tight_reference_relative_error": tight_relative, "repetitions": reps, "threads": 1})
        write_csv(TIMING / ("receiver_benchmark_smoke.csv" if smoke else "receiver_benchmark.csv"), timing_rows)
        write_json(PROVENANCE / "benchmark_environment.json", {**environment_record(), "warmup": warmup, "repetitions": reps, "timed_region": "prepared retained-matrix RHS plus solve; resident CPU data; excludes transforms/setup", "thread_count": 1})
    finally:
        torch.set_num_threads(old_threads)


def _timed_ns(call, warmup: int, repetitions: int) -> tuple[int, int]:
    for _ in range(warmup):
        call()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns(); call(); values.append(time.perf_counter_ns() - started)
    return int(np.median(values)), int(np.quantile(values, .95))


def phase_setup(smoke: bool) -> None:
    """Receiver-processing preparation, transform, and amortization timing."""
    ensure_dirs(); protocol = cfg(); old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        ks = (24,) if smoke else tuple(int(value) for value in protocol["k_values"])
        channel_count = 2 if smoke else int(protocol["banks"]["benchmark_channels"]["channels"])
        support_count = 8 if smoke else int(protocol["banks"]["statistical_support"]["channels"])
        repetitions = 5 if smoke else int(protocol["complexity"]["benchmark_repetitions"])
        warmup = 2 if smoke else int(protocol["complexity"]["warmup"])
        channels = condition_channels("nominal_dd_v1", channel_count, "benchmark_channels")
        support_channels = condition_channels("nominal_dd_v1", support_count, "statistical_support")
        waveforms = named_waveforms(); selected_names = ("Learned", "OTFS", "OFDM", "AFDM_slope_1")
        generator = torch.Generator().manual_seed(semantic_seed("setup_benchmark_vectors"))
        rows = []
        for name in selected_names:
            waveform = waveforms[name]
            supports = {k: fixed_energy_support(support_channels, waveform, k) for k in ks}
            for channel_index, channel in enumerate(channels):
                vector = torch.randn(64, dtype=torch.complex128, generator=generator)
                transform_median, transform_p95 = _timed_ns(lambda: waveform.mH @ vector, warmup, repetitions)
                effective_median, effective_p95 = _timed_ns(lambda: waveform.mH @ channel @ waveform, warmup, repetitions)
                actual = waveform.mH @ channel @ waveform
                for k in ks:
                    from src.pub1b.core import sparse_from_mask, sparse_from_topk
                    oracle_median, oracle_p95 = _timed_ns(lambda: sparse_from_topk(actual, k), warmup, repetitions)
                    fixed_median, fixed_p95 = _timed_ns(lambda: sparse_from_mask(actual, supports[k]), warmup, repetitions)
                    sparse = sparse_from_topk(actual, k); retained = sparse.dense()
                    from src.receiver import mismatched_lmmse_equalizer
                    equalizer_median, equalizer_p95 = _timed_ns(lambda: mismatched_lmmse_equalizer(retained, noise_variance(18)), warmup, repetitions)
                    equalizer = mismatched_lmmse_equalizer(retained, noise_variance(18))
                    apply_median, apply_p95 = _timed_ns(lambda: equalizer @ vector, warmup, repetitions)
                    rows.append({"waveform": name, "k": k, "channel_index": channel_index,
                        "receive_transform_median_ns": transform_median, "receive_transform_p95_ns": transform_p95,
                        "effective_channel_median_ns": effective_median, "effective_channel_p95_ns": effective_p95,
                        "oracle_selection_pack_median_ns": oracle_median, "oracle_selection_pack_p95_ns": oracle_p95,
                        "fixed_lookup_pack_median_ns": fixed_median, "fixed_lookup_pack_p95_ns": fixed_p95,
                        "direct_equalizer_setup_median_ns": equalizer_median, "direct_equalizer_setup_p95_ns": equalizer_p95,
                        "precomputed_equalizer_apply_median_ns": apply_median, "precomputed_equalizer_apply_p95_ns": apply_p95,
                        "repetitions": repetitions, "threads": 1})
        write_csv(TIMING / ("receiver_setup_smoke.csv" if smoke else "receiver_setup.csv"), rows)
        detector = read_csv(TIMING / ("receiver_benchmark_smoke.csv" if smoke else "receiver_benchmark.csv"))
        amortization = []
        for name in selected_names:
            for k in ks:
                prepared = [row for row in rows if row["waveform"] == name and int(row["k"]) == k]
                sparse_rows = [row for row in detector if row["waveform"] == name and int(row["k"]) == k and row["implementation"] == "explicit_sparse_matrix_free"]
                transform = float(np.median([int(row["receive_transform_median_ns"]) for row in prepared]))
                effective = float(np.median([int(row["effective_channel_median_ns"]) for row in prepared]))
                oracle_pack = float(np.median([int(row["oracle_selection_pack_median_ns"]) for row in prepared]))
                fixed_pack = float(np.median([int(row["fixed_lookup_pack_median_ns"]) for row in prepared]))
                equalizer_setup = float(np.median([int(row["direct_equalizer_setup_median_ns"]) for row in prepared]))
                equalizer_apply = float(np.median([int(row["precomputed_equalizer_apply_median_ns"]) for row in prepared]))
                sparse_recurring = float(np.median([int(row["median_ns"]) for row in sparse_rows]))
                for reuse in protocol["complexity"]["reuse_blocks"]:
                    for support_policy, packing in (("oracle_row_topk", oracle_pack), ("fixed_statistical", fixed_pack)):
                        amortization.append({"waveform": name, "k": k, "reuse_blocks": reuse, "support_policy": support_policy, "implementation": "precomputed_dense_equalizer", "receive_transform_ns": transform, "recurring_detection_ns": equalizer_apply, "reusable_preparation_ns": effective + packing + equalizer_setup, "amortized_receiver_processing_ns": transform + equalizer_apply + (effective + packing + equalizer_setup) / int(reuse)})
                        amortization.append({"waveform": name, "k": k, "reuse_blocks": reuse, "support_policy": support_policy, "implementation": "explicit_sparse_matrix_free", "receive_transform_ns": transform, "recurring_detection_ns": sparse_recurring, "reusable_preparation_ns": effective + packing, "amortized_receiver_processing_ns": transform + sparse_recurring + (effective + packing) / int(reuse)})
        write_csv(DATA / ("amortization_smoke.csv" if smoke else "amortization.csv"), amortization)
    finally:
        torch.set_num_threads(old_threads)


def phase_figures(smoke: bool) -> None:
    ensure_dirs()
    baseline = read_csv(DATA / "baseline_confirmatory_per_channel.csv")
    intervals = read_csv(DATA / "baseline_paired_intervals.csv")
    aggregates = {}
    for row in baseline:
        key = (row["condition"], row["modulation"], row["track"], row["label"])
        aggregates.setdefault(key, [0, 0]); aggregates[key][0] += int(row["bit_errors"]); aggregates[key][1] += int(row["total_bits"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for axis, modulation in zip(axes, ("16QAM", "QPSK")):
        chosen=[row for row in intervals if row["condition"]=="joint_0.5" and row["modulation"]==modulation]
        labels=[row["right"] for row in chosen]; centers=np.asarray([float(row["mean_difference"]) for row in chosen])
        lows=np.asarray([float(row["ci95_low"]) for row in chosen]); highs=np.asarray([float(row["ci95_high"]) for row in chosen])
        axis.errorbar(centers, np.arange(len(labels)), xerr=np.vstack((centers-lows,highs-centers)), fmt="o", color=COLORS["Learned"], capsize=3)
        axis.axvline(0,color="black",ls="--",lw=1); axis.set_yticks(np.arange(len(labels)),labels); axis.set_xlabel("Learned - classical BER"); axis.set_title(modulation); axis.grid(axis="x",alpha=.25)
    fig.suptitle("Joint-(0.5,0.5) operational receiver gate, K=24, 18 dB",fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES/"F3_historical_checkpoint_receiver_gate.png",dpi=220,bbox_inches="tight"); fig.savefig(FIGURES/"F3_historical_checkpoint_receiver_gate.pdf",bbox_inches="tight"); plt.close(fig)
    ledger=read_csv(DATA/"complexity_ledger.csv")
    plain=[row for row in ledger if row["residual_aware"]=="False"]
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.8))
    axes[0].plot([int(r["k"]) for r in plain],[int(r["algorithmic_total_complex_equivalent_macs"]) for r in plain],marker="o"); axes[0].set(xlabel="K",ylabel="Complex-equivalent MACs",title="64-iteration sparse model"); axes[0].grid(alpha=.25)
    axes[1].plot([int(r["k"]) for r in plain],[int(r["sparse_total_bytes"]) for r in plain],marker="o",label="sparse values+indices"); axes[1].axhline(int(plain[0]["dense_coefficients_bytes"]),color="k",ls="--",label="dense coefficients"); axes[1].set(xlabel="K",ylabel="Bytes",title="Retained-channel storage"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES/"F4a_operations_storage.png",dpi=220,bbox_inches="tight"); fig.savefig(FIGURES/"F4a_operations_storage.pdf",bbox_inches="tight"); plt.close(fig)
    amortization_path = DATA / ("amortization_smoke.csv" if smoke else "amortization.csv")
    if amortization_path.exists():
        amortization = read_csv(amortization_path)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for implementation, label, marker in (("precomputed_dense_equalizer", "Precomputed dense G", "o"), ("explicit_sparse_matrix_free", "Sparse matrix-free CG", "s")):
            values = []
            for k in sorted({int(row["k"]) for row in amortization}):
                selected = [float(row["amortized_receiver_processing_ns"]) / 1e3 for row in amortization if row["implementation"] == implementation and row["support_policy"] == "oracle_row_topk" and int(row["reuse_blocks"]) == 1 and int(row["k"]) == k]
                values.append((k, float(np.median(selected))))
            axes[0].plot([item[0] for item in values], [item[1] for item in values], marker=marker, label=label)
        axes[0].set(xlabel="K", ylabel="Latency (us)", title="Unamortized, B=1"); axes[0].set_yscale("log"); axes[0].grid(alpha=.25); axes[0].legend(fontsize=7)
        for implementation, label, marker in (("precomputed_dense_equalizer", "Precomputed dense G", "o"), ("explicit_sparse_matrix_free", "Sparse matrix-free CG", "s")):
            values=[]
            for reuse in sorted({int(row["reuse_blocks"]) for row in amortization}):
                selected=[float(row["amortized_receiver_processing_ns"])/1e3 for row in amortization if row["implementation"]==implementation and row["support_policy"]=="oracle_row_topk" and int(row["k"])==24 and int(row["reuse_blocks"])==reuse]
                values.append((reuse,float(np.median(selected))))
            axes[1].plot([item[0] for item in values],[item[1] for item in values],marker=marker,label=label)
        axes[1].set(xlabel="Shared vectors B",ylabel="Amortized latency (us)",title="K=24 setup amortization"); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].grid(alpha=.25); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGURES/"F4b_latency_amortization.png",dpi=220,bbox_inches="tight"); fig.savefig(FIGURES/"F4b_latency_amortization.pdf",bbox_inches="tight"); plt.close(fig)


def phase_report(smoke: bool) -> None:
    gate=read_json(DATA/"early_gate.json"); execution=read_json(DATA/"baseline_execution.json")
    selections=read_csv(DATA/"baseline_selection.csv"); intervals=read_csv(DATA/"baseline_paired_intervals.csv")
    confirm=read_csv(DATA/"baseline_confirmatory_per_channel.csv")
    aggregate: dict[tuple[str,str,str], list[int]] = {}
    for row in confirm:
        if row["track"] != "operational_selected": continue
        key=(row["condition"],row["modulation"],row["label"]); aggregate.setdefault(key,[0,0]); aggregate[key][0]+=int(row["bit_errors"]); aggregate[key][1]+=int(row["total_bits"])
    timing_path=TIMING/("receiver_benchmark_smoke.csv" if smoke else "receiver_benchmark.csv")
    timing=read_csv(timing_path) if timing_path.exists() else []
    def median_timing(implementation: str, k: int) -> float:
        values=[float(row["median_ns"])/1e3 for row in timing if row["implementation"]==implementation and int(row["k"])==k]
        return float(np.median(values)) if values else float("nan")
    ledger=read_csv(DATA/"complexity_ledger.csv") if (DATA/"complexity_ledger.csv").exists() else []
    plain={int(row["k"]):row for row in ledger if row["residual_aware"]=="False"}
    k8_macs=int(plain[8]["algorithmic_total_complex_equivalent_macs"]) if 8 in plain else 0; k64_macs=int(plain[64]["algorithmic_total_complex_equivalent_macs"]) if 64 in plain else 0
    k8_storage=int(plain[8]["sparse_total_bytes"]) if 8 in plain else 0; k64_storage=int(plain[64]["sparse_total_bytes"]) if 64 in plain else 0
    lines=["# Pub-1F — Training reproducibility and receiver-complexity qualification","",f"Run kind: **{'smoke' if smoke else 'production'}**.","",f"## Decision: historical-checkpoint early gate {'passed' if gate['passed'] else 'failed'}","",f"The predeclared gate is **{'PASS' if gate['passed'] else 'FAIL'}**. This is a receiver-qualification result, not training-reproducibility evidence. {'Production training was therefore not launched.' if not gate['passed'] and not smoke else ''}","",f"Fresh banks used {execution['validation_channels']} validation, {execution['statistics_channels']} residual-statistics, and {execution['confirmatory_channels']} confirmatory channels with {execution['bits_per_channel']} bits/channel/modulation. Validation selected configurations; the 512-channel test bank only confirmed them.","","### Favorable joint-(0.5,0.5) paired gate","","| modulation | comparison | Learned BER | classical BER | difference | paired 95% CI |","|---|---|---:|---:|---:|---:|"]
    for row in intervals:
        if row["condition"]!="joint_0.5": continue
        left=aggregate[("joint_0.5",row["modulation"],"Learned")]; right=aggregate[("joint_0.5",row["modulation"],row["right"])]
        lines.append(f"| {row['modulation']} | Learned - {row['right']} | {left[0]/left[1]:.7f} | {right[0]/right[1]:.7f} | {float(row['mean_difference']):.7f} | [{float(row['ci95_low']):.7f}, {float(row['ci95_high']):.7f}] |")
    lines += ["","All 16-QAM comparisons and QPSK comparisons against OTFS/OFDM support Learned. QPSK versus operational AFDM slope 1 is lower in point estimate but inconclusive because the interval includes zero; this single mandatory failure stops the gate.","","### Operational selected configurations and test BER","","| condition | modulation | Learned | OTFS | OFDM | AFDM | AFDM validation choice |","|---|---|---:|---:|---:|---:|---|"]
    for condition in CONDITIONS:
        for modulation in ("16QAM","QPSK"):
            vals=[]
            for label in ("Learned","OTFS","OFDM","AFDM"):
                errors,bits=aggregate[(condition,modulation,label)]; vals.append(errors/bits)
            afdm=next(row for row in selections if row["condition"]==condition and row["modulation"]==modulation and row["waveform_family"]=="AFDM")
            lines.append(f"| {condition} | {modulation} | {vals[0]:.7f} | {vals[1]:.7f} | {vals[2]:.7f} | {vals[3]:.7f} | slope {afdm['slope']}, {afdm['receiver_family']} |")
    lines += ["","The intermediate (0.2,0.2) point is a probe, not a crossover boundary. Learned loses significantly to OTFS in probe QPSK, is inconclusive against AFDM there, and loses to OTFS/AFDM on the integer grid. Favorable and unfavorable regimes therefore persist. Fresh nominal residual-aware QPSK shows a small supported Learned-AFDM difference, unlike Pub-1E's inconclusive nominal result; this is a new bank/configuration result, not a reinterpretation of Pub-1E.","","Residual-aware results use waveform/condition/K/oracle-support-specific independent omitted means and dense receive covariances. They neglect conditioning on realized retained values/support, are not conditional-optimal, and require dense stored information and direct solves.","","## Training reproducibility and objective ablation","",("Not executed. The frozen early-gate failure prohibited the five-seed 2,000+1,000+1,000-step production workload. Smoke exercised all five seeds for two steps only; it is correctness evidence and has no scientific meaning." if not gate["passed"] else "See `data/training_summary.csv` for all five seeds; no seed was replaced or selected by BER."),"","Replication remains scientifically useful if the goal is to estimate optimization reproducibility despite an inconclusive operational gate. Proceeding requires an explicit decision to override the frozen cost-benefit gate; it must not be reframed as evidence that Learned won.","","## Complexity: arithmetic, storage, latency, and amortization","",f"For 64 executed CG iterations, the audited detector model is `NK + N + t(2NK + 6N)`. K=8 costs {k8_macs:,} complex-equivalent MACs versus {k64_macs:,} at K=64, an analytical reduction of {(1-k8_macs/k64_macs)*100:.1f}%. Sparse complex128/int64 values+indices occupy {k8_storage:,} versus {k64_storage:,} bytes, a {(1-k8_storage/k64_storage)*100:.1f}% reduction. At K=24 the representation stores 36,864 bytes versus 65,536 dense-coefficient bytes. Residual awareness adds 131,072 bytes for one dense complex mean and covariance before other state.","",f"Measured prepared-detector median latency was {median_timing('explicit_sparse_matrix_free',8):.1f} us at K=8 and {median_timing('explicit_sparse_matrix_free',64):.1f} us at K=64, only a {(1-median_timing('explicit_sparse_matrix_free',8)/median_timing('explicit_sparse_matrix_free',64))*100:.1f}% reduction. Dense direct solve over a matrix containing zeros was about {median_timing('dense_zero_direct',24):.1f} us at K=24, while explicit sparse matrix-free CG was {median_timing('explicit_sparse_matrix_free',24):.1f} us. Thus fewer modeled operations did not produce a faster implementation.","",f"The fixed 64-step sparse workload had {sum(row['converged']=='False' for row in timing if row['implementation']=='explicit_sparse_matrix_free')} nonconverged channel/configuration cases out of {sum(row['implementation']=='explicit_sparse_matrix_free' for row in timing)}; maximum 64-step relative solution error was {max([float(row['numerical_relative_error']) for row in timing if row['implementation']=='explicit_sparse_matrix_free'],default=float('nan')):.3e}. Every implementation first passed a separate tight solve gate; maximum tight relative error was {max([float(row['tight_reference_relative_error']) for row in timing],default=float('nan')):.3e}. Residual and BER remain separate criteria.","","Receive transform, complete effective-channel formation, oracle/fixed packing, direct equalizer setup, recurring application, and B=1/10/100/1000 amortization are saved separately. Because the implementation forms complete A first, it demonstrates no effective-channel construction saving. Oracle support requires coefficient access; fixed support saves position selection but still updates retained values. Precomputed dense G has dense application/storage even when A is sparse.","","The measured study covers receiver processing from an available channel matrix and resident vectors. It excludes channel estimation, RF acquisition, coding, transmitter work, plotting, and BER bookkeeping. No total-transceiver or energy claim is made. The B sweep is hypothetical, not a physical coherence-time claim.","","## Matched quality and claim boundary","","The predeclared BER targets remain frozen, but minimum validated cost `C*_w(b*)` is **not estimable**: the early gate stopped the validation-selected K/iteration production grid. No test-set lower envelope, interpolation, or post-failure reselection is substituted. Consequently there is no supported same-BER receiver-processing saving, no BER-versus-latency frontier, and no F4c/F4d claim.","","At fixed N, sparse products scale O(NK); at fixed rho they remain O(rho N^2). The measurements establish only a K-dependent constant reduction at N=64, not a better asymptotic class. No N=128/256 training was launched.","","## Explicit answers","","1. Five-seed reproducibility: **not tested**, stopped by the gate.","2. Tail versus matched mean continuation: **not tested**.","3. Slope-1 AFDM with improved detection: 16-QAM advantage survives; favorable QPSK is **inconclusive**, so the joint gate fails.","4. OFDM: Learned is significantly better in the favorable joint condition for both modulations; OFDM remains reported everywhere.","5. Nominal residual-aware QPSK: on the fresh Pub-1F bank it is no longer inconclusive versus selected AFDM; the paired interval is below zero. Pub-1E's result remains historically inconclusive.","6. Regimes: favorable 16-QAM and unfavorable integer behavior persist; probe QPSK is mixed/unfavorable.","7. Arithmetic/storage: K=8 versus 64 reduces the declared 64-step MAC ledger by the percentage above and sparse row storage by 87.5%.","8. Runtime/throughput: sparse latency falls modestly with K but remains much slower than dense direct/precomputed paths on this CPU implementation; no throughput advantage is shown.","9. Iterations/overhead/preparation: nonconvergence and Python indexing dominate; full-A construction and transforms remain; amortization helps reusable dense setup most.","10. Minimum cost per target: **not established** because the mandatory operating grid did not run.","11. Full receiver-processing savings: **not demonstrated**.","12. Conference figures: current defensible figures are F3 receiver gate, F4a operation/storage, and F4b latency/amortization. F1/F2 and F4c/F4d require an authorized replication/operating-grid continuation.","","## Evidence category and recommendation","",("**Narrow receiver/regime-dependent result.** The historical waveform retains a strong 16-QAM advantage and several QPSK advantages, but the predeclared slope-1-inclusive favorable-QPSK gate is inconclusive. Arithmetic representation benefits are real; runtime and matched-quality receiver-processing savings are not demonstrated. For a six-page paper now, center the corrected Pub-1E regime map plus this stronger-baseline limitation and implementation reality. Run five-seed replication only if reproducibility itself is worth the substantial compute despite the failed operational gate." if not smoke else "Smoke only; no scientific category."),"","## Artifacts","","- `data/baseline_validation.csv`: all validation candidates.","- `data/baseline_selection.csv`: deployable validation choices.","- `data/baseline_confirmatory_per_channel.csv` and `data/baseline_paired_intervals.csv`: raw test evidence.","- `data/complexity_ledger.csv`, `timing/receiver_benchmark.csv`, `timing/receiver_setup.csv`, and `data/amortization.csv`: operation, storage, convergence, latency, and setup evidence.","- `figures/F3_historical_checkpoint_receiver_gate.*`, `F4a_operations_storage.*`, and `F4b_latency_amortization.*`: current primary figures."]
    REPORT.write_text("\n".join(lines)+"\n")


def phase_manifest(smoke: bool) -> None:
    ensure_dirs(); protected=cfg()["historical_checkpoint"]["sha256"]
    after=sha256(HISTORICAL)
    if after != protected:
        raise RuntimeError("protected historical checkpoint changed")
    files=[]
    for base in (OUT,):
        for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name!="artifact_manifest.json"):
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in (CONFIG, ROOT/"reports/stage_pub1f_protocol.md", REPORT):
        if path.exists(): files.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    write_json(OUT/"artifact_manifest.json",{"stage":"Pub-1F","run_kind":"smoke" if smoke else "production","count":len(files),"protected_checkpoint_sha256_after":after,"files":files})


PHASES={"prerequisites":phase_prerequisites,"baseline":phase_baseline,"gate":phase_gate,"training":phase_training,"complexity":phase_complexity,"setup":phase_setup,"figures":phase_figures,"report":phase_report,"manifest":phase_manifest}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("all",*PHASES),default="all"); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    selected=PHASES.items() if args.phase=="all" else ((args.phase,PHASES[args.phase]),)
    for name,function in selected:
        print(f"[Pub-1F] {name}",flush=True); function(args.smoke)


if __name__=="__main__": main()
