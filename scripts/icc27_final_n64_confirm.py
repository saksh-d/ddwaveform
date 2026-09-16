"""Run the fresh evaluation-only ICC27 N=64 confirmation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.channel import COMPLEX_DTYPE
from src.icc27_final import (
    NAMESPACE,
    communication_rows,
    fixed_energy_support,
    receiver_state,
    sample_channels,
    semantic_seed,
    summarize,
)
from src.metrics import qam16_constellation, qpsk_constellation
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/icc27_final_n64_confirm.json"
ATTACHMENT = ROOT / ".context/attachments/cXTT8G/pasted_text_2026-09-16_11-37-02.txt"
OUTPUT = ROOT / "outputs/icc27_final_n64_confirm"
STRESS_CONDITIONS = ("wider_delay", "wider_doppler", "more_paths", "integer_grid")


def read_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_waveform(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("waveform", "matrix", "best_matrix"):
            if key in value:
                return value[key].to(COMPLEX_DTYPE)
        raise ValueError(f"no waveform tensor in {path}")
    return value.to(COMPLEX_DTYPE)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or tuple(rows[0]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def checkpoint_records(configuration: dict[str, object]) -> list[dict[str, object]]:
    learned = configuration["waveforms"]["Learned"]
    records = [{"role": "authoritative", "path": learned["path"], "expected_sha256": learned["sha256"]}]
    reproducibility = configuration["reproducibility"]
    for seed_id in reproducibility["seed_ids"]:
        records.append(
            {
                "role": "reproducibility",
                "seed_id": seed_id,
                "path": reproducibility["checkpoint_template"].format(seed_id=seed_id),
                "expected_sha256": reproducibility["checkpoint_sha256"][str(seed_id)],
            }
        )
    return records


def validate_inputs(configuration: dict[str, object]) -> list[dict[str, object]]:
    if configuration["namespace"] != NAMESPACE:
        raise RuntimeError("config and implementation namespaces differ")
    records = checkpoint_records(configuration)
    for record in records:
        path = ROOT / record["path"]
        if not path.exists():
            raise RuntimeError(f"missing checkpoint: {record['path']}")
        actual = file_sha256(path)
        if actual != record["expected_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {record['path']} {actual}")
        waveform = load_waveform(path)
        identity = torch.eye(configuration["n"], dtype=COMPLEX_DTYPE)
        residual = float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity))
        if residual > 1e-10:
            raise RuntimeError(f"non-unitary checkpoint: {record['path']} {residual}")
        record["tensor_sha256"] = tensor_sha256(waveform)
        record["unitarity_residual"] = residual
    return records


def seed_ledger(configuration: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {
        "support_bank": semantic_seed("support_channels", condition="nominal"),
        "nominal_channels": semantic_seed("test_channels", condition="nominal"),
        "stress_channels": {
            condition: semantic_seed("test_channels", condition=condition) for condition in STRESS_CONDITIONS
        },
        "symbols": {},
        "noise": {},
    }
    roles = (("nominal", "nominal_confirmation"),) + tuple(
        (condition, "stress_confirmation") for condition in STRESS_CONDITIONS
    )
    for condition, bank_role in roles:
        for modulation in configuration["modulations"]:
            key = f"{condition}:{modulation}"
            fields = {
                "bank_role": bank_role,
                "condition": condition,
                "modulation": modulation,
                "snr_db": configuration["snr_db"],
            }
            values["symbols"][key] = semantic_seed("symbols", **fields)
            values["noise"][key] = semantic_seed("noise", **fields)
    return values


def runtime_config(
    configuration: dict[str, object],
    checkpoint_info: list[dict[str, object]],
    waveforms: dict[str, torch.Tensor],
    run_kind: str,
) -> dict[str, object]:
    status = git_output("status", "--short")
    return {
        **configuration,
        "run_kind": run_kind,
        "source": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": status.splitlines(),
            "config_sha256": file_sha256(CONFIG_PATH),
            "runner_sha256": file_sha256(Path(__file__)),
            "core_sha256": file_sha256(ROOT / "src/icc27_final.py"),
            "request_sha256": file_sha256(ATTACHMENT),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "all_seeds": seed_ledger(configuration),
        "checkpoint_records": checkpoint_info,
        "primary_waveform_tensor_sha256": {
            name: tensor_sha256(waveform) for name, waveform in waveforms.items()
        },
    }


def named_waveforms(configuration: dict[str, object]) -> dict[str, torch.Tensor]:
    learned_path = ROOT / configuration["waveforms"]["Learned"]["path"]
    return {
        "Learned": load_waveform(learned_path),
        "OFDM": ofdm_waveform(),
        "OTFS": otfs_waveform(8, 8),
        "AFDM": afdm_waveform(c1=float(configuration["waveforms"]["AFDM"]["c1"]), c2=float(configuration["waveforms"]["AFDM"]["c2"])),
    }


def evaluate_nominal(
    configuration: dict[str, object],
    waveforms: dict[str, torch.Tensor],
    supports: dict[str, torch.Tensor],
    count: int,
) -> list[dict[str, object]]:
    channels = sample_channels("nominal", count, semantic_seed("test_channels", condition="nominal")).matrices
    rows: list[dict[str, object]] = []
    for waveform_name, waveform in waveforms.items():
        for mode in configuration["receiver"]["nominal_modes"]:
            state = receiver_state(channels, waveform, mode, configuration["k"], configuration["snr_db"], supports[waveform_name])
            for constellation in (qpsk_constellation(), qam16_constellation()):
                counts = communication_rows(
                    channels,
                    waveform,
                    constellation,
                    state,
                    condition="nominal",
                    bank_role="nominal_confirmation",
                    snr_db=configuration["snr_db"],
                    bits_per_channel=configuration["bits_per_channel"],
                )
                receiver_k = configuration["n"] if mode == "full_k64" else configuration["k"]
                rows.extend(
                    {
                        "channel_id": row["channel_id"],
                        "waveform": waveform_name,
                        "modulation": constellation.name,
                        "receiver_mode": mode,
                        "snr_db": configuration["snr_db"],
                        "K": receiver_k,
                        "bit_errors": row["bit_errors"],
                        "bits": row["bits"],
                        "retained_energy": row["retained_energy"],
                        "analytic_mse": row["analytic_mse"],
                    }
                    for row in counts
                )
        print(f"nominal {waveform_name}", flush=True)
    return rows


def evaluate_stress(
    configuration: dict[str, object],
    waveforms: dict[str, torch.Tensor],
    supports: dict[str, torch.Tensor],
    count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in STRESS_CONDITIONS:
        channels = sample_channels(condition, count, semantic_seed("test_channels", condition=condition)).matrices
        for waveform_name, waveform in waveforms.items():
            state = receiver_state(
                channels,
                waveform,
                "fixed_statistical_k24",
                configuration["k"],
                configuration["snr_db"],
                supports[waveform_name],
            )
            for constellation in (qpsk_constellation(), qam16_constellation()):
                counts = communication_rows(
                    channels,
                    waveform,
                    constellation,
                    state,
                    condition=condition,
                    bank_role="stress_confirmation",
                    snr_db=configuration["snr_db"],
                    bits_per_channel=configuration["bits_per_channel"],
                )
                rows.extend(
                    {
                        "channel_id": row["channel_id"],
                        "condition": condition,
                        "waveform": waveform_name,
                        "modulation": constellation.name,
                        "receiver_mode": "nominal_fixed_statistical_k24",
                        "snr_db": configuration["snr_db"],
                        "K": configuration["k"],
                        "bit_errors": row["bit_errors"],
                        "bits": row["bits"],
                        "retained_energy": row["retained_energy"],
                        "analytic_mse": row["analytic_mse"],
                    }
                    for row in counts
                )
            print(f"stress {condition} {waveform_name}", flush=True)
    return rows


def evaluate_reproducibility(
    configuration: dict[str, object], support_channels: torch.Tensor, count: int
) -> list[dict[str, object]]:
    channels = sample_channels("nominal", count, semantic_seed("test_channels", condition="nominal")).matrices
    rows: list[dict[str, object]] = []
    reproducibility = configuration["reproducibility"]
    for seed_id in reproducibility["seed_ids"]:
        path = ROOT / reproducibility["checkpoint_template"].format(seed_id=seed_id)
        waveform = load_waveform(path)
        support = fixed_energy_support(support_channels, waveform, configuration["k"])
        state = receiver_state(
            channels,
            waveform,
            "fixed_statistical_k24",
            configuration["k"],
            configuration["snr_db"],
            support,
        )
        checkpoint_hash = file_sha256(path)
        for constellation in (qpsk_constellation(), qam16_constellation()):
            counts = communication_rows(
                channels,
                waveform,
                constellation,
                state,
                condition="nominal",
                bank_role="nominal_confirmation",
                snr_db=configuration["snr_db"],
                bits_per_channel=configuration["bits_per_channel"],
            )
            rows.extend(
                {
                    "seed_id": seed_id,
                    "checkpoint_hash": checkpoint_hash,
                    "channel_id": row["channel_id"],
                    "modulation": constellation.name,
                    "bit_errors": row["bit_errors"],
                    "bits": row["bits"],
                    "retained_energy": row["retained_energy"],
                    "analytic_mse": row["analytic_mse"],
                }
                for row in counts
            )
        print(f"reproducibility seed {seed_id}", flush=True)
    return rows


def run(smoke: bool) -> Path:
    configuration = read_config()
    checkpoint_info = validate_inputs(configuration)
    output = OUTPUT / "smoke" if smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    waveforms = named_waveforms(configuration)
    write_json(
        output / "config.json",
        runtime_config(configuration, checkpoint_info, waveforms, "smoke" if smoke else "production"),
    )
    support_count = 8 if smoke else configuration["banks"]["support_channels"]
    nominal_count = 4 if smoke else configuration["banks"]["nominal_channels"]
    stress_count = 3 if smoke else configuration["banks"]["stress_channels_per_condition"]
    support_channels = sample_channels(
        "nominal", support_count, semantic_seed("support_channels", condition="nominal")
    ).matrices
    supports = {name: fixed_energy_support(support_channels, waveform, configuration["k"]) for name, waveform in waveforms.items()}
    started = time.perf_counter()
    nominal = evaluate_nominal(configuration, waveforms, supports, nominal_count)
    write_csv(output / "nominal_per_channel.csv", nominal)
    write_csv(
        output / "nominal_summary.csv",
        summarize(nominal, ("waveform", "modulation", "receiver_mode", "snr_db", "K")),
    )
    stress = evaluate_stress(configuration, waveforms, supports, stress_count)
    write_csv(output / "stress_per_channel.csv", stress)
    write_csv(
        output / "stress_summary.csv",
        summarize(stress, ("condition", "waveform", "modulation", "receiver_mode", "snr_db", "K")),
    )
    reproducibility = evaluate_reproducibility(configuration, support_channels, nominal_count)
    write_csv(
        output / "reproducibility_per_channel.csv",
        reproducibility,
        ("seed_id", "checkpoint_hash", "channel_id", "modulation", "bit_errors", "bits", "retained_energy", "analytic_mse"),
    )
    write_csv(
        output / "reproducibility_summary.csv",
        summarize(reproducibility, ("seed_id", "checkpoint_hash", "modulation")),
    )
    artifacts = []
    for path in sorted(output.glob("*")):
        if path.is_file():
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(
        output / "manifest.json",
        {
            "experiment": configuration["experiment"],
            "run_kind": "smoke" if smoke else "production",
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": artifacts,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run(args.smoke)
    print(f"ICC27 confirmation complete: {output}")


if __name__ == "__main__":
    main()
