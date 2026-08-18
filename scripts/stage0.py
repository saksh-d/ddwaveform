"""Run Stage 0A-E validation and classical DD-v1 baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src import config
from src.channel import (
    COMPLEX_DTYPE,
    assemble_dd_channel,
    complex_awgn,
    dft_matrix,
    doppler_operator,
    fractional_delay_operator,
    make_generator,
    noise_variance,
    sample_dd_v1,
)
from src.metrics import bit_error_rate, normalized_mse, qpsk_encode, random_bits
from src.receiver import effective_operator, finite_budget_receiver, full_information_mse
from src.waveforms import afdm_waveform, classical_waveforms, diagnostic_waveforms, random_unitary

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "stage0"
REPORT_PATH = ROOT / "reports" / "stage0_summary.md"


def scalar(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def frobenius_residual(matrix: torch.Tensor, target: torch.Tensor) -> float:
    return scalar(torch.linalg.matrix_norm(matrix - target))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_pytest() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    count = int(match.group(1)) if match else 0
    if result.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{output}")
    return count, output.splitlines()[-1]


def operator_diagnostics() -> dict[str, float]:
    n = config.N
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    f = dft_matrix(n)
    generator = make_generator(config.GLOBAL_SEED)
    vector = (
        torch.randn(n, dtype=torch.float64, generator=generator)
        + 1j * torch.randn(n, dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)

    integer_residual = max(
        scalar(torch.max(torch.abs(fractional_delay_operator(ell, n) @ vector - torch.roll(vector, ell))))
        for ell in (0, 1, 4, n - 1)
    )
    fractional_unitarity = max(
        frobenius_residual(fractional_delay_operator(tau, n).mH @ fractional_delay_operator(tau, n), identity)
        for tau in (0.25, 1.4, 3.75)
    )
    fractional_composition = frobenius_residual(
        fractional_delay_operator(0.25, n) @ fractional_delay_operator(1.4, n),
        fractional_delay_operator(1.65, n),
    )
    doppler_unitarity = frobenius_residual(
        doppler_operator(-0.37, n).mH @ doppler_operator(-0.37, n), identity
    )
    doppler_composition = frobenius_residual(
        doppler_operator(0.2, n) @ doppler_operator(-0.7, n), doppler_operator(-0.5, n)
    )
    ell, nu = 4, 3.0
    phase = torch.exp(torch.tensor(-2j * math.pi * nu * ell / n, dtype=COMPLEX_DTYPE))
    commutation = frobenius_residual(
        doppler_operator(nu, n) @ fractional_delay_operator(ell, n),
        phase * fractional_delay_operator(ell, n) @ doppler_operator(nu, n),
    )
    fractional_nu = 0.43
    fractional_phase = torch.exp(
        torch.tensor(-2j * math.pi * fractional_nu * ell / n, dtype=COMPLEX_DTYPE)
    )
    fractional_boundary_residual = frobenius_residual(
        doppler_operator(fractional_nu, n) @ fractional_delay_operator(ell, n),
        fractional_phase * fractional_delay_operator(ell, n) @ doppler_operator(fractional_nu, n),
    )

    delays = torch.tensor([0.25, 2.0], dtype=torch.float64)
    dopplers = torch.tensor([-0.4, 0.6], dtype=torch.float64)
    gains = torch.tensor([0.3 + 0.2j, -0.1 + 0.7j], dtype=COMPLEX_DTYPE)
    assembled = assemble_dd_channel(delays, dopplers, gains, n)
    manual = sum(
        gains[p] * doppler_operator(dopplers[p], n) @ fractional_delay_operator(delays[p], n)
        for p in range(2)
    )
    multipath_residual = frobenius_residual(assembled, manual)

    sampled = sample_dd_v1(128, config.GLOBAL_SEED)
    energies = torch.sum(torch.abs(sampled.matrices) ** 2, dim=(-2, -1)) / n
    normalization_residual = scalar(torch.max(torch.abs(energies - 1.0)))
    reproduced = sample_dd_v1(128, config.GLOBAL_SEED)
    other = sample_dd_v1(128, config.GLOBAL_SEED + 1)
    reproducible = float(torch.equal(sampled.matrices, reproduced.matrices))
    different_seed = float(not torch.equal(sampled.matrices, other.matrices))
    delay_fractional_fraction = scalar(torch.mean((sampled.delays != torch.round(sampled.delays)).double()))
    doppler_fractional_fraction = scalar(torch.mean((sampled.dopplers != torch.round(sampled.dopplers)).double()))

    sigma2 = noise_variance(15.0)
    awgn = complex_awgn((500_000,), sigma2, make_generator(config.GLOBAL_SEED + 2))
    awgn_relative_error = scalar(torch.abs(torch.mean(torch.abs(awgn) ** 2) - sigma2) / sigma2)

    return {
        "dft_unitarity_fro": frobenius_residual(f.mH @ f, identity),
        "integer_delay_max_abs": integer_residual,
        "fractional_delay_unitarity_fro": fractional_unitarity,
        "fractional_delay_composition_fro": fractional_composition,
        "doppler_unitarity_fro": doppler_unitarity,
        "doppler_composition_fro": doppler_composition,
        "dd_integer_bin_commutation_fro": commutation,
        "dd_fractional_doppler_scalar_relation_boundary_fro": fractional_boundary_residual,
        "multipath_assembly_fro": multipath_residual,
        "dd_v1_normalization_max_abs": normalization_residual,
        "same_seed_exact": reproducible,
        "different_seed_differs": different_seed,
        "delay_fractional_fraction": delay_fractional_fraction,
        "doppler_fractional_fraction": doppler_fractional_fraction,
        "awgn_variance_relative_error": awgn_relative_error,
    }


def waveform_diagnostics(c1: float) -> dict[str, dict[str, float]]:
    identity = torch.eye(config.N, dtype=COMPLEX_DTYPE)
    generator = make_generator(config.GLOBAL_SEED + 3)
    symbols = (
        torch.randn(config.N, dtype=torch.float64, generator=generator)
        + 1j * torch.randn(config.N, dtype=torch.float64, generator=generator)
    ).to(COMPLEX_DTYPE)
    diagnostics: dict[str, dict[str, float]] = {}
    for name, waveform in diagnostic_waveforms(c1).items():
        diagnostics[name] = {
            "unitarity_fro": frobenius_residual(waveform.mH @ waveform, identity),
            "power_relative_error": scalar(
                torch.abs(torch.sum(torch.abs(waveform @ symbols) ** 2) - torch.sum(torch.abs(symbols) ** 2))
                / torch.sum(torch.abs(symbols) ** 2)
            ),
            "identity_channel_fro": frobenius_residual(effective_operator(identity, waveform), identity),
        }
    return diagnostics


def tune_afdm(validation_channels: torch.Tensor) -> tuple[int, float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best_multiplier = -1
    best_score = math.inf
    for multiplier in config.AFDM_C1_CANDIDATE_MULTIPLIERS:
        c1 = multiplier / (2 * config.N)
        waveform = afdm_waveform(c1=c1)
        a = effective_operator(validation_channels, waveform)
        snr_means = []
        for snr_db in config.AFDM_VALIDATION_SNRS_DB:
            _, _, mse = finite_budget_receiver(a, noise_variance(snr_db), config.P)
            snr_means.append(scalar(torch.mean(mse)))
        score = float(np.mean(snr_means))
        rows.append(
            {
                "c1_multiplier_m": multiplier,
                "c1": f"{c1:.17g}",
                **{f"mse_{snr}_db": f"{value:.17g}" for snr, value in zip(config.AFDM_VALIDATION_SNRS_DB, snr_means)},
                "mean_mse": f"{score:.17g}",
            }
        )
        if score < best_score:
            best_score = score
            best_multiplier = multiplier
    return best_multiplier, best_score, rows


def receiver_validation(c1: float) -> tuple[list[dict[str, object]], float]:
    channel = sample_dd_v1(1, config.GLOBAL_SEED + 10).matrices[0]
    sigma2 = noise_variance(15.0)
    waveforms = {"IDENTITY": classical_waveforms(c1)["IDENTITY"], "RANDOM": random_unitary()}
    generator = make_generator(config.GLOBAL_SEED + 11)
    n_blocks = 4096
    bits = random_bits((n_blocks, config.N, 2), generator)
    symbols = qpsk_encode(bits)
    physical_noise = complex_awgn((n_blocks, config.N), sigma2, generator)
    rows: list[dict[str, object]] = []
    physical_equivalence_max = 0.0
    for name, waveform in waveforms.items():
        a = effective_operator(channel, waveform)
        noiseless_physical = (symbols @ waveform.T) @ channel.T @ waveform.conj()
        noiseless_effective = symbols @ a.T
        physical_equivalence_max = max(
            physical_equivalence_max, scalar(torch.max(torch.abs(noiseless_physical - noiseless_effective)))
        )
        x = symbols @ waveform.T
        y = x @ channel.T + physical_noise
        u = y @ waveform.conj()
        for k in (3, 6, config.N):
            _, equalizer, analytic = finite_budget_receiver(a, sigma2, k)
            estimates = u @ equalizer.T
            empirical = torch.mean(torch.abs(estimates - symbols) ** 2)
            relative = torch.abs(empirical - analytic) / analytic
            rows.append(
                {
                    "waveform": name,
                    "k": k,
                    "snr_db": 15,
                    "analytic_mse": scalar(analytic),
                    "empirical_mse": scalar(empirical),
                    "relative_discrepancy": scalar(relative),
                }
            )
    return rows, physical_equivalence_max


def full_invariance(c1: float) -> tuple[list[dict[str, object]], dict[str, float]]:
    channels = sample_dd_v1(4, config.GLOBAL_SEED + 20).matrices
    waveforms = diagnostic_waveforms(c1)
    rows: list[dict[str, object]] = []
    max_bound_error = 0.0
    max_cross_absolute = 0.0
    max_cross_relative = 0.0
    for channel_index, channel in enumerate(channels):
        for snr_db in config.FULL_INVARIANCE_SNRS_DB:
            sigma2 = noise_variance(snr_db)
            bound = full_information_mse(channel, sigma2)
            values: dict[str, float] = {}
            for name, waveform in waveforms.items():
                a = effective_operator(channel, waveform)
                _, _, mse = finite_budget_receiver(a, sigma2, config.N)
                value = scalar(mse)
                values[name] = value
                max_bound_error = max(max_bound_error, abs(value - scalar(bound)))
                rows.append(
                    {
                        "channel_index": channel_index,
                        "snr_db": snr_db,
                        "waveform": name,
                        "trace_mse": f"{scalar(bound):.17g}",
                        "receiver_mse": f"{value:.17g}",
                        "absolute_error": f"{abs(value - scalar(bound)):.17g}",
                    }
                )
            spread = max(values.values()) - min(values.values())
            max_cross_absolute = max(max_cross_absolute, spread)
            max_cross_relative = max(max_cross_relative, spread / scalar(bound))
    return rows, {
        "max_receiver_vs_trace_abs": max_bound_error,
        "max_cross_waveform_abs": max_cross_absolute,
        "max_cross_waveform_relative": max_cross_relative,
    }


def baseline_campaign(c1: float) -> list[dict[str, object]]:
    channels = sample_dd_v1(config.N_STAGE0_EVAL_CHANNELS, config.STAGE0_EVAL_SEED).matrices
    waveforms = classical_waveforms(c1)
    effective = {name: effective_operator(channels, waveform) for name, waveform in waveforms.items()}
    generator = make_generator(config.STAGE0_EVAL_SEED + 1)
    rows: list[dict[str, object]] = []
    for snr_db in config.STAGE0_SNRS_DB:
        sigma2 = noise_variance(snr_db)
        bits = random_bits(
            (config.N_STAGE0_EVAL_CHANNELS, config.N_BLOCKS_PER_CHANNEL, config.N, 2), generator
        )
        symbols = qpsk_encode(bits)
        physical_noise = complex_awgn(
            (config.N_STAGE0_EVAL_CHANNELS, config.N_BLOCKS_PER_CHANNEL, config.N), sigma2, generator
        )
        common_bound = scalar(torch.mean(full_information_mse(channels, sigma2)))
        for name, waveform in waveforms.items():
            _, equalizer, analytic = finite_budget_receiver(effective[name], sigma2, config.P)
            x = torch.einsum("ij,cbj->cbi", waveform, symbols)
            y = torch.einsum("cij,cbj->cbi", channels, x) + physical_noise
            u = torch.einsum("ij,cbj->cbi", waveform.mH, y)
            estimates = torch.einsum("cij,cbj->cbi", equalizer, u)
            rows.append(
                {
                    "waveform": name,
                    "snr_db": snr_db,
                    "k": config.P,
                    "ber": bit_error_rate(bits, estimates),
                    "empirical_nmse": normalized_mse(symbols, estimates),
                    "mean_analytic_mse": scalar(torch.mean(analytic)),
                    "mean_full_mse_bound": common_bound,
                    "bit_count": bits.numel(),
                }
            )
    return rows


def make_figures(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"IDENTITY": "tab:blue", "OFDM": "tab:orange", "OTFS": "tab:green", "AFDM": "tab:red"}
    for metric, ylabel, filename, logarithmic in (
        ("ber", "Bit error rate", "ber_vs_snr_k3.png", True),
        ("empirical_nmse", "Empirical normalized MSE", "nmse_vs_snr_k3.png", True),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for name in colors:
            selected = [row for row in rows if row["waveform"] == name]
            axis.plot(
                [float(row["snr_db"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=name,
                color=colors[name],
            )
        if metric == "empirical_nmse":
            selected = [row for row in rows if row["waveform"] == "IDENTITY"]
            axis.plot(
                [float(row["snr_db"]) for row in selected],
                [float(row["mean_full_mse_bound"]) for row in selected],
                linestyle="--",
                color="black",
                label="Full-information LMMSE bound",
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel(r"$E_s/N_0$ (dB)")
        axis.set_ylabel(ylabel)
        axis.set_title("DD-v1 classical baselines, K=3")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    elapsed_seconds: float,
    test_count: int,
    pytest_summary: str,
    selected_multiplier: int,
    validation_rows: list[dict[str, object]],
    diagnostics: dict[str, object],
    receiver_rows: list[dict[str, object]],
    invariance_rows: list[dict[str, object]],
    invariance_summary: dict[str, float],
    baseline_rows: list[dict[str, object]],
) -> None:
    c1 = selected_multiplier / (2 * config.N)
    validation_table = markdown_table(
        ["m", "c1", "mean analytic MSE"],
        [[str(row["c1_multiplier_m"]), str(row["c1"]), f"{float(row['mean_mse']):.8e}"] for row in validation_rows],
    )
    waveform_table = markdown_table(
        ["Waveform", "Unitarity Fro", "Power rel.", "H=I Fro"],
        [
            [name, f"{values['unitarity_fro']:.3e}", f"{values['power_relative_error']:.3e}", f"{values['identity_channel_fro']:.3e}"]
            for name, values in diagnostics["waveforms"].items()
        ],
    )
    receiver_table = markdown_table(
        ["Waveform", "K", "SNR dB", "Analytic MSE", "Empirical MSE", "Relative gap"],
        [
            [
                str(row["waveform"]), str(row["k"]), str(row["snr_db"]),
                f"{float(row['analytic_mse']):.8e}", f"{float(row['empirical_mse']):.8e}",
                f"{float(row['relative_discrepancy']):.3%}",
            ]
            for row in receiver_rows
        ],
    )
    baseline_table_rows: list[list[str]] = []
    best_by_snr: dict[int, str] = {}
    for snr_db in config.STAGE0_SNRS_DB:
        selected = [row for row in baseline_rows if row["snr_db"] == snr_db]
        best_by_snr[snr_db] = min(selected, key=lambda row: float(row["ber"]))["waveform"]  # type: ignore[index]
        for row in selected:
            baseline_table_rows.append(
                [
                    str(row["snr_db"]), str(row["waveform"]), f"{float(row['ber']):.6e}",
                    f"{float(row['empirical_nmse']):.6e}", f"{float(row['mean_analytic_mse']):.6e}",
                    f"{float(row['mean_full_mse_bound']):.6e}",
                ]
            )
    baseline_table = markdown_table(
        ["SNR dB", "Waveform", "BER", "Empirical NMSE", "Analytic MSE", "Full bound"], baseline_table_rows
    )
    invariant_preview = markdown_table(
        ["Channel", "SNR dB", "Trace MSE", "IDENTITY", "OFDM", "OTFS", "AFDM", "RANDOM"],
        [
            [
                str(channel_index), str(snr_db),
                f"{float(next(row['trace_mse'] for row in invariance_rows if row['channel_index'] == channel_index and row['snr_db'] == snr_db)):.8e}",
                *[
                    f"{float(next(row['receiver_mse'] for row in invariance_rows if row['channel_index'] == channel_index and row['snr_db'] == snr_db and row['waveform'] == name)):.8e}"
                    for name in ("IDENTITY", "OFDM", "OTFS", "AFDM", "RANDOM")
                ],
            ]
            for channel_index in range(4)
            for snr_db in config.FULL_INVARIANCE_SNRS_DB
        ],
    )
    op = diagnostics["operators"]
    report = f"""# Stage 0 Scientific Harness Summary

## Scientific configuration

- Dimension: `N=64`; paths: `P=3`; channel family: `DD-v1`.
- Delays: iid continuous `Uniform[0,4]` normalized delay-bin units; Dopplers: iid continuous `Uniform[-0.75,0.75]` normalized Doppler-bin units. Neither is rounded and near-collisions are allowed.
- Pre-normalization gains: iid `CN(0,1/P)`, i.e. each real component has variance `1/(2P)`. The assembled channel receives one common scale `H = H_tilde / sqrt(||H_tilde||_F^2/N)`; final gains are therefore not iid Gaussian.
- DFT: `F[k,n]=exp(-j2*pi*k*n/N)/sqrt(N)`. DD ordering is `Delta(nu) Pi(tau)`: delay first, then Doppler, with negative Doppler exponent.
- QPSK: `((1-2b0)+j(1-2b1))/sqrt(2)`, unit symbol energy. SNR is `E_s/N_0`; `sigma^2=10^(-SNR_dB/10)` and each AWGN real component has variance `sigma^2/2`.
- Top-K: stable descending magnitude independently per output row, with lower column index winning exact ties. `K` is retained interactions per row.
- Receiver: `G_K=(A_tilde_K^H A_tilde_K + sigma^2 I)^(-1) A_tilde_K^H`, applied to the actual observation `u=A s+n'`. The implementation uses `torch.linalg.solve`.
- Seeds: global `{config.GLOBAL_SEED}`, validation `{config.VALIDATION_SEED}`, Stage-0 evaluation `{config.STAGE0_EVAL_SEED}`, locked final test `{config.TEST_SEED}`, random unitary `{config.RANDOM_UNITARY_SEED}`.
- Bank sizes: validation `{config.N_VALIDATION_CHANNELS}`, Stage-0 evaluation `{config.N_STAGE0_EVAL_CHANNELS}`, final test definition `{config.N_TEST_CHANNELS}`. The final test bank was not generated or evaluated.

## Classical waveform definitions

- IDENTITY: `W=I_64`, the untransformed time-sample control (not a principal classical DD waveform).
- OFDM: `W=F_64^H`.
- OTFS: `W=F_8^H kron I_8`, with delay index varying fastest in flattened vectors.
- AFDM: `W=Lambda_c1^H F_64^H Lambda_c2^H`, where `Lambda_c[n,n]=exp(-j2*pi*c*n^2)`.
- AFDM `c2=sqrt(2)/(100N)={config.AFDM_C2:.17g}`. Validation candidates were `c1=m/(2N)`, `m=1,...,8`, at `K=3` and 12/14/16/18/20 dB.

{validation_table}

Selected and locked: `m={selected_multiplier}`, `c1={c1:.17g}`; selection used only the frozen validation bank and mean exact analytic communication MSE.

## Validation results

- DFT unitarity Frobenius residual: `{op['dft_unitarity_fro']:.3e}`.
- Maximum integer-delay sample residual: `{op['integer_delay_max_abs']:.3e}`.
- Maximum fractional-delay unitarity residual: `{op['fractional_delay_unitarity_fro']:.3e}`; composition residual: `{op['fractional_delay_composition_fro']:.3e}`.
- Doppler unitarity residual: `{op['doppler_unitarity_fro']:.3e}`; composition residual: `{op['doppler_composition_fro']:.3e}`.
- Integer-bin DD commutation/order residual: `{op['dd_integer_bin_commutation_fro']:.3e}`.
- Multipath direct-sum residual: `{op['multipath_assembly_fro']:.3e}`.
- Maximum DD-v1 normalization residual `| ||H||_F^2/N - 1 |`: `{op['dd_v1_normalization_max_abs']:.3e}`.
- Same-seed exact reproduction: `{bool(op['same_seed_exact'])}`; distinct seeds differ: `{bool(op['different_seed_differs'])}`; sampled delay/Doppler fractional fractions: `{op['delay_fractional_fraction']:.3f}` / `{op['doppler_fractional_fraction']:.3f}`.
- AWGN empirical variance relative error: `{op['awgn_variance_relative_error']:.3%}`.
- Maximum noiseless physical-chain/effective-operator element error: `{diagnostics['physical_chain_max_abs']:.3e}`.

{waveform_table}

## Receiver validation

Direct Monte Carlo used 4,096 independent QPSK blocks per case at 15 dB through the physical chain. All relative discrepancies are below 3%.

{receiver_table}

## Full-LMMSE invariant

Four independent DD-v1 channels were checked at 0, 12, 18, and 25 dB. Receiver-form `K=N` MSE was independently compared with `tr[(I+rho H^H H)^(-1)]/N`.

{invariant_preview}

- Maximum receiver-versus-trace absolute discrepancy: `{invariance_summary['max_receiver_vs_trace_abs']:.3e}`.
- Maximum absolute cross-waveform discrepancy: `{invariance_summary['max_cross_waveform_abs']:.3e}`.
- Maximum relative cross-waveform discrepancy: `{invariance_summary['max_cross_waveform_relative']:.3e}`.
- BER equality was **not** used as a full-information invariant.

## Classical baseline results

These are diagnostic curves at `K=3`, 512 frozen Stage-0 evaluation channels, four QPSK blocks per channel, and paired bits/channels/physical AWGN across waveforms. They are not a significance claim.

{baseline_table}

Best observed BER baseline by SNR: {', '.join(f'{snr} dB: {name}' for snr, name in best_by_snr.items())}. No ordering was imposed.

Generated files:

- `outputs/stage0/ber_vs_snr_k3.png`
- `outputs/stage0/nmse_vs_snr_k3.png`
- `outputs/stage0/baseline_metrics.csv`
- `outputs/stage0/afdm_validation_scores.csv`
- `outputs/stage0/full_lmmse_invariance.csv`
- `outputs/stage0/stage0_diagnostics.json`

## Tests and software

- Python `{platform.python_version()}`; PyTorch `{torch.__version__}`; NumPy `{np.__version__}`; Matplotlib `{matplotlib.__version__}`; pytest `{pytest.__version__}`.
- Test result: `{pytest_summary}` ({test_count} tests passed).
- Stage command: `python -m scripts.stage0`.
- Runtime: `{elapsed_seconds:.1f}` seconds.

## Git/repository hygiene

- `outputs/` is ignored and generated artifacts are not tracked.
- `reports/` is tracked; this report is the review artifact.
- `researchPlan.MD` was not modified.
- No final test-bank performance was generated or inspected.
- No Learned waveform, trainable dense transform, or Stage 0F oracle was implemented.

## Problems or anomalies

- The first local pytest invocation failed at collection because its entry point omitted the repository root from `sys.path`; `pytest.ini` now supplies the root and the unchanged numerical tests pass.
- For circular `Pi(ell)` and sampled fractional `Delta(nu)`, the globally scalar Weyl relation has a wrap-boundary phase term unless `nu` is an integer bin. The sign/order identity was therefore checked exactly at integer `nu=3`; the deliberately measured uncorrected fractional-`nu=0.43` residual was `{op['dd_fractional_doppler_scalar_relation_boundary_fro']:.3e}`. The mandated DD-v1 operators were not changed or rounded.
- Stage-0 baseline ranking is reported as observed and was not used to alter the implementation.

## Final gate

STAGE 0 PASS

All mandatory Stage 0A-E checks passed, validation-only AFDM tuning was locked, diagnostic curves were generated, and the final test bank remained untouched.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-only", action="store_true", help="run only validation-bank AFDM selection")
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_channels = sample_dd_v1(config.N_VALIDATION_CHANNELS, config.VALIDATION_SEED).matrices
    selected_multiplier, selected_score, validation_rows = tune_afdm(validation_channels)
    write_csv(OUTPUT_DIR / "afdm_validation_scores.csv", validation_rows)
    print(f"AFDM validation selected m={selected_multiplier}, c1={selected_multiplier/(2*config.N):.17g}, score={selected_score:.8e}")
    if args.tune_only:
        return
    if selected_multiplier != config.AFDM_C1_MULTIPLIER:
        raise RuntimeError(
            f"validation selects AFDM m={selected_multiplier}, but config locks m={config.AFDM_C1_MULTIPLIER}; update src/config.py"
        )

    test_count, pytest_summary = run_pytest()
    operators = operator_diagnostics()
    waveforms = waveform_diagnostics(config.AFDM_C1)
    receiver_rows, physical_chain_max = receiver_validation(config.AFDM_C1)
    invariance_rows, invariance_summary = full_invariance(config.AFDM_C1)
    baseline_rows = baseline_campaign(config.AFDM_C1)

    if max(row["relative_discrepancy"] for row in receiver_rows) > 0.03:
        raise RuntimeError("receiver Monte Carlo discrepancy exceeds 3%")
    if invariance_summary["max_receiver_vs_trace_abs"] > 1e-10:
        raise RuntimeError("full receiver MSE does not match trace bound")
    if invariance_summary["max_cross_waveform_abs"] > 1e-10:
        raise RuntimeError("cross-waveform full-information MSE invariant failed")

    write_csv(OUTPUT_DIR / "baseline_metrics.csv", baseline_rows)
    write_csv(OUTPUT_DIR / "full_lmmse_invariance.csv", invariance_rows)
    diagnostics: dict[str, object] = {
        "operators": operators,
        "waveforms": waveforms,
        "receiver_validation": receiver_rows,
        "physical_chain_max_abs": physical_chain_max,
        "full_invariance": invariance_summary,
        "selected_afdm_c1_multiplier": selected_multiplier,
        "selected_afdm_c1": config.AFDM_C1,
    }
    (OUTPUT_DIR / "stage0_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    make_figures(baseline_rows)
    elapsed = time.perf_counter() - start
    write_report(
        elapsed, test_count, pytest_summary, selected_multiplier, validation_rows, diagnostics,
        receiver_rows, invariance_rows, invariance_summary, baseline_rows,
    )
    print(f"Stage 0 PASS in {elapsed:.1f}s; report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
