"""Scientific and execution primitives for Pub-1F."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import platform
import time

import numpy as np
import torch

from src import config
from src.channel import COMPLEX_DTYPE, REAL_DTYPE, complex_awgn, make_generator, noise_variance, sample_dd_v1
from src.learning import hard_topk_communication_mse, stage1a_learning_rate, validation_objective
from src.metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from src.oracle import polar_project
from src.pub1b.core import SparseRowMatrix, cg_mac_count, sparse_cg_lmmse, sparse_from_topk
from src.pub1e.core import ResidualStatistics, controlled_fractionality_2d, residual_statistics
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer, rowwise_topk
from src.tail_learning import per_stream_exact_mse, stage1a2_learning_rate, stream_cvar, validation_tail_metrics


NAMESPACE = "pub1f-v1"


def environment_record() -> dict[str, object]:
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
        "dtype": "complex128/float64", "device": "cpu",
    }


def semantic_seed(role: str, *, namespace: str = NAMESPACE, **fields: object) -> int:
    """Stable role-separated seed; semantic fields are sorted, never positional."""
    encoded = "|".join([namespace, role] + [f"{key}={fields[key]}" for key in sorted(fields)])
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def condition_channels(condition: str, count: int, role: str, *, namespace: str = NAMESPACE, **fields: object) -> torch.Tensor:
    seed = semantic_seed(role, namespace=namespace, condition=condition, **fields)
    if condition == "nominal_dd_v1" or condition == "full_information_nominal":
        return sample_dd_v1(count, seed).matrices
    values = {"joint_0.5": (0.5, 0.5), "joint_0.2_probe": (0.2, 0.2), "integer": (0.0, 0.0)}
    if condition not in values:
        raise ValueError(f"unknown Pub-1F condition: {condition}")
    epsilon_tau, epsilon_nu = values[condition]
    return controlled_fractionality_2d(count, seed, epsilon_tau, epsilon_nu).matrices


def unitary_residual(waveform: torch.Tensor) -> float:
    identity = torch.eye(waveform.shape[-1], dtype=waveform.dtype)
    return float(torch.linalg.matrix_norm(waveform.mH @ waveform - identity))


@dataclass(frozen=True)
class TrainingProtocol:
    seed_id: int
    stage: str
    steps: int
    batch_size: int
    cadence: int
    k: int
    parent_sha256: str | None
    namespace: str = NAMESPACE


def _stage_metrics(waveform: torch.Tensor, validation: torch.Tensor, k: int) -> tuple[float, float]:
    metrics = validation_tail_metrics(waveform, validation, k=k)
    return float(metrics["mean_mse"]), float(metrics["cvar95"])


def train_stage(
    initial: torch.Tensor,
    validation: torch.Tensor,
    protocol: TrainingProtocol,
    *,
    reference_parent_mean: float | None = None,
    resume: dict[str, object] | None = None,
    save_callback=None,
) -> dict[str, object]:
    """One resumable projected-Adam stage with explicit selection semantics."""
    if protocol.stage not in {"parent", "mean_continuation", "tail_continuation"}:
        raise ValueError("invalid training stage")
    parameter = torch.nn.Parameter(polar_project(initial.to(COMPLEX_DTYPE)))
    continuation = protocol.stage != "parent"
    optimizer = torch.optim.Adam([parameter], lr=stage1a2_learning_rate(1) if continuation else stage1a_learning_rate(1))
    initial_mean, initial_cvar = _stage_metrics(parameter.detach(), validation, protocol.k)
    if resume is None:
        start_step = 1
        best = parameter.detach().clone()
        best_mean, best_cvar, best_step = initial_mean, initial_cvar, 0
        endpoint = parameter.detach().clone()
        trace = [{"step": 0, "validation_mean_mse": initial_mean, "validation_cvar95": initial_cvar, "eligible": True}]
        runtime_seconds = 0.0
        max_unitarity = unitary_residual(parameter.detach())
        nonfinite = False
    else:
        if resume["protocol"] != asdict(protocol):
            raise RuntimeError("incompatible Pub-1F resumable training state")
        parameter.data.copy_(resume["parameter"])
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"]) + 1
        best = resume["best"].clone()
        best_mean, best_cvar, best_step = float(resume["best_mean"]), float(resume["best_cvar"]), int(resume["best_step"])
        endpoint = resume["parameter"].clone()
        trace = list(resume["trace"])
        runtime_seconds = float(resume["runtime_seconds"])
        max_unitarity = float(resume["max_unitarity"])
        nonfinite = bool(resume["nonfinite"])
    started = time.perf_counter()
    for step in range(start_step, protocol.steps + 1):
        optimizer.param_groups[0]["lr"] = stage1a2_learning_rate(step) if continuation else stage1a_learning_rate(step)
        channel_seed = semantic_seed("training_channel", namespace=protocol.namespace, seed_id=protocol.seed_id, stage="continuation" if continuation else "parent", step=step)
        snr_seed = semantic_seed("training_snr", namespace=protocol.namespace, seed_id=protocol.seed_id, stage="continuation" if continuation else "parent", step=step)
        channels = sample_dd_v1(protocol.batch_size, channel_seed).matrices
        snr_generator = make_generator(snr_seed)
        snr_db = 12.0 + 8.0 * torch.rand(protocol.batch_size, dtype=REAL_DTYPE, generator=snr_generator)
        if protocol.stage == "tail_continuation":
            streams = per_stream_exact_mse(parameter, channels, noise_variance(snr_db), protocol.k)
            means = streams.mean(dim=-1)
            objective = (means + 0.40 * (stream_cvar(streams, 0.95) - means)).mean()
        else:
            objective = hard_topk_communication_mse(parameter, channels, noise_variance(snr_db), protocol.k).mean()
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(objective):
            nonfinite = True
            raise RuntimeError(f"non-finite {protocol.stage} objective at seed {protocol.seed_id}, step {step}")
        objective.backward()
        if not torch.isfinite(parameter.grad.real).all() or not torch.isfinite(parameter.grad.imag).all():
            nonfinite = True
            raise RuntimeError(f"non-finite {protocol.stage} gradient at seed {protocol.seed_id}, step {step}")
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(polar_project(parameter))
            max_unitarity = max(max_unitarity, unitary_residual(parameter))
        if step % protocol.cadence == 0 or step == protocol.steps:
            mean_mse, cvar95 = _stage_metrics(parameter.detach(), validation, protocol.k)
            eligible = reference_parent_mean is None or mean_mse <= 1.02 * reference_parent_mean
            if protocol.stage == "tail_continuation":
                improved = eligible and (cvar95, mean_mse, step) < (best_cvar, best_mean, best_step)
            else:
                improved = (mean_mse, step) < (best_mean, best_step)
            if improved:
                best = parameter.detach().clone()
                best_mean, best_cvar, best_step = mean_mse, cvar95, step
            endpoint = parameter.detach().clone()
            trace.append({"step": step, "training_objective": float(objective), "validation_mean_mse": mean_mse, "validation_cvar95": cvar95, "eligible": eligible, "selected_so_far": best_step})
            runtime_seconds += time.perf_counter() - started
            state = {
                "protocol": asdict(protocol), "step": step, "parameter": endpoint,
                "optimizer": optimizer.state_dict(), "best": best, "best_mean": best_mean,
                "best_cvar": best_cvar, "best_step": best_step, "trace": trace,
                "runtime_seconds": runtime_seconds, "max_unitarity": max_unitarity, "nonfinite": nonfinite,
            }
            if save_callback is not None:
                save_callback(state)
            started = time.perf_counter()
    endpoint_mean, endpoint_cvar = _stage_metrics(endpoint, validation, protocol.k)
    return {
        "selected": best, "endpoint": endpoint, "best_step": best_step,
        "best_mean": best_mean, "best_cvar95": best_cvar,
        "endpoint_mean": endpoint_mean, "endpoint_cvar95": endpoint_cvar,
        "trace": trace, "runtime_seconds": runtime_seconds,
        "max_unitarity": max_unitarity, "nonfinite": nonfinite,
        "protocol": asdict(protocol),
    }


def receiver_estimate(
    actual: torch.Tensor,
    observations: torch.Tensor,
    k: int,
    sigma2: float,
    family: str,
    *,
    regularization_scale: float = 1.0,
    stats: ResidualStatistics | None = None,
    support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    retained = actual if family == "full" else (
        rowwise_topk(actual, k) if support is None else torch.where(support, actual, torch.zeros((), dtype=actual.dtype))
    )
    if family == "residual_aware":
        if stats is None:
            raise ValueError("residual-aware receiver requires matching statistics")
        b = retained + stats.mean
        eye = torch.eye(actual.shape[-1], dtype=COMPLEX_DTYPE)
        system = b @ b.mH + stats.receive_covariance + sigma2 * eye
        equalizer = torch.linalg.solve(system, b).mH
    else:
        scale = regularization_scale if family == "tuned" else 1.0
        equalizer = mismatched_lmmse_equalizer(retained, scale * sigma2)
    return torch.einsum("cij,cbj->cbi", equalizer, observations), retained


def communication_counts(
    channels: torch.Tensor,
    waveform: torch.Tensor,
    constellation: Constellation,
    snr_db: int,
    k: int,
    family: str,
    bits_per_channel: int,
    *,
    condition: str,
    bank_role: str,
    regularization_scale: float = 1.0,
    stats: ResidualStatistics | None = None,
    support: torch.Tensor | None = None,
    namespace: str = NAMESPACE,
) -> list[dict[str, object]]:
    blocks = math.ceil(bits_per_channel / (config.N * constellation.bits_per_symbol))
    bit_shape = (len(channels), blocks, config.N, constellation.bits_per_symbol)
    bits = random_bits(bit_shape, make_generator(semantic_seed("symbols", namespace=namespace, bank_role=bank_role, condition=condition, modulation=constellation.name, snr_db=snr_db)))
    symbols, true_indices = constellation_encode(bits, constellation)
    noise = complex_awgn((len(channels), blocks, config.N), noise_variance(snr_db), make_generator(semantic_seed("noise", namespace=namespace, bank_role=bank_role, condition=condition, modulation=constellation.name, snr_db=snr_db)))
    physical = torch.einsum("ij,cbj->cbi", waveform, symbols)
    received = torch.einsum("cij,cbj->cbi", channels, physical) + noise
    observations = torch.einsum("ij,cbj->cbi", waveform.mH, received)
    actual = effective_operator(channels, waveform)
    estimates, retained = receiver_estimate(actual, observations, k, float(noise_variance(snr_db)), family, regularization_scale=regularization_scale, stats=stats, support=support)
    decoded_bits, decoded_indices = nearest_neighbor_decode(estimates, constellation)
    bit_errors = torch.sum(decoded_bits != bits, dim=(1, 2, 3))
    symbol_errors = torch.sum(decoded_indices != true_indices, dim=(1, 2))
    tested_bits = blocks * config.N * constellation.bits_per_symbol
    tested_symbols = blocks * config.N
    retained_energy = torch.sum(torch.abs(retained) ** 2, dim=(-2, -1)) / torch.sum(torch.abs(actual) ** 2, dim=(-2, -1))
    return [{
        "channel_index": index, "bit_errors": int(bit_errors[index]), "total_bits": tested_bits,
        "ber": int(bit_errors[index]) / tested_bits, "symbol_errors": int(symbol_errors[index]),
        "total_symbols": tested_symbols, "ser": int(symbol_errors[index]) / tested_symbols,
        "retained_energy": float(retained_energy[index]),
    } for index in range(len(channels))]


def aggregate_counts(rows: list[dict[str, object]]) -> dict[str, object]:
    errors = sum(int(row["bit_errors"]) for row in rows)
    bits = sum(int(row["total_bits"]) for row in rows)
    return {"bit_errors": errors, "total_bits": bits, "ber": errors / bits, "n_channels": len(rows), "mean_retained_energy": float(np.mean([row["retained_energy"] for row in rows]))}


def paired_interval(left: list[dict[str, object]], right: list[dict[str, object]], comparison: str, resamples: int = 10_000) -> dict[str, object]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired channel rows must align")
    differences = np.asarray([float(a["ber"]) - float(b["ber"]) for a, b in zip(left, right)])
    rng = np.random.default_rng(semantic_seed("bootstrap", comparison=comparison))
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {"comparison": comparison, "mean_difference": float(differences.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "resamples": resamples}


def operation_ledger(n: int, k: int, iterations: int) -> dict[str, int]:
    model = cg_mac_count(n, k, iterations)
    return {
        "n": n, "k": k, "iterations": iterations,
        "rhs_sparse_complex_macs": n * k,
        "initial_inner_product_complex_equivalent_macs": n,
        "per_iteration_sparse_complex_macs": 2 * n * k,
        "per_iteration_vector_complex_equivalent_macs": 6 * n,
        "algorithmic_total_complex_equivalent_macs": model.total,
        "dense_zero_matvec_complex_macs_per_pair": 2 * n * n,
    }


def storage_ledger(n: int, k: int, *, residual_aware: bool) -> dict[str, int]:
    complex_bytes = torch.empty((), dtype=COMPLEX_DTYPE).element_size()
    index_bytes = torch.empty((), dtype=torch.int64).element_size()
    values = n * k * complex_bytes
    indices = n * k * index_bytes
    dense = n * n * complex_bytes
    residual = 2 * dense if residual_aware else 0
    return {
        "dense_coefficients_bytes": dense, "sparse_values_bytes": values,
        "sparse_indices_bytes": indices, "sparse_total_bytes": values + indices,
        "waveform_bytes": dense, "fixed_support_bool_bytes": n * n,
        "residual_mean_covariance_bytes": residual,
        "persistent_receiver_state_bytes": values + indices + dense + residual,
    }


def benchmark_paths(matrix: torch.Tensor, observation: torch.Tensor, sigma2: float, *, repetitions: int, warmup: int, max_iterations: int) -> list[dict[str, object]]:
    sparse = sparse_from_topk(matrix, min(max_iterations, matrix.shape[-1]))
    retained = sparse.dense()
    def dense_call():
        return mismatched_lmmse_equalizer(retained, sigma2) @ observation
    def sparse_call():
        return sparse_cg_lmmse(sparse, observation, sigma2, max_iterations=max_iterations, tolerance=1e-10).estimate
    direct = dense_call()
    iterative = sparse_call()
    relative = float(torch.linalg.vector_norm(iterative - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(REAL_DTYPE).tiny))
    if relative > 1e-8:
        raise RuntimeError(f"benchmark estimators disagree: {relative}")
    rows = []
    for name, function in (("dense_zero_direct", dense_call), ("explicit_sparse_matrix_free", sparse_call)):
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns(); function(); samples.append(time.perf_counter_ns() - started)
        rows.append({"implementation": name, "median_ns": int(np.median(samples)), "p95_ns": int(np.quantile(samples, .95)), "repetitions": repetitions, "numerical_relative_error": 0.0 if name == "dense_zero_direct" else relative})
    return rows


def environment_record() -> dict[str, object]:
    return {
        "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "precision": "complex128/float64",
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
    }
