"""Scientific core for Pub-1B compute validation and N-scaling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from src.channel import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    ChannelBatch,
    assemble_dd_channel,
    make_generator,
    noise_variance,
)
from src.metrics import Constellation, constellation_encode, nearest_neighbor_decode, random_bits
from src.oracle import polar_project
from src.receiver import analytic_communication_mse, effective_operator, mismatched_lmmse_equalizer
from src.stage2d import ambiguity_coupling, apply_support, score_support_mask
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


PARENT_COMMIT = "737c720b89b64f9bea11b2fd825b8730a77cb14c"
N64_CHECKPOINT_SHA256 = "31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc"
PUB1A_TREE_SHA256 = "a127827f1f045a5bfe1dff17f404f93d39233ac5edd7abc8003cdba251492344"
K_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
COMPUTE_K = (8, 16, 24, 32, 48, 64)
ITERATION_BUDGETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
BER_THRESHOLDS = (1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3)
TAIL_BETA = 0.95
TAIL_LAMBDA = 0.40


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(f"{file_sha256(item)}  {item.relative_to(path)}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ScalingDefinition:
    n: int
    paths: int
    delay_max: float
    doppler_max: float
    primary_k: int
    k_values: tuple[int, ...]
    n_tau: int
    n_nu: int


def otfs_grid(n: int) -> tuple[int, int]:
    """Frozen near-balanced grids; delay is the fastest Kronecker index."""
    grids = {64: (8, 8), 128: (16, 8), 256: (16, 16)}
    if n not in grids:
        raise ValueError(f"Pub-1B defines OTFS grids only for N=64/128/256, got {n}")
    return grids[n]


def scaling_definition(n: int) -> ScalingDefinition:
    if n not in (64, 128, 256):
        raise ValueError("Pub-1B scaling is limited to N=64/128/256")
    n_tau, n_nu = otfs_grid(n)
    return ScalingDefinition(
        n=n,
        paths=3,
        delay_max=4.0 * n / 64,
        doppler_max=0.75 * n / 64,
        primary_k=round(0.375 * n),
        k_values=tuple(round(value * n) for value in K_FRACTIONS),
        n_tau=n_tau,
        n_nu=n_nu,
    )


def scaling_rule_record() -> dict[str, object]:
    return {
        "status": "frozen before Learned-vs-classical test BER",
        "operator_derivation": {
            "Pi": "F^H diag(exp(-j 2 pi k tau/N)) F; tau/N is circular block-delay fraction",
            "Delta": "diag(exp(-j 2 pi nu n/N)); nu/N is normalized cycles per sample",
            "conclusion": "linear tau and nu bounds preserve tau/N and nu/N under exact repository definitions",
        },
        "definitions": [asdict(scaling_definition(n)) for n in (64, 128, 256)],
        "otfs_rule": "most balanced frozen factorization with delay as fastest index and delay dimension compatible with support",
    }


def sample_scaled_dd(count: int, seed: int, n: int) -> ChannelBatch:
    definition = scaling_definition(n)
    generator = make_generator(seed)
    delays = definition.delay_max * torch.rand((count, definition.paths), dtype=REAL_DTYPE, generator=generator)
    dopplers = (2 * torch.rand((count, definition.paths), dtype=REAL_DTYPE, generator=generator) - 1) * definition.doppler_max
    scale = math.sqrt(1.0 / (2 * definition.paths))
    gains = scale * (
        torch.randn((count, definition.paths), dtype=REAL_DTYPE, generator=generator)
        + 1j * torch.randn((count, definition.paths), dtype=REAL_DTYPE, generator=generator)
    )
    gains = gains.to(COMPLEX_DTYPE)
    raw = assemble_dd_channel(delays, dopplers, gains, n=n)
    scales = torch.sqrt(torch.sum(torch.abs(raw) ** 2, dim=(-2, -1)) / n)
    return ChannelBatch(raw / scales[:, None, None], delays, dopplers, gains, scales)


def scaled_path_samples(n: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    definition = scaling_definition(n)
    unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(count).to(REAL_DTYPE)
    return definition.delay_max * unit[:, 0], (2 * unit[:, 1] - 1) * definition.doppler_max


def fixed_scaled_masks(
    waveforms: dict[str, torch.Tensor], n: int, ks: tuple[int, ...], samples: int, seed: int
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, np.ndarray]]:
    delays, dopplers = scaled_path_samples(n, samples, seed)
    scores = {name: ambiguity_coupling(waveform, delays, dopplers) for name, waveform in waveforms.items()}
    masks = {name: {k: score_support_mask(score, k) for k in ks} for name, score in scores.items()}
    return masks, scores


@dataclass(frozen=True)
class SparseRowMatrix:
    """Exactly K stored entries per dense row; no dense measured-path Gram."""

    columns: torch.Tensor
    values: torch.Tensor

    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def k(self) -> int:
        return self.values.shape[1]

    @property
    def nnz(self) -> int:
        return self.n * self.k

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim == 1:
            return torch.sum(self.values * vector[self.columns], dim=-1)
        return torch.sum(self.values[None] * vector[:, self.columns], dim=-1)

    def adjoint_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim == 1:
            output = torch.zeros(self.n, dtype=COMPLEX_DTYPE)
            rows = torch.arange(self.n, dtype=torch.int64)[:, None].expand(self.n, self.k)
            output.index_add_(0, self.columns.reshape(-1), (self.values.conj() * vector[rows]).reshape(-1))
            return output
        output = torch.zeros((vector.shape[0], self.n), dtype=COMPLEX_DTYPE)
        columns = self.columns[None].expand(vector.shape[0], self.n, self.k)
        products = self.values.conj()[None] * vector[:, :, None]
        output.scatter_add_(1, columns.reshape(vector.shape[0], -1), products.reshape(vector.shape[0], -1))
        return output

    def dense(self) -> torch.Tensor:
        output = torch.zeros((self.n, self.n), dtype=COMPLEX_DTYPE)
        output.scatter_(1, self.columns, self.values)
        return output


def sparse_from_mask(matrix: torch.Tensor, mask: torch.Tensor) -> SparseRowMatrix:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or mask.shape != matrix.shape:
        raise ValueError("matrix and mask must be equal square matrices")
    counts = mask.sum(dim=-1)
    if not torch.all(counts == counts[0]) or int(counts[0]) < 1:
        raise ValueError("mask must retain one common positive K per row")
    columns = torch.argsort((~mask).to(torch.int64), dim=-1, stable=True)[:, : int(counts[0])]
    return SparseRowMatrix(columns=columns, values=torch.gather(matrix, 1, columns))


def sparse_from_topk(matrix: torch.Tensor, k: int) -> SparseRowMatrix:
    columns = torch.argsort(torch.abs(matrix), dim=-1, descending=True, stable=True)[:, :k]
    return SparseRowMatrix(columns=columns, values=torch.gather(matrix, 1, columns))


@dataclass(frozen=True)
class MacCount:
    sparse_detector: int
    vector_dot: int
    preconditioner: int
    setup: int

    @property
    def execution(self) -> int:
        return self.sparse_detector + self.vector_dot + self.preconditioner

    @property
    def total(self) -> int:
        return self.execution + self.setup


def cg_mac_count(n: int, k: int, iterations: int) -> MacCount:
    """Complex-equivalent MACs: two sparse products and six N-vector ops/iteration."""
    return MacCount(
        sparse_detector=2 * n * k * iterations,
        vector_dot=n + 6 * n * iterations,
        preconditioner=0,
        setup=n * k,
    )


@dataclass
class CGResult:
    estimate: torch.Tensor
    iterations: int
    relative_residual: float
    converged: bool
    macs: MacCount
    setup_seconds: float
    execution_seconds: float


@dataclass
class CGBatchSnapshot:
    estimate: torch.Tensor
    iterations: torch.Tensor
    relative_residual: torch.Tensor
    converged: torch.Tensor


def sparse_cg_batch_snapshots(
    matrix: SparseRowMatrix,
    observations: torch.Tensor,
    sigma2: float | torch.Tensor,
    budgets: tuple[int, ...] = ITERATION_BUDGETS,
    tolerance: float = 1e-10,
) -> tuple[dict[int, CGBatchSnapshot], float, float]:
    """Vectorized independent CG solves, retaining every requested iteration snapshot."""
    if observations.ndim != 2 or observations.shape[-1] != matrix.n:
        raise ValueError("observations must have shape [batch,N]")
    if tuple(sorted(set(budgets))) != budgets or budgets[0] < 1:
        raise ValueError("budgets must be strictly increasing positive integers")
    started = time.perf_counter()
    rhs = matrix.adjoint_matvec(observations)
    setup_seconds = time.perf_counter() - started
    sigma = torch.as_tensor(sigma2, dtype=REAL_DTYPE)
    estimate = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.sum(residual.conj() * residual, dim=-1).real
    rhs_norm = torch.sqrt(residual_sq).clamp_min(torch.finfo(REAL_DTYPE).tiny)
    relative = torch.sqrt(residual_sq) / rhs_norm
    converged = relative <= tolerance
    iterations = torch.zeros(len(observations), dtype=torch.int64)
    snapshots: dict[int, CGBatchSnapshot] = {}
    started = time.perf_counter()
    for iteration in range(1, budgets[-1] + 1):
        active = ~converged
        product = matrix.adjoint_matvec(matrix.matvec(direction)) + sigma * direction
        denominator = torch.sum(direction.conj() * product, dim=-1).real
        valid = active & torch.isfinite(denominator) & (denominator > 0)
        alpha = torch.where(valid, residual_sq / denominator.clamp_min(torch.finfo(REAL_DTYPE).tiny), 0.0)
        estimate = estimate + alpha[:, None] * direction
        residual = residual - alpha[:, None] * product
        next_sq = torch.sum(residual.conj() * residual, dim=-1).real
        iterations[valid] = iteration
        relative = torch.sqrt(next_sq) / rhs_norm
        converged |= relative <= tolerance
        beta = torch.where(valid & ~converged, next_sq / residual_sq.clamp_min(torch.finfo(REAL_DTYPE).tiny), 0.0)
        direction = residual + beta[:, None] * direction
        residual_sq = next_sq
        if iteration in budgets:
            snapshots[iteration] = CGBatchSnapshot(
                estimate=estimate.clone(), iterations=iterations.clone(),
                relative_residual=relative.clone(), converged=converged.clone(),
            )
    return snapshots, setup_seconds, time.perf_counter() - started


def sparse_cg_lmmse(
    matrix: SparseRowMatrix,
    observation: torch.Tensor,
    sigma2: float | torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float = 1e-10,
    clock: Callable[[], float] = time.perf_counter,
) -> CGResult:
    """CG on A^H A + sigma2 I using only sparse A/A^H products."""
    started = clock()
    rhs = matrix.adjoint_matvec(observation)
    setup_seconds = clock() - started
    sigma = torch.as_tensor(sigma2, dtype=REAL_DTYPE)

    started = clock()
    estimate = torch.zeros(matrix.n, dtype=COMPLEX_DTYPE)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.vdot(residual, residual).real
    rhs_norm = max(float(torch.sqrt(residual_sq)), torch.finfo(REAL_DTYPE).tiny)
    relative = float(torch.sqrt(residual_sq)) / rhs_norm
    converged = relative <= tolerance
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        product = matrix.adjoint_matvec(matrix.matvec(direction)) + sigma * direction
        denominator = torch.vdot(direction, product).real
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            break
        alpha = residual_sq / denominator
        estimate = estimate + alpha * direction
        residual = residual - alpha * product
        next_sq = torch.vdot(residual, residual).real
        iterations = iteration
        relative = float(torch.sqrt(next_sq)) / rhs_norm
        converged = relative <= tolerance
        if converged:
            residual_sq = next_sq
            break
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
    execution_seconds = clock() - started
    return CGResult(
        estimate=estimate,
        iterations=iterations,
        relative_residual=relative,
        converged=converged,
        macs=cg_mac_count(matrix.n, matrix.k, iterations),
        setup_seconds=setup_seconds,
        execution_seconds=execution_seconds,
    )


def direct_lmmse_estimate(retained: torch.Tensor, observation: torch.Tensor, sigma2: float | torch.Tensor) -> torch.Tensor:
    equalizer = mismatched_lmmse_equalizer(retained, sigma2)
    return equalizer @ observation


def detector_counts(
    reference_bits: torch.Tensor,
    reference_indices: torch.Tensor,
    estimates: torch.Tensor,
    constellation: Constellation,
) -> tuple[int, int, int, int]:
    decoded_bits, decoded_indices = nearest_neighbor_decode(estimates, constellation)
    return (
        int(torch.sum(decoded_bits != reference_bits)),
        reference_bits.numel(),
        int(torch.sum(decoded_indices != reference_indices)),
        reference_indices.numel(),
    )


def direct_analytic_mse(actual: torch.Tensor, retained: torch.Tensor, snr_db: float) -> float:
    sigma2 = noise_variance(snr_db)
    equalizer = mismatched_lmmse_equalizer(retained, sigma2)
    return float(analytic_communication_mse(actual, equalizer, sigma2))


def afdm_candidate_multipliers(n: int) -> tuple[int, int]:
    definition = scaling_definition(n)
    endpoint = torch.tensor([-definition.doppler_max, definition.doppler_max], dtype=REAL_DTYPE)
    alpha = torch.floor(endpoint + 0.5).to(torch.int64)
    alpha_max = int(torch.max(torch.abs(alpha)))
    return 2 * alpha_max + 1, 2 * (alpha_max + 1) + 1


def scaled_waveforms(n: int, learned: torch.Tensor, afdm_multiplier: int) -> dict[str, torch.Tensor]:
    definition = scaling_definition(n)
    return {
        f"Learned-N{n}": learned,
        "OTFS": otfs_waveform(definition.n_tau, definition.n_nu),
        "AFDM": afdm_waveform(n=n, c1=afdm_multiplier / (2 * n), c2=math.sqrt(2.0) / (100 * n)),
        "OFDM": ofdm_waveform(n),
    }


def stream_mse_scaled(waveform: torch.Tensor, channels: torch.Tensor, sigma2: torch.Tensor, k: int) -> torch.Tensor:
    effective = waveform.mH @ channels @ waveform
    columns = torch.argsort(torch.abs(effective), dim=-1, descending=True, stable=True)[..., :k]
    mask = torch.zeros_like(effective, dtype=torch.bool)
    mask.scatter_(-1, columns, True)
    retained = torch.where(mask, effective, torch.zeros((), dtype=COMPLEX_DTYPE))
    n = waveform.shape[-1]
    identity = torch.eye(n, dtype=COMPLEX_DTYPE)
    sigma = torch.as_tensor(sigma2, dtype=REAL_DTYPE)
    if sigma.ndim == 0:
        sigma = sigma.expand(channels.shape[0])
    system = retained.mH @ retained + sigma[:, None, None] * identity
    equalizer = torch.linalg.solve(system, retained.mH)
    residual = equalizer @ effective - identity
    return torch.sum(torch.abs(residual) ** 2, dim=-1) + sigma[:, None] * torch.sum(torch.abs(equalizer) ** 2, dim=-1)


def stream_cvar(values: torch.Tensor, beta: float = TAIL_BETA) -> torch.Tensor:
    count = max(1, math.ceil((1 - beta) * values.shape[-1]))
    return torch.topk(values, count, dim=-1, largest=True, sorted=True).values.mean(dim=-1)


def validation_metrics_scaled(
    waveform: torch.Tensor,
    channels: torch.Tensor,
    k: int,
    snrs_db: tuple[int, ...] = (12, 15, 18, 20),
    chunk: int = 8,
) -> tuple[float, float]:
    mean_total = 0.0
    cvar_total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(channels), chunk):
            selected = channels[start : start + chunk]
            for snr in snrs_db:
                streams = stream_mse_scaled(waveform, selected, noise_variance(snr), k)
                mean_total += float(streams.mean(dim=-1).sum())
                cvar_total += float(stream_cvar(streams).sum())
                count += len(selected)
    return mean_total / count, cvar_total / count


def clean_reconstruction_error(waveform: torch.Tensor) -> float:
    n = waveform.shape[-1]
    generator = make_generator(16001 + n)
    symbols = torch.randn(n, dtype=COMPLEX_DTYPE, generator=generator)
    return float(torch.linalg.vector_norm(waveform.mH @ (waveform @ symbols) - symbols) / torch.linalg.vector_norm(symbols))


def full_mse_invariance(channels: torch.Tensor, waveforms: dict[str, torch.Tensor], snr_db: float = 18) -> float:
    values = []
    with torch.no_grad():
        for waveform in waveforms.values():
            actual = effective_operator(channels, waveform)
            equalizer = mismatched_lmmse_equalizer(actual, noise_variance(snr_db))
            values.append(float(analytic_communication_mse(actual, equalizer, noise_variance(snr_db)).mean()))
    return max(values) - min(values)
