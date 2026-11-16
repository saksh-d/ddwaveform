"""Historical positive-sign CFO, rectangular W and distinct receive analysis R.

V=R^H is the receive-column convention; A=V^H H W=R H W. The historical
einsum returns A^T (input,output), which is saved explicitly for comparison.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .artifacts import sha

PAYLOAD_HASH = "380da00b389b047699c53cab6f1f80c28675dfc8b4835a70e54f7be1325ed64c"


def operator(eps, m=64):
    eps = torch.as_tensor(eps, dtype=torch.float32)
    samples = torch.arange(m, dtype=torch.float32)
    return torch.diag_embed(torch.exp(2j * math.pi * eps[..., None] * samples / m).to(torch.complex64))


def apply_residual_cfo(x, eps):
    # Historical rows are blocks, columns are samples; positive CFO sign.
    return x * operator(eps, x.shape[-1]).diagonal(dim1=-2, dim2=-1)


def effective(w, r, eps):
    return r @ operator(eps, w.shape[0]) @ w


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def payload_hash(payload):
    values = {k: _json_safe(payload.get(k, {})) for k in
              ["format", "base_seed", "config", "provenance", "stage1_acceptance_summary"]}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode())
    for key in ["learned_tx", "learned_rx"]:
        array = payload[key].detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_qualified(root):
    root = Path(root)
    path = root / "residual_cfo/stage1_checkpoint.pt"
    manifest = json.loads((root / "LOCAL_MANIFEST.json").read_text())
    expected = manifest["files"]["residual_cfo/stage1_checkpoint.pt"]["sha256"]
    if sha(path) != expected:
        raise ValueError("CFO checkpoint file hash differs")
    # This known local payload uses NumPy config arrays and a Path; no source import.
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct, np.ndarray,
            np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType, type(Path())]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "stage1_linear_receiver_v2" or payload_hash(payload) != PAYLOAD_HASH or payload["payload_hash_sha256"] != PAYLOAD_HASH:
        raise ValueError("CFO payload format/hash differs")
    w, r = payload["learned_tx"], payload["learned_rx"]
    if w.shape != (64, 45) or r.shape != (45, 64) or w.dtype != torch.complex64 or r.dtype != torch.complex64:
        raise ValueError("CFO shape/dtype differs")
    if not torch.isfinite(w).all() or not torch.isfinite(r).all():
        raise ValueError("Nonfinite CFO basis")
    acceptance = payload["stage1_acceptance_summary"]
    if not acceptance["passed"]:
        raise ValueError("Historical acceptance failed")
    a0 = effective(w, r, 0.)
    eye = torch.eye(45, dtype=torch.complex64)
    identity = float((a0 - eye).abs().square().sum() / 45)
    power = a0.abs().square()
    offdiag = float((power.sum() - power.diagonal().sum()).clamp_min(0) / power.sum())
    saved = acceptance["metrics"]
    if abs(identity - saved["clean_identity_loss"]) > 2e-7 or abs(offdiag - saved["clean_offdiag_leakage"]) > 2e-7:
        raise ValueError("CFO clean operator does not reproduce saved acceptance metrics")
    checks = {"checkpoint_file_sha256": expected, "payload_sha256": PAYLOAD_HASH,
              "W_shape": list(w.shape), "R_shape": list(r.shape), "dtype": str(w.dtype),
              "column_energy_max_error": float((w.abs().square().sum(0) - 1).abs().max()),
              "clean_identity_loss": identity, "clean_offdiag_leakage": offdiag,
              "historical_acceptance": acceptance,
              "qualification": "Hash, geometry and clean operator only; saved BER not rerun"}
    return w, r, checks
