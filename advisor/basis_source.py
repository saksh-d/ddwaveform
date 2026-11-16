"""Basis-selection boundary. Training is deliberately unavailable in this branch."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch
from .artifacts import load_reference
from .selection import identity


def require_available_mode(mode):
    if mode == "train_new":
        raise NotImplementedError(
            "train_new is unavailable: training belongs in the separately authorized follow-on branch. "
            "No training started and no existing basis was silently loaded.")
    if mode != "load_existing":
        raise ValueError("MODE must be load_existing or train_new (currently unavailable)")


@dataclass(frozen=True)
class SelectedBasis:
    """Common provider result: synthesis W, its fixed-K support and provenance.

    A later trainer must return this contract into the same downstream workflow.
    It must supply support belonging to its returned W, not borrow historical
    Learned support/curves. No trainer or fitting routine is defined here.
    """
    matrix: torch.Tensor
    fixed_q_mask: torch.Tensor
    provenance: dict

    def validate(self):
        w, mask = self.matrix, self.fixed_q_mask
        n = w.shape[0] if isinstance(w, torch.Tensor) and w.ndim == 2 else 0
        k = 3*n//8
        if not isinstance(w, torch.Tensor) or n not in (64,128,256) or w.shape != (n,n) or w.dtype != torch.complex128 or w.device.type != "cpu":
            raise ValueError("Selected basis must be CPU complex128 square W[N,N], N=64/128/256")
        if not torch.isfinite(w).all() or torch.linalg.matrix_norm(w.mH @ w - torch.eye(n, dtype=w.dtype)) > 1e-10:
            raise ValueError("Selected basis must be finite and unitary")
        if not isinstance(mask, torch.Tensor) or mask.shape != (n,n) or mask.dtype != torch.bool or mask.device.type != "cpu" or not (mask.sum(-1) == k).all():
            raise ValueError("Provider must supply its own CPU boolean fixed-K support")
        if not isinstance(self.provenance, dict) or not self.provenance.get("source"):
            raise ValueError("Basis provider provenance requires a source")
        return self

    def describe(self):
        digest = hashlib.sha256(self.matrix.detach().contiguous().numpy().tobytes()).hexdigest()
        return {**self.provenance, "label": "Learned", "shape": list(self.matrix.shape),
                "matrix_content_sha256": digest}

    def matches_saved_reference(self, artifacts):
        if self.matrix.shape != (64,64): return False
        w = load_reference(artifacts)
        with np.load(Path(artifacts) / "focused/data/supports.npz", allow_pickle=False) as data:
            mask = torch.from_numpy(data["Learned_mask"].copy())
        return torch.equal(self.matrix, w) and torch.equal(self.fixed_q_mask, mask)


def select_basis(mode, artifacts):
    """Only implemented provider: verified existing export. Never a train fallback."""
    require_available_mode(mode)
    with np.load(Path(artifacts) / "focused/data/supports.npz", allow_pickle=False) as data:
        mask = torch.from_numpy(data["Learned_mask"].copy())
    return SelectedBasis(load_reference(artifacts), mask,
        {**identity(), "source": "reference/learned_w.npy", "basis_mode": "load_existing",
         "training_status": "Not run: loaded an existing trained basis"}).validate()
