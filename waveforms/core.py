"""Explicitly authorized reference arithmetic; one thread, no device fallback."""
import hashlib
import json
import torch

REAL = torch.float64
COMPLEX = torch.complex128


def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(REAL)
    return {"device": "cpu", "real_dtype": str(REAL), "complex_dtype": str(COMPLEX),
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "authorization": "Owner authorized CPU float64/complex128 for this package, 2026-11-16"}


def seed(*namespace):
    # Same namespaces reproduce completed WMS draws, including dependence on count.
    value = json.dumps(["wms-v1", *namespace], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


def rng(*namespace):
    return torch.Generator(device="cpu").manual_seed(seed(*namespace))
