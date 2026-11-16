"""Hash-checked local provisioning. No downloads, implicit searches or pickle loads."""
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

EXPECTED_W = "11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    return path


def verify(root, manifest):
    root = Path(root)
    for name, record in manifest["files"].items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"Missing or altered local artifact: {name}. Run bootstrap.")
    return {"files": len(manifest["files"]), "bytes": sum(x["bytes"] for x in manifest["files"].values())}


def provision(bundle, destination, manifest):
    """Validate every file in staging before publishing a read-only local snapshot."""
    destination = Path(destination)
    if "bundle_sha256" in manifest and sha(bundle) != manifest["bundle_sha256"]:
        raise ValueError("Bundle SHA256 differs from tracked artifact manifest")
    if destination.exists():
        raise FileExistsError(f"Destination exists; use a fresh directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "snapshot"
        stage.mkdir()
        with tarfile.open(bundle, "r:*") as archive:
            seen = set()
            for member in archive:
                name = member.name
                relative = safe_relative(name)
                if not member.isfile() or name not in manifest["files"] or name in seen:
                    raise ValueError(f"Unexpected bundle member: {name}")
                seen.add(name)
                if member.size != manifest["files"][name]["bytes"]:
                    raise ValueError(f"Wrong size: {name}")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        verify(stage, manifest)
        write_json(stage / "LOCAL_MANIFEST.json", manifest)
        for path in stage.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.rename(stage, destination)
    return verify(destination, manifest)


def load_reference(root):
    """Original array: columns synthesize samples, never transposed or permuted."""
    import numpy as np
    import torch
    path = Path(root) / "reference/learned_w.npy"
    if not path.exists():
        raise FileNotFoundError("Local Learned export absent. Provision the bundle; code-only smoke needs none.")
    if sha(path) != EXPECTED_W:
        raise ValueError("Historical Learned file hash differs")
    array = np.load(path, allow_pickle=False)
    if array.shape != (64, 64) or array.dtype != np.complex128:
        raise ValueError("Historical Learned shape/dtype differs")
    return torch.from_numpy(array.copy())
