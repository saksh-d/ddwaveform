"""Seal only completed N64 evidence from explicit read-only source directories.

Checks each file before/after copying and rejects a changing source. Larger-N,
logs, caches, pilot results and generated figures are deliberately not selected.
"""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waveforms.artifacts import sha, verify, write_json


def selected(wms, cfo):
    pairs = []
    def add(source, name):
        pairs.append((source, name))
    def tree(base, relative, patterns, destination):
        for pattern in patterns:
            for path in sorted((base / relative).glob(pattern)):
                if path.is_file() and "__pycache__" not in path.parts and "scaling_stress" not in path.name:
                    add(path, destination + "/" + str(path.relative_to(base / relative)))
    add(wms / "reference/artifacts/learned_w.npy", "reference/learned_w.npy")
    tree(wms, "results/focused_figures/data", ["bases.npz", "channels.npz", "supports.npz"], "focused/data")
    tree(wms, "results/focused_figures", ["*.csv", "*.json", "*.md"], "focused")
    tree(wms, "results/focused_figures/communication", ["**/batch*.npz", "**/per_channel.npz", "**/complete.json"], "focused/communication")
    tree(wms, "results/focused_figures/snr25", ["*.csv", "README.md"], "focused/snr25")
    for sub in ["orthogonality_bridge", "block_frequency"]:
        tree(wms, "results/focused_figures/" + sub, ["*.npz", "*.csv", "summary.json", "association.json", "protocol.json", "release*.json", "REPORT.md"], "supplementary/" + sub)
    tree(wms, "results/main", ["*.csv", "REPORT.md", "STATUS.json", "config.json", "manifest.json", "rng_manifest*.json", "verification.json", "environment.json", "structural_revision*.json", "intervention_bases.npz", "projector_profile_comparison.npz"], "supplementary/main")
    tree(wms, "results/main/communication", ["*.npz", "*.json"], "supplementary/main/communication")
    tree(wms, "results/main/training", ["**/trace.csv", "**/selected.json", "**/best.pt"], "supplementary/main/training")
    tree(wms, "results/main/discovery", ["*.npz"], "supplementary/main/discovery")
    for file in ["README.md", "REPORT.md", "PROTOCOL.md", "PROVENANCE.md", "STRUCTURAL_AMENDMENT.md", "requirements.lock.txt"]:
        add(wms / file, "sources/wms/" + file)
    # Archived source is evidence, never imported at runtime.
    tree(wms, "wms", ["*.py"], "sources/wms/wms")
    tree(wms, "tests", ["*.py"], "sources/wms/tests")
    for relative in ["results/main/executed_source", "results/main/executed_source_v2"]:
        tree(wms, relative, ["**/*.py", "**/*.json", "*.md"], "sources/wms/" + Path(relative).name)
    for stage in ["stage1_fullframe", "stage2_awgn/residual_cfo", "stage3_multipath"]:
        tree(cfo, stage, ["channel.py", "transmitter.py", "receiver.py", "comm_core.py", "README.md"], "sources/cfo/" + stage)
    for name in ["stage1_checkpoint.pt", "report.md", "summary_metrics.csv", "operator_diagnostics.csv", "ber_vs_cfo.csv", "ber_vs_snr_by_cfo.csv", "stage_summary.csv"]:
        add(cfo / "stage2_awgn/stage1_linear_outputs/16qam_n45_r9" / name, "residual_cfo/" + name)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wms", required=True, type=Path)
    parser.add_argument("--cfo", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("local/sealed"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/artifacts.json"))
    parser.add_argument("--bundle", type=Path, default=Path("local/advisor-artifacts.tar.gz"))
    args = parser.parse_args()
    if args.out.exists() or args.bundle.exists():
        parser.error("Use new snapshot/bundle destinations; existing evidence is never overwritten")
    pairs = selected(args.wms, args.cfo)
    records = {}
    for source, name in pairs:
        before = source.stat()
        digest = sha(source)
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest != sha(source) or digest != sha(target):
            raise RuntimeError(f"Changing source rejected: {source}")
        records[name] = {"sha256": digest, "bytes": before.st_size,
                         "source": ("wms/" + str(source.relative_to(args.wms))) if source.is_relative_to(args.wms) else "cfo/" + str(source.relative_to(args.cfo))}
    manifest = {"version": 1, "pinned_ddwaveform_commit": "9aae23ed6f9ac6e2a2e3e057f38693a148886400", "scope": "Completed N64 only; no active scaling files", "files": records}
    verify(args.out, manifest)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.bundle, "w:gz") as archive:
        for name in sorted(records):
            archive.add(args.out / name, arcname=name, recursive=False)
    manifest["bundle_sha256"] = sha(args.bundle)
    write_json(args.manifest, manifest)
    write_json(args.out / "LOCAL_MANIFEST.json", manifest)
    for path in args.out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records.values()), "bundle": str(args.bundle), "sha256": manifest["bundle_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
