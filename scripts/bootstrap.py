"""Provision a supplied local bundle, without source projects or network access."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advisor.artifacts import provision, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path)
parser.add_argument("--dest", type=Path, default=Path("local/artifacts"))
parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs/artifacts.json")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
if args.verify:
    result = verify(args.dest, manifest)
elif args.bundle:
    result = provision(args.bundle, args.dest, manifest)
else:
    parser.error("Supply --bundle PATH or --verify. No automatic artifact download.")
print(json.dumps(result, indent=2))
