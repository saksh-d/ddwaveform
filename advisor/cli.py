"""Small explicit commands: no train/all/background workflows."""
import argparse
import json
from pathlib import Path
from .artifacts import write_json
from .core import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "export-dd", "export-cfo", "figures", "ber", "verify-artifacts", "internal-supplementary", "internal-figures", "internal-studies"])
    parser.add_argument("--artifacts", type=Path, help="Provisioned local snapshot; required for Learned results")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--snrs", type=float, nargs="+", default=[0., 18.])
    parser.add_argument("--eps", type=float, nargs="+", default=[0., .05, .1])
    parser.add_argument("--snr-max", type=float, default=25., help="Display limit only; --full-range retains every saved SNR")
    parser.add_argument("--full-range", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("local/internal") / args.command.removeprefix("internal-") if args.command.startswith("internal-") else Path("local/results")
    environment = configure()
    from .experiments import ber, check, export_cfo, export_dd
    if args.command in ["export-cfo", "figures", "verify-artifacts", "internal-supplementary", "internal-figures", "internal-studies"] and not args.artifacts:
        parser.error("This command requires --artifacts PATH. Run scripts/bootstrap.py first.")
    if args.command == "smoke":
        result = check(args.artifacts)
    elif args.command == "export-dd":
        result = export_dd(args.out, args.artifacts, args.channels)
    elif args.command == "export-cfo":
        result = export_cfo(args.out, args.artifacts, args.eps)
    elif args.command == "ber":
        result = ber(args.out, args.artifacts, args.channels, args.blocks, args.snrs)
    elif args.command == "figures":
        from .figures import figures
        result = figures(args.artifacts, args.out, None if args.full_range else args.snr_max)
    elif args.command == "internal-supplementary":
        from .supplementary import figures
        result = figures(args.artifacts, args.out)
    elif args.command == "internal-figures":
        from .internal_figures import figures
        result = figures(args.artifacts, args.out, None if args.full_range else args.snr_max)
    elif args.command == "internal-studies":
        from .internal_studies import figures
        result = figures(args.artifacts, args.out)
    else:
        from .artifacts import verify
        result = verify(args.artifacts, json.loads((args.artifacts / "LOCAL_MANIFEST.json").read_text()))
    print(json.dumps({"environment": environment, "result": result}, indent=2))


if __name__ == "__main__":
    main()
