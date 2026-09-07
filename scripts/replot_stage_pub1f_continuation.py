"""Replot Pub-1F continuation figures from saved source tables only."""

import argparse

from scripts.stage_pub1f_continuation import phase_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    phase_figures(args.smoke)


if __name__ == "__main__":
    main()
