"""Internal descriptive review of existing paired counts; no simulations/selection fit."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advisor.core import configure
from advisor.artifacts import sha, write_json
from advisor.experiments import output_dir
from advisor.selection import INTERNAL_ID, identity


def review(root, out):
    import numpy as np
    import pandas as pd
    root, out = Path(root), output_dir(out)
    path = root / "supplementary/main/communication_summary.csv"
    table = pd.read_csv(path)
    baselines = ["OFDM", "OTFS", "AFDM", "IDENTITY", "HAAR0", "HAAR1", "HAAR2"]
    candidates = table[~table.waveform.isin(baselines)]
    primary = candidates[(candidates.snr_db == 18) & (candidates.slicer == "raw") & (candidates.receiver == "fixed")].copy()
    pivot = primary.pivot(index="waveform", columns="modulation", values="ber")[["QPSK", "16QAM"]]
    if len(primary) != 2 * len(pivot) or set(primary.bits) != {1048576}:
        raise ValueError("Missing/unequal candidate budgets")
    reference = pivot.loc[INTERNAL_ID]
    dominators = pivot.index[((pivot <= reference).all(axis=1)) & ((pivot < reference).any(axis=1))].tolist()
    # Reconcile original counts and common channel dimensions, without resampling.
    for name in pivot.index:
        with np.load(root / f"supplementary/main/communication/{name}.npz", allow_pickle=False) as data:
            si = data['snrs_db'].tolist().index(18)
            ri = data['receivers'].tolist().index('fixed')
            for mi, mod in enumerate(data['modulations'].tolist()):
                row = primary[(primary.waveform == name) & (primary.modulation == mod)].iloc[0]
                if data['errors'].shape[-1] != 1024 or int(data['errors'][si, ri, mi].sum()) != int(row.errors):
                    raise ValueError("Stored count mismatch")
    primary.to_csv(out / "primary_counts.csv", index=False)
    pivot.to_csv(out / "candidate_tradeoff.csv")
    # Retain every saved SNR, modulation and receiver in this separate internal view.
    candidates.to_csv(out / "all_candidate_conditions.csv", index=False)
    result = {**identity(), "internal_selected_id": INTERNAL_ID,
        "criterion_declared_post_hoc": "Raw uncoded BER jointly for QPSK and16QAM at18dB Es/N0, fixed discovery-Q24; no modulation weights specified",
        "secondary": "Full-A LMMSE and complete saved SNR/receiver/modulation table retained; no per-condition switching",
        "candidate_count": len(pivot), "channels_per_point": 1024, "bits_per_point": 1048576,
        "selected_primary_ber": reference.to_dict(), "point_estimate_dominators": dominators,
        "best_primary_QPSK": str(pivot.QPSK.idxmin()), "best_primary_16QAM": str(pivot['16QAM'].idxmin()),
        "source_sha256": sha(path),
        "limits": "Existing BER evidence viewed before retaining reference; descriptive post-hoc choice, not independent test evidence or statistically proven Pareto optimality. Main common bank only; focused adaptive bank not pooled. No new bootstrap or campaign."}
    write_json(out / "selection_review.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("local/internal/selection_review"))
    args = parser.parse_args()
    configure()
    print(json.dumps(review(args.artifacts, args.out), indent=2))
