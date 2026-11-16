# Local inputs for existing results

DD archived-result modes in [Waveform_Exploration.ipynb](../Waveform_Exploration.ipynb) need `local/advisor-scaling-artifacts.tar.gz`:
64 verified nominal numerical/provenance files,90,150,353 compressed bytes,
SHA256 `71a342c505ddfbafefc2d2bc368dc9071ebc1996cfb9414130f62bf0f3d9edd5`.
Tracked `configs/scaling_artifacts.json` binds bundle/file hashes. Set its path
in the notebook configuration; Run All safely provisions missing local inputs
under this checkout and verifies existing inputs without overwriting. A clean
checkout has no generated arrays: provision this local bundle explicitly. No
network/source-project fallback or silently substituted size.

Snapshot policy and scientific identity: [coupling/size notes](COUPLING.md).
Includes own NPY matrices, all nominal structural NPZs (Q/M/examples/supports),
raw per-channel nominal BER archives, complete metadata/summary CSV, selected
training metadata and exact executed source snapshot (not imported at runtime).
No `.pt` required for inference; original optimizer/checkpoint histories stay
untouched in the read-only source project. Snapshot utility:
`python scripts/snapshot_scaling.py --help`. It rejects changed/incomplete inputs
and existing destinations. Do not rerun a campaign to rebuild an artifact bundle.

All generated numerical arrays, plots and archives stay ignored. Test code-only
behavior with `python -m pytest tests -q`; local full checks add
`--artifacts local/artifacts --scaling-artifacts local/scaling-artifacts`.
Each marker skip explicitly distinguishes missing local evidence from success.
CFO requires the legacy bundle below. New DD training is independent of both bundles.
The sealed historical bundle filenames are retained to preserve input identity.

---

# Local artifact contract

Primary use: open Waveform_Exploration.ipynb. Its configuration cell supplies the
local bundle/destination; Run All calls the same hash-checked provisioning code
automatically when inputs are absent, then regenerates all figures inline.
The script commands below are optional technical/internal utilities.

Git contains code, docs, notebook source, dependency lock and checksum manifest.
**Numerical inputs/results are intentionally ignored.** A clone supports
classical smoke/export only. Learned regeneration requires the matching local
`advisor-n64-artifacts.tar.gz`; there is no download URL or silent old-project
dependency. Owner reviews it locally before any handoff.

Provision with `python scripts/bootstrap.py --bundle PATH`. Default destination
`local/artifacts`; `--dest PATH` supports another local location. Bundle SHA and
every member size/SHA checked against `configs/artifacts.json` before staging is
published. Unsafe names, symlinks/non-files, duplicates, unexpected/missing files,
corruption and existing destinations rejected. Files chmod444; directory remains
owner-writable. This is a checksum-verified snapshot, not a signed immutable
storage system. `python scripts/bootstrap.py --verify` rechecks it later.

Sealed release:645files,182,919,587uncompressed bytes; gzip bundle176,447,435bytes.
Bundle SHA256 `693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.

| Local bundle subtree | Purpose |
|---|---|
| `reference/learned_w.npy` | Essential verified historical N64 input, original bytes |
| `focused/data/{bases,channels,supports}.npz` | Essential three W's, exact example H/A/paths, grouping/order, discovery Q/masks |
| `focused/*.csv,*.json,*.md` | Final BER/paired intervals, protocol/SNR freeze, selection/provenance and interpretation |
| `focused/communication/*` |27complete points, original per-batch errors/paths and concatenated per-channel counts; duplicates intentional for direct audit of original saved concatenations |
| `focused/snr25` | Source display-window CSV/README; full-range data retained above |
| `supplementary/{orthogonality_bridge,block_frequency}` | Complex bridge/phase control/pair diagnostics, frequency profiles/outliers, completed reports |
| `supplementary/main` | Completed training/intervention/transfer summaries and per-channel BER evidence; selected checkpoints/traces and discovery inputs |
| `residual_cfo` | Qualified genuine W/R checkpoint, original saved diagnostics/report/CFO BER tables |
| `sources` | Relevant source and executed closures, archived for inspection only. No runtime imports. Active scaling modules excluded. |

Example loading, after provisioning:

```python
from pathlib import Path
import numpy as np
r = Path('local/artifacts')
b = np.load(r/'focused/data/bases.npz', allow_pickle=False)
d = np.load(r/'focused/data/channels.npz', allow_pickle=False)
W, H, A = b['Learned'], d['H'], d['Learned_example']
assert np.allclose(W.conj().T @ H @ W, A)
counts = np.load(r/'focused/communication/QPSK_25dB/per_channel.npz', allow_pickle=False)
# errors axes: waveform[Learned,OTFS,AFDM], receiver[fixed,full],
# slicer[raw,gain], independent physical channel. bits_per_channel=1024.
```

Raw batch paths identify `(focused-v1,final-channel,batch,index)`;512channels
per batch, last75at cap. WMS RNG is SHA256 of compact JSON
`['wms-v1',*namespace]`, first8bytes big-endian modulo2^63-1. Final H namespace
`('focused-v1','final-channel',batch)`. Bits/noise add phase, modulation,SNR,batch;
exact source protocol retained. Main supplementary NPZ names receivers,
modulations,SNRs and per-channel denominators explicitly.

## Rebuilding a local bundle from authorized sources

Normally provision the sealed bundle; no source directories needed afterward.
If the owner intentionally wants a new snapshot from the same read-only roots:

```bash
python scripts/snapshot.py --wms /path/to/waveform-mechanism-study \
  --cfo /path/to/ofdm/residualCFO --out local/new-sealed \
  --bundle local/new-bundle.tar.gz --manifest local/new-manifest.json
python scripts/bootstrap.py --bundle local/new-bundle.tar.gz \
  --manifest local/new-manifest.json --dest local/new-artifacts
```

This selects only named completed N64 paths, checks copy stability and records
relative source paths/hashes. A newly built archive can differ in gzip/tar
metadata even with equal file content; its own manifest/bundle hash must match.
Do not overwrite the tracked manifest or promote newly changing studies during
ordinary figure regeneration. No training/recovery/bootstrap from network.
