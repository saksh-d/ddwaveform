# Included inputs

**A normal Git clone contains every input needed by the public notebook.**
No separate bundle, hidden Mac directory, manual extraction or download is required.
Install the dependencies, open the root notebook and Run All.

| Workflow | Numerical inputs used, all included unless stated |
|---|---|
| Saved DD N64 / N128 / N256 | `dd_N64.npz` / `dd_N128.npz` / `dd_N256.npz`, plus `dd_replay.csv` and manifest |
| Fresh DD with archived frozen Learned | Same size-specific NPZ and manifest; BER is recomputed, never read from replay |
| Newly trained DD + fresh tests | **None**; training generates W, supports, groups and new test results |
| Fresh DD with your trained checkpoint | Your matching `.npz` + `.json` pair under `local/`, matching N/K |
| Supported frozen CFO + fresh tests | `cfo_stage1.pt` and manifest: original W[64,45], R[45,64], A=RHW |

The integrity check verifies all six files together even when a mode uses only a
subset. The manifest binds the complete small runtime dataset; restore the entire
`waveforms/data/` directory from Git if any file is missing or altered. All N are
packaged together, including in installed wheels. No learned matrix is resized or
substituted. CFO training and CFO saved replay remain unsupported. Fresh numerical
work requires MPS/FP32; saved DD replay preserves the original reference arithmetic.

## Actual tracked payload

| File under `waveforms/data/` | Bytes |
|---|---:|
| `dd_N64.npz` | 160,178 |
| `dd_N128.npz` | 632,511 |
| `dd_N256.npz` | 2,521,290 |
| `dd_replay.csv` | 92,292 |
| `cfo_stage1.pt` | 55,141 |
| `manifest.json` | 21,231 |
| **Total** | **3,482,643** |

Each DD NPZ preserves the exact original Learned W, fixed supports for Learned/OTFS/
AFDM, native group mapping, train/test Q and mean channel-power statistics, and
original example paths/gains. Classical bases and redundant example H/A/spectra
are reconstructed by the unchanged numerical routines. The CSV retains all 288
nominal/raw/fixed-or-full replay rows: original counts, intervals, budgets, stopping
and uncertainty flags. It makes no new empirical claim. Fresh mode never uses
these BER values; channel/support/checkpoint changes cannot reuse saved BER.

`manifest.json` records file sizes/SHA256, per-array content hashes, original source
file/key/hash, learned checkpoint identity and selection, channel protocol and
source bundle hashes. CFO is the **byte-identical original checkpoint**, with its
original payload hash and clean-operator acceptance checks. Neither DD nor CFO was
retrained, selected on BER, phase-adjusted or replaced for this packaging change.
See [basis provenance](COUPLING.md) and [scientific limits](CONFIGURABLE.md).

## Optional historical audit evidence

The full archives retain per-channel campaigns, internal investigations, training
histories and source snapshots. They are **not required for the public notebook**
and are deliberately not added to Git:

- `advisor-scaling-artifacts.tar.gz`: 90,150,353 bytes; SHA256
  `71a342c505ddfbafefc2d2bc368dc9071ebc1996cfb9414130f62bf0f3d9edd5`.
- `advisor-n64-artifacts.tar.gz`: 176,447,435 bytes; SHA256
  `693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.

Existing holders may keep these intact under ignored `local/` and provision them
with `support/scripts/bootstrap.py`, choosing the matching manifest in support/configs/.
The full legacy tests take explicit `--artifacts` and `--scaling-artifacts` paths;
those optional tests skip without the audit archives. The delivered-input tests
run without them. Sources inside archives are never imported.

Maintainer extraction is reproducible from the sealed, verified snapshots using
`support/scripts/build_runtime_inputs.py --scaling local/scaling-artifacts
--legacy local/artifacts --out <new-directory>`. This only copies/subsets exact
evidence; it does not simulate BER or train a model. Per-array hashes bind content
independently of ZIP compression metadata. Full archives remain untouched.

Generated results/checkpoints, provisioned audit directories, archives, executed
notebooks and caches stay gitignored. Only the six named runtime files are tracked.
