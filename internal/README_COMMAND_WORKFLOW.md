# Advisor waveform reproducibility

Inspect a physical channel **H**, synthesis basis **W**, and complex effective
channel **A**; regenerate the owner's N64 basis/matrix/BER figures from completed
data. Separate DD and residual-CFO models. No training or factorization campaign.

Advisor DD presentation: exactly one verified waveform, **Learned**, with OTFS
and AFDM. The same matrix is used in every plot, SNR, modulation and receiver.
[Selection rationale](../docs/SELECTION.md): existing evidence trades QPSK against
16QAM; retaining the prior reference is a post-hoc presentation choice, not a
claim of universal superiority or independent test validation.

## Install and code-only smoke

Python **3.10.17**, tested macOS arm64. Exact dependency closure:
`requirements.lock.txt`. Existing `pytorch` conda environment already matches.

```bash
conda activate pytorch
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PIP_CACHE_DIR="$PWD/local/.pip-cache" MPLCONFIGDIR="$PWD/local/.matplotlib"
# For a fresh environment:
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps -e .

python -m waveforms.cli smoke
python -m pytest tests -q --basetemp=local/test-tmp
python -m waveforms.cli export-dd --channels 2 --out local/classical_dd
```

CPU float64/complex128 explicitly authorized by the owner for this reference
package. Torch/BLAS threads1; no fallback. CFO retains its historical
float32/complex64 operator. Code-only smoke uses classical controls and needs no
ignored data. Artifact tests explicitly skip until `--artifacts` is supplied.

## Provision the local evidence bundle

**A clone does not contain Learned weights or BER inputs.** Owner must provision
the local bundle before reviewing those outputs. No automatic download/upload.
See [artifact contract](../docs/ARTIFACTS.md); exact file hashes in
[configs/artifacts.json](../configs/artifacts.json).

```bash
# Supply the accompanying local advisor-n64-artifacts.tar.gz at any path:
python scripts/bootstrap.py --bundle /path/to/advisor-n64-artifacts.tar.gz
python scripts/bootstrap.py --verify
python -m pytest tests -q --basetemp=local/test-tmp --artifacts local/artifacts
python -m waveforms.cli smoke --artifacts local/artifacts
```

Bootstrap rejects wrong hashes, unsafe paths and existing destinations. The final
bundle seals645files/182,919,587uncompressed bytes, all source-stability checked.
It contains **completed N64 evidence only**, not N128/N256 results or active
scaling modules. See manifest for bundle size/hash.
Snapshot files read-only; make a new output directory for each regeneration.

## Essential advisor walkthrough

```bash
# Save complex H/Hraw, paths, scales, W, A, C, H_f, Gram matrices, frozen masks/order:
python -m waveforms.cli export-dd --artifacts local/artifacts --channels 2 --out local/dd_export

# Replot exactly FIVE essential300DPI PNGs; no BER reevaluation:
python -m waveforms.cli figures --artifacts local/artifacts --out local/figures
# Preserve high-SNR reversals: full30dB QPSK /40dB16QAM display:
python -m waveforms.cli figures --artifacts local/artifacts --full-range --out local/figures_full

# Optional, explicitly bounded NEW-bank BER qualification, not saved-curve reproduction:
python -m waveforms.cli ber --artifacts local/artifacts --channels 8 --blocks 2 --snrs 0 18 --out local/ber_smoke
```

Read [model/receiver conventions](../docs/SCIENCE.md) before interpreting figures.
Exactly five figures: physical H and effective A comparison; Learned time-domain
basis; Learned frequency-domain basis; QPSK BER;16QAM BER. Same selected Learned
throughout, alongside OTFS/AFDM. BER uses raw nearest-point slicing; fixed-Q and
full receivers occupy separate panels. Saved full-range counts remain available.
No intervention, transfer, grouping, bridge or gain-diagnostic figure is generated
by the advisor command.

Load matrices without this package: `np.load('local/dd_export/matrices.npz',
allow_pickle=False)`. H has shape `[channel,64,64]`, each `*_W` `[64,64]`,
`*_A` `[channel,64,64]`; complex128, native symbol order. Exact export metadata
contains seeds, normalization and checksum. Frozen discovery permutation is an
additional view, never refit per channel. The saved illustrative H is separately
in `local/artifacts/focused/data/channels.npz` with all three saved complex A's.

Open [Waveform_Exploration.ipynb](../Waveform_Exploration.ipynb) after provisioning. It loads
saved H/W/A and BER evidence and displays only these five figures; no hidden run.

## Internal research and provenance

All exploratory studies, diagnostics, variants, checkpoints and working data
remain intact in the separate [internal workflow](../internal/README.md). Its
explicit `internal-*` commands are never invoked by the advisor figure command
or notebook. Original source directories remain read-only and unchanged.
The distinct rectangular CFO model/export is also documented there; it is never
substituted for the selected DD Learned matrix.

[Provenance](../docs/PROVENANCE.md) maps definitions to pinned sources, recovered DD
export and the qualified genuine CFO checkpoint. Original DD `.pt` remains
unrecovered. Historical stage/workflow files remain accessible at base commit
`9aae23ed6f9ac6e2a2e3e057f38693a148886400`; original worktrees unchanged.

All artifact bundles, plots, numerical outputs, notebook outputs, caches and
checkpoints stay local/ignored. Source/docs/tests/config only in Git. Nothing
pushed or shared. [Validation](../docs/VALIDATION.md) records actual bounded checks
and limitations for owner review.
