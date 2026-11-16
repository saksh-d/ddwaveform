# Owner review validation

Host: Mac M3Pro18GB; conda `pytorch`, Python3.10.17, Torch2.7.0,
NumPy2.2.6, Matplotlib3.10.3, pandas2.2.3. DD CPU float64/complex128 expressly
authorized for this package. CFO CPU float32/complex64 preserves source model.
Torch/BLAS thread cap1. No training, large simulations or active-job changes.

## Executed commands and results

Environment prefix for all numerical commands:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`, within
`conda run -n pytorch` (or activated environment).

| Command from project root | Observed result |
|---|---|
| `python scripts/bootstrap.py --bundle local/advisor-n64-artifacts.tar.gz` (fresh destination) |645files,182,919,587bytes validated |
| `python -m pytest tests -q --basetemp=local/test-code-only` |14passed,5artifact checks explicitly skipped,0.57s |
| `python -m pytest tests -q --basetemp=local/test-final --artifacts local/artifacts` |19passed,0.95s |
| `python -m waveforms.cli smoke --artifacts local/artifacts` |Learned unitarity2.23303e-14, max3fresh-channel bridge relative1.44e-14 |
| `python -m waveforms.cli export-dd --artifacts local/artifacts --channels 2 --out local/review/dd_export` |complex H/Hraw/W/A/C/H_f/Grams, paths/scales/seeds and frozen support/order exported |
| `python -m waveforms.cli export-cfo --artifacts local/artifacts --out local/review/cfo_export` |file/payload match, W64x45/R45x64, clean identity7.8946375e-4, all41stored leakage diagnostics tested |
| `python -m waveforms.cli ber --artifacts local/artifacts --channels 8 --blocks 2 --snrs 0 18 --out local/review/ber_smoke` |36points; QPSK2048bits/point,16QAM4096bits/point; qualification only |
| `python -m waveforms.cli figures --artifacts local/artifacts --out local/review/figures` |9PNG,300DPI, saved counts only,25dB display |
| same with `--full-range --out local/review/figures_full` |9PNG, entire30/40dB saved grids |
| `python -m waveforms.cli supplementary --artifacts local/artifacts --out local/review/supplementary_figures` |4PNG: training, interventions, structural transfer, frequency groups |

Tests independently cover negative DD sign, delay-first order, integer-delay
action, normalized path sums, finite-sum frequency kernel, exact OTFS/AFDM,
full-MSE versus stream geometry, full physical observation versus truncated
receiver model, stable support ties, Gray mapping and RNG replay. All2016Learned
column pairs checked, max off-diagonal inner product below2e-15. Saved example
H and complex A reconstructed. Every27focused point's per-channel count total,
denominator and summary matches; unfavorable full-receiver ordering retained.

Provisioning tests cover fresh load, refusal to overwrite, corrupt input and
unsafe archive traversal. No `.pt` is required for code-only tests. Real CFO
test verifies distinct analysis, positive sign, transpose convention, payload and
all41stored operator diagnostics. Saved CFO BER not rerun.

## Visual review

Core and supplementary PNGs inspected as contact sheets; axes/legends/labels
readable, no overlaps/clipping. Core matrix/basis/main BER and full-range figures
also available individually. DejaVu Sans18pt axes/colorbars,17panel titles,
15ticks,16legends,300DPI. No typography overrides. Absolute64-bin spectral
energies; no per-column normalization. Full versus fixed models and raw versus
gain slicing separated. Original saved matrix selection and all SNR/count data
unchanged. Figure manifests record input and plotting-source hashes.

## Clean-copy verification

Copied only29staged source/docs/test/config files into an empty local directory;
no ignored outputs or Git-history imports. Installed editable into an isolated
local venv with existing dependencies and `PIP_NO_INDEX=1`, then ran Python with
`-I` and empty PYTHONPATH. Original project directories never used at runtime.

- Before provisioning:14passed/5explicit artifact skips (0.52s). The first check
  exposed pytest's missing local temp parent; fixed in conftest, rerun with the
  entire local directory absent. No artifact fallback introduced.
- Bootstrap from the sealed bundle validated645files. After provisioning:
 19passed (0.75s initial clean check; final repeated after temp-parent fix).
- DD export, genuine CFO export and36-point BER smoke reproduced every saved
 array exactly. All9clean-copy core PNGs byte-identical to root exports.
- Source notebook executed in the isolated copy:10cells/5code cells,0errors.
 Executed notebook retained only under `local/review`; committed notebook has
 no outputs. A temporary local kernel spec selected the isolated interpreter;
 kernel shut down after execution.
- All645selected source files rehashed against originals:0mismatches. Bundle
 compressed176,447,435bytes; SHA256
 `693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.

`python -m compileall -q waveforms scripts tests` and `git diff --cached --check`
passed. Final cache-boundary suite19passed in12.24s, including first local
font-cache initialization; all22release PNGs remained byte-identical. Root runtime source has no absolute source-project paths or imports from
the retired `src`/WMS packages. Original main remains a12d24f, remote default
origin/main. Local audit logs/figures/bundle ignored; only29intended source,
documentation, test, notebook and configuration files form the new tree.

## Limits and pending review

Owner review remains before handoff. This is a numerical reference package, not
physical validation, original DD checkpoint recovery, training replay, a new
factorization/compression result or a universal waveform advantage. Full2048
bridge campaign not rerun; relevant small checks independently pass. Original
DD `.pt` unavailable. CFO stored AWGN example qualified; stage3 multipath
checkpoint/model unqualified here. N128/N256 active results excluded.

No generated outputs/artifacts included in Git. Bundle must accompany any future
owner-approved local handoff; clean clone alone intentionally lacks Learned data.
