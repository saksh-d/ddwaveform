# Current advisor presentation validation

This revision narrows only the presentation. The original full validation record
is preserved in `support/internal/VALIDATION_BEFORE_PRESENTATION.md`; source history and
all internal working arrays/checkpoints remain intact.

## Selection and presentation contract

One verified N64 matrix labeled Learned, file SHA256
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`,
with OTFS and AFDM. Same matrix for time/frequency, channel/effective channel,
bridge, transfer, all BER SNRs/modulations/receivers. Source identity checked
before rendering; no fallback to another candidate when data are absent.

Existing common-bank review:43learned candidates,1024physical channels and
1,048,576bits/point, raw fixed-Q24 at18dB Es/N0. Joint QPSK/16QAM criterion has
no supplied modulation weights. Retained reference has no point-estimate
candidate dominator in both measures. Different modulations favor different
alternatives; full-A and complete SNR tables retained internally. This later
choice is post hoc, not independent test evidence or statistical proof of
optimality. Detailed candidate IDs/tables stay internal. No new simulation.

## Executed commands

Use conda pytorch, explicit authorized CPU float64/complex128 and one thread:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.
Historical CFO keeps float32/complex64 in its separate workflow.

| Command | Result |
|---|---|
| `python support/scripts/review_selection.py --artifacts local/artifacts --out local/internal/selection_review` |43candidate count totals reconciled; no simulation/bootstrap; full condition table retained internally |
| `python -m pytest support/tests -q --basetemp=local/test-presentation --artifacts local/artifacts` |24passed in2.03s |
| `python -m pytest support/tests -q --basetemp=local/test-presentation-code-only` |16passed,8explicit artifact skips in0.47s |
| `python -m waveforms.cli figures --artifacts local/artifacts --out local/review_single/figures` |9core PNGs, Learned/OTFS/AFDM only |
| same with `--full-range --out local/review_single/figures_full` |9core PNGs, full30/40dB grids retained |
| `python -m waveforms.cli supplementary --artifacts local/artifacts --out local/review_single/supplementary` |2advisor PNGs: exact selected transfer and frequency groups |
| `python -m waveforms.cli internal-studies --artifacts local/artifacts --out local/internal/studies` |All4original multi-variant plots regenerate internally |
| `python -m waveforms.cli verify-artifacts --artifacts local/artifacts` |645files/182,919,587bytes, all hashes unchanged |

Regression tests reject matrix substitution and missing/ambiguous source labels;
inspect every advisor plot's text for exactly the selected Learned and baselines;
compare selected transfer rows exactly to original saved source rows; and confirm
internal training/intervention/transfer labels and four-figure renderer remain.
Original DD/path/receiver/BER-count/CFO qualification checks still pass.

## Visual and data preservation checks

Inspected all11advisor PNGs as a contact sheet: no clipping, one Learned plus
baselines, shared scales. Time solid/dashed denote real/imaginary coefficients;
same column indices across waveforms, discovered groups apply only to Learned.
Bridge rows use Learned/OTFS/AFDM. DejaVu Sans18pt axes/colorbars,17panel labels,
15ticks,16legends;300DPI PNG. BER curves/counts unchanged, including adverse
full-receiver outcomes and high-SNR reversals. All-strength structural transfer
CSV is filtered for presentation; full source table remains unchanged.

Inventoried2133pre-existing local array/checkpoint/manifest files: none removed
or modified. The645file bundle is unchanged, SHA256
`693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.
Original multi-variant notebook/index/renderer retained separately, with explicit
internal audience metadata. Advisor notebook no longer displays training or
alternative-candidate tables, or mixes the distinct CFO matrix into its DD view.

## Clean-copy and notebook checks

New39file staged-source copy installed in its own local venv with no network,
`-I` and empty PYTHONPATH. Before provisioning:16pass/8explicit artifact skips
(11.52s including first local font cache). Hash-checked bootstrap succeeds;
then24tests pass1.83s. All11advisor PNGs byte-identical to the root regeneration.
All8BER PNGs across25dB/full-range raw/gain views remain byte-identical to the
previous version; counts and curves did not change.

Revised advisor notebook:8cells/4code cells,0errors, no internal candidate IDs in
text/table output. The full original10cell notebook remains internal. Earlier
executed notebook/gallery copies preserved under local/internal/presentation_archive;
existing review aliases point to the corrected advisor view. Prior working plots
are intact and explicitly marked internal. Source notebook stays output-free.

## Limits

Owner review pending. No push/merge/main/default-branch changes. Original DD .pt
still unavailable; verified export retained. CFO AWGN operator qualified in the
separate workflow, with no new training/BER replay or stage3 claim. This remains
a circular perfect-CSI numerical model, not physical validation, factorization,
independent confirmation of a post-hoc choice, or universal waveform superiority.
