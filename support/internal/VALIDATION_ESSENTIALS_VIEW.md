# Essential advisor package validation

Advisor figure generation is limited to exactly five PNGs:

1. `channel_matrices_native.png`: physical H, Learned/OTFS/AFDM effective A.
2. `basis_time.png`: selected Learned and baseline time-domain basis columns.
3. `basis_frequency.png`: corresponding absolute DFT-bin energies.
4. `ber_QPSK_raw.png`: Learned/OTFS/AFDM BER versus Es/N0.
5. `ber_16QAM_raw.png`: same comparison for16QAM.

Same verified Learned SHA256
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`
for every plot/SNR/receiver/modulation. Raw slicing only in advisor BER; fixed-Q24
and full LMMSE separated into panels. Full30/40dB views remain available through
`--full-range`, generating the same five essential figure types. No exploratory
figures, complex-bridge plot bundle or filtered transfer table generated here.

## Checks executed

CPU float64/complex128 explicitly authorized, Torch/BLAS threads1; existing
pytorch environment. Bounded numerical tests only; no training, production BER reevaluation or new
research campaign.

| Command | Result |
|---|---|
| `python -m pytest support/tests -q --basetemp=local/test-essentials --artifacts local/artifacts` |26passed in1.92s |
| `python -m waveforms.cli figures --artifacts local/artifacts --out local/review_essentials/figures` |Exactly5essential PNGs,300DPI |
| same with `--full-range --out local/review_essentials/figures_full` |Same5figure types; raw BER PNGs unchanged from prior full-range views |

Regression tests enforce the exact five-name output whitelist, reject matrix
substitution/internal candidate leakage, and ensure advisor CLI/notebook cannot
invoke study generation. They also retain the original DD/receiver/BER-count/CFO
qualification checks. Internal9figure diagnostics,2selected-reference study plots
and4multi-variant plots still regenerate in tests, with internal audience metadata.

Visual review of all5essential PNGs: readable axes/legends, no clipping;
DejaVu Sans18pt axes/colorbars,17panel labels,15ticks,16legends,300DPI.
Raw BER PNGs byte-identical to earlier25dB and full-range outputs; no numerical
values, counts, uncertainty intervals, adverse outcomes or curves changed.

## Clean-copy, notebook and preservation checks

The isolated source copy passes26artifact tests in1.81s and17code-only tests
with9explicit artifact skips in0.44s. All5PNG outputs are byte-identical to
root regeneration. Revised notebook:6cells/3code cells,0errors, exactly5image
outputs, no supplementary table or study generation. Kernel shut down afterward.

All2598pre-existing numerical/checkpoint/bundle files unchanged in size/mtime;
all645bundle file hashes verified. Bundle SHA256 remains
`693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.
Prior gallery/notebook copies archived internally; existing review aliases now
point to the essentials-only view. Previous plots remain intact and marked internal.

## Internal work preserved

Previous full plotting source retained in `waveforms/internal_figures.py`.
`internal-supplementary` and `internal-studies` retain the selected-reference and
multi-variant studies. Prior notebook, plotting outputs, all variants and working
data remain internal; original directories remain read-only and unchanged.
Previous validation records are preserved under support/internal/.

The advisor command is only `figures`; the former `supplementary` CLI route is
replaced by explicit `internal-supplementary`. No internal study runs automatically.
The advisor notebook has only6cells/3code cells: H/W/A, the five essential figures
and saved BER counts. Selection rationale remains explicitly post hoc; no claim
of independent confirmation, universal winner or physical validation.

No push, merge, main/default-branch change or external handoff. Owner review pending.
