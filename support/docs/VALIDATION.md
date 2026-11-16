> Historical validation record for earlier notebook revisions. Current training,
> delivered-input support and qualification are in [VALIDATION_CONFIGURABLE.md](VALIDATION_CONFIGURABLE.md).
> Earlier disabled-training/input-bundle statements below describe those earlier revisions.

# Current overlay review validation

- `python -m pytest support/tests -q --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts --basetemp local/audit/overlay-review-tests-tmp`: **63passed,5.49s**, CPU float64/complex128, one thread.
- Tests check exact8/N/2 counts, every rendered original curve and full key, true real/imaginary values, absolute energy vs normalized magnitude, native/group/rank mapping, invalid configuration rejection and unchanged W/H/BER under changed review selections.
- Clean-copy notebook defaults execute at N64/N128/N256 with **0errors/13inline figures** each; bounded changed-channel N256 fresh likewise. Changed N128 indices and enabled subset execute **0errors/10figures**, exactly matching configured variants. train_new remains expected NotImplementedError/0figures/0archives.
- Default/custom N128 matrices.npz arrays and BER tables are exactly identical while review index mappings change. All8 or32/64/128 selected curves remain together per applicable panel; no curve splitting/omission. Plot/index mappings retained in local CSVs/run metadata.
- Inspected8-curve and half-basis exports, including128-entry keys;62current PNGs verified300dpi/source hashes. Dense half overlays remain visually crowded by design for review; no smoothing/shape selection/automatic pruning. All74preceding archive files checksum-identical.
- Executed notebooks/ledger/comparison viewer: local/notebook_delivery/overlay-review. Audit: local/audit/overlay-review-final-audit.json. Tracked notebook remains output-free/one config cell. No training/full campaign, source data edits, branch/default changes or publishing.

The inline comparison viewer uses downscaled display previews to fit its25MiB
limit; full300dpi PNGs remain embedded in executed notebooks/local archives.

See [review variants](PLOT_VARIANTS.md). Owner selection for final presentation remains pending.

---

# Prior individually identifiable basis validation (before overlay alternatives)

- Focused suite: `python -m pytest support/tests -q --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts --basetemp local/audit/basis-examples-tests-tmp`: **55passed,4.80s**, CPU float64/complex128, one thread.
- Tests verify native-ID/group/permuted-rank mapping, exact displayed time/coupling columns, labeled grouped-spectrum highlights, selection bounds and unchanged W/H/A/BER after changing examples. At most two examples per group; invalid density fails before archive creation.
- Clean-copy source notebook executions: N64/N128/N256 replay, bounded changed-channel N256 fresh and a changed-example N128 replay each **0errors/9inline figures**. Disabled training: expected NotImplementedError/0figures/0archives. One configuration cell; source notebook output-free.
- Changed-example N128 execution was compared against default: different examples, exactly identical H/Hraw/W/A/groups/permutation/BER. Mapping also verified against example_native_columns in saved matrices and examples.csv. Defaults chosen by native ID, not appearance/BER.
- Visually inspected identifiable examples at all sizes and changed N128 indices; grouped overview remains all-member, with separately labeled highlights.45current PNG exports verified300dpi/source hashes;52preceding archive files remain checksum-identical.
- Executed review notebooks: local/notebook_delivery/basis-examples. Execution ledger: execution_verification.json; audit: local/audit/basis-examples-final-audit.json. No source projects/jobs changed, training/full campaign or publishing.

---

# Prior grouped-member spectral validation (before identifiable examples)

- Full focused suite: `python -m pytest support/tests -q --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts --basetemp local/audit/grouped-spectra-test-tmp`: **51passed,3.79s**; CPU float64/complex128, one thread. Checks distinguish absolute coefficient energy from20log10 amplitude normalization, preserve all group members and actual size-specific permutations, and retain complex coefficients.
- Final clean-copy notebook executions at N64/N128/N256 replay: **0errors/8inline figures** each. Changed-channel N256 fresh also0errors/8figures; only2channels/1block/2SNRs. train_new expected NotImplementedError,0figures/0archives. No new full simulation/training.
- Final executed notebooks and execution_verification.json: local/notebook_delivery/grouped-spectra-final. Each source figure has300dpi PNG, and all32exports match current source hashes. Optional grouped_spectra.npz retains complex coefficients, absolute powers, peak-normalized dB and index mapping. Tracked source notebook output-free, one configuration cell.
- Visually inspected grouped plots at all three sizes; shortened overlapping axis labels while keeping formulas in notebook/docs. All44files in the preceding scaling-run archives remain checksum-identical. Audit: local/audit/grouped-spectra-final-audit.json. Textual reference concept implemented; no original reference pixels inspected.

---

# Prior spectral/scaling validation (before grouped member view)

- `python -m pytest support/tests -q --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts --basetemp local/audit/scaling-test-release-tmp`: **47 passed,3.19s**, CPU float64/complex128, one thread.
- Fresh source-only copy: `python -m pytest support/tests -q --basetemp local/audit/code-only-spectral-tmp`: **28 passed,19 artifact skips,11.22s**. Skips are not numerical qualification.
- Clean-copy notebook auto-provisions only the scaling bundle; all cells execute for N64/N128/N256 replay, each **0errors/7inline figures**. Changed-model N256 fresh uses only2BER channels/1block/2SNRs; same0errors/7figures. Disabled training gives expected NotImplementedError before loading,0figures/0archives.
- Independent DTFT finite-sum/DFT-knot/energy checks at all sizes; complex spectral A bridge max errors6.89e-16/7.47e-16/1.12e-15. Matched source H/paths and complete A for all three waveforms. All288 nominal raw fixed/full summary error totals exactly match saved per-channel counts; own groups/permutation/support verified. Q row/column sums and normalized ensemble total checked.
- Full notebook executions/rendered outputs under local/notebook_delivery/spectral-final, with execution_verification.json; local/audit/spectral-final-audit.json binds28PNG exports to current sources and300dpi. All3070 earlier recorded numerical/checkpoint/bundle files unchanged;64sealed source files still match. Tail checkpoint identities/exact exported-array equality independently checked read-only at N128/256.
- No training, full BER campaign, stress campaign, new grouping/factorization research or external publishing. Tests of tiny fresh banks do not improve source campaign precision or measure training-seed variation.

See [current definitions/limits](COUPLING.md). Source notebook remains output-free;
executed copies are local/ignored. Main/origin default remain unchanged.

---

> Current notebook extension: [N64/128/256 own bases and spectral/coupling views](COUPLING.md).
> Current replay uses the completed scaling-release protocol; earlier N64 evidence below remains archival.

# Primary notebook workflow validation

The single advisor entry point is `Waveform_Exploration.ipynb`: configure, Run All,
inspect H/W/A and five inline figures. No pre-rendered PNG reads, plot-folder
hunting or generation CLI needed. Supporting modules stay behind notebook cells.
Tracked source has no outputs; executed review copies remain local/ignored.

## Executed end to end

Both copies of the same source notebook executed in the isolated source copy,
using a local pytorch kernel and no original projects. Each completed12cells,
6code cells,0errors, exactly5inline image/png outputs. Image bytes extracted
from notebook outputs match their optional300DPI archives; all10outputs visually
inspected for labels, clipping and readability. Kernels shut down afterward.

| Mode | What actually ran | Configuration verified |
|---|---|---|
| saved_replay | Automatic provisioning from the supplied local bundle into an absent artifact directory;645file hash verification; reconstruct H from frozen paths/gains; recompute complex A; load saved BER counts; regenerate five inline figures |Nominal N64/P3/delay[0,4]/Doppler[-.75,.75]; no BER simulation or training |
| fresh | New DD matrix bank and bounded paired uncoded BER, using the same fixed Learned/OTFS/AFDM and nominal discovery-Q24 masks; five inline figures |P2,delay[.25,2],Doppler[-.3,.2],seed432,3matrix channels/index1,4BER channels,3blocks/channel,SNR[0,12,24],columns[1,3,7,9] |

Default source notebook selects MODE=load_existing and EVIDENCE_MODE=saved_replay. Fresh execution is a deliberately
small qualification check, not a replacement/reproduction of the saved campaign.
Saved plots label “Saved BER replay”; fresh plots label “Fresh BER smoke”.
Fresh bands describe +/-1.96channel SE; they are not the saved adaptive-bootstrap
intervals. Zero errors remain unresolved, with separate conservative bounds.

## Configuration propagation and rejection

- MODEL path count/ranges apply to both fresh matrix and BER channel banks;
  normalization, tensor shapes/ranges, shared method budgets and deterministic
  seed replay tested. Changing seed changes generated matrices.
- DISPLAY columns pass to basis rendering; SNR display cap passes to BER plots.
  Saved counts/grid stay intact. N remains64 because the Learned export is64x64.
- Replay rejects changed model parameters and any FRESH seed/count/grid payload;
  missing/unknown settings and fresh budget violations fail before numerical work.
  MODE=train_new fails explicitly as unavailable; it never loads a fallback. No hidden support recalibration or checkpoint change.
- Configured H/Hraw/W/A remain in notebook memory. Optional unique local archives
  retain complex arrays, paths/gains/normalizations, fixed masks/permutation,
  displayed columns, counts, settings, RNG namespaces/seeds and source hashes.
  There is no overwrite of prior research evidence.

## Tests and execution environment

Root focused suite:34passed in2.38s. Isolated source copy:34passed in2.58s.
Coverage includes replay refusing to call the BER simulator, exact saved-count
loading, complex H/A reconstruction, nondefault fresh propagation/reproducibility,
five inline outputs without pre-existing plots, archive integrity and earlier
DD/receiver/count/CFO/internal-workflow checks. No training or large campaign.

Python3.10.17, Torch2.7.0, NumPy2.2.6. Owner-authorized CPU float64/complex128,
Torch/BLAS threads1; no fallback. Exact dependency closure remains pinned.
DejaVu Sans18pt axes/colorbars,17panel headings,15ticks,16legends;300DPI PNG
bytes embedded in notebook outputs. Source files and data remain local.

## Scope and preserved work

Only the essentials appear inline: physical/effective matrices, time/frequency
basis, QPSK/16QAM BER. One globally fixed Learned matrix, SHA256
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`,
with OTFS/AFDM. Selection remains explicitly post hoc, not universal optimality
or independent confirmation. No intervention/transfer/other exploratory figures.

Preservation audit:2618pre-existing numerical/checkpoint/bundle files unchanged;
all645artifact hashes verified. All original internal variants/studies/data remain intact; prior validation and
notebook versions preserved under support/internal/. Only a disabled training selection and common basis-result contract were added;
no trainer, training campaign or next workflow was started. No push, merge, main/default-branch change or sharing.
Owner review remains before any handoff.

## Final basis-source interface correction

One config cell separates MODE=load_existing/train_new from the independent
EVIDENCE_MODE=saved_replay/fresh. MODE=train_new is clearly marked unavailable.
It fails before provisioning, basis loading or run creation, without fabricating
output. All subsequent notebook cells are the same for either provider.

SelectedBasis carries W[64,64], its own boolean row-Q24 mask and source
provenance. The common workflow consumes that supplied matrix/support for all
H/A, basis figures and BER; it does not reload historical Learned behind the
provider. Saved curves are rejected for a different W or mask. Historical group
views are not attached to a different supplied basis. The follow-on trainer is
not implemented by this interface design. Prior execution evidence above refers
to the earlier evidence-setting interface; current interface checks follow below.

The implemented load_existing basis route has now executed in both replay and
bounded fresh evidence settings, with0errors/5inline figures in each. A third
execution selecting train_new stops with the expected NotImplementedError,
0figures and no run archive created. No training is called or emulated. Tests
also clear a previous notebook workflow before a failed selection, preventing
manual continuation from silently reusing an old loaded basis.

Current final interface verification:37focused tests pass2.10s. Re-executed the
final source notebook with load_existing + saved_replay and load_existing + fresh:
both0errors/5inline images. train_new produces only its expected unavailable
error, with0images/0new run directories. All3070pre-existing numerical/checkpoint/
bundle files unchanged; all645artifact hashes verified. Exactly one config cell;
tracked notebook output-free. Existing review aliases now use this final interface.
