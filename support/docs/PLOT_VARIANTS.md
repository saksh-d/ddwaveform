# Basis overlay alternatives for owner review

Refinement of the same notebook, not a second task/campaign. Current defaults
retain earlier core/coupling/grouped/individually identified figures and add four
overlay alternatives. Owner decides which remain after visual review; no automatic
figure pruning or numerical-result selection.

| PLOT_VARIANTS entry | Curves together per panel | Representation |
|---|---|---|
|overlay8_complex|8 native columns|Real/imaginary time and actual complex DTFT, four panels|
|overlay_half_complex|N/2 native columns|Same four complex-component panels|
|overlay8_energy_magnitude|Same8 columns|Absolute DFT energy and separately normalized spectral magnitude|
|overlay_half_energy_magnitude|SameN/2 columns|Same two energy/magnitude panels|
|grouped_spectra|Every member, one row per actual coupling group|Original reference-concept energy/normalized-magnitude view, black left group mean and selected-ID right highlights|

All curves in the selected subset are overlaid **together** in each applicable
panel; no splitting a requested128-curve overlay into smaller batches. Each
complex basis has real and imaginary components in separate shared panels rather
than hiding phase in a magnitude plot. One selected Learned W per N throughout.
The existing Learned/OTFS/AFDM comparisons and individual time/spectral/coupling
rows remain in the downstream notebook path.

## Configuration and identity

One configuration cell holds PLOT_VARIANTS, OVERLAY8_COLUMNS and HALF_COLUMNS.
Index counts must be exactly8 andN/2 (32/64/128). Duplicate/out-of-range IDs,
unknown variants or malformed controls fail clearly. Default deterministic
index sets are0,N/8,... and0,2,..., independent of channel/BER/spectral appearance.
Users can explicitly inspect different IDs; W/training is unchanged.

Display follows the frozen coupling permutation. Every native ID appears in a
complete figure key as j/native ID, G/coupling group and r/permuted rank, plus
inline index tables. Curve color depends on native j/N; line style depends on
frozen rank modulo4. Those styles stay fixed between the8/half and complex/
energy-magnitude layouts at a given N, even when subsets change. They identify
individual curves, while the existing grouped/coupling figures use group colors.
Group counts/sizes are read from each size's qualified records, never prescribed.

A32/64/128-entry key grows vertically below the plots using the approved16pt
legend font, rather than squeezing labels over data. Dense overlays can still be
hard to trace at crossings; inspect the8-curve and individually labeled figures
for comparison. This tradeoff is visible for owner review, not repaired by
omitting curves, smoothing spectra or selecting favorable shapes.

## Meaning of the axes

Complex layouts: W[n,j] real/imaginary versus time sample n; actual complex
finite-block DTFT S_j(f)=Σ_n W[n,j]exp(-j2πfn)/√N real/imaginary versus periodic
f/Fs∈[0,1). No peak or per-curve amplitude normalization. Frequency curves use the
same32N-point grid stored for grouped spectra; original DFT knots equal C=FW.

Energy/magnitude layouts: left is absolute N-point |C[k,j]|²; unit-norm W columns
have summed energy1. Right is20log10(|S_j(f)|/max_grid|S_j|), each curve normalized
independently, clipped at−60dB for display. Right shapes do not measure absolute
strength or energy. Zero padding interpolates a finite-block spectrum and adds
no physical resolution; no window/continuous-time pulse model is introduced.

No changes to physical H, complex A, Q, group discovery/supports, modulation,
receiver or BER sampling/counts. Saved replay remains replay, no full campaign.
K/N remains3/8, using own verified N64/N128/N256 basis records. Optional local
archives include overlay8_mapping.csv/half_mapping.csv and exact index settings
in run.json, bound by input/source/file hashes; complex coefficients are retained
in grouped_spectra.npz and full W in matrices.npz. Outputs stay ignored/local.
