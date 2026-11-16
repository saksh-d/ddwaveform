> Historical completed-exploration API notes. Current notebook uses the configurable
> workflow: [current controls and support](CONFIGURABLE.md). Training limitations and
> CPU-only fresh descriptions below apply to the preserved legacy API, not the new entry point.

# Notebook usage details

See the short [entry guide](../README.md). One notebook/configuration cell; all
The core figures plus enabled review alternatives appear inline sequentially
(13 with current defaults). No previous figure workflow is removed.

MODE=load_existing selects one verified Learned matrix at N64/128/256, with own
K24/48/96 support and frozen groups. train_new is unavailable and fails before
loading/provisioning. Future training returns W, own support and provenance to
the same downstream path; no trainer runs here. See [basis interface](BASIS_INTERFACE.md).

EVIDENCE_MODE=saved_replay reconstructs structural channel index0 and loads
nominal scaling-release BER counts; it never reruns that campaign. Model must
stay P3, delay[0,4], Doppler[-.75,.75]. Current replay differs from the older
focused N64 protocol, retained internally. N changes select each size's own data,
not a resized N64 matrix. Display columns/SNR limit may change.

EVIDENCE_MODE=fresh uses configured channel seed/ranges/path count, matrix index,
BER channels/blocks and SNR grid for a bounded paired simulation with frozen W.
Caps: matrix channels≤16,BER channels≤64,blocks≤16,SNR points≤8 and
BER channels×blocks×SNR≤4096. No training/support/group recalibration. The coupling
figures still identify archived512-channel mean powers and2048-atom Q as **nominal**;
they are not estimated anew from a tiny fresh bank. Fresh confidence bands are
descriptive channel-SE bands, not the saved campaign's bootstrap intervals.

EXAMPLE_COLUMNS in the configuration cell selects native Learned basis IDs for
individually labeled time/spectral/coupling rows and labeled highlights on the
grouped spectra. Up to two per group/eight total; display order follows the
frozen coupling permutation. The selected table links group, native j and rank.
This control changes only displayed examples; all-member overviews remain intact.
For complex real/imaginary comparison panels of those same IDs, set BASIS_COLUMNS[N]
to the same tuple; both are display controls for the same frozen Learned matrix.

The first displayed basis column selects transmit j for received-basis/spectral
propagation. Group colors/permutation match basis-panel titles and the inline
native-ID mapping table. The grouped member view adds one row per actual group: every member absolute
N-point DFT energy plus black mean, alongside32N-point individually peak-normalized
dB magnitude. Periodic[0,1); zero padding interpolates without adding physical
resolution. Complex DTFT real/imaginary curves preserve phase;
power/energy panels are labeled separately. [Definitions/limits](COUPLING.md).

One required input bundle: local/advisor-scaling-artifacts.tar.gz. Set SCALING_BUNDLE
and SCALING_ARTIFACT_DIR in the configuration cell. If absent, the directory is
provisioned automatically and all64 hashes checked. No source imports, downloads,
prior PNGs or legacy bundle are needed. [Artifact provisioning](ARTIFACTS.md).

CPU float64/complex128, one thread explicitly authorized. Tracked source notebook
is output-free; executed notebooks/plots/caches/archives remain local/ignored.
SAVE_RUN optionally preserves H/Hraw/W/A, paths/normalization, supports, groups,
complex C/H_f/A reconstruction, nominal Q/M, BER counts and input/source hashes.
No output-folder steps are required to view results. Clear outputs/restart/Run All
after configuration changes; failed train selection cannot reuse a stale workflow.

[Runtime evidence](RUNTIMES.md) · [scientific conventions](SCIENCE.md) ·
[provenance](PROVENANCE.md) · [internal research](../internal/README.md).


## Overlay alternatives before advisor simplification

PLOT_VARIANTS selects `overlay8_complex`, `overlay_half_complex`,
`overlay8_energy_magnitude`, `overlay_half_energy_magnitude`, `grouped_spectra`.
All five are enabled by default for comparison. Disable alternatives in that tuple
after review; the notebook renders/checks exactly the configured set. Existing
coupling and individual-example figures remain. [Definitions](PLOT_VARIANTS.md).

OVERLAY8_COLUMNS must contain8 unique native IDs; HALF_COLUMNS exactlyN/2
(32/64/128). Defaults use everyN/8-th native ID and every second native ID,
respectively, without appearance/BER selection. Every selected member is overlaid
together per panel; the complete native/group/rank key and tables remain readable
with a height that grows at higher N. Dense line crossings are preserved for this
readability comparison. These controls change display only, not W/H/A/BER.
