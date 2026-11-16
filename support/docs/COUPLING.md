# Spectra, coupling and sizes

The archived evidence below supports N64/128/256, each with its own Learned W
and nominal support, K=3N/8. New DD training is implemented separately; see
[the current controls](CONFIGURABLE.md). Every run keeps one selected W across
all downstream figures/SNRs; new fits do not inherit archived BER or groups.

## Qualified available evidence

| N/K | Basis source | Nominal group sizes | Learned NPY SHA256 |
|---|---|---|---|
|64/24|Historical verified export; original optimizer checkpoint missing|23/21/20|11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f|
|128/48|Fresh nominal OFDM-start, one trial, validation-selected tail step1000|25/44/24/35|cf19e5eb5d02607a68f412c164e979a79654099f33a6f7ac741d55e9b0ca1bdf|
|256/96|Fresh nominal OFDM-start, one trial, validation-selected tail step1000|63/62/79/52|52de3ac3fca6c34716cd1f7ea9b0c515d34cb1199cee1cd9e5d077c650cf6b6c|

Source: completed waveform-mechanism-study/results/scaling_dd_transfer release,
2026-11-16. Source thread was read only and reported no active run. Its delivery
hashes, structural NPZ hashes, basis identities and completed minimum coverage
were checked before a two-pass hash/size/mtime-stable local snapshot. Tracked
support/configs/scaling_artifacts.json binds all64 copied files and the local bundle.
No source import, mutation, training or campaign restart.

Higher-N training:2000 mean+1000 tail steps each; parent mean and tail selections
use nominal validation, tail mean≤1.02×same-N parent. No BER-based checkpoint
selection. One trial each: no training-seed uncertainty or dimension-only causal
claim; N64 is historical while higher N are fresh replications. W[sample,symbol]
is synthesis orientation; no transpose, phase adjustment or per-plot permutation.

Current replay uses this release's **nominal** channel index0, shared across
waveforms, fixed before BER. It replaces the notebook's earlier focused-N64
illustrative sample/curves with a single consistent scaling-release protocol;
the focused data and original internal workflows remain intact. All nominal BER
arms have4768channels/4,882,432bits/point,8SNRs(0,5,10,12,15,18,20,25),raw slicing,
fixed-nominal/full LMMSE,2000 channel-bootstrap resamples. All cells stopped
budget_censored_after_complete_minimum. Some points have<300errors; flags/counts
remain visible. Pointwise intervals approximate adaptive stopping. Zero-event
markers show1/bits, not zero BER or a confidence limit; channel upper bounds
remain in the table. No universal Learned win: full-receiver QPSK can favor OTFS.

## Complex frequency representation

C=FW contains complex unitary-DFT coefficients. Plot real/imaginary components of
the finite-block DTFT S_j(f)=Σ_n W[n,j]exp(-j2πfn)/√N, sampled at8N frequencies,
no window/peak normalization. S_j(k/N)=C[k,j]. Magnitude |C|, power |C|² and summed
energy are distinct. Σ_k|C[k,j]|²=1; summing8N dense powers gives8, not1. Zero
padding interpolates a periodic DTFT, not additional independent measurements or
a continuous-time pulse-shaping model. Original complex arrays remain available.

Time and complex spectral panels use matched native column IDs, all baselines,
and Learned group-colored headings. Higher-N defaults choose the lowest native
index in each group, independent of BER. N64 keeps its previously frozen six IDs.
DISPLAY.basis_columns[0] selects transmit j for the propagation view.

## Coupling and frozen groups

A[i,j]=w_i^H H w_j: transmit column j contributes to received analysis coefficient i.
Hw_j=Σ_i w_i A[i,j]. Native A stays authoritative. The additional group view uses
the frozen stable discovery permutation on both axes, never per-channel sorting.
The inline table explicitly maps every plotted rank to native basis ID/group.
Colors are consistent across group strips, column headings, received-index bars
and group spectra. Group IDs/colors have meaning only within each size.

Three objects stay separate:

- One sample |A|², computed from the displayed H.
- M=mean(|W^H H W|²), from512 independently drawn nominal normalized multipath
  channels; the source averages **squared magnitudes**, not complex entries.
- Q_test=mean(|W^H T(τ,ν)W|²), from2048 nominal unitary DD atoms. Discovery Q uses
  a separate1024-atom bank and determines fixed support/groups. Q_test differs
  from the normalized multipath second moment M.

All use output-row/input-column orientation. Source groups passed split-half
and independent structural-test gates. They are approximate/leaky, can occupy
split/wrapped frequency regions and do not imply a unique analytical waveform.
No N64 three-group assumption is applied at higher N. Fresh model changes reuse
these **labeled nominal** statistics/groups rather than silently recalibrating.
Fresh individual-channel H/A/propagation are newly generated; archived nominal
M/Q are replayed, not recomputed by the notebook.

H_f=FHF^H and A=C^H H_f C are reconstructed with complex coefficients, with maximum
entry error printed inline. |C|² alone cannot reconstruct A. The propagation view
shows |A[i,j]|² in frozen receive order and coherent group contributions to the
received spectrum. Group contribution powers do **not** add to total power:
cross terms remain. The full complex sum reconstructs H w_j.

Same H, bits and physical time-domain noise across Learned/OTFS/AFDM at a given
size/SNR. Across sizes, source normalized primitive paths are paired, but fixed
sample delays and Doppler-bin ranges are not identical physical impairments at
fixed sample rate: DopplerHz=nu*Fs/N. Fixed K/N does not resolve that distinction.
OTFS grids are8×8,8×16,16×16 (delay index fastest), W=kron(F_nu^H,I_tau).
AFDM retains c1=1/(2N),c2=sqrt(2)/(100N), without retuning. See SCIENCE.md.

## Scope and inputs

The notebook retains nine core DD views plus four optional overlays (13 by default): common physical/effective matrices,
time and complex spectral bases, coupling/group view, grouped member energy/magnitude, identifiable basis examples, received-basis/spectrum
propagation, QPSK/16QAM BER. No intervention/transfer campaign figures return.
Internal histories, variants, stress results and original sources are preserved.

Existing DD results require local/advisor-scaling-artifacts.tar.gz;
CFO uses local/advisor-n64-artifacts.tar.gz. New DD training requires neither bundle.
The scaling bundle is90,150,353bytes, SHA256
71a342c505ddfbafefc2d2bc368dc9071ebc1996cfb9414130f62bf0f3d9edd5.
Run All provisions it under local/ and checks every manifest entry. No ignored
input is assumed to appear in a clean checkout. See ARTIFACTS.md. Unavailable N,
missing bundle, altered basis/support or unsupported replay settings fail clearly.
Only numerical/provenance inputs, selected phase metadata and exact source snapshots
are copied, not unrelated galleries or optimizer checkpoint histories. Original
checkpoints remain untouched in the source directory; exported W is sufficient
for archived-basis evaluation. New DD training uses its own independent recipe
and requires neither historical bundle; it does not replay the source training.


## Grouped member energy / normalized magnitude

Additional figure `grouped_spectra`, in the same sequential notebook path, follows
the user's textual reference concept (original pixels were not inspected).
Actual frozen size-specific labels determine row count/member count; no prescribed
three groups or23/21/20 assumption. Rows/group colors and the rank ranges in the
headings match the permuted coupling matrix and full native-ID mapping table.
Every member is plotted, in that same frozen within-group order.

Left column: absolute |C[k,j]|² at the N unitary-DFT knots k/N; thin colored
member curves and black arithmetic mean **within each group**. No individual
peak scaling, no magnitude-as-energy substitution. Every unit-norm basis column's
N energies sum to1. Both columns span periodic normalized f/Fs in[0,1);1 marks
the wrap boundary, with no extra independent sample at1.

Right column: finite-block complex DTFT sampled by32N-point zero-padded FFT,
then magnitude |S_j(f)|; each column independently divided by its maximum on
that32N-point grid. Plot20log10 of that ratio (amplitude dB), clipped at−60dB.
No black mean on this side. The right plot compares normalized shapes, not
absolute strength/energy. Zero padding only interpolates the finite-block spectrum;
it adds no physical frequency resolution. No window or group-conditioned filtering.

The separate actual complex real/imaginary basis-spectrum figure stays unchanged.
Optional local `grouped_spectra.npz` preserves complex DFT/dense coefficients,
absolute energy, grid peaks, normalized dB, frequencies, labels and permutation;
all derive from the same selected Learned W. This is a new visualization of frozen
inputs, not a training run, new grouping, BER campaign or new experimental evidence.


## Individually identifiable selected bases

The one configuration cell exposes EXAMPLE_COLUMNS per N, native synthesis-column
IDs rather than permuted ranks. Defaults are the lowest native ID in each frozen
group (N64:0/15/44; N128:0/8/22/46; N256:0/40/87/148), chosen independently of
BER or visual quality. Up to two examples per group/eight total; unsupported or
overcrowded selections fail before creating a run. Users can choose other members,
including multiple groups or a subset of groups, without changing the learned W.
Display sorts selected IDs into the existing frozen coupling permutation, not a
new outcome-dependent order. The inline mapping table identifies group/native ID/
permuted rank and marks selected examples in the full native-ID table.

Grouped overview still shows **every** member and the black absolute-energy group
mean. On its right side only, selected members additionally have black solid/dashed
curves labeled with native j and rank (at most two per row). Group label/color is
unchanged. This makes specific shapes identifiable without crowding the overview.

The additional `basis_examples` figure gives each selected member its own row:
real/imaginary time basis, individually peak-normalized spectral magnitude and
its exact coupling column |A[i,j]|² in the frozen receive order. Every title repeats
Learned/group/native j/rank. Same displayed physical H; output-group bar colors
match the coupling/group plots. These are actual matrix columns, not a group mean
or a substituted "representative" waveform. The separate complex-frequency basis
comparison stays intact. Selection changes no H/A arrays, Q/M, support, receiver,
BER counts, basis/checkpoint or group inference; tests verify this invariance.

Optional archives retain example_native_columns in matrices.npz/grouped_spectra.npz
and a full selected mapping in examples.csv, bound by run.json input/source/file
hashes. All original internal data remains untouched. No new training/campaign.
