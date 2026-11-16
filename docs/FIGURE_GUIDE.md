# Reading the figures

The root [Waveform Exploration notebook](../Waveform_Exploration.ipynb) is the entry point.
The notes below expand its short captions; they describe definitions, not new performance claims.

## 3. Basis spectra and coupling-derived groups

Same native column IDs across time/frequency and waveforms; panel-title colors
identify **Learned's own frozen groups**, not baseline groups. Solid/dashed curves
show real/imaginary coefficients. Frequency curves sample the finite-block complex
DTFT S_j(f)=ΣₙW[n,j]exp(-j2πfn)/√N with eightfold zero padding, no window or peak
normalization. At DFT knots S_j(k/N)=C[k,j], C=FW. Magnitude is |C|; power is |C|²;
Σₖ|C[k,j]|²=1. Dense DTFT samples are correlated interpolation, not new evidence.

The following permutation table explicitly maps plotted ranks to native basis IDs.
Archived groups were inferred from **nominal coupling**; new groups use the configured
train/discovery bank, independently of the displayed test channel.
Archived grouping is size-specific; new grouping is derived from train/discovery multipath
coupling by thresholded graph components, with diagonal excluded. Count can be one.
Freeze these groups across all views. They are approximate,
leaky and may occupy split/wrapped frequency regions, not a unique analytical construction.

A[i,j] couples transmitted basis j into received basis i: Hw_j=Σᵢw_i A[i,j].
The group view separates one |A|² sample, mean |A|² over multipath channels,
and Q=mean|WᴴT(τ,ν)W|² over unitary atoms. Titles give actual counts/laws. Q is not the normalized
multipath second moment. Neither statistic is |mean(complex A)|².
Replay retains nominal statistics. Fresh recomputes independent diagnostic test statistics
under the evaluation law, without changing W/support or selecting from test data.

The complex bridge Cᴴ(FHFᴴ)C=A is checked below; power spectra alone cannot reconstruct A.
The propagation view uses DISPLAY's first native column j and the same H as the channel
figure: received-index powers and coherently synthesized group spectra locate the leakage.
Group spectral **powers are not additive**: complex contributions have cross terms.


**Grouped energy/magnitude view (additional, not a replacement for complex spectra):**
one row per actual coupling group, matching the group colors/permutation/native-ID table.
Left: **every member**'s absolute N-point DFT coefficient energy |C[k,j]|², with
black arithmetic group mean. No individual peak normalization on this side; each
unit-norm column sums to1 over N DFT bins. Right: every member's finite-block
spectral **magnitude**, 20log₁₀(|S_j(f)|/max_f|S_j(f)|), individually peak-normalized
on the displayed32N-point grid and clipped at−60dB. Shapes here do not compare
absolute strength. Both axes use periodic f/Fs in[0,1); the endpoint1 labels the
wrap boundary, not an extra sample. Zero padding interpolates the finite-block
spectrum; it adds no physical frequency resolution. No window or retraining.
Rows/counts come from the frozen size-specific grouping, not the reference example.
The separate complex real/imaginary frequency figure above is retained.


**Individually identifiable examples:** EXAMPLE_COLUMNS in the single configuration
cell selects native Learned column IDs, independently of DISPLAY.basis_columns.
The selected examples are shown in frozen coupling order, at most two per group,
with an inline table linking **group, native j and permuted rank**. Archived defaults use the lowest native ID of each frozen group; new-fit defaults
show native ID0. Neither choice is selected for BER or appearance.
Black labeled lines in the grouped plot's right column identify those same examples
against all member curves. Each has its own row below: real/imaginary time basis,
readable spectral shape, and that exact transmitted column's |A[i,j]|² across the
received bases using the same displayed H. Titles repeat group/native j/rank;
received-basis bar colors match the frozen groups. Changing examples changes only
what is displayed; it does not change W, H, A, group inference, receiver or BER.
The separate actual complex-frequency comparison remains above.


**Readability variations for user review before final simplification:** PLOT_VARIANTS
chooses the enabled alternatives in this same configuration cell. OVERLAY8_COLUMNS
selects exactly8 native basis IDs; HALF_COLUMNS selects exactlyN/2 (32/64/128).
Every selected curve is overlaid together in each applicable panel, not split into
small subsets. Display follows the frozen coupling permutation; the full curve
key and inline index tables identify native j, group G and rank r. Color is a
stable native-ID color, not the group-color palette used in the group overview.
Line style is fixed by coupling rank. Default index choices are deterministic,
independent of appearance/BER. Dense overlays are retained for readability review;
use the labeled individual examples when curves overlap too heavily to trace.

The complex layouts separate real/imaginary time samples and actual complex DTFT
coefficients, without peak normalization. Energy/magnitude layouts instead show
absolute N-point |C|² and individually peak-normalized20log10 magnitude (−60dB
floor). Same32N-point interpolation grid; zero padding adds no physical resolution.
The coupling-group reference view remains an enabled alternative by default, and
all earlier coupling/individual/complex-basis figures remain available. These are
views of one Learned matrix per N, not comparisons among learned variants.


CFO shows distinct complex transmit W and receive Rᴴ columns. DD coupling-group,
unitary bridge and 8/N/2 DD overlays are inapplicable to rectangular W/R.
