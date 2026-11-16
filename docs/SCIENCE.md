> Current notebook extension: [N64/128/256 own bases and spectral/coupling views](COUPLING.md).
> Current replay uses the completed scaling-release protocol; earlier N64 evidence below remains archival.

# Model and interpretation

## Delay–Doppler

Column j of W is the time waveform for symbol j: `x=Ws`, `y=Hx+n`,
`z=W^H y=As+W^H n`, `A=W^H H W`. Native ordering authoritative.

```
F[k,n] = exp(-j2πkn/N)/sqrt(N), k,n=0,...,N-1
Pi(tau) = F^H diag(exp(-j2π k tau/N)) F
Delta(nu) = diag(exp(-j2π nu n/N))
T(tau,nu) = Delta(nu) Pi(tau)          # delay first
Hraw = sum_p h_p T(tau_p,nu_p)
scale = sqrt(||Hraw||_F²/N), H = Hraw/scale
```

Nominal N64, P3, iid tau U[0,4], nu U[-.75,.75], independent pre-normalization
gains CN(0,1/3). Circular finite-block, perfect CSI. No continuous-time front end,
CP/pilot overhead, FEC, spectral-mask, RF or over-the-air validation.

Historical OTFS `kron(F8^H,I8)`, delay index fastest; no implicit reshape/shift.
AFDM `diag(exp(+j2πc1 n²)) F64^H diag(exp(+j2πc2 k²))`, c1=1/128,
c2=sqrt(2)/6400. These are historical calibrated parameters, not a universal
optimal AFDM setting or fresh tuning. OFDM F64^H is a classical control.

## Three distinct matrices and the complex frequency bridge

`W^H W=I`: orthogonal transmitted columns, even if samples look irregular or
spectra overlap. `A=W^H H W`: channel coupling, often dense.
`W^H H^H H W=A^H A`: Gram matrix after propagation; multipath can destroy
orthogonality. A single unitary DD path can create dense A while preserving this
Gram matrix. These facts describe different physical questions.

`C=FW`, `H_f=FHF^H`, **`A=C^H H_f C`**. Every complex coefficient matters.
The independent `frequency_paths` finite sum checks path order/sign without
using a plotted magnitude. For independent zero-mean gains, the raw expectation
E|Araw_ij|² equals the corresponding average atom-response power Qij; the
realization-dependent normalization generally changes that identity for A.

Completed N64 bridge reports max normalized column inner product1.01e-15 over
2016pairs, reconstruction max4.80e-14 across2048structural channels. This package
independently checks all column pairs, saved example and small fresh path banks;
it does not rerun the full2048-channel bridge campaign. Measured groups23/21/20
have approximate/leaky frequency localization. Projector-bin overlaps
98.80/97.19/98.07% describe those saved subspaces, not unique analytical bands.
DFT-bin energy equality does not imply equality between bins or equal coupling.

## Receiver, mapping and uncertainty

`G=(Amodel^H Amodel+sigma² I)^-1 Amodel^H`, exact dense solve for every arm.
Full model uses A. Realized-K picks strongest realized entries per row. Fixed-Q
uses stable row-Top24 positions from an independent discovery Q; it keeps each
realized **complex value**, not Q as channel coefficients. Actual observations
always use complete A. K counts modeled interactions per row; all64symbols and
observations remain. No end-to-end complexity/speed equivalence established.

Full square-unitary LMMSE average MSE is basis-invariant; individual stream
reliability and finite-alphabet BER need not be. Exact mismatched stream error is
diag[(GA-I)(GA-I)^H+sigma²GG^H]. Concentration alone omits complex cross-products
and does not determine decision-region errors.

DD axis **Es/N0**, sigma²=10^(-SNR/10), unit-average-energy QPSK and Gray16QAM.
QPSK labels map0 to positive real/imag;16QAM Gray indices `[00,01,10,11]` map
`[-3,-1,+3,+1]/sqrt(10)`. Raw nearest-point slicing is primary. Supplementary
gain diagnostic divides by true complex diag(GA), guard1e-12, under perfect CSI.
No waveform-specific slicer or BER-based checkpoint selection.

Focused saved bank: same H, bits and time-domain noise across all3waveforms,
2receiver models and2slicers. Each independent channel1024bits (QPSK8blocks,
16QAM4blocks). Start2048channels; continue together until every arm300errors or
cap19531channels/19,999,744bits.27frozen SNR points total. Pilot disjoint, grid
frozen before final counts. Saved pointwise2000-resample channel-bootstrap
intervals approximate under adaptive stopping; not simultaneous intervals.
Per-channel counts and original batches/paths remain in bundle. Source protocol
and namespace seeds are preserved; count affects RNG draw shapes.

Zero errors unresolved: marker1/bits is a resolution indicator, not a bound.
Conservative independent-channel95% upper bound `1-.05^(1/C)` is separately
reported, leaving degenerate zero bootstrap intervals undefined. Small new BER
smoke reports descriptive channel SE and counts, never promotes it to a curve
replication. Same comparison budgets/seeds within each check.

Learned's fixed-support advantage is conditional. Completed full-QPSK target
1e-3 estimates: Learned18.02dB, AFDM16.59dB, OTFS14.70dB, from measured brackets
only. Fixed-QPSK Learned18.57dB; controls unbracketed through30dB. Fixed mismatch
can flatten or worsen as regularization weakens at high SNR. Full40dB16QAM
AFDM-vs-Learned difference unresolved by paired interval; OTFS lower. Full saved
curves and unfavorable outcomes retained; default25dB display is just a view.

## Residual CFO: separate model

Historical `apply_residual_cfo`: `H=diag(exp(+j2πeps n/M))`, M64, float32/
complex64. eps measured in1/Msample-rate units. No DD channel normalization.
Qualified stored W64x45 and receive analysis R45x64 are distinct;
`V=R^H`, `A=V^H H W=R H W`. Rows of R already contain receive conjugation.
Historical `einsum('mk,bm,nm->bkn', W, phase, R)` stores A transpose; package
exports both conventions. Noise after analysis has covariance sigma²RR^H.
W rectangular/column normalized, no square-unitary or whiteness assumption.

The checkpoint models45data streams in a56-dimensional payload subspace of a
64sample frame,4reserved pilots and4guards; redundancy11. Pilots were not used
for estimation. Historical BER axis Eb/N0, sigma²=1/(4*10^(EbN0/10)) for16QAM.
Stored CFO BER is archival, not rerun or compared directly to DD Es/N0 plots.
At operator level only, matching exponent to DD would require
`nu=-eps*N/M`; different bases, dimensions, noise and channel semantics remain.
DD Doppler-only structural transfer is not this historical W/R experiment.

No NOFS baseline invented. Official NOFS weights/spec unavailable in completed
source audit. SEFDM is a different nonorthogonal construction and omitted from
the compact core; original exploratory evidence remains in original directory.
