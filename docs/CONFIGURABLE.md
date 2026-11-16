# Configurable workflow contract

Base: `c6bc03b720a97948d28036579f673eee98bdb95b`, completed waveform-exploration.
Implementation is confined to this isolated worktree. The stopped wrong-base task
is not a source. Archived internal/source snapshots remain inspection-only.

## Physical models

DD supports only `dd_v1`: circular fractional delay followed by negative Doppler,
complex Gaussian P-path gains, per-realization Frobenius normalization to N.
P1–6, delays within[0,8] samples, Dopplers within[-2,2] DFT bins; endpoints increasing.
No extra named fading families, path-loss, CP or hardware model are invented.
N64/128/256 have own W; compatible OTFS grids8x8/8x16/16x16. AFDM keeps the
existing chirp rule. Every DD arm uses unit symbols, identical physical H/noise,
the same K per row and the same stopped channel prefix. Full and fixed-support
LMMSE stay distinct. Actual observations always retain all channel interactions.

Residual CFO supports only64samples/45streams and the hash-qualified stage1 W/R.
H=diag(exp(+j2πεn/64)), ε uniform in configured range within[-.5,.5].
W[64,45], R[45,64], A=RHW. R differs from Wᴴ and noise covariance after analysis
is σ²RRᴴ. Fresh BER transmits through W/H, adds physical white noise, then uses
historical direct R analysis and raw slicing. No extra LMMSE is inserted.
The new axis is Es/N0; historical CFO Eb/N0 tables are not reused. There are no
CFO classical-comparison, pilot-estimator, compensation, multipath or sparse-receiver
claims. CFO training is rejected before provisioning/output: the archived staged
joint W/R optimizer, constraints and acceptance gates have not been reconstructed
and numerically qualified here. A DD unitary trainer would be physically wrong.

## Controls and scope

| Controls | Effect |
|---|---|
| SCENARIO, MODEL | Validated physical evaluation law; changes do not train W |
| MODE | Load one archived/checkpoint W, or train a new DD W |
| N, K_FRACTION/K | Own dimensions, row support budget; default3/8; no resizing |
| SUPPORT_POLICY=frozen | Retain selected basis's own fixed discovery mask |
| SUPPORT_POLICY=recalibrate + DISCOVERY | Explicit new train-independent discovery law/bank for loaded W, all DD baselines same bank/K; fresh only |
| TRAINING.model/validation_model | Separate fit/selection channel laws, same N as test |
| TRAINING seeds/iterations/lr/layers/init/channels/cadence/time | Actual new fit and one validation selection; never BER-based |
| EVALUATION seed/SNR/modulations/counts/errors/time | Independent new test evidence, paired across DD arms |
| EVALUATION diagnostic counts | Independent fresh atom/multipath test statistics; no refit |
| DISPLAY, REVIEW | Native IDs, complex/energy plots,8/N/2 overlays only |
| CHECKPOINT, OUTPUT_ROOT, SAVE_RUN | Local inference checkpoint and optional fresh archive |

Unknown/inapplicable keys fail. Replay requires nominal law, exact archived basis,
K/support; evaluation/discovery/training controls must be absent. New W/support,
channel law or evaluation grid cannot acquire historical BER. Archived full grids
remain available; snr_max_db only limits display. CFO has no DD K/support/group knobs.

K changes on archived W require explicit recalibration; train_new learns its own
support at the chosen integral K. Frozen masks on changed test laws remain frozen,
with no hidden recalibration. One W per run/N is used for all SNRs and figures.

## New DD training recipe

This is a documented new bounded experiment, not a replay of historical training.
Initialize from OFDM/OTFS/AFDM and small seeded phases. Alternate learnable time
and Fourier-domain diagonal phase layers: every factor is unitary. Adam updates
real FP32 phase parameters; no unconstrained rectangular transform or resizing.

Train bank: normalized multipath H sampled from the full training law. Let
Mtr=mean|WᴴHW|². Choose K largest entries per row (stable native-index ties).
Loss =1−sum(mask·Mtr)/sum(Mtr). Support selection is detached; gradients flow
through selected coupling energies. This concentration objective has no SNR or
modulation parameter and is not a BER objective. Path count/ranges affect the
actual training and validation channel draws.
Validation uses its own multipath bank and current **train-derived** mask, minimizes
held-out leakage, includes step0, earliest ties. One selected W is exported.
Failure to beat initialization is retained. Finite loss/gradient, unitarity and
solve-residual gates stop on numerical failure; no device/precision fallback.

Train/validation/test seeds must differ, and role namespaces separate banks.
Test matrix, atomic/multipath diagnostics and BER have distinct namespaces.
Groups are connected components of symmetrized train/discovery multipath coupling,
threshold relative to each endpoint's strongest off-diagonal edge; diagonal excluded.
Group count is derived, including one diffuse group or singleton groups. Group
permutation is frozen stable native-ID order. Groups have no exact bandwidth claim.
Fresh diagnostic Q and multipath mean power use configured test law/counts,
labeled separately; neither is |mean(complex A)|². Complex group contributions
reconstruct the received signal; their powers omit cross terms.

## Budgets, precision and portability

Fresh default8channels/point,2blocks,4SNRs, both constellations; at most20training
steps/120s. Error target requires all DD waveform/receiver arms; all stop together.
Time is checked after each bounded batch, so a batch can exceed the cap and the
minimum coverage can be unmet (flagged). Training time includes setup/validation
and may exceed its cap by one step/validation. These are resource limits, not speed
or statistical precision promises. Larger bounded caps are validated explicitly.
Keep enough observations for inference; small runs remain qualification only.

Descriptive channel-SE intervals are approximate under error/time stopping. Counts,
minimum/target/stopping flags and1/bits resolution are retained. Zero-error bounds
use independent channels conservatively, not an independent-bit assumption.
No training-seed uncertainty or universal learned advantage is established.

All fresh simulation/optimization/receiver solves use MPS FP32 real/imaginary pairs
(no native complex requirement). LMMSE uses a2N real block solve, checks relative
residual ≤3e-3; unitarity max-entry gate5e-4. High-SNR/ill-conditioned cases may fail
rather than silently changing precision. Archived replay/reference checks keep
their pre-existing CPU float64/complex128 route. NumPy/Matplotlib postprocess
exported arrays for figures; files use ordinary CPU I/O. Checkpoint inference
uses the exact exported complex64 matrix and mask, evaluated via MPS FP32.

Training checkpoints are safe numeric NPZ plus JSON format/hash/provenance. No
optimizer resume is implemented. Load via MODE=load_existing, EVIDENCE_MODE=fresh,
CHECKPOINT=local/path.npz, TRAINING=None, matching N/K. Existing checkpoints are
never overwritten. With SAVE_RUN=True a default training checkpoint is saved in
its unique run archive; an explicit new CHECKPOINT overrides its destination.
All output/checkpoint paths resolve within this worktree local/, including symlinks.
Artifacts are copied, hash-verified and never mutated; no runtime original-project
paths or external sharing. Source notebook remains output-free.
