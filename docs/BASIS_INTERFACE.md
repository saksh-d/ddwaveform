> Historical completed-exploration API notes. Current notebook uses the configurable
> workflow: [current controls and support](CONFIGURABLE.md). Training limitations and
> CPU-only fresh descriptions below apply to the preserved legacy API, not the new entry point.

# Shared basis-source contract

One notebook configuration cell selects MODE=load_existing or train_new.
load_existing currently returns the selected, verified matrix for N64/128/256.
train_new is unavailable: it raises before provisioning/loading/run creation,
with no fallback or campaign. A future separate training branch should implement
that provider boundary, not a second downstream plotting/evaluation workflow.

SelectedBasis supplies:

- CPU complex128 synthesis W[N,N], finite/unitary, N=64/128/256.
- Its own boolean fixed-K support[N,N], K=3N/8 entries per row.
- Source/selection/training provenance, with content digest calculated at use.

Shared downstream cells consume that result for H/A, matched time/complex-frequency
basis figures, coupling views and paired BER. Configured N must match W exactly;
no resizing, transposition, substituted size or hidden checkpoint choice. See
[qualified size identities](COUPLING.md). New providers cannot reuse another W's
saved curves, support or group labels. Group views require qualified statistics
belonging to the exact selected W/support; future training must additionally
supply or explicitly qualify its own discovery statistics/groups.

EVIDENCE_MODE controls replay/fresh evaluation independently of basis source.
Replay verifies exact W/support identity against size-specific saved evidence.
Fresh draws physical channels/bits/noise using declared settings and retains
nominal support/groups, without training/recalibration. Saved nominal Q/mean powers
remain identified as archived nominal statistics even when fresh channels change.
Future training objective/seed/schedule/initialization changes produce a new basis;
channel/evaluation/display controls only evaluate the frozen selected basis.

After a mode/config change, clear outputs/restart/Run All. The notebook clears
its workflow variable before selection, so unavailable training cannot leave a
stale basis available for downstream execution. Source notebook stays output-free.
