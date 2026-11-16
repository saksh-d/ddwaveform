# One selected Learned waveform

All advisor-facing DD outputs use the same verified N64 synthesis matrix,
labeled **Learned**. File SHA256:
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`.
This applies to time/frequency columns, physical/effective-channel figures
and every BER SNR/receiver/modulation. Internal analyses retain the same
provenance but are excluded from advisor figure generation.
OTFS and AFDM are the comparison baselines. No per-plot, per-SNR or per-modulation
replacement, phase alignment, reordered checkpoint or relabeled fresh matrix.

The existing reference is retained after reviewing the saved common-bank
evidence. The declared performance criterion is **raw uncoded BER jointly for
QPSK and16QAM at18dB Es/N0 under fixed discovery-Q24 reception**. No preference
weights between the two modulations were supplied. Full-A LMMSE and the complete
saved SNR range are secondary checks, retained separately.

The saved alternatives trade QPSK against16QAM performance. The selected reference
is not dominated in both primary BER measures by another evaluated learned
candidate. This is a descriptive point-estimate observation, not a statistical
proof of optimality. A criterion favoring QPSK alone or16QAM alone could choose a
different candidate; no universal winner follows. Keeping the prior reference
also preserves the already qualified matrix/paths/support/group/bridge/curve
provenance as one coherent package.

This review is **post hoc**: results were visible before the presentation choice.
It is not independent test evidence for the selection, a newly frozen prospective
selection protocol, or a claim that Learned always beats the baselines. Original
training selections were validation-based; this later presentation decision must
not be confused with those selections. No new simulations, training, bootstrap
or checkpoint optimization were performed. No numerical data changed.

Only the original main study's common bank is used to assess candidates:
1024physical channels and1,048,576bits/point, common bits/noise and each waveform's
own discovery-Q24 support. The separate focused adaptive-count bank is used for
the selected reference's advisor curves and is never pooled with that bank to
rank candidates. Original counts, full-range reversals and uncertainty remain.

The complete candidate provenance, tables, exploratory figures and working data
remain available under the separate [internal workflow](../internal/README.md).
Advisor plots/notebook do not expose those multi-candidate comparisons.

The archived residual-CFO W/R example is a separate rectangular model with its
own provenance. It is not the N64 selected matrix and is never mixed into this
DD comparison or substituted into its plots.
