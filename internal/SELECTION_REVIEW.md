# Internal existing-evidence review

Criterion declared post hoc: raw uncoded BER at18dB Es/N0, fixed discovery-Q24,
QPSK and16QAM jointly, with no modulation preference weights. Main common bank
only:1024physical channels and1,048,576bits/point. All43learned candidates,
including intervention directions, have reconciled per-channel counts. Baseline
and focused adaptive-bank counts are not pooled into candidate ranking.

Retain HISTORICAL as the advisor's one Learned matrix, SHA256
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`.
Raw primary BER: QPSK0.0011348724365234,16QAM0.0245466232299804.
No evaluated learned candidate has both primary point estimates no worse and at
least one strictly better. This is descriptive, not statistical Pareto proof.

Across the examined candidates, T1_tail has lowest primary QPSK while T1_q has
lowest primary16QAM. For the original display trial, T0_tail similarly favors
QPSK while T0_q favors16QAM. The historical reference lies between these
objectives. Selecting a best individual trial after observing BER would add
selection bias; no independently confirmed overall winner is established.

Full-A reception and all saved SNRs/receivers/modulations are retained as
secondary evidence, with adverse orderings and uncertainties intact. There is no
switch of matrix across figures or conditions, and no simulation or bootstrap.
Original checkpoint selection was validation-only; this later presentation
choice is explicitly post hoc and cannot reuse those BER counts as an
independent test of the choice.

Machine-readable review under `local/internal/selection_review`:43candidate
tradeoff table,86primary rows, all-condition table and JSON provenance. Rebuild
with the command in README using a fresh output folder. These IDs/tables are
internal and never relabeled into the advisor plotting input.
