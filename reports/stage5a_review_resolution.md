# Stage 5A independent review resolution

Separate read-only Codex process, session `01a19346-d797-7922-9fb4-fff336f32887`, reported runtime model `gpt-6-luna`, low reasoning effort. No requested model was substituted. Original review is retained unchanged in `outputs/stage5a/review/initial_independent_review.md`.

| Finding | Resolution / evidence |
|---|---|
| Cold synthetic qualification fails | Accepted. All 12 failures retained. Six planted-neighborhood checks diagnose optimization uncertainty; no architectural impossibility or successful-compression claim. |
| Partial fit grid could be frozen | Fixed. Evaluation requires completion marker, original protocol hash and exact unique 72-row grid. Actual selection followed completed fitting. |
| Exact identifiability does not imply approximate DD/BER equivalence | Accepted. Report proves exact labeled-family identifiability, gives only a one-way approximation bound and leaves weighted operator fitting open. Final independent operator and BER banks retained. |
| Historical population narrower than requested | Accepted, prominent limitation. User-specified archive searched; unavailable repeats/Q-only not retrained or replaced. |
| Channel bootstrap has finite-block/rare-event limits | Estimand clarified. Independent channel/bit/noise clusters are resampled intact; empirical cluster variability includes finite-block variability. No iid-bit, exact-tail or simultaneous preservation claim. All counts/error-bearing-channel counts retained. |
| Scalar gain correction retains interference | Accepted as labeling clarification. Same full LMMSE; optional per-stream diagonal-gain correction, no joint-detection claim. QPSK counts identical for every channel/point. |
| Nesting only protected in one restart, may worsen | Rejected with executable evidence. Both restart sequences use `incumbent`; first closure records step zero before optimizer updates. All 24 target/topology/restart sequences are monotone. Exact identity-nesting test passes for both topologies. |
| Native loss and global-phase diagnostic differ | Accepted distinction already implemented. Native complex loss remains the sole fit selection criterion; alignment never changes transmitted waveforms. |
| Compilation timing measured only clones | Fixed before final evaluation. Actual `model.compile()` repeated and timed separately. Nominal arithmetic ledger explicitly excludes gather/allocation work, measured CPU latency includes it. Complete-chain timing and receiver setup added. |
| No final results yet verified | Resolved by saved final counts and independent physical replay/aggregation. Follow-up review checks final report and results. |

The initial review recommended delaying communication until fitter qualification. Lead instead recorded a prospective clarification before final-bank generation: evaluate attained frozen matrices diagnostically, with no capacity or successful-compression inference. The user's brief asks separately whether actual fixed candidates preserve communication; optimizer nonoptimality does not invalidate a direct BER measurement of those candidates. No candidate was changed after those data were inspected.

Independent numerical implementation verified 196,608 raw rows, 192 summary groups, 512 exact error-count replays through a different effective-coordinate solve, a paired noninferiority interval, mean-MSE invariance and 42 protected historical files. This validates calculations and provenance; it does not establish a successful optimizer or an independent scientific replication.

Final separate read-only review: `outputs/stage5a/review/final_independent_review.md`. Conclusion: **no material issue found within the reviewed evidence**. It verified the main fidelity, BER/noninferiority, gauge and implementation-cost statements against the final artifacts, retaining all scientific limitations above.
