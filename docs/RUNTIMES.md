> Current notebook extension: [N64/128/256 own bases and spectral/coupling views](COUPLING.md).
> Current replay uses the completed scaling-release protocol; earlier N64 evidence below remains archival.

# Runtime evidence and limits

The README numbers are archival full-workload measurements/planning estimates,
not notebook-smoke timings and not a promise for future training. No new timing
campaign or training was run for this documentation change.

Host assumptions: this Apple M3 Pro Mac,18GB unified memory; CPU float64/
complex128, one Torch/BLAS thread. Historical environment records Python3.10.17,
Torch2.7.0, NumPy2.2.6. No MPS/FP32 speed extrapolation. N64/P3 nominal normalized
DD channel, perfect CSI, unit-symbol-energy QPSK/16QAM.

## Full focused BER

Source: provisioned `local/artifacts/focused/`:
`protocol_pre_pilot.json`, `final_run_manifest.json`, `runtime_estimate.json`,
`budget.json` and27completed communication points. No training in this sweep.

- Three frozen bases, two receiver models (fixed-Q24/full), two slicers (raw/gain).
  Original full SNR grids total27points; display limits do not reduce stored data.
- 1024 bits per physical channel; each point starts 2048 channels/2,097,152 bits.
  Shared adaptive stopping: every arm at least300errors after minimum coverage,
  otherwise cap19531channels/19,999,744bits per arm/point. Same comparison budgets.
- **Measured evaluation:**428.345929s =7.14min. This is the ledger's complete
  `focused-evaluate` segment; preparation, pilot, plotting and summarization
  are separately recorded. Actual adaptive counts did not all reach the cap.
- **Extrapolated maximum workload:**512-channel complete batch benchmark1.546395s
  ×27points ×39batches =1628.353759s (27.14min), plus900s reserve =42.14min.
  This is a worst-case planning estimate, not measured full-cap execution.
  Original cumulative numerical hard cap10800s (3h), not an expected duration.

The current notebook's fresh mode is deliberately bounded to much smaller
budgets and raw fixed/full evaluation. It does not implement this full adaptive
campaign. Its smoke runtime must not be substituted for full BER timing.
Saved replay does not simulate BER at all; it reconstructs H/A and plots counts.

## Full training reference recipe

Sources: `local/artifacts/supplementary/main/`:
`training_selected_summary.csv`, `config.json`, `manifest.json` and saved selected
phase metadata. These are three completed fresh OFDM-start replications in the
prior study, not recovery of the original historical Learned optimizer run.

Each trial executed2000mean steps,1000mean-continuation steps,1000tail steps and
2000Q-only steps:6000steps/trial. Twelve completed phases across three trials.
Receiver steps use16channels; Q steps use128atoms. Validation uses256channels,
SNR12/15/18/20dB and cadence25steps; normalization/projection/selection overhead
belongs to that source protocol. A step is not a uniform unit of compute.

- **Measured full phase-runtime sums:** trial0=269.202064s, trial1=278.078738s,
  trial2=299.774230s, or4.49/4.63/5.00min. All three total847.055032s (14.12min).
  These sums include the saved phases' validation/checkpoint overhead and exclude
  separate BER/structural/plotting work.
- **Original projected training per trial:**276.076653s (4.60min), from measured
  production-shaped kernels and validation cost. It was an estimate before the
  completed study, not a fresh benchmark of this package's unavailable trainer.
- **Historical mixed-study planning:**4000.697383s (66.68min) for three trials
  plus evaluation/structure/reserve, under7200s (2h) cumulative numerical cap.
  The reported completed mixed-study ledger was2042.9s (34.05min).
  Neither mixed-study duration is a training-only runtime.

`MODE="train_new"` currently raises NotImplementedError before basis loading or
run creation. Its future objective, schedule, precision and implementation are
not frozen/benchmarked here, so no defensible runtime estimate exists for that
new implementation yet. A follow-on can benchmark its declared complete recipe
and state a new budget before training. This branch starts no such workflow.

## Current scaling-release replay

The notebook now replays the separate completed scaling release (N64/128/256,
nominal only). That source's complete training+BER+structural/transfer/figure study
accounted10108.5s/168.47min of a10800s/180min budget, including a15s conservative
qualification allowance; measured numerical-command ledger10093.5s. Production
used4 CPU threads, complex128. This is **not** isolated nominal BER time, not a
one-thread notebook prediction and not a forecast for train_new. The current
package runs only one-thread replay or bounded fresh checks; no new full campaign.
