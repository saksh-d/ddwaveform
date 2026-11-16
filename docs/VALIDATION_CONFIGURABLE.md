# Configurable-workflow validation

Exact commands/results and executed notebooks are retained under ignored local/.
Local command ledger: `local/COMMANDS.md`; release evidence: `local/qualification-release/`.

```bash
conda run -n pytorch python scripts/bootstrap.py --bundle local/advisor-n64-artifacts.tar.gz
conda run -n pytorch python -c "from advisor.scaling_evidence import prepare_scaling; prepare_scaling('.', 'local/scaling-artifacts', 'local/advisor-scaling-artifacts.tar.gz')"
conda run --no-capture-output -n pytorch python -m pytest tests -q --basetemp local/pytest-release-final --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts
conda run --no-capture-output -n pytorch python scripts/qualify_notebook.py --out local/qualification-release
conda run --no-capture-output -n pytorch python scripts/audit_configurable.py --qualification local/qualification-release
```

Notebook qualifier copies the source notebook into local/, changes only its single
configuration cell, executes every cell, saves executed notebooks/inline PNGs and
an execution ledger. It never writes outputs into notebooks/results.ipynb.
Use a new --out path each time; checkpoint overwrite is deliberately rejected.
Default N64 training/evaluation settings are executed as presented. N128/N256
training qualification is bounded to3steps/3train and3validation channels, with
2test channels/point; these short runs do not qualify production training quality.

Baseline all63tests passed with both artifact sets before changes. Existing test
cases remain; notebook assertions now recognize working training and centralized
sequential figures. New tests qualify configuration rejection, native dimensions,
independent train/validation/test banks, deterministic seeds/gradients/reference
operators, actual optimization, checkpoint inference equivalence, changed-law/K
propagation, shared-arm budgets, zero-error uncertainty and distinct CFO behavior.

Reference tests retain the repository's explicit CPU double arithmetic; fresh
simulation and optimization tests verify actual MPS FP32 tensors, fallback disabled.
No full training sweep, multi-seed confidence study, long precision BER campaign,
physical-radio or CFO staged-training qualification is performed.

## Release results — 2026-11-16

- Full local artifacts suite: **97 passed in19.31s**, including all63baseline cases.
- Notebook matrix: **14successful executions +1expected CFO-training rejection**,
  **166inline PNGs**,172.93s summed wall time; no unexpected cell errors.
- N64 default:20training updates, selected step20;0.845s trainer,11.35s complete
  notebook. N128/N256:3updates each, selected steps3/2;12.91/21.91s notebooks.
- Changed DD train/eval laws, changed CFO range, N256 changed evaluation,
  explicit K16recalibration and independent checkpoint reload all executed.
- Reloaded N64 checkpoint: matrices, masks, mapping arrays, per-channel counts
  and BER exactly equal; timing metadata excluded from BER equality.
- Fresh trained max-entry unitarity/complex bridge errors ≤1.32e-6 across sizes.
  New short fits yielded one connected coupling group each; no forced target count.
- Audit verifies every archived file/source digest, inline/export PNG byte equality,
  all166PNG exports300DPI,645legacy+64scaling input hashes and output-free source notebook.
- Manually inspected core/coupling/grouped/individual/8-and-N/2overlay layouts
  across sizes and CFO W/R views; fixed discovered clipping/roundoff-scale issues.

Evidence: `local/full-tests-release-final.txt`, `local/notebook-release.txt`,
`local/release-audit.txt`, `local/qualification-release/{execution,audit}.json`,
executed notebooks and their referenced unique `local/notebook_runs/` archives.
The command ledger retains failed/provisional iterations as well as final checks.

Ready for Saksham's separate branch/code/notebook review. This qualifies the
configurable workflow, not production BER precision, historical training replay,
physical validation or a Learned performance advantage.
