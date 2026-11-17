# Historical verification before CPU-only correction

This record predates the owner's CPU-only instruction. Device selectors and
accelerator execution described below are retired; current runtime is CPU-only.
See ../docs/VALIDATION_CONFIGURABLE.md for the current verification.

# Configurable-workflow validation

## Explicit CPU backend and frozen delivery qualification — 2026-11-16

- Owner instruction requires MPS by default and no silent CPU fallback; permanent
  MPS-only behavior was an implementation choice. Latest request authorizes explicit
  CPU FP32. BACKEND="mps" remains default; BACKEND="cpu" is an explicit selection.
- Full130tests pass17.10s; integrated target repeats130pass/15.98s. Frozen clean
  clone95pass/35optional archive skips.
- With original inputs/previous packages blocked and MPS deliberately unavailable:
  default replay at N64/128/256 succeeds,39figures/41.52s. Explicit CPU fresh DD at
  all sizes,20step train/reload, changed DD/CFO and expected CFO-training rejection
  execute101figures/88.25s. CPU input-independent N128/N2562step training also passes.
- Real MPS7notebooks/83figures/79.77s; all83PNG bytes/572array fields match prior
  references. Historical input bytes unchanged; CPU fresh results never reuse saved BER.
- Checkpoint CPU↔MPS transfer preserves W/support and original training provenance;
  evaluation records its selected device. Seeds repeat within a backend; streams
  differ between backends. Budget/uncertainty rules and CFO dimensions unchanged.
- Source/output hashes, checkpoint reload equivalence and300DPI exports pass audits;
  CPU coupling/BER/CFO figures inspected. Exact commands/evidence: local/portability/.
- CPU and MPS tested on macOS Apple Silicon only; Linux/Windows instructions remain
  untested, CUDA unsupported, CFO training unsupported. No production campaign claim.

## Fresh-machine setup recheck — 2026-11-16

- Actual Git clone with a new Python3.10 venv installed dependencies and optional
  JupyterLab4.6.6; constraints preserved every locked version. Generic Python3 and
  registered Waveform Exploration kernels both execute inside that venv.
- Original artifact roots and previous environment packages OS-blocked:85tests
  pass,35optional archive checks skip. Full local-artifact suite:120pass/16.04s.
- DD64/128/256 replay,20step training/reload and CFO:6notebooks/70figures/69.47s;
  70PNG bytes and434array fields match retained references, CSV exact except timing.
- Runtime code/data and every notebook cell unchanged; only generic kernel display
  metadata and setup/provenance documentation changed. Public readable runtime,
  manifests and current docs contain no absolute machine paths.
- Exact install/probe/test commands and logs retained under ignored local/fresh-setup/.
  This verifies a clean environment on the qualified Mac platform, not another OS.

## Current self-contained delivery — 2026-11-16

The default notebook now uses tracked waveforms/data/; no bundle/download is
required. Full archived data remain optional for the older audit-only tests.

- Full suite with optional audit inputs: **120passed,16.41s**, all102prior cases retained.
  Integrated target repeats **120passed,16.56s**; replay/train/CFO13/13/5figures exact.
- Actual shallow Git clone in a temporary directory, with OS-denied reads to original
  worktrees/Documents/conductor roots: **85passed,35optional archive checks skipped**.
- From that clone: **12successful notebooks +1expected CFO-training rejection**,
  **140inline figures**,146.32s driver wall time. Default saved DD and fresh frozen
  DD work at N64/128/256; changed DD/CFO,20step training and reload also execute.
- A second sandbox additionally denies the clone's waveforms/data/: training and
  checkpoint reload still execute,26figures,23.36s; no pretrained dependency.
- Against retained pre-cleanup/full-input runs: **114PNG byte matches,785exact array
  fields**, CSV values exact except timing, training selection/trace unchanged.
- Runtime manifest verifies all3,482,643tracked bytes and original field identities.
  Installed wheel loads all3DD sizes and genuine CFO with repository inputs blocked.
- Rendered reading flow and DD64/DD256/train/CFO plots inspected;13DD/5CFO views
  retained, including complex/grouped spectra and8/N/2overlays. Source output-free.

Exact command arrays, sandbox policies, blocked-read probes, logs, execution/audit
ledgers and equivalence results are retained under ignored local/runtime-handoff/.
These checks qualify distribution and unchanged computation, not model quality,
long BER precision, physical validation or unsupported CFO training.

Clean-clone checks (no archives):
```bash
python -m pytest support/tests -q
python support/scripts/qualify_notebook.py --out local/qualification
python support/scripts/audit_configurable.py --qualification local/qualification
```

## Earlier configurable-workflow qualification

Exact historical commands/results and executed notebooks are retained under ignored local/.
The equivalents below use the project environment created in the root README.
Local command ledger: `local/COMMANDS.md`; release evidence: `local/qualification-release/`.

```bash
python support/scripts/bootstrap.py --bundle local/advisor-n64-artifacts.tar.gz
python -c "from waveforms.scaling_evidence import prepare_scaling; prepare_scaling('.', 'local/scaling-artifacts', 'local/advisor-scaling-artifacts.tar.gz')"
python -m pytest support/tests -q --basetemp local/pytest-release-final --artifacts local/artifacts --scaling-artifacts local/scaling-artifacts
python support/scripts/qualify_notebook.py --out local/qualification-release
python support/scripts/audit_configurable.py --qualification local/qualification-release
```

Notebook qualifier copies the source notebook into local/, changes only its single
configuration cell, executes every cell, saves executed notebooks/inline PNGs and
an execution ledger. It never writes outputs into Waveform_Exploration.ipynb.
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

Ready for local branch/code/notebook review. This qualifies the
configurable workflow, not production BER precision, historical training replay,
physical validation or a Learned performance advantage.
