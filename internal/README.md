# Internal research workspace

Not the advisor-facing presentation. All original local variants, arrays,
checkpoints, traces, counts, supports and reports remain intact. Nothing was
deleted to simplify the presentation. The original multi-variant renderer is
preserved in `advisor/internal_studies.py`; its output is explicitly internal.

```bash
conda activate pytorch
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m advisor.cli internal-figures --artifacts local/artifacts --out local/internal/diagnostic_figures
python -m advisor.cli internal-supplementary --artifacts local/artifacts --out local/internal/selected_studies
python -m advisor.cli internal-studies --artifacts local/artifacts --out local/internal/studies
python scripts/review_selection.py --artifacts local/artifacts --out local/internal/selection_review
```

Use fresh output directories for reruns. `internal-studies` retains all3trials,
all four training phases, HISTORICAL/T0_tail intervention comparisons and
T0_tail/T0_q structural transfer. These are never exported by the advisor `figures` command.
`internal-figures` preserves the previous nine-figure diagnostics (bridge, gain
slicing and reordered matrices); `internal-supplementary` preserves selected
transfer/frequency-group analyses. The previous `supplementary` CLI name is
replaced by this explicit internal name; the Python module/data are retained. [Original study index](STUDIES.md) lists preserved data.
[Original results notebook](notebooks/variants.ipynb) retains the internal tables
and separate CFO example; run with project-root working directory after install.

The selection review reads only the completed main common-bank counts:
raw fixed-Q24 at18dB, QPSK and16QAM jointly. It reconciles all candidate counts,
retains the full saved SNR/receiver table and records point-estimate dominators.
It fits nothing, runs no simulation and does not turn a post-hoc choice into
independent test evidence. Keep its tables and plots internal.

All local artifact bundles and prior working outputs remain unchanged. The
historical stage tree remains available in the pinned Git tree and original
read-only worktrees, as before this correction; this change removes no further
internal functionality or research artifacts. Only the advisor entry point,
plot/notebook filters and explicit audience metadata change.

The previous selected-reference notebook is preserved in
`internal/notebooks/selected_studies.ipynb`. It includes the structural-transfer
table removed from the minimal advisor walkthrough. All numerical data and prior
figures remain intact. The original nine-figure plotting source is preserved as
`advisor/internal_figures.py`; advisor plots now use only essential outputs.

Separate CFO matrices, without invoking advisor plots:
`python -m advisor.cli export-cfo --artifacts local/artifacts --eps 0 .05 .1
--out local/cfo_export`. This rectangular W/R model remains distinct from N64 DD.

The primary advisor delivery is now the configuration-first
`notebooks/results.ipynb`, which executes its complete replay/fresh workflow and
shows all five figures inline. `internal/notebooks/essentials_saved_view.ipynb`
preserves the former saved-PNG viewer, not the current advisor entry point.
No configurable-training workflow or campaign was implemented here.
