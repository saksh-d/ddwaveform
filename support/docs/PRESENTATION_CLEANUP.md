# Research presentation and neutral naming

Presentation-only cleanup on waveform-config, based on local commit
`aca235e627c13319287fd0c2c17ae931b7c7cc81`. No numerical/model/training changes.

## Layout

- One root entry: [Waveform_Exploration.ipynb](../../Waveform_Exploration.ipynb).
  The former notebooks/results.ipynb is moved, with no duplicate entry notebook.
- Python implementation is waveforms/, not an audience-specific package.
  Packaging, imports, CLI, root discovery, scripts, configs and docs use that name.
- Seven numbered sections, separate figure cells, short captions, five-row run
  overview and collapsed full metadata/tables. Long theory moved to FIGURE_GUIDE.md.
- One grouped configuration cell retains every expression/control/default.
  README preserves the user's removal of named conda/kernel assumptions.
- Historical notebooks/studies remain under support/internal/ and in Git history. Sealed
  bundle names, input hashes and RNG namespaces are intentionally unchanged.

## Verification

Before cleanup:97tests pass13.34s. After cleanup:102tests pass11.63s, including the
same97cases plus5presentation contracts. Code-only71pass/31explicit artifact skips.
The renamed package builds/installs as a wheel; all23submodules import from both
the checkout and isolated local wheel installation. CLI help works.

Fresh before/after executions of DD replay, DD20-step training and frozen CFO:

| Case | Figures before / after | Array fields exactly equal | PNG files byte-identical |
|---|---:|---:|---:|
| DD N64 replay |13 /13|42|13|
| DD N64 train + fresh test |13 /13|139|13|
| CFO frozen W/R + fresh test |5 /5|30|5|

All CSV numerical values/counts/intervals match exactly except runtime seconds.
Training steps, selected step, validation score and trace are identical.
AST comparison of all23pre-existing package modules allows only documented naming,
docstring and presentation-label changes;15modules are byte-identical. The complete
configuration AST is unchanged except the root discovery directory name.

Rendered DD notebooks have13images and5collapsed detail panels; CFO5images and3
collapsed panels. None of the detail panels opens by default. The largest main
markdown cell is395characters, versus5363before cleanup. Total main markdown is
2465characters instead of9443. Rendered reading flow and unchanged plots inspected.
Full-resolution HTML/PNG/notebook outputs retained; scaled display-only HTML previews
avoid the25MiB inline viewer limit. Notebook itself has no custom CSS/JavaScript.

Evidence under ignored `local/presentation-cleanup/`: before/after executed notebooks,
source/config snapshot, comparison.json, reading-flow.json, full/code-only test logs,
wheel/import checks, source/input/archive audit and rendered HTML. Exact commands in
COMMANDS.md. Source notebook stays output-free and generated results remain ignored.

Scientific scope is unchanged: surrogate fitting, independent train/validation/test
banks, one Learned, no performance guarantee or training-seed uncertainty claim,
fixed/full distinction, replay/fresh separation and unsupported CFO training. Existing
results need the provided bundles; new DD training remains bundle-independent.

## Minimal root follow-up

Existing manifests, utilities, tests, notes and archives are consolidated under
support/. Root HISTORY.md is a short current log; the full earlier record is retained
byte-exact in support/HISTORY.md. All79previous tracked files are accounted for.
Relocated paths pass102tests; repeated DD replay/train/CFO outputs again match all
31PNG bytes and211array fields exactly. No numerical/training changes.
