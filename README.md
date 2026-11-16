# Advisor notebook — start here

```bash
conda activate pytorch
# One-time dependency setup, if needed:
python -m pip install -r requirements.lock.txt
```

1. Open **[notebooks/results.ipynb](notebooks/results.ipynb)** in Jupyter; select the `pytorch` kernel.
2. Edit its **one CONFIGURATION cell**, including `N` (64/128/256) and the supplied local scaling-bundle path. Inputs are verified/provisioned automatically.
3. Clear old outputs, restart the kernel, and run cells **in order from the top**.

The same downstream cells show the essential plots **inline**: H/effective A,
time/frequency bases, and QPSK/16QAM BER for one **Learned** with OTFS/AFDM.
Grouped member energy/magnitude and coupling views link the frozen groups to
complex spectra and received basis energy. `EXAMPLE_COLUMNS` selects individually labeled bases, matched to
their coupling columns. `PLOT_VARIANTS` enables review overlays of 8 or N/2 bases;
indices are editable in the same cell. No output-folder steps.

**Basis/training controls:** `MODE="load_existing"` works. `"train_new"` is
**unavailable pending the separate follow-on implementation** and fails clearly.
Future objective/initialization/training-seed/step changes require a newly trained
basis and its own support; those controls are not implemented here. Each available N uses its own verified basis/support; unsupported sizes fail.

**Frozen-basis evaluation controls:** channel ranges/path count, evaluation seed,
BER sample/block budgets, SNR grid and display settings do not retrain W.
Use `EVIDENCE_MODE="fresh"` for bounded new evaluation; `"saved_replay"` loads
completed BER evidence and rejects incompatible model changes. Optimizing W for
a changed channel law requires future training; evaluating frozen W does not.

[Size provenance and coupling notes](docs/COUPLING.md). Saved replay uses the completed
scaling release; timings below concern the earlier reference workloads.

**Reference runtimes** — M3 Pro/18GB, CPU float64/complex128, one thread, N64/P3:

| Workload | Recorded timing / planning estimate |
|---|---|
| Full BER: 27 SNR points, 3 bases, 2 receivers/2 slicers, 2.1–20M bits/arm/point | 7.1 min **measured**; 42 min worst-case **extrapolated** incl. reserve; 3 h cap |
| Full reference training recipe: 6,000 steps/trial | 4.5–5.0 min/trial **measured**; 4.6 min/trial **projected**; 2 h cap covered the entire historical mixed study |

These are historical workloads, **not smoke timings or a forecast for the
unimplemented trainer**. [Runtime assumptions](docs/RUNTIMES.md).
[Usage/details](docs/NOTEBOOK_GUIDE.md) · [provenance](docs/PROVENANCE.md) · [internal research](internal/README.md).
