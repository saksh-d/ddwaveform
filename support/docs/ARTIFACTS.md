# Inputs: exactly what is needed

An input bundle is one `.tar.gz` file containing **previously learned matrices and
saved numerical evidence**. It is not another Python dependency. The notebook verifies
and unpacks it automatically; no individual-file scavenging or manual extraction.

## Current notebook requirements

| What you run | Minimum external input with the current loader |
|---|---|
| Saved DD replay, N64 / N128 / N256 | `local/advisor-scaling-artifacts.tar.gz` — one archive covers all three sizes |
| Fresh evaluation of an archived DD Learned, any supported N | The same scaling archive; new channels/BER are computed, W stays frozen |
| New DD training + fresh evaluation | **None**; the code creates W, support, groups and new test results |
| Fresh evaluation of your newly trained DD checkpoint | Its `.npz` **and matching `.json`**, matching N/K; no historical archive |
| CFO frozen W[64,45], R[45,64], fresh direct-R evaluation | `local/advisor-n64-artifacts.tar.gz` only; no scaling archive |

If the complete verified `local/scaling-artifacts/` or `local/artifacts/` directory
is already provisioned, its archive is no longer required at runtime. **Both archives
are needed only when you want both archived DD and CFO workflows** (or the full
historical artifact test suite). CFO training and configurable CFO saved replay remain
unsupported. The current DD notebook does not use the legacy focused-N64 data path.

## How to run

1. Install the requirements and open the root notebook.
2. For an existing-result mode, copy the required **provided archive**, intact, into
   the checkout's `local/` folder. The person distributing the saved results supplies
   this file once; Git alone does not contain or download it.
3. Edit the one configuration cell and Run All. Existing verified inputs are checked;
   missing inputs are provisioned locally without modifying the archive.

To run from code alone, select `MODE="train_new"`, `EVIDENCE_MODE="fresh"`, DD.
For your own checkpoint, use `load_existing` + `fresh`, set CHECKPOINT to its local
`.npz`, TRAINING=None, and matching N/K. Fresh work requires MPS/FP32; no CPU fallback.
A newly trained basis does not reproduce the historical Learned or its saved BER.

## Integrity and storage

- Scaling archive:90,150,353bytes;64files,91,830,414uncompressed bytes.
  SHA256 `71a342c505ddfbafefc2d2bc368dc9071ebc1996cfb9414130f62bf0f3d9edd5`.
- Legacy N64/CFO archive:176,447,435bytes;645files,182,919,587uncompressed bytes.
  SHA256 `693bbb6cca27d8b0f83dcebbe33e37ef9e51e14ddb2f47bda01374d97d09e535`.

Tracked manifests in support/configs/ bind every file. Current provisioning verifies
these **complete** snapshots, even when a mode consumes only a subset. Merely copying
a W array out of an archive is not a supported replacement. Historical bundle names
and sealed bytes are retained. Sources in the bundles are inspection-only, never
imported. [Size/basis provenance](COUPLING.md) · [Scientific contract](CONFIGURABLE.md).

All bundles, provisioned inputs, generated arrays/figures/checkpoints, executed
notebooks and caches remain gitignored. The source notebook is output-free.

## Lean distribution recommendation — assessed, not implemented

For a share-and-run repo, ship a **small checked-in, hash-verified runtime dataset**
instead of requiring the full historical evidence archives. Keep those archives as
the audit record outside Git. Do not add the large `.tar.gz` files to Git.

The three DD W arrays total1,376,640bytes, but weights alone cannot replay BER or
reproduce coupling-statistic figures. A proposed all-N subset retaining Learned W,
fixed supports for all baselines, group/native mappings, Q/mean-power statistics,
example paths and the complete saved BER summary contains4,794,361bytes of existing
array/CSV payload, before future metadata/packaging. The genuine CFO checkpoint is
55,141bytes; preserve its distinct W/R and acceptance/provenance checks.

This is an inventory-based estimate, not a qualified new input format. It requires
a narrow loader/manifest and exact replay/evaluation/figure-equivalence tests before
being called self-contained. Current validated loaders still require the inputs in
the table above. No new dataset/archive was added to Git in this cleanup.
