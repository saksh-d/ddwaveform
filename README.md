# Waveform Exploration

Explore learned delay–Doppler waveforms and residual CFO in one research notebook.
**Pretrained inputs are included** — no separate bundle or download.

```bash
python -m pip install -r requirements.lock.txt
# Optional notebook frontend: python -m pip install jupyterlab
```

Open **[Waveform_Exploration.ipynb](Waveform_Exploration.ipynb)**, edit the single
configuration cell, restart, **Run All**. Figures appear inline; complete results
and metadata are saved under ignored `local/`.

| Choose | What runs |
|---|---|
| Default: `load_existing` + `saved_replay` | Exact archived DD basis and saved BER, N64/128/256 |
| `load_existing` + `fresh` | New channels and BER for the frozen DD basis, or supported CFO |
| `train_new` + `fresh` | New DD basis, held-out selection and fresh tests; no input assets needed |

Python 3.10 / Apple Silicon is the qualified environment. Fresh evaluation and
training require **MPS/FP32**, without CPU fallback; saved replay keeps reference
arithmetic. One Learned per N/run, each N has its own W. Changed channels/supports
require fresh BER. Training fits a surrogate; bounded trials make no performance
guarantee. **CFO training is unsupported**; CFO retains its distinct rectangular W/R.

```text
Waveform_Exploration.ipynb   ← start here
README.md
HISTORY.md
pyproject.toml
requirements.lock.txt
waveforms/                 # implementation + 3.48 MB verified data/
support/                   # tests, utilities, technical notes, preserved history
```

The six small tracked files in `waveforms/data/` supply all public pretrained runs.
[Input provenance and exact contents](support/docs/ARTIFACTS.md) ·
[Scientific controls and limits](support/docs/CONFIGURABLE.md).
Full historical audit archives are optional. They, generated figures/arrays,
checkpoints, executed notebooks, caches and build products are **gitignored**.
The source notebook stays output-free.
