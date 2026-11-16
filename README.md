# Waveform Exploration

A research notebook for configurable delay–Doppler waveforms and a distinct
residual-CFO model. Inspect complex bases and channel coupling, then evaluate BER.

## Start

```bash
python -m pip install -r requirements.lock.txt
# Optional notebook frontend: python -m pip install jupyterlab
```

Open **[Waveform_Exploration.ipynb](Waveform_Exploration.ipynb)** in your notebook
frontend. Edit the **one grouped configuration cell**, restart, and **Run All**.
Numbered sections lead through the run summary, channel, bases, coupling, overlays
and error rates. Full metadata and tables are collapsed; results are archived locally.

## Choose a workflow

| Configuration | What runs | Required local inputs |
|---|---|---|
| DD `load_existing` + `saved_replay` | Exact archived scenario and BER counts | Scaling bundle |
| DD `load_existing` + `fresh` | New evaluation of a frozen basis/checkpoint | Scaling bundle or your checkpoint |
| DD `train_new` + `fresh` | Surrogate fit, held-out selection, independent test | No bundle |
| CFO `load_existing` + `fresh` | Frozen W[64,45], R[45,64], A=RHW | Legacy N64 bundle |

Existing results require the **provided bundles** under ignored `local/`; code alone
does not contain them. [Input provisioning](docs/ARTIFACTS.md). Fresh computation
requires **MPS/FP32**, without CPU fallback; archived replay retains reference arithmetic.

Evaluation controls test W; training controls create W. DD supports N64/128/256 with
own matrices, default K/N=3/8 and one Learned per run. Changed W/channel/support cannot
reuse old BER. Training optimizes a concentration surrogate, not BER; limited trials
and bounded evaluation establish no performance advantage. **CFO training is unsupported.**

## Repository

```text
Waveform_Exploration.ipynb   # Single research entry point; output-free in Git
waveforms/                  # Python models, training, evaluation and presentation
configs/                    # Defaults and sealed input manifests
scripts/                    # Provisioning and reproducibility checks
tests/                      # Numerical and workflow regression tests
docs/                       # Methods, controls, figure definitions and validation
internal/                   # Archived notebooks and historical studies
pyproject.toml              # Package and command-line entry point
requirements.lock.txt       # Reproducible dependency setup
HISTORY.md                  # Compressed development record
local/                      # Ignored bundles, checkpoints and generated results
```

[Controls & methods](docs/CONFIGURABLE.md) · [Figure guide](docs/FIGURE_GUIDE.md) ·
[Validation](docs/VALIDATION_CONFIGURABLE.md)
