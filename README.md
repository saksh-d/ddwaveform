# Waveform Exploration

Explore learned delay–Doppler waveforms and residual CFO in one research notebook.
**Pretrained inputs are included** — no separate bundle or download.

Use **64-bit Python 3.10**. Tested on macOS / Apple Silicon with MPS and CPU;
Linux/Windows instructions are provided but **not tested here**. Install Python 3.10
first if the command below is unavailable. Choose the **waveform-exploration**
branch when cloning or downloading ZIP:

```bash
git clone --branch waveform-exploration --single-branch https://github.com/saksh-d/ddwaveform.git
cd ddwaveform
```

For an existing download, open its repository root. Then create the environment:

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```
On Windows, use `.\.venv\Scripts\python.exe` wherever `python` appears below.
Choose either notebook frontend:

- **VS Code:** install the **Python** and **Jupyter** extensions, open this repository
  folder, open the notebook, then **Select Kernel → Python Environments → .venv**.
- **JupyterLab:** optionally install the frontend and register this environment:

  ```bash
  python -m pip install jupyterlab -c requirements.lock.txt
  python -m ipykernel install --user --name waveforms --display-name "Waveform Exploration"
  python -m jupyter lab Waveform_Exploration.ipynb
  ```

  Choose **Kernel → Change Kernel → Waveform Exploration**.

In **[Waveform_Exploration.ipynb](Waveform_Exploration.ipynb)**, edit the single
configuration cell, **Restart Kernel**, then **Run All**. Figures appear inline;
complete results and metadata are saved under ignored `local/`.

| Choose | What runs |
|---|---|
| Default: `load_existing` + `saved_replay` | Exact archived DD basis and saved BER, N64/128/256 |
| `load_existing` + `fresh` | New channels and BER for the frozen DD basis, or supported CFO |
| `train_new` + `fresh` | New DD basis, held-out selection and fresh tests; no input assets needed |

**Default Run All needs no accelerator or external input:** it replays saved DD
results using CPU float64/complex128 reference arithmetic. Fresh evaluation and
DD training use FP32: keep `BACKEND="mps"` on an MPS-capable Mac, or explicitly
set `BACKEND="cpu"`. No automatic device/precision fallback; CUDA is unsupported.
The run summary records the actual device/dtype. Seeded draws reproduce within a backend;
CPU and MPS random streams/results need not match. Large CPU runs may be slower.

One Learned per N/run; each N has its own W. Changed channels/supports require
fresh BER. Training fits a surrogate; bounded trials make no performance guarantee.
**CFO training is unsupported**; CFO retains its distinct rectangular W/R.

```text
Waveform_Exploration.ipynb   ← start here
README.md
HISTORY.md
pyproject.toml
requirements.lock.txt
waveforms/                 # implementation + 3.48 MB verified data/
support/                   # tests, utilities, technical notes, preserved history
```

The six tracked files in `waveforms/data/` supply all public pretrained runs.
[Input provenance](support/docs/ARTIFACTS.md) · [Scientific controls](support/docs/CONFIGURABLE.md).
Optional full audit archives, generated results/checkpoints, executed notebooks,
caches and build products are **gitignored**. The source notebook stays output-free.
