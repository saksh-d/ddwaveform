> Current notebook extension: [N64/128/256 own bases and spectral/coupling views](COUPLING.md).
> Current replay uses the completed scaling-release protocol; earlier N64 evidence below remains archival.

# Provenance and scope

Base historical repository: `https://github.com/saksh-d/ddwaveform`, commit
`9aae23ed6f9ac6e2a2e3e057f38693a148886400`. This worktree verified clean on
`advisor-reproducibility` at that commit before changes. Main stayed
`a12d24f07c03a1319ee5fa8773efcf3e84086747`. No default-branch changes, external
sharing, other-thread messages, interruption, push or PR.

Read-only sources on this Mac:

- Historical checkout `/Users/saksh/Documents/Local Documents/IBM/delayDopplerWaveform`;
  pinned code inspected from this worktree's base. Original checkout not reset.
- Standalone `/Users/saksh/Documents/Local Documents/IBM/waveform-mechanism-study`;
  inspected actual README/PROTOCOL/PROVENANCE/REPORT/HISTORY, relevant modules/tests,
  focused saved inputs/counts,25dB exports, bridge and completed main status.
- `/Users/saksh/Documents/Local Documents/IBM/ofdm/residualCFO`;
  bounded stage1_fullframe/stage2_awgn/stage3_multipath source/artifact inspection.
  Read stage2_awgn/AGENTS.md. **TECHNICAL.md never read.** No source writes/runs.
- Research record README and `projects/delay-doppler-research.md` consulted as
  evidence pointers; relevant claims checked against supplied code/reports.

| Package definition | Authoritative source | Fidelity |
|---|---|---|
| `delay_doppler.dft/atoms/assemble/sample` | pinned `src/channel.py`, `src/config.py`; WMS `channel.py` | Exact signs, delay-first, unshifted fractional-delay indexing, per-realization normalization and iid path law. WMS namespace RNG preserved. |
| `delay_doppler.classical` | pinned `src/waveforms.py`, `src/config.py`; WMS `bases.py` | Exact OTFS ordering and historical calibrated AFDM. No Haar population in core. |
| `receivers.support/receiver/stream_mse` | pinned receiver/learning; WMS `receivers.py` | Stable original-index ties; realized coefficients; full physical observations and exact mismatch error. |
| QPSK/16QAM encode/decode | pinned `src/metrics.py`; WMS `modulation.py` | Historical normalized Gray points and raw nearest slicing. |
| Complex spectral bridge | WMS `orthogonality_bridge.py` | Same identity; independent finite-sum frequency kernel instead of sinc ratio. Small tests independently verify signs/order. |
| Core figure renderer | WMS `focused.py/focused_plots.py`, saved data | New compact layout of saved inputs; same counts, selected columns and example. No new BER during plotting. Original detailed plots remain in source directory. |
| Supplementary index/renderer | WMS completed main/bridge/block_frequency | Saved evidence, corrected grouping protocol, all3trials/directions. Archived sources included for inspection, no runtime dependency. |
| CFO operator/load qualification | residualCFO stage1/2/3 `channel.py`; stage2 `comm_core.py`, transmitter/receiver | Positive sign; complex64, separate W/R, payload-hash algorithm and clean metrics reproduced. Historical operator output is A transpose. |

## Verified artifacts

Historical DD `reference/artifacts/learned_w.npy` SHA256:
`11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`.
N64 complex128, finite, columns synthesize x=Ws, no transpose/gauge/permutation
on load. Unitarity Frobenius2.23303e-14. Original
`tail_from_ofdm_best.pt` SHA256
`31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`
**not recovered**. Array export is available, optimizer/training serialization
not recovered. WMS provenance traces export to the named phoenix prior-evidence
tree; this package uses the supplied verified WMS snapshot, not an open-ended
artifact search or invented replacement training.

Genuine CFO checkpoint:
`stage2_awgn/stage1_linear_outputs/16qam_n45_r9/stage1_checkpoint.pt`.
File SHA256 `8f0bf24d6b593d2c186a50b03c10a8bf1c781514c3add52c5ce9ca91859afb07`;
payload SHA256 `380da00b389b047699c53cab6f1f80c28675dfc8b4835a70e54f7be1325ed64c`.
Stored v2 acceptance passed. W64x45, R45x64, complex64; clean identity
7.8946375e-4, off-diagonal fraction3.7322321e-4 reproduced. All41stored
off-diagonal operator diagnostics checked. Checkpoint loaded with restricted
Torch loader and explicit NumPy/Path config allowlist; source code not imported.
This qualifies the stored AWGN+CFO operator example, **not original training or
BER regeneration**. Saved CFO metrics/report retained with their historical
Eb/N0 axis. Stage3 checkpoint files exist in the bounded search, but their full
model/provenance was not qualified; no learned multipath reproduction claimed.
No serialized stage1_fullframe weights found in the bounded file-pattern scan.

## Completed and active study distinction

Read-only thread inspection:

- [Completed focused/bridge thread](https://app.t3.codes/threads/039b7173-6310-4cf2-a867-e89b75030401/91fd05e1-9cd4-4788-b9c9-ef96c524670f).
- [Active larger-N study](https://app.t3.codes/threads/039b7173-6310-4cf2-a867-e89b75030401/b969399d-3ecb-42ff-b8fb-f0edae03151e).

The latter was extending N128/N256 sampling at snapshot time. Those files and
active scaling source/test modules excluded. Every included source file checked
for size/mtime/hash stability before/after copy; content hashes retained. No
messages sent, jobs changed, or waiting for optional results.

## Retired branch tree and retained scientific negatives

123original tracked files replaced by a compact package. Every retired file,
including historical HISTORY/research plan/stage reports/notebooks/tests/scripts,
remains in the pinned Git tree. Example:

```bash
git ls-tree -r --name-only 9aae23ed6f9ac6e2a2e3e057f38693a148886400
git show 9aae23ed6f9ac6e2a2e3e057f38693a148886400:reports/stage3d_summary.md
```

No unique generated evidence existed in this initially clean worktree. Original
source directories/worktrees and their ignored artifacts unchanged. Old pipeline
not required for imports, tests or artifact provisioning. Passing this package's
focused suite does not certify the retired historical stage suite or recover its
missing checkpoints.

Retained prior findings: Stage2C no simple ceil(N/K) law; Stage3A raw DD coupling
law already derived; Stage3B Q concentration improved finite-K communication
without one common grouping; Stage3C modulation-dependent residual geometry;
Stage3D extra geometry objective not consistently improving; Stage5 compact
reconstruction unqualified. Current package starts no continuation of those
campaigns. NOFS official weights/spec remain unavailable in the completed audit;
SEFDM is not relabeled as NOFS.

## Completed scaling extension

Current notebook inputs come from the completed scaling release described in
[COUPLING.md](COUPLING.md), sealed under configs/scaling_artifacts.json. Additional
read-only qualification verified each larger-N source tail/best.pt SHA against
selected.json and exact array equality to learned_N128/N256.npy. Source checkpoint
hashes are39beb604fdf82ad1d63da1a76c960c4e62306036ec3e43681849d81ef4081d64
(N128) and8791397ea52234ecab5eb8eff2fc860243d661d293dfdaec4d5440e930c2b0ff
(N256). These fresh checkpoints are distinct from the still-unrecovered historical
N64 checkpoint. No optimizer state is required/copied for this inference package.
