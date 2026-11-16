# Internal selected-reference studies (preserved)

Exactly one selected N64 matrix, labeled **Learned**, throughout. Same OTFS and
AFDM baselines as the core figures. Selection rationale and post-hoc limitations
in [selection rationale](../docs/SELECTION.md). No training/intervention candidate comparisons
in the advisor command or notebook. All working data remain unchanged.

| Output | Saved evidence | Interpretation |
|---|---|---|
| Core complex bridge | `focused/data/{bases,channels}.npz`, `supplementary/orthogonality_bridge` | W^H W, W^H H W and W^H H^H H W are different. Complex C=FW and H_f=FHF^H reconstruct A; spectral powers alone do not. Same frozen Learned and baselines. |
| `frequency_groups.png` | `focused/data/bases.npz`, frozen labels/permutation; `supplementary/block_frequency` | Learned's approximate/leaky spectral groups, with baseline DFT-coefficient heatmaps. Group labels/order apply only to Learned, not baseline streams. No analytical uniqueness or hard frequency-band claim. |
| `structural_transfer.png` | exact selected reference rows in `supplementary/main/impairment_transfer.csv` | Same Learned under identity, delay, Doppler, one DD path and normalized multipath. OTFS/AFDM kept. Display strength1; CSV retains all .25/.5/1 values. Structural statistics only, not transfer BER or genuine CFO W/R. |

`python -m waveforms.cli internal-supplementary --artifacts local/artifacts --out
local/supplementary_figures` writes only these two supplementary PNGs, a filtered
CSV and an internal audience/selection/input-hash manifest. It rejects missing selected
reference data rather than relabeling another candidate.

Full-range core BER retains unfavorable outcomes and high-SNR reversals. No
checkpoint changes across modulation, SNR or receiver. No new campaign.

The [internal study index](STUDIES.md), renderer and original notebook
preserve complete training trajectories, all intervention directions and all
candidate transfer/comparison tables. Internal IDs remain there, with original
counts, uncertainties, failures and provenance. None of that data was deleted.
