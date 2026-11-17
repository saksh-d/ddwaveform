# Shared basis-source contract

[Waveform_Exploration.ipynb](../../Waveform_Exploration.ipynb) uses
`waveforms.entry.start_workflow`. Load and train modes feed the same downstream
plotting/evaluation methods; `waveforms.presentation` only formats their outputs.

DD SelectedBasis supplies its own square W, boolean fixed-K support and selection
provenance. Archived exports retain their verified complex128 representation;
new training exports complex64 from CPU FP32 real pairs. Each N has its own W;
no resizing, transposition, per-SNR selection or borrowed support/groups.

Replay requires exact archived W/support/model identity. Fresh evaluation generates
new evidence without retraining W. Training uses separate fit/validation/test banks,
selects one W with train-derived support, and requires fresh evaluation. Its
concentration surrogate and limited trials imply no BER guarantee.

CFO is a separate rectangular W[64,45]/R[45,64] contract with A=RHW, not a DD
SelectedBasis. CFO training and DD sparse/unitary substitutions remain unsupported.

The legacy replay API and historical notebooks remain recoverable under support/internal/
and Git history. [Full scientific contract](CONFIGURABLE.md).
