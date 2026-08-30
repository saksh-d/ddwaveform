# Pub-1C Phase-1 publication completion

## Classification: SUPPORTED

Frozen claim tested: under fractional/off-grid DD channels, the authoritative Learned waveform remains more recoverable when every waveform receives its own oracle strongest interaction fraction. This is regime-specific, not a universal waveform or compute claim.

## Controlled result

- Joint epsilon=0.5, 16-QAM, rho=0.375 BER: Learned 0.0214958, AFDM 0.113434, OTFS 0.13406.
- Corresponding retained energy: Learned 0.998337, AFDM 0.952848, OTFS 0.929165.
- Paired 95% channel-bootstrap BER differences (Learned minus baseline): Learned-AFDM [-0.0969506, -0.0869446]; Learned-OTFS [-0.11871, -0.106449].
- Paired 95% retained-energy differences: Learned-AFDM [0.0421191, 0.0487467]; Learned-OTFS [0.064103, 0.0740332].
- On-grid 16-QAM BER: Learned 0.0246277, AFDM 0.0210495, OTFS 0.0202522; classical behavior is reported rather than suppressed.
- Budget curves include rho=1 and must be interpreted jointly with retained energy; no fixed K=24 scaling claim is made.

## Representative 15 dB constellation geometry

- P7 compares independently selected physical-median GRID-DD, FRACTIONAL-DOPPLER, and DD-v1 channels; selection uses waveform-invariant full-information MSE only.
- GRID-DD raw EVM/BER: Learned 0.296/0.0575, OTFS 0.288/0.05545, AFDM 0.286/0.0535.
- FRACTIONAL-DOPPLER raw EVM/BER: Learned 0.300/0.0589, OTFS 0.286/0.05545, AFDM 0.334/0.07706.
- DD-v1 OFF-GRID raw EVM/BER: Learned 0.284/0.05005, OTFS 0.363/0.09314, AFDM 0.380/0.09918.
- These single-channel panels diagnose cluster geometry; the controlled multi-channel P1 sweep remains the population-level causal evidence.

## Structural interpretation

- The same authoritative full map at K=24 yields 3 spectral communities, modularity 0.6162, and within-community energy 0.9646.
- Counts change across thresholds; the figures test rather than assume an exact three-group law.
- Separately trained K=16/24/32 tensors are compared only through full untruncated maps. Their detected group counts are K=16: 6, K=24: 5, K=32: 3.

## Method boundaries

- Each waveform and realization uses its own row-top-K support. Learned support is never imposed on AFDM or OTFS.
- Raw 16-QAM outputs use identical unit-power symbols/noise and no phase, gain, or constellation alignment.
- Learned training steps: zero. Fixed-Q remains secondary. Phase 2 physical-front-end and imperfect-CSI work was not specified or executed.
