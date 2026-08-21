# Stage 4A — Structural Generator and Waveform-Family Forensics

## Result: Outcome C

Only functional DD-representation equivalence is established: the successful population is Q-concentrated, but no common compact atom/generator family survives the controls.

No waveform/channel training or BER-based fitting was performed. All fitted candidates were frozen before paired communication evaluation.

## Interpretation gate

1. **Simple canonical generator:** not established as a complete law. Learned G_D is concentrated, but G_S is diffuse; the best compressed native orbit was `single_dd_atom` with mean column overlap only 0.2604.
2. **DD-sparse/local generators:** mixed. Learned top-24 DD energy is 0.0749 for G_S and 0.7005 for G_D; the asymmetric signature is not shared by all six Stage-3B solutions.
3. **Compressed orbit:** not accurate enough to claim an orbit law; best phase-aligned orbit NMSE is 1.4792.
4. **Fourier/chirp/phase family:** best tested aligned fit is `FOURIER_POLY_D3` with NMSE 1.1073 and 7 free real parameters.
5. **Prototype/Gabor family:** evaluated as an exact critical-lattice tight orbit with O(N) prototype storage; its fidelity is reported without dense hidden parameters.
6. **Compact Q fidelity:** no tested candidate met the predeclared eta/retained-energy gates.
7. **Communication fidelity:** no compact candidate met the paired two-modulation, four-SNR gate without BER refitting.
8. **Common Stage-3B family:** no. Low matrix overlap, weak generator-support agreement, and inconsistent generator sparsity rule out a shared tested family despite common Q concentration.
9. **Haar separation:** partial, not universal. Learned G_D and OFDM-start solutions separate strongly from Haar, while Haar-start Q optima retain diffuse generators.
10. **N=64 sufficiency:** insufficient; multi-N and multi-solution evidence is required before proposing a waveform law.

## Best structural fits to authoritative Learned

| Family | real parameters | aligned NMSE | mean squared overlap | eta24 Q | fixed-Q retained |
|---|---:|---:|---:|---:|---:|
| FOURIER_POLY_D3 | 7 | 1.1073 | 0.2136 | 0.990885 | 0.990529 |
| FOURIER_POLY_D2 | 5 | 1.1092 | 0.2135 | 0.990914 | 0.990531 |
| FOURIER_POLY_D1 | 3 | 1.1176 | 0.2088 | 0.990779 | 0.990312 |
| CHIRP_FOURIER | 2 | 1.1193 | 0.2084 | 0.990948 | 0.990557 |
| GABOR_TIGHT_PROTOTYPE | 126 | 1.3601 | 0.1059 | 0.721465 | 0.718688 |
| SHALLOW_2FFT | 7 | 1.5338 | 0.0603 | 0.983025 | 0.982936 |

## Paired communication fidelity at 18 dB

| Waveform | QPSK fixed-Q BER | 16-QAM fixed-Q BER | QPSK oracle BER | 16-QAM oracle BER |
|---|---:|---:|---:|---:|
| Learned | 0.00117493 | 0.0250053 | 0.00110722 | 0.0243654 |
| FIT_FOURIER_POLY_D3 | 0.00547981 | 0.0453329 | 0.00514126 | 0.0423241 |
| FIT_FOURIER_POLY_D2 | 0.00553799 | 0.0457735 | 0.0050869 | 0.0424242 |
| FIT_FOURIER_POLY_D1 | 0.00539589 | 0.046175 | 0.00501537 | 0.0430164 |
| FIT_CHIRP_FOURIER | 0.00539207 | 0.0457096 | 0.00504208 | 0.0427189 |
| FIT_GABOR_TIGHT_PROTOTYPE | 0.151214 | 0.277888 | 0.0837641 | 0.211531 |
| FIT_SHALLOW_2FFT | 0.0065136 | 0.0621319 | 0.00369358 | 0.0513401 |

## Same matrix, same family, and same function

- **Same matrix:** no. Learned-to-Stage-3B aligned squared overlap spans 0.1444-0.1894.
- **Same mathematical family:** not established. Best-fit family counts across the six Stage-3B solutions are {'FOURIER_POLY_D3': 6}; successful-pair G_S/G_D top-24 Jaccard spans 0.000-0.067 / 0.067-0.231.
- **Same functional DD representation:** partially yes. Successful-pair Q top-24 row overlap spans 0.376-0.761, independently of low atom overlap.
- Shared-support generator phase coherence spans 0.000-1.000 for G_S and 0.190-0.874 for G_D; no universal phase law is asserted.

## Protocol and preservation

- Structural fitting steps: 400; fit objective used matrix/atom structure only.
- Q bank: 2048 Sobol DD atoms, seed 11130; communication bank: 512 independent DD-v1 channels, seed 11131.
- Each communication point targeted at least 1,000,000 paired bits with Full, Oracle-K24, and frozen-Q-K24 receivers.
- Authoritative Learned SHA-256: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`; all six Stage-3B hashes verified.
- All 250 prior artifacts remained byte-identical; Stage 2E absent and IDENTITY excluded from figures.
- Fitted matrices are isolated under `outputs/stage4a/fitted/`; no source checkpoint was overwritten.

## Conclusion

Only functional DD-representation equivalence is established: the successful population is Q-concentrated, but no common compact atom/generator family survives the controls.

**STOP.**
