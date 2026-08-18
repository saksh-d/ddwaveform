"""Focused deterministic Stage 2B forensic checks."""

import numpy as np
import torch

from src import config
from src.stage2b import (
    adjusted_rand_index,
    afdm_path_audit,
    dense_atom_spectra,
    ensemble_operator_statistics,
    positive_shift_coordinates,
    representative_atom_indices,
    spectral_communities,
)
from src.waveforms import afdm_waveform, identity_waveform, ofdm_waveform


def test_reference_orientation_positive_shift_is_below_diagonal_with_top_origin():
    rows, columns = positive_shift_coordinates(8, 1)
    assert (rows[0], columns[0]) == (1, 0)
    assert (rows[-1], columns[-1]) == (0, 7)


def test_afdm_tuned_integer_path_centers_match_established_law():
    waveform = afdm_waveform(c1=config.AFDM_K24_TUNED_MULTIPLIER / (2 * config.N))
    rows = afdm_path_audit(waveform, ((0, 0), (2, 0), (2, 1), (4, -1)), config.AFDM_C1)
    assert all(row["matches"] for row in rows)
    assert min(float(row["minimum_row_peak"]) for row in rows) > 1 - 1e-12


def test_representative_atoms_and_dense_spectra_are_deterministic():
    waveform = ofdm_waveform()
    first, _ = representative_atom_indices(waveform)
    second, _ = representative_atom_indices(waveform)
    frequencies, spectra = dense_atom_spectra(waveform, first, fft_size=1024)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 5
    assert frequencies.shape == (1024,)
    assert spectra.shape == (1024, 5)
    assert np.allclose(spectra.max(axis=0), 0.0)


def test_ensemble_energy_and_support_normalization():
    channels = torch.stack((torch.eye(8, dtype=torch.complex128), torch.eye(8, dtype=torch.complex128)))
    waveforms = {"I": identity_waveform(8)}
    # Utility uses the frozen N=64 accumulator, so exercise the production dimension.
    channels64 = torch.stack((torch.eye(64, dtype=torch.complex128), torch.eye(64, dtype=torch.complex128)))
    energy, support = ensemble_operator_statistics(channels64, {"I": identity_waveform()}, k=24)
    assert np.isclose(energy["I"].sum(), 64.0)
    assert np.all((support["I"] >= 0) & (support["I"] <= 1))
    assert np.allclose(support["I"].sum(axis=1), 1.0)


def test_spectral_community_and_adjusted_rand_sanity():
    similarity = np.block([[np.ones((3, 3)), np.zeros((3, 2))], [np.zeros((2, 3)), np.ones((2, 2))]])
    labels, audit = spectral_communities(similarity, maximum_groups=3)
    assert audit["group_count"] == 2
    assert len(np.unique(labels)) == 2
    assert adjusted_rand_index(labels, labels) == 1.0
