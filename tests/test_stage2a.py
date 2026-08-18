"""Focused checks for frozen Stage 2A qualification utilities."""

import numpy as np
import torch

from src import config
from src.channel import sample_dd_v1
from src.stage2a import (
    aggregate_rows,
    learned_structure,
    qualification_waveforms,
    representative_channel_index,
)
from src.waveforms import random_unitary


def test_qualification_names_and_frozen_tensor_are_not_mutated():
    learned = random_unitary(8, 15000)
    before = learned.clone()
    waveforms = qualification_waveforms(random_unitary(config.N, 15001))
    assert tuple(waveforms) == ("IDENTITY", "OFDM", "OTFS", "AFDM", "Learned")
    assert torch.equal(learned, before)


def test_representative_channel_rule_uses_identity_median_only():
    channels = sample_dd_v1(9, 15002, n=8).matrices
    index, values = representative_channel_index(channels, k=4)
    assert index == int(np.argmin(np.abs(values - np.median(values))))


def test_communication_aggregation_uses_error_counts():
    rows = [
        {"modulation": "QPSK", "snr_db": 18, "waveform": "Learned", "k": 24,
         "bit_errors": 1, "total_bits": 10, "symbol_errors": 1, "total_symbols": 5},
        {"modulation": "QPSK", "snr_db": 18, "waveform": "Learned", "k": 24,
         "bit_errors": 2, "total_bits": 30, "symbol_errors": 2, "total_symbols": 15},
    ]
    result = aggregate_rows(rows, ("modulation", "snr_db", "waveform", "k"))[0]
    assert result["ber"] == 3 / 40
    assert result["ser"] == 3 / 20


def test_structure_diagnostics_are_finite_and_detect_fourier_modulation():
    waveform = torch.fft.fft(torch.eye(8, dtype=torch.complex128), dim=0, norm="ortho")
    rows, summary, arrays = learned_structure(waveform)
    assert len(rows) == 8
    assert arrays["magnitude"].shape == (8, 8)
    assert summary["modulation_similarity_p95"] > 1 - 1e-12
    assert all(np.isfinite(float(value)) for value in summary.values())
