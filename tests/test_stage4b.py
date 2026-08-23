"""Focused checks for Stage 4B population comparisons."""

from pathlib import Path
import re

import numpy as np
import torch

from scripts.stage4b import CONTROLS, LEARNED, TRAIN_SEEDS, comparison_pairs
from src.stage3b import atomic_operators
from src.stage4b import aligned_operator_metrics, matrix_pair_metrics, paired_bootstrap_interval
from src.waveforms import random_unitary


def test_matrix_and_complex_operator_alignment_remove_only_equivalences():
    n = 8
    left = random_unitary(n, 16000)
    permutation = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    phases = torch.exp(1j * torch.arange(n, dtype=torch.float64))
    right = left[:, permutation] * phases[None, :]
    operators = atomic_operators(torch.tensor([0.2, 1.4, 3.1]), torch.tensor([-.4, .1, .6]), n=n)
    transformed = left.mH @ operators @ left
    q = torch.mean(torch.abs(transformed).square(), dim=0).numpy()
    matrix = matrix_pair_metrics(left, right)
    operator = aligned_operator_metrics(left, right, operators, q, q, k=4)
    assert matrix["aligned_matrix_nmse"] < 1e-25
    assert operator["d_b"] < 1e-25
    assert operator["complex_coefficient_coherence"] > 1 - 1e-12


def test_paired_bootstrap_is_deterministic_and_keeps_pairing():
    left = np.asarray([0.0, 0.1, 0.2, 0.3])
    right = left + np.asarray([0.02, -0.01, 0.03, 0.00])
    first = paired_bootstrap_interval(left, right, seed=16001, resamples=500)
    second = paired_bootstrap_interval(left, right, seed=16001, resamples=500)
    assert first == second
    assert abs(first["mean_delta"] - 0.01) < 1e-15


def test_population_comparison_scope_and_training_seeds_are_predeclared():
    pairs = comparison_pairs()
    assert len(TRAIN_SEEDS) == 4 and len(set(TRAIN_SEEDS)) == 4
    assert len([pair for pair in pairs if pair[2] == "Learned-Learned"]) == 10
    assert len([pair for pair in pairs if pair[2] == "Learned-Control"]) == len(LEARNED) * len(CONTROLS)


def test_stage4b_training_path_has_no_ber_or_new_loss_selection():
    source = Path("scripts/stage4b.py").read_text()
    training = source[source.index("def train_members"):source.index("def load_population")]
    assert re.search(r"(^|[^a-z])ber([^a-z]|$)", training.lower()) is None
    assert "q_objective" not in training
    assert "exact_error_geometry" not in training
