"""Naive regression baselines for vowel density.

These check whether the fine-tuned model is actually learning something, rather
than the target being trivially predictable from a heuristic.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .vowels import first_word_density, last_word_density, vowel_density


def global_density(texts: Sequence[str]) -> float:
    """Vowel density of the whole (train) corpus concatenated together."""
    return vowel_density(" ".join(texts))


def baseline_predictions(
    train_texts: Sequence[str], test_texts: Sequence[str]
) -> dict[str, np.ndarray]:
    """Predictions on ``test_texts`` for each baseline.

    The ``global`` baseline is fit on ``train_texts`` to avoid leakage.
    """
    g = global_density(train_texts)
    return {
        "global": np.full(len(test_texts), g, dtype=float),
        "first_word": np.array([first_word_density(t) for t in test_texts], dtype=float),
        "last_word": np.array([last_word_density(t) for t in test_texts], dtype=float),
    }
