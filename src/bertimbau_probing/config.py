"""Experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters and paths for the probing experiments."""

    data_path: str = "data/B2W-Reviews01.csv"
    text_column: str = "review_text"
    model_name: str = "neuralmind/bert-base-portuguese-cased"

    samples: int = 10_000          # rows to subsample (0 = use all)
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-5
    num_classes: int = 3

    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.25         # fraction of the post-test split

    artifacts_dir: str = "artifacts"
