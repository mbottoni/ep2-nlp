"""Dataset loading, target construction, splitting and balancing.

Depends only on pandas + scikit-learn (no torch), so it is import-light and
unit-testable without a deep-learning environment.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .vowels import density_band, vowel_density


def load_reviews(path: str, text_column: str = "review_text") -> pd.DataFrame:
    """Read the B2W CSV and reduce it to a clean single ``text`` column."""
    return preprocess(pd.read_csv(path, low_memory=False), text_column)


def preprocess(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Keep the review text, rename it to ``text``, drop blanks/NaNs."""
    out = df[[text_column]].rename(columns={text_column: "text"}).dropna()
    out = out[out["text"].str.strip().astype(bool)]
    return out.reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``density`` (regression) and ``label`` (3-band) target columns."""
    out = df.copy()
    out["density"] = out["text"].map(vowel_density)
    out["label"] = out["density"].map(density_band)
    return out


def subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Randomly subsample ``n`` rows (``n <= 0`` or ``n >= len`` keeps all)."""
    if n and 0 < n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def balance_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Down-sample every class to the size of the smallest one."""
    n = int(df["label"].value_counts().min())
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test frames.

    When ``stratify`` is set the split is stratified on the ``label`` column
    (useful for the classification task).
    """
    strat = df["label"] if stratify else None
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    strat = train["label"] if stratify else None
    train, val = train_test_split(train, test_size=val_size, random_state=seed, stratify=strat)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
