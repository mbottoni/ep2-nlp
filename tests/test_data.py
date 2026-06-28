import numpy as np
import pandas as pd

from bertimbau_probing import data
from bertimbau_probing.baselines import baseline_predictions, global_density


def _raw():
    return pd.DataFrame(
        {
            "review_text": ["ótima qualidade", "produto ruim", None, "  ", "entrega"],
            "other": [1, 2, 3, 4, 5],
        }
    )


def test_preprocess_drops_blank_and_nan():
    out = data.preprocess(_raw())
    assert list(out.columns) == ["text"]
    assert len(out) == 3
    assert "  " not in out["text"].tolist()


def test_add_targets_columns_and_ranges():
    out = data.add_targets(data.preprocess(_raw()))
    assert {"text", "density", "label"} <= set(out.columns)
    assert ((out["density"] >= 0) & (out["density"] <= 1)).all()
    assert set(out["label"].unique()) <= {0, 1, 2}


def test_balance_labels_equalizes_classes():
    df = pd.DataFrame({"text": list("abcdefghij")})
    df["label"] = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    balanced = data.balance_labels(df, seed=0)
    counts = balanced["label"].value_counts()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 2  # smallest class had 2


def test_split_sizes_sum_to_total():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)]})
    df["density"] = 0.5
    df["label"] = [i % 3 for i in range(100)]
    train, val, test = data.split(df, test_size=0.2, val_size=0.25, seed=42)
    assert len(train) + len(val) + len(test) == 100
    assert len(test) == 20
    assert len(val) == 20  # 25% of the remaining 80


def test_baseline_predictions_shapes_and_global():
    train = ["casa", "aeiou"]
    test = ["casa xyz", "bb"]
    preds = baseline_predictions(train, test)
    assert set(preds) == {"global", "first_word", "last_word"}
    for arr in preds.values():
        assert isinstance(arr, np.ndarray) and len(arr) == 2
    assert np.allclose(preds["global"], global_density(train))
    assert preds["first_word"][0] == 0.5  # "casa"
