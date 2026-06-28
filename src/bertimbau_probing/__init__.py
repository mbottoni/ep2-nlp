"""Probing BERTimbau for vowel density on B2W product reviews.

Submodules are kept import-light: `vowels`, `data`, `baselines` and `metrics`
have no deep-learning dependencies, so they can be used (and tested) without
torch/transformers installed. The `models`, `datasets` and `train` modules pull
in torch and are only needed to actually fine-tune.
"""

__version__ = "0.1.0"
