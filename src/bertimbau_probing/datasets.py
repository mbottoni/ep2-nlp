"""Torch datasets and dataloaders (requires torch + transformers)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


def load_tokenizer(model_name: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_name)


class ReviewDataset(Dataset):
    """Tokenize review text on the fly, paired with a numeric target."""

    def __init__(self, texts, targets, tokenizer, max_length=512, target_dtype=torch.float):
        self.texts = list(texts)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        target = torch.tensor(self.targets[idx], dtype=self.target_dtype)
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), target


def build_loader(
    texts: Sequence[str],
    targets: Sequence[float],
    tokenizer,
    batch_size: int = 16,
    shuffle: bool = False,
    max_length: int = 512,
    target_dtype=torch.float,
) -> DataLoader:
    dataset = ReviewDataset(texts, targets, tokenizer, max_length, target_dtype)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
