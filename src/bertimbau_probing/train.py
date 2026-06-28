"""Training and prediction loops (requires torch + transformers)."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from .config import Config
from .datasets import build_loader, load_tokenizer
from .models import BertForClassification, BertForRegression


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, optimizer, device, *, train: bool) -> float:
    model.train() if train else model.eval()
    total = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, attention_mask, target in tqdm(loader, leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            if train:
                optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, target)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _fit(model, train_loader, val_loader, criterion, cfg, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(cfg.epochs):
        tr = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch {epoch + 1}/{cfg.epochs}  train_loss={tr:.4f}  val_loss={va:.4f}")
    return model


def train_regression(train_df, val_df, cfg: Config, device=None):
    """Fine-tune a regression head to predict vowel density."""
    device = device or get_device()
    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    model = BertForRegression(cfg.model_name).to(device)
    train_loader = build_loader(
        train_df["text"], train_df["density"], tokenizer,
        cfg.batch_size, True, cfg.max_length, torch.float,
    )
    val_loader = build_loader(
        val_df["text"], val_df["density"], tokenizer,
        cfg.batch_size, False, cfg.max_length, torch.float,
    )
    _fit(model, train_loader, val_loader, nn.MSELoss(), cfg, device)
    return model, tokenizer


def train_classification(train_df, val_df, cfg: Config, device=None):
    """Fine-tune a classification head over the density bands."""
    device = device or get_device()
    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    model = BertForClassification(cfg.model_name, cfg.num_classes).to(device)
    train_loader = build_loader(
        train_df["text"], train_df["label"], tokenizer,
        cfg.batch_size, True, cfg.max_length, torch.long,
    )
    val_loader = build_loader(
        val_df["text"], val_df["label"], tokenizer,
        cfg.batch_size, False, cfg.max_length, torch.long,
    )
    _fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), cfg, device)
    return model, tokenizer


@torch.no_grad()
def predict_regression(model, loader, device):
    model.eval()
    preds, trues = [], []
    for input_ids, attention_mask, target in loader:
        output = model(input_ids.to(device), attention_mask.to(device))
        preds.extend(np.atleast_1d(output.cpu().numpy()).ravel())
        trues.extend(np.atleast_1d(target.numpy()).ravel())
    return np.array(trues), np.array(preds)


@torch.no_grad()
def predict_classification(model, loader, device):
    model.eval()
    preds, trues = [], []
    for input_ids, attention_mask, target in loader:
        output = model(input_ids.to(device), attention_mask.to(device))
        preds.extend(output.argmax(dim=1).cpu().numpy())
        trues.extend(target.numpy())
    return np.array(trues), np.array(preds)
