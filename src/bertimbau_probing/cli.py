"""Command-line entry points for the probing experiments.

    bertimbau-probing baselines        # naive baselines, no torch required
    bertimbau-probing regression       # fine-tune the regression head
    bertimbau-probing classification   # fine-tune the classifier (--balanced)

torch/transformers are imported lazily so the ``baselines`` command runs in a
lightweight environment.
"""
from __future__ import annotations

import argparse
import json

from . import data
from .baselines import baseline_predictions
from .config import Config
from .metrics import classification_metrics, regression_metrics


def _prepare(cfg: Config, *, balanced: bool, stratify: bool):
    df = data.load_reviews(cfg.data_path, cfg.text_column)
    df = data.subsample(df, cfg.samples, cfg.seed)
    df = data.add_targets(df)
    if balanced:
        df = data.balance_labels(df, cfg.seed)
    return data.split(df, cfg.test_size, cfg.val_size, cfg.seed, stratify=stratify)


def _cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    for field in ("data_path", "model_name", "samples", "epochs", "batch_size"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    return cfg


def cmd_baselines(args: argparse.Namespace) -> None:
    cfg = _cfg_from_args(args)
    train_df, _, test_df = _prepare(cfg, balanced=False, stratify=False)
    preds = baseline_predictions(train_df["text"].tolist(), test_df["text"].tolist())
    y_true = test_df["density"].to_numpy()
    report = {name: regression_metrics(y_true, p) for name, p in preds.items()}
    print(json.dumps(report, indent=2))


def cmd_regression(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415

    from .datasets import build_loader  # noqa: PLC0415  (lazy: torch)
    from .train import get_device, predict_regression, train_regression  # noqa: PLC0415

    cfg = _cfg_from_args(args)
    train_df, val_df, test_df = _prepare(cfg, balanced=False, stratify=False)
    device = get_device()
    model, tokenizer = train_regression(train_df, val_df, cfg, device)
    test_loader = build_loader(
        test_df["text"], test_df["density"], tokenizer,
        cfg.batch_size, False, cfg.max_length, torch.float,
    )
    y_true, y_pred = predict_regression(model, test_loader, device)
    print(json.dumps({"model": regression_metrics(y_true, y_pred)}, indent=2))


def cmd_classification(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415

    from .datasets import build_loader  # noqa: PLC0415  (lazy: torch)
    from .train import get_device, predict_classification, train_classification  # noqa: PLC0415

    cfg = _cfg_from_args(args)
    train_df, val_df, test_df = _prepare(cfg, balanced=args.balanced, stratify=True)
    device = get_device()
    model, tokenizer = train_classification(train_df, val_df, cfg, device)
    test_loader = build_loader(
        test_df["text"], test_df["label"], tokenizer,
        cfg.batch_size, False, cfg.max_length, torch.long,
    )
    y_true, y_pred = predict_classification(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred, cfg.num_classes)
    print(json.dumps({"model": metrics}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bertimbau-probing", description=__doc__)

    def add_common(sub):
        sub.add_argument("--data-path", help="path to B2W-Reviews01.csv")
        sub.add_argument("--model-name", help="HuggingFace model id")
        sub.add_argument("--samples", type=int, help="rows to subsample (0 = all)")
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--batch-size", type=int)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baselines", help="evaluate naive density baselines")
    add_common(p)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("regression", help="fine-tune the regression probe")
    add_common(p)
    p.set_defaults(func=cmd_regression)

    p = sub.add_parser("classification", help="fine-tune the classification probe")
    add_common(p)
    p.add_argument("--balanced", action="store_true", help="down-sample to balance classes")
    p.set_defaults(func=cmd_classification)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
