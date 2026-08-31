#!/usr/bin/env python3
"""Run spaGFM fine-tuning from the command line."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch


from spaGFM.stRoamer.training import run_train_ft as run_train_ft_module
from spaGFM.stRoamer.training.run_train_ft import run_ft


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for spaGFM fine-tuning.")
    parser.add_argument(
        "train_file_fold",
        help="Folder containing train.pt and val.pt, or a project-specific fine-tuning split folder.",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Optional model label used in output paths, e.g. 36M.",
    )
    parser.add_argument(
        "model_weight_file",
        nargs="?",
        default=None,
        help="Optional pretrained checkpoint path. Prefer --model-file for multiple checkpoints.",
    )
    parser.add_argument(
        "--model-file",
        action="append",
        default=None,
        help="Pretrained checkpoint path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--train-file-name",
        default="train.pt",
        help="Training graph filename inside train_file_fold (default: train.pt).",
    )
    parser.add_argument(
        "--val-file-name",
        default="val.pt",
        help="Validation graph filename inside train_file_fold (default: val.pt).",
    )
    parser.add_argument(
        "--train-graph",
        default=None,
        help="Explicit training graph .pt path. Overrides train_file_fold/train-file-name.",
    )
    parser.add_argument(
        "--val-graph",
        default=None,
        help="Explicit validation graph .pt path. Overrides train_file_fold/val-file-name.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Run train-only fine-tuning without weighted validation or best_model.pt.",
    )
    parser.add_argument(
        "--save-folder",
        default=None,
        help="Checkpoint output directory. Defaults to <train_file_fold>/ft_checkpoints.",
    )
    parser.add_argument("--frozen-backbone", action="store_true", help="Freeze the backbone encoders.")
    parser.add_argument("--batch-size", type=int, default=80, help="Fine-tuning batch size.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=50,
        help="Log train/validation progress every N batches; use 0 to disable batch logs.",
    )
    parser.add_argument(
        "--save-epoch-wise-size",
        type=int,
        default=1,
        help="Save every N epochs; use 0 to save only best_model.pt when validation is enabled.",
    )
    parser.add_argument(
        "--walk-lengths",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="Walk lengths to fine-tune (default: 1 2 3 4 5 6 7 8).",
    )
    parser.add_argument("--n-total-rw", type=int, default=None, help="Override n_total_rw.")
    parser.add_argument(
        "--task-type",
        choices=("classification", "regression"),
        default="classification",
        help="Fine-tuning task type.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=None,
        help="Decoder output dimension. Defaults to number of unique labels for classification.",
    )
    parser.add_argument(
        "--label-attr",
        default="niche",
        help="Graph attribute copied to graph.y before training (default: niche).",
    )
    parser.add_argument(
        "--subgraph-pooling",
        choices=("attn", "mean", "cls"),
        default="attn",
        help="Subgraph pooling strategy (default: attn).",
    )
    parser.add_argument(
        "--subgraph-cls-token",
        action="store_true",
        help="Enable a fine-tuning CLS token.",
    )
    parser.add_argument(
        "--nonlinear-decoder",
        action="store_true",
        help="Use a two-layer decoder instead of the default linear decoder.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index if CUDA is available.")
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit torch device, e.g. cpu, cuda, or cuda:1. Overrides --gpu-id.",
    )
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision fine-tuning.")
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="AMP dtype for fine-tuning (default: bfloat16).",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    return parser.parse_args(args)


def resolve_device(device_arg: str | None, gpu_id: int) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def resolve_model_files(args: argparse.Namespace) -> list[Path]:
    raw_paths = list(args.model_file or [])
    if args.model_weight_file:
        raw_paths.append(args.model_weight_file)
    if not raw_paths:
        raise ValueError("Provide a pretrained checkpoint with model_weight_file or --model-file.")

    model_files = [Path(path).expanduser() for path in raw_paths]
    missing = [str(path) for path in model_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoint(s): {missing}")
    return model_files


def load_graph(path: Path):
    graph = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(graph, (list, tuple)):
        if len(graph) != 1:
            raise ValueError(f"Expected one graph in {path}, found {len(graph)}.")
        graph = graph[0]
    return graph


def prepare_graph_targets(graph, label_attr: str, task_type: str) -> int:
    if hasattr(graph, label_attr):
        graph.y = getattr(graph, label_attr)
    elif not hasattr(graph, "y"):
        raise AttributeError(f"Graph is missing both '{label_attr}' and 'y' labels.")

    if task_type == "classification":
        graph.y = graph.y.long()
        labels = torch.unique(graph.y.detach().cpu())
        if torch.any(labels < 0):
            raise ValueError("Classification labels must be non-negative integers.")
        return int(labels.numel())

    graph.y = graph.y.float()
    return int(graph.y.shape[-1]) if graph.y.ndim > 1 else 1


def model_label_for(path: Path, args: argparse.Namespace, index: int, model_count: int) -> str:
    if args.model_name and model_count == 1:
        return args.model_name
    stem = path.stem
    return stem if model_count == 1 else f"{index:02d}_{stem}"


def add_file_handler(log_file: Path | None):
    if log_file is None:
        return None
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    run_train_ft_module.logger.addHandler(handler)
    return handler


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    device = resolve_device(parsed.device, parsed.gpu_id)
    train_folder = Path(parsed.train_file_fold).expanduser()
    train_graph_path = Path(parsed.train_graph).expanduser() if parsed.train_graph else train_folder / parsed.train_file_name
    val_graph_path = Path(parsed.val_graph).expanduser() if parsed.val_graph else train_folder / parsed.val_file_name
    if not train_graph_path.is_file():
        raise FileNotFoundError(f"Training graph not found: {train_graph_path}")
    if parsed.save_epoch_wise_size < 0:
        raise ValueError("--save-epoch-wise-size must be >= 0.")
    if not parsed.no_validation and not val_graph_path.is_file():
        raise FileNotFoundError(
            f"Validation graph not found: {val_graph_path}. "
            "Pass --no-validation to run without val.pt."
        )

    model_files = resolve_model_files(parsed)
    save_root = Path(parsed.save_folder).expanduser() if parsed.save_folder else train_folder / "ft_checkpoints"
    graph = load_graph(train_graph_path)
    inferred_output_dim = prepare_graph_targets(graph, parsed.label_attr, parsed.task_type)
    val_graph = None
    if not parsed.no_validation:
        val_graph = load_graph(val_graph_path)
        prepare_graph_targets(val_graph, parsed.label_attr, parsed.task_type)
    output_dim = parsed.output_dim or inferred_output_dim

    backbone_state = "frozen" if parsed.frozen_backbone else "unfrozen"
    for model_idx, model_file in enumerate(model_files):
        model_label = model_label_for(model_file, parsed, model_idx, len(model_files))
        for walk_length in parsed.walk_lengths:
            save_path = save_root / backbone_state / model_label / f"walk_{walk_length}"
            ft_params = {
                "mode": "train",
                "task_type": parsed.task_type,
                "subgraph_cls_token": parsed.subgraph_cls_token,
                "subgraph_pooling": parsed.subgraph_pooling,
                "output_dim": output_dim,
                "linear_decoder": not parsed.nonlinear_decoder,
                "frozen_backbone": parsed.frozen_backbone,
                "ft_batch_size": parsed.batch_size,
                "epochs": parsed.epochs,
                "save_epoch_wise_size": parsed.save_epoch_wise_size,
                "log_every_n_batches": parsed.log_every_n_batches,
                "lr": parsed.lr,
                "device": device,
                "use_amp": parsed.use_amp,
                "amp_dtype": parsed.amp_dtype,
                "walk_length": walk_length,
                "save_path": str(save_path),
            }
            if parsed.n_total_rw is not None:
                ft_params["n_total_rw"] = parsed.n_total_rw

            log_file = Path(parsed.log_file).expanduser() if parsed.log_file else save_path / "train.log"
            handler = add_file_handler(log_file)
            try:
                print(f"Training {model_label} walk_length={walk_length}; saving to {save_path}")
                run_ft(
                    model_file=str(model_file),
                    train_graph=graph,
                    val_graph=val_graph,
                    ft_params=ft_params,
                    logger_override=run_train_ft_module.logger,
                )
            finally:
                if handler is not None:
                    run_train_ft_module.logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    main()
