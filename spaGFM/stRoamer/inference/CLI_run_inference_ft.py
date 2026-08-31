#!/usr/bin/env python3
"""Run spaGFM fine-tuned inference from the command line."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from spaGFM.stRoamer.inference.inference_ft import inference_ft
from spaGFM.stRoamer.utils.helper import set_seed
from spaGFM.stRoamer.utils.rw_sampler import get_patterns


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for spaGFM fine-tuned inference.")
    parser.add_argument(
        "--graph-paths",
        nargs="+",
        default=None,
        help="One or more test graph .pt files.",
    )
    parser.add_argument(
        "--test-file-folder",
        default=None,
        help="Folder containing fold subdirectories with test graph files.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Fold names or numbers to read under --test-file-folder, e.g. 1 2 3 4.",
    )
    parser.add_argument(
        "--test-file-name",
        default="test.pt",
        help="Test graph filename inside each fold directory (default: test.pt).",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="One or more fine-tuned checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=None,
        help="Glob pattern for fine-tuned checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Root directory searched recursively for checkpoint-file-name.",
    )
    parser.add_argument(
        "--checkpoint-file-name",
        default="best_model.pt",
        help="Checkpoint filename used with --checkpoint-root (default: best_model.pt).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for inference outputs.")
    parser.add_argument(
        "--output-format",
        choices=("h5ad", "csv", "pt"),
        default="h5ad",
        help="Output format for predictions (default: h5ad).",
    )
    parser.add_argument("--label-attr", default="niche", help="Graph label attribute for output.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument(
        "--pattern-batch-size",
        type=int,
        default=8192,
        help="Batch size used while sampling random-walk patterns.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index if CUDA is available.")
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit torch device, e.g. cpu, cuda, or cuda:1. Overrides --gpu-id.",
    )
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision during inference.")
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="AMP dtype for inference (default: bfloat16).",
    )
    parser.add_argument("--save-attention", action="store_true", help="Save attention weights as .pt files.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for pattern generation.")
    return parser.parse_args(args)


def resolve_device(device_arg: str | None, gpu_id: int) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def fold_dir_name(fold: str) -> str:
    return fold if fold.startswith("fold") else f"fold{fold}"


def resolve_graph_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.graph_paths:
        paths.extend(Path(path).expanduser() for path in args.graph_paths)

    if args.test_file_folder:
        root = Path(args.test_file_folder).expanduser()
        if args.folds:
            paths.extend(root / fold_dir_name(str(fold)) / args.test_file_name for fold in args.folds)
        else:
            paths.extend(sorted(root.glob(f"fold*/{args.test_file_name}")))

    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("Provide --graph-paths or --test-file-folder.")

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing test graph(s): {missing}")
    return paths


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    raw_paths: list[str] = []
    if args.checkpoint_paths:
        raw_paths.extend(args.checkpoint_paths)
    if args.checkpoint_pattern:
        raw_paths.extend(sorted(glob.glob(args.checkpoint_pattern)))
    if args.checkpoint_root:
        root = Path(args.checkpoint_root).expanduser()
        raw_paths.extend(str(path) for path in sorted(root.rglob(args.checkpoint_file_name)))

    paths = sorted(dict.fromkeys(Path(path).expanduser() for path in raw_paths))
    if not paths:
        raise ValueError("Provide --checkpoint-paths, --checkpoint-pattern, or --checkpoint-root.")

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return paths


def load_graph(path: Path, label_attr: str):
    graph = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(graph, (list, tuple)):
        if len(graph) != 1:
            raise ValueError(f"Expected one graph in {path}, found {len(graph)}.")
        graph = graph[0]
    if hasattr(graph, label_attr):
        graph.y = getattr(graph, label_attr)
    return graph


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run_checkpoint_inference(
    graph,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    pattern_batch_size: int,
    seed: int,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "ft_params" not in checkpoint:
        raise KeyError(f"Checkpoint is missing ft_params: {checkpoint_path}")

    ft_params = dict(checkpoint["ft_params"])
    ft_params.pop("random_walks", None)
    ft_params.pop("train_random_walks", None)
    ft_params.pop("val_random_walks", None)
    ft_params["device"] = device
    ft_params["use_amp"] = use_amp
    ft_params["amp_dtype"] = amp_dtype

    set_seed(seed)
    patterns = get_patterns(graph, ft_params, batch_size=pattern_batch_size).to(device)
    graph = graph.to(device)
    attn_weights, logits = inference_ft(
        graph=graph,
        pattern=patterns,
        model_file=str(checkpoint_path),
        params=ft_params,
        batch_size=batch_size,
    )
    return attn_weights.detach().cpu(), logits.detach().cpu()


def safe_name(path: Path) -> str:
    parts = [part for part in path.with_suffix("").parts[-4:] if part not in (os.sep, "")]
    name = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def graph_obs_frame(graph, label_attr: str) -> pd.DataFrame:
    n_obs = int(graph.x.size(0))
    obs = pd.DataFrame(index=[str(i) for i in range(n_obs)])
    if hasattr(graph, label_attr):
        labels = getattr(graph, label_attr)
        obs["label"] = labels.detach().cpu().numpy()
    elif hasattr(graph, "y"):
        obs["label"] = graph.y.detach().cpu().numpy()
    if hasattr(graph, "is_assigned"):
        obs["is_assigned"] = graph.is_assigned.detach().cpu().numpy()
    return obs


def save_prediction_table(obs: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        obs.to_csv(output_path)
        return
    if output_format == "h5ad":
        import anndata as ad

        adata = ad.AnnData(X=np.empty((obs.shape[0], 0)), obs=obs)
        adata.write_h5ad(output_path)
        return
    torch.save({"obs": obs}, output_path)


def output_path_for(graph_path: Path, output_dir: Path, output_format: str) -> Path:
    suffix = ".pt" if output_format == "pt" else f".{output_format}"
    return output_dir / f"{graph_path.stem}_ft_predictions{suffix}"


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    device = resolve_device(parsed.device, parsed.gpu_id)
    output_dir = Path(parsed.output_dir).expanduser()
    graph_paths = resolve_graph_paths(parsed)
    checkpoint_paths = resolve_checkpoint_paths(parsed)

    print(f"Loaded {len(graph_paths)} graph path(s) and {len(checkpoint_paths)} checkpoint path(s).")
    for graph_path in graph_paths:
        graph = load_graph(graph_path, parsed.label_attr)
        obs = graph_obs_frame(graph, parsed.label_attr)
        attention_outputs = {}

        for checkpoint_path in checkpoint_paths:
            checkpoint_name = safe_name(checkpoint_path)
            print(f"Running {graph_path} with {checkpoint_path}")
            attn_weights, logits = run_checkpoint_inference(
                graph=graph,
                checkpoint_path=checkpoint_path,
                device=device,
                batch_size=parsed.batch_size,
                pattern_batch_size=parsed.pattern_batch_size,
                seed=parsed.seed,
                use_amp=parsed.use_amp,
                amp_dtype=parsed.amp_dtype,
            )
            probabilities = torch.softmax(logits, dim=1)
            obs[f"{checkpoint_name}_prediction"] = torch.argmax(logits, dim=1).numpy()
            obs[f"{checkpoint_name}_confidence"] = torch.max(probabilities, dim=1).values.numpy()
            attention_outputs[checkpoint_name] = attn_weights

        output_path = output_path_for(graph_path, output_dir, parsed.output_format)
        save_prediction_table(obs, output_path, parsed.output_format)
        print(f"Saved {output_path}")

        if parsed.save_attention:
            attention_path = output_dir / f"{graph_path.stem}_attention.pt"
            attention_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attention_outputs, attention_path)
            print(f"Saved {attention_path}")


if __name__ == "__main__":
    main()
