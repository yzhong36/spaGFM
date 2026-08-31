#!/usr/bin/env python3
"""Run spaGFM inference from the command line.

Example:
    python CLI_inference.py \
        --model-pattern "/path/to/run_dir/step_150000.pt" \
        --graph-paths /path/to/test1.pt /path/to/test2.pt \
        --save-path /path/to/output \
        --gpu-id 0 \
        --walk-length 8 \
        --batch-size 256
"""

from __future__ import annotations

import argparse
import os

import torch

from spaGFM.stRoamer.inference.run_inference import (
    load_checkpoint,
    load_graph_inputs,
    output_path_for_result,
    resolve_model_paths,
    run_inference,
    save_inference_result,
)


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for spaGFM inference")

    parser.add_argument(
        "--model-pattern",
        type=str,
        required=True,
        help="Checkpoint glob pattern or file path (e.g., /run_dir/step_150000.pt)",
    )
    parser.add_argument(
        "--graph-paths",
        nargs="+",
        required=True,
        help="One or more .pt graph files or .h5ad files with adata.uns['pyG_graph']",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save inference outputs",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default: 256)",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=None,
        help="Override model params.walk_length if provided",
    )
    parser.add_argument(
        "--n-total-rw",
        type=int,
        default=None,
        help="Override model params.n_total_rw if provided",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="spaGFM.pt",
        help="Suffix for saved outputs (default: spaGFM.pt)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision during inference",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="AMP dtype for inference (default: bfloat16)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        default=True,
        help="Use torch.load(..., weights_only=True) for checkpoint loading (default: True)",
    )
    parser.add_argument(
        "--no-weights-only",
        action="store_false",
        dest="weights_only",
        help="Disable weights_only checkpoint loading",
    )

    return parser.parse_args(args)


def main(args=None) -> None:
    args = parse_args(args)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    model_paths = resolve_model_paths(args.model_pattern)
    print(f"Matched {len(model_paths)} model checkpoint(s)")

    graph_inputs = load_graph_inputs(args.graph_paths, device=device)
    print(f"Loaded {len(graph_inputs)} graph(s)")

    for ckpt_path in model_paths:
        model_index = os.path.basename(ckpt_path)
        print(f"Running checkpoint: {model_index}")

        model_paras = load_checkpoint(ckpt_path, device=device, weights_only=args.weights_only)

        if args.walk_length is not None:
            if "params" not in model_paras or not isinstance(model_paras["params"], dict):
                raise KeyError("Checkpoint is missing params dict; cannot set walk_length")
            model_paras["params"]["walk_length"] = args.walk_length

        if args.n_total_rw is not None:
            if "params" not in model_paras or not isinstance(model_paras["params"], dict):
                raise KeyError("Checkpoint is missing params dict; cannot set n_total_rw")
            model_paras["params"]["n_total_rw"] = args.n_total_rw

        for i, (graph, graph_path, adata) in enumerate(graph_inputs):
            inference_input = adata if adata is not None else graph
            result = run_inference(
                inference_input,
                model=model_paras,
                device=device,
                batch_size=args.batch_size,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
            )
            output_path = output_path_for_result(
                graph=graph,
                graph_path=graph_path,
                adata=adata,
                model_index=model_index,
                save_path=args.save_path,
                output_suffix=args.output_suffix,
                input_index=i,
            )
            save_inference_result(result, output_path)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
