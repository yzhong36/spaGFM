#!/usr/bin/env python3
"""Command-line evaluation for spaGFM graph embeddings.

Example:
    python CLI_evaluation.py \
        --embedding-dir /path/to/corpus_full_317M_8_8 \
        --output-suffix spaGFM.pt \
        --save-path /path/to/output \
        --method linear_probe_node
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


from spaGFM.stRoamer.utils.data_split import get_split
from spaGFM.stRoamer.utils.eval_filter import filter_low_abundance_labels, is_regression_metric
from spaGFM.stRoamer.utils.helper import set_seed
from spaGFM.stRoamer.utils.metric_eval import normalize_metric_names
from spaGFM.stRoamer.utils.model_downstream import kmeans_node, knn_node, linear_probe_node


logger = logging.getLogger(__name__)


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for spaGFM evaluation")

    parser.add_argument(
        "--embedding-dir",
        type=str,
        required=True,
        help="Directory containing saved embedding graphs",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="spaGFM.pt",
        help="Input file suffix used to discover embedding graphs inside --embedding-dir",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="linear_probe_node",
        choices=["linear_probe_node", "knn_node", "kmeans_node"],
        help="Evaluation method to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for evaluation, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="low",
        help="Split strategy passed to get_split",
    )
    parser.add_argument(
        "--split-repeat",
        type=int,
        default=10,
        help="Number of repeated splits",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        help="Metric name stored in params and passed to evaluation",
    )
    parser.add_argument(
        "--dataset-attr",
        type=str,
        default="dataset",
        help="Graph attribute used as the dataset key, or .obs column when loading .h5ad",
    )
    parser.add_argument(
        "--label-attr",
        type=str,
        required=True,
        help="Graph attribute used as the node label source, or .obs column when loading .h5ad",
    )
    parser.add_argument(
        "--embedding-attr",
        type=str,
        required=True,
        help="Graph attribute used as the embedding input, or .obsm key when loading .h5ad",
    )
    parser.add_argument(
        "--linear-probe-lr",
        type=float,
        default=0.01,
        help="Learning rate for linear probing",
    )
    parser.add_argument(
        "--linear-probe-weight-decay",
        type=float,
        default=0.001,
        help="Weight decay for linear probing",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=100,
        help="Number of epochs for linear probing",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Number of neighbors for kNN evaluation",
    )
    parser.add_argument(
        "--knn-batch-size",
        type=int,
        default=2048,
        help="Query batch size used inside knn_node",
    )
    parser.add_argument(
        "--kmeans-num-clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters. Defaults to the number of labels.",
    )
    parser.add_argument(
        "--kmeans-n-init",
        type=int,
        default=10,
        help="Number of KMeans initializations.",
    )
    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=500,
        help="Maximum KMeans iterations.",
    )
    parser.add_argument(
        "--kmeans-random-state",
        type=int,
        default=0,
        help="Random state used by KMeans and PCA.",
    )
    parser.add_argument(
        "--kmeans-num-runs",
        type=int,
        default=10,
        help="Number of full-dataset KMeans runs with different seeds.",
    )
    parser.set_defaults(pca_embeddings=True)
    parser.add_argument(
        "--pca-embeddings",
        dest="pca_embeddings",
        action="store_true",
        help="Apply PCA before KMeans evaluation. Enabled by default.",
    )
    parser.add_argument(
        "--no-pca-embeddings",
        dest="pca_embeddings",
        action="store_false",
        help="Disable PCA before KMeans evaluation.",
    )
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=64,
        help="Maximum number of PCA components used when --pca-embeddings is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used before split generation",
    )
    parser.add_argument(
        "--min-label-ratio",
        type=float,
        default=0.01,
        help=(
            "Drop instances whose label frequency is below this dataset-level ratio before "
            "generating splits, but only when the label count is also below "
            "--min-label-count. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=100,
        help=(
            "Drop instances from labels whose instance count is below this threshold, but only "
            "when the label frequency is also below --min-label-ratio."
        ),
    )

    return parser.parse_args(args)


def to_data_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "to_data_list"):
        return obj.to_data_list()
    return [obj]


def _to_tensor_label(values: Any, is_regression: bool) -> torch.Tensor:
    array = np.asarray(values)
    if is_regression:
        return torch.as_tensor(array)

    if array.dtype.kind in {"U", "S", "O"}:
        _, encoded = np.unique(array.astype(str), return_inverse=True)
        return torch.as_tensor(encoded, dtype=torch.long)

    return torch.as_tensor(array, dtype=torch.long)


def _to_embedding_tensor(values: Any) -> torch.Tensor:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def load_h5ad_file(
    path: str,
    dataset_attr: str,
    label_attr: str,
    embedding_attr: str,
    metric: str,
) -> List[Any]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Loading .h5ad input requires the 'anndata' package.") from exc

    adata = ad.read_h5ad(path)
    if dataset_attr not in adata.obs.columns:
        raise KeyError(f"{path} is missing obs column: {dataset_attr}")
    if label_attr not in adata.obs.columns:
        raise KeyError(f"{path} is missing obs column: {label_attr}")
    if embedding_attr not in adata.obsm:
        raise KeyError(f"{path} is missing obsm key: {embedding_attr}")

    labels = _to_tensor_label(adata.obs[label_attr].to_numpy(), is_regression_metric(metric))
    embeddings = _to_embedding_tensor(adata.obsm[embedding_attr])
    dataset_values = adata.obs[dataset_attr].astype(str).to_numpy()
    unique_datasets = np.unique(dataset_values)
    if unique_datasets.size != 1:
        raise ValueError(
            f"{path} must contain exactly one dataset value in obs['{dataset_attr}'], found {unique_datasets.size}."
        )
    dataset_name = unique_datasets[0]

    return [
        SimpleNamespace(
            dataset=dataset_name,
            y=labels,
            **{label_attr: labels, embedding_attr: embeddings},
        )
    ]


def load_graph_file(
    path: str,
    dataset_attr: str,
    label_attr: str,
    embedding_attr: str,
    metric: str,
) -> List[Any]:
    if path.endswith(".h5ad"):
        return load_h5ad_file(
            path,
            dataset_attr=dataset_attr,
            label_attr=label_attr,
            embedding_attr=embedding_attr,
            metric=metric,
        )

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return to_data_list(loaded)


def group_embedding_paths(embedding_dir: str, output_suffix: str) -> Dict[str, List[str]]:
    embedding_paths = sorted(glob.glob(os.path.join(embedding_dir, f"*{output_suffix}")))
    if not embedding_paths:
        raise FileNotFoundError(f"No embedding files matched: {embedding_dir}/*{output_suffix}")

    grouped: Dict[str, List[str]] = defaultdict(list)
    for path in embedding_paths:
        match = re.search(r"(step_\d+)", Path(path).name)
        if match:
            grouped[match.group(1)].append(path)
        else:
            grouped[Path(path).stem].append(path)

    return dict(grouped)


def build_dataset_label_map(graphs: Iterable[Any], label_attr: str) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)

    for graph in graphs:
        dataset = getattr(graph, "dataset", None)
        label_value = getattr(graph, label_attr, None)
        if dataset is None or label_value is None:
            continue
        grouped[str(dataset)].append(label_value)

    dataset_label_map: Dict[str, torch.Tensor] = {}
    for dataset, label_list in grouped.items():
        if len(label_list) == 1:
            dataset_label_map[dataset] = label_list[0]
        else:
            dataset_label_map[dataset] = torch.cat(label_list)

    return dataset_label_map


def build_eval_params(args: argparse.Namespace, y: torch.Tensor) -> Dict[str, Any]:
    metric_names = normalize_metric_names(args.metric)
    primary_metric = metric_names[0]
    output_dim = int(y.size(-1)) if is_regression_metric(primary_metric) and y.ndim > 1 else 1
    if not is_regression_metric(primary_metric):
        output_dim = len(torch.unique(y))

    params: Dict[str, Any] = {
        "linear_probe_lr": args.linear_probe_lr,
        "linear_probe_weight_decay": args.linear_probe_weight_decay,
        "linear_probe_epochs": args.linear_probe_epochs,
        "split": args.split,
        "split_repeat": args.split_repeat,
        "metric": metric_names if len(metric_names) > 1 else primary_metric,
        "output_dim": output_dim,
        "knn_batch_size": args.knn_batch_size,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "kmeans_random_state": args.kmeans_random_state,
        "kmeans_num_runs": args.kmeans_num_runs,
        "PCA_embeddings": args.pca_embeddings,
        "PCA_n_components": args.pca_n_components,
    }
    if args.kmeans_num_clusters is not None:
        params["kmeans_num_clusters"] = args.kmeans_num_clusters
    return params


def extract_metric_summaries(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "metrics" in result:
        return result["metrics"]
    return {str(result["metric"]): result}


def build_metric_specific_results(step_results: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    metric_specific_results: Dict[str, Any] = {}
    for dataset_name, result in step_results.items():
        if "metrics" in result:
            if metric_name not in result["metrics"]:
                raise KeyError(f"Metric '{metric_name}' not found in result for dataset '{dataset_name}'.")
            metric_specific_results[dataset_name] = result["metrics"][metric_name]
        else:
            if str(result["metric"]) != metric_name:
                raise KeyError(
                    f"Metric mismatch for dataset '{dataset_name}': expected '{metric_name}', got '{result['metric']}'."
                )
            metric_specific_results[dataset_name] = result
    return metric_specific_results


def run_step_evaluation(
    step_name: str,
    embedding_paths: List[str],
    dataset_label_map: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    step_start = time.perf_counter()
    logger.info("Evaluating %s with %d embedding file(s)", step_name, len(embedding_paths))

    step_results: Dict[str, Any] = {}
    for embedding_path in embedding_paths:
        dataset_start = time.perf_counter()
        test_graphs = load_graph_file(
            embedding_path,
            dataset_attr=args.dataset_attr,
            label_attr=args.label_attr,
            embedding_attr=args.embedding_attr,
            metric=normalize_metric_names(args.metric)[0],
        )
        if not test_graphs:
            raise ValueError(f"No graphs loaded from {embedding_path}")

        dataset_results: Dict[str, Any] = {}
        for test_graph in test_graphs:
            dataset_name = str(getattr(test_graph, "dataset", Path(embedding_path).stem))
            if dataset_name not in dataset_label_map:
                raise KeyError(f"No labels found for dataset: {dataset_name}")

            test_graph.y = dataset_label_map[dataset_name].clone()
            if not hasattr(test_graph, args.embedding_attr):
                raise AttributeError(f"Graph {dataset_name} is missing embedding attribute: {args.embedding_attr}")
            embeddings = getattr(test_graph, args.embedding_attr)
            labels = test_graph.y

            primary_metric = normalize_metric_names(args.metric)[0]
            if not is_regression_metric(primary_metric):
                embeddings, labels = filter_low_abundance_labels(
                    dataset_name=dataset_name,
                    embeddings=embeddings,
                    y=labels,
                    min_label_ratio=args.min_label_ratio,
                    min_label_count=args.min_label_count,
                    logger=logger,
                )

            eval_params = build_eval_params(args, labels)

            set_seed(args.seed)
            if args.method == "kmeans_node":
                splits = []
            else:
                split_graph = SimpleNamespace(num_nodes=int(labels.size(0)))
                splits = [get_split(split_graph, args.split) for _ in range(args.split_repeat)]

            eval_start = time.perf_counter()
            if args.method == "linear_probe_node":
                result = linear_probe_node(
                    embeddings=embeddings,
                    y=labels,
                    splits=splits,
                    params=eval_params,
                    device=args.device,
                )
            elif args.method == "knn_node":
                result = knn_node(
                    embeddings=embeddings,
                    y=labels,
                    splits=splits,
                    params=eval_params,
                    device=args.device,
                    k=args.knn_k,
                )
            else:
                result = kmeans_node(
                    embeddings=embeddings,
                    y=labels,
                    splits=splits,
                    params=eval_params,
                    device=args.device,
                )

            eval_seconds = time.perf_counter() - eval_start
            logger.info(
                "%s | %s | emb=%s | metrics=%s | %.2fs",
                step_name,
                dataset_name,
                args.embedding_attr,
                ",".join(extract_metric_summaries(result).keys()),
                eval_seconds,
            )
            for metric_name, metric_result in extract_metric_summaries(result).items():
                logger.info(
                    "%s | %s | metric=%s | val=%.4f | test=%.4f",
                    step_name,
                    dataset_name,
                    metric_name,
                    float(metric_result.get("val", 0.0)),
                    float(metric_result.get("test", 0.0)),
                )
            dataset_results[dataset_name] = result

        dataset_seconds = time.perf_counter() - dataset_start
        logger.info("Finished %s file %s in %.2fs", step_name, Path(embedding_path).name, dataset_seconds)
        step_results.update(dataset_results)

    step_seconds = time.perf_counter() - step_start
    logger.info("Completed %s in %.2fs", step_name, step_seconds)
    return step_results


def main(args=None) -> None:
    args = parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not 0 <= args.min_label_ratio < 1:
        raise ValueError("--min-label-ratio must be in the range [0, 1).")
    if args.min_label_count < 1:
        raise ValueError("--min-label-count must be at least 1.")
    if args.kmeans_num_clusters is not None and args.kmeans_num_clusters < 2:
        raise ValueError("--kmeans-num-clusters must be at least 2.")
    if args.kmeans_n_init < 1:
        raise ValueError("--kmeans-n-init must be at least 1.")
    if args.kmeans_max_iter < 1:
        raise ValueError("--kmeans-max-iter must be at least 1.")
    if args.kmeans_num_runs < 1:
        raise ValueError("--kmeans-num-runs must be at least 1.")
    if args.pca_n_components < 1:
        raise ValueError("--pca-n-components must be at least 1.")

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    primary_metric = normalize_metric_names(args.metric)[0]
    if is_regression_metric(primary_metric) and args.min_label_ratio > 0:
        logger.info(
            "Ignoring --min-label-ratio=%.4f and --min-label-count=%d because metric '%s' is regression.",
            args.min_label_ratio,
            args.min_label_count,
            primary_metric,
        )
    elif args.min_label_ratio > 0:
        logger.info(
            "Low-abundance label filtering enabled with min ratio %.4f and min count %d",
            args.min_label_ratio,
            args.min_label_count,
        )

    embedding_groups = group_embedding_paths(args.embedding_dir, args.output_suffix)
    step_groups = embedding_groups
    logger.info("Found %d step group(s) under %s using suffix '%s'", len(step_groups), args.embedding_dir, args.output_suffix)

    for step_file, embedding_paths in sorted(step_groups.items()):
        step_name = Path(step_file).stem
        load_start = time.perf_counter()
        step_graphs: List[Any] = []
        for graph_path in embedding_paths:
            step_graphs.extend(
                load_graph_file(
                    graph_path,
                    dataset_attr=args.dataset_attr,
                    label_attr=args.label_attr,
                    embedding_attr=args.embedding_attr,
                    metric=primary_metric,
                )
            )
        dataset_label_map = build_dataset_label_map(step_graphs, args.label_attr)
        load_seconds = time.perf_counter() - load_start
        logger.info(
            "Loaded %d embedding graph(s), %d dataset label group(s) for %s using '%s' in %.2fs",
            len(step_graphs),
            len(dataset_label_map),
            step_name,
            args.label_attr,
            load_seconds,
        )

        step_results = run_step_evaluation(step_name, embedding_paths, dataset_label_map, args)

        for metric_name in normalize_metric_names(args.metric):
            output_name = f"{step_name}_{args.method}_spaGFM_{metric_name}_stat.pt"
            output_path = os.path.join(args.save_path, output_name)
            torch.save(build_metric_specific_results(step_results, metric_name), output_path)
            logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
