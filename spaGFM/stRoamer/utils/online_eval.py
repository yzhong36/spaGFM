from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
import wandb
from torch.amp import autocast

from spaGFM.stRoamer.utils.data_split import get_split
from spaGFM.stRoamer.utils.eval_filter import is_regression_metric, prepare_filtered_labels
from spaGFM.stRoamer.utils.model_downstream import knn_node, linear_probe_node
from spaGFM.stRoamer.utils.rw_sampler import get_patterns


VALID_EVAL_EMBEDDINGS = {"node_emb", "subgraph_emb"}


def get_rank_eval_slice(num_nodes, rank, world_size):
    start = (num_nodes * rank) // world_size
    end = (num_nodes * (rank + 1)) // world_size
    return start, end


def normalize_eval_labels(eval_label):
    if eval_label is None:
        return []
    if isinstance(eval_label, str):
        labels = [eval_label]
    elif isinstance(eval_label, (list, tuple)):
        labels = list(eval_label)
    else:
        raise TypeError("eval_label must be a string or a list of strings.")

    if not labels:
        raise ValueError("eval_label must contain at least one label name.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Each eval label must be a non-empty string.")
    if len(set(labels)) != len(labels):
        raise ValueError("eval_label contains duplicate label names.")
    return labels


def normalize_eval_embedding_map(eval_labels, eval_embedding_type):
    if not eval_labels:
        return {}

    if eval_embedding_type is None:
        return {label: "subgraph_emb" for label in eval_labels}

    if isinstance(eval_embedding_type, str):
        if eval_embedding_type not in VALID_EVAL_EMBEDDINGS:
            raise ValueError(
                f"eval_embedding_type must be one of {sorted(VALID_EVAL_EMBEDDINGS)}."
            )
        return {label: eval_embedding_type for label in eval_labels}

    if isinstance(eval_embedding_type, (list, tuple)):
        if len(eval_embedding_type) != len(eval_labels):
            raise ValueError("eval_embedding_type list must match eval_label length.")
        embedding_map = dict(zip(eval_labels, eval_embedding_type))
    elif isinstance(eval_embedding_type, dict):
        missing_labels = [label for label in eval_labels if label not in eval_embedding_type]
        if missing_labels:
            raise ValueError(
                f"eval_embedding_type is missing labels: {missing_labels}"
            )
        embedding_map = {label: eval_embedding_type[label] for label in eval_labels}
    else:
        raise TypeError(
            "eval_embedding_type must be a string, a list, or a dict."
        )

    invalid = {
        label: embedding_name
        for label, embedding_name in embedding_map.items()
        if embedding_name not in VALID_EVAL_EMBEDDINGS
    }
    if invalid:
        raise ValueError(
            f"Invalid eval embedding selections: {invalid}. "
            f"Valid options are {sorted(VALID_EVAL_EMBEDDINGS)}."
        )
    return embedding_map


def build_eval_patterns(eval_graph, eval_params):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed = eval_params.get("eval_seed", 2025)
    try:
        torch.manual_seed(seed)
        if cuda_rng_states is not None:
            torch.cuda.manual_seed_all(seed)
        return get_patterns(eval_graph, eval_params, batch_size=8192)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def build_eval_splits(eval_graph, split_name, split_repeat, seed):
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return [get_split(eval_graph, split_name) for _ in range(max(int(split_repeat), 1))]
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def build_online_eval_params(params, y):
    metric = params.get("eval_metric", "f1_macro")
    output_dim = len(torch.unique(y)) if not is_regression_metric(metric) else params.get("output_dim", 1)
    eval_params = {
        "linear_probe_lr": params.get("linear_probe_lr", 0.01),
        "linear_probe_weight_decay": params.get("linear_probe_weight_decay", 0.001),
        "linear_probe_epochs": params.get("linear_probe_epochs", 100),
        "metric": metric,
        "output_dim": output_dim,
        "knn_weights": params.get("knn_weights", "uniform"),
        "knn_metric": params.get("knn_metric", "minkowski"),
        "knn_p": params.get("knn_p", 2),
        "knn_n_jobs": params.get("knn_n_jobs", -1),
    }
    if params.get("num_tasks") is not None:
        eval_params["num_tasks"] = params["num_tasks"]
    return eval_params


def build_online_eval_state(params, device, logger: logging.Logger | None = None):
    active_logger = logger or logging.getLogger(__name__)

    eval_dataset_path = params.get("eval_dataset")
    eval_every_steps = params.get("eval_every_steps")
    run_eval = eval_dataset_path is not None and eval_every_steps is not None and eval_every_steps > 0
    if not run_eval:
        return None

    eval_label = params.get("eval_label")
    if eval_label is None:
        raise ValueError(
            "Online evaluation requires eval_label when eval_dataset and eval_every_steps are set."
        )

    eval_embedding_type = params.get("eval_embedding_type", "subgraph_emb")
    eval_split = params.get("eval_split", "low")
    eval_split_repeat = params.get("eval_split_repeat", 10)
    eval_seed = params.get("eval_seed", 2025)
    eval_metric = params.get("eval_metric", "f1_macro")
    min_label_ratio = params.get("min_label_ratio", 0.01)
    min_label_count = params.get("min_label_count", 100)

    if not 0 <= min_label_ratio < 1:
        raise ValueError("min_label_ratio must be in the range [0, 1).")
    if min_label_count < 1:
        raise ValueError("min_label_count must be at least 1.")

    eval_label_names = normalize_eval_labels(eval_label)
    eval_embedding_map = normalize_eval_embedding_map(eval_label_names, eval_embedding_type)
    eval_graph = torch.load(eval_dataset_path, map_location="cpu", weights_only=False)
    missing_eval_labels = [label for label in eval_label_names if not hasattr(eval_graph, label)]
    if missing_eval_labels:
        raise AttributeError(
            f"Evaluation graph is missing label attribute(s): {missing_eval_labels}"
        )

    eval_params = dict(params)
    eval_params["device"] = device
    eval_patterns = build_eval_patterns(eval_graph, eval_params)

    eval_targets = {}
    for label in eval_label_names:
        target_y = getattr(eval_graph, label).detach().cpu()
        keep_mask = None
        if not is_regression_metric(eval_metric) and min_label_ratio > 0:
            keep_mask, _, target_y = prepare_filtered_labels(
                dataset_name=f"online_eval/{label}",
                y=target_y,
                min_label_ratio=min_label_ratio,
                min_label_count=min_label_count,
                logger=active_logger,
            )
        split_graph = SimpleNamespace(num_nodes=int(target_y.size(0)))
        eval_targets[label] = {
            "y": target_y,
            "keep_mask": keep_mask,
            "splits": build_eval_splits(split_graph, eval_split, eval_split_repeat, eval_seed),
        }

    if is_regression_metric(eval_metric) and min_label_ratio > 0:
        active_logger.info(
            "Ignoring --min-label-ratio=%.4f and --min-label-count=%d for online evaluation "
            "because eval_metric '%s' is regression.",
            min_label_ratio,
            min_label_count,
            eval_metric,
        )
    elif min_label_ratio > 0:
        active_logger.info(
            "Online evaluation low-abundance filtering enabled with min ratio %.4f and min count %d",
            min_label_ratio,
            min_label_count,
        )

    return {
        "eval_graph": eval_graph,
        "eval_patterns": eval_patterns,
        "eval_label_names": eval_label_names,
        "eval_embedding_map": eval_embedding_map,
        "eval_targets": eval_targets,
        "eval_batch_size": params.get("eval_batch_size", 256),
        "eval_knn_k": params.get("knn_k", 5),
        "num_eval_nodes": int(eval_graph.x.size(0)),
    }


def run_online_evaluation_step(
    core_model,
    params,
    device,
    rank,
    world_size,
    is_distributed,
    current_step,
    eval_state,
    logger: logging.Logger | None = None,
):
    active_logger = logger or logging.getLogger(__name__)

    eval_graph = eval_state["eval_graph"]
    eval_patterns = eval_state["eval_patterns"]
    eval_label_names = eval_state["eval_label_names"]
    eval_embedding_map = eval_state["eval_embedding_map"]
    eval_targets = eval_state["eval_targets"]
    eval_batch_size = eval_state["eval_batch_size"]
    eval_knn_k = eval_state["eval_knn_k"]
    num_eval_nodes = eval_state["num_eval_nodes"]

    core_model.eval()
    eval_graph_dev = eval_graph.to(device)
    eval_patterns_dev = eval_patterns.to(device)

    if is_distributed:
        start_idx, end_idx = get_rank_eval_slice(num_eval_nodes, rank, world_size)
    else:
        start_idx, end_idx = 0, num_eval_nodes

    local_node_ids = torch.arange(start_idx, end_idx)
    local_node_embeddings_list = []
    local_subgraph_embeddings_list = []
    amp_dtype = torch.float16 if params.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
    use_amp = bool(params.get("use_amp", False)) and (
        device.type == "cuda" or (device.type == "cpu" and amp_dtype == torch.bfloat16)
    )
    with torch.no_grad():
        for batch_start in range(0, local_node_ids.numel(), eval_batch_size):
            cur_nodes = local_node_ids[batch_start: batch_start + eval_batch_size].to(device)
            if cur_nodes.numel() == 0:
                continue
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                node_emb, subgraph_emb = core_model.inference(
                    graph=eval_graph_dev,
                    nodes=eval_patterns_dev[:, cur_nodes, :],
                )
            local_node_embeddings_list.append(node_emb.detach().cpu())
            local_subgraph_embeddings_list.append(subgraph_emb.detach().cpu())

    hidden_dim = params["hidden_dim"]
    if local_node_embeddings_list:
        local_node_embeddings = torch.cat(local_node_embeddings_list, dim=0)
    else:
        local_node_embeddings = torch.empty((0, hidden_dim), dtype=torch.float32)

    if local_subgraph_embeddings_list:
        local_subgraph_embeddings = torch.cat(local_subgraph_embeddings_list, dim=0)
    else:
        local_subgraph_embeddings = torch.empty((0, hidden_dim), dtype=torch.float32)

    if is_distributed:
        gathered_node_embeddings = [None] * world_size if rank == 0 else None
        gathered_subgraph_embeddings = [None] * world_size if rank == 0 else None
        dist.gather_object(local_node_embeddings, object_gather_list=gathered_node_embeddings, dst=0)
        dist.gather_object(local_subgraph_embeddings, object_gather_list=gathered_subgraph_embeddings, dst=0)
    else:
        gathered_node_embeddings = [local_node_embeddings]
        gathered_subgraph_embeddings = [local_subgraph_embeddings]

    if rank == 0:
        node_embeddings = torch.cat(gathered_node_embeddings, dim=0)
        subgraph_embeddings = torch.cat(gathered_subgraph_embeddings, dim=0)
        wandb_payload = {"evaluation/step": current_step}

        for current_eval_label in eval_label_names:
            label_embedding_attr = eval_embedding_map[current_eval_label]
            eval_embeddings = (
                node_embeddings if label_embedding_attr == "node_emb" else subgraph_embeddings
            )
            eval_target = eval_targets[current_eval_label]
            keep_mask = eval_target["keep_mask"]
            if keep_mask is not None:
                eval_embeddings = eval_embeddings[keep_mask]
            y = eval_target["y"]
            online_eval_params = build_online_eval_params(params, y)
            eval_splits = eval_target["splits"]

            linear_probe_result = linear_probe_node(
                embeddings=eval_embeddings,
                y=y,
                splits=eval_splits,
                params=online_eval_params,
                device=device,
            )
            knn_result = knn_node(
                embeddings=eval_embeddings,
                y=y,
                splits=eval_splits,
                params=online_eval_params,
                device=device,
                k=eval_knn_k,
            )
            metric_name = linear_probe_result["metric"]

            active_logger.info(
                "Online eval step %d [%s/%s]: linear_probe val=%.4f test=%.4f | knn@%d val=%.4f test=%.4f",
                current_step,
                current_eval_label,
                label_embedding_attr,
                linear_probe_result["val"],
                linear_probe_result["test"],
                eval_knn_k,
                knn_result["val"],
                knn_result["test"],
            )

            wandb_payload.update({
                f"evaluation/{current_eval_label}/{label_embedding_attr}/linear_probe_node/val_{metric_name}": linear_probe_result["val"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/linear_probe_node/val_std": linear_probe_result["val_std"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/linear_probe_node/test_{metric_name}": linear_probe_result["test"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/linear_probe_node/test_std": linear_probe_result["test_std"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/knn_node/val_{metric_name}": knn_result["val"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/knn_node/val_std": knn_result["val_std"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/knn_node/test_{metric_name}": knn_result["test"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/knn_node/test_std": knn_result["test_std"],
                f"evaluation/{current_eval_label}/{label_embedding_attr}/knn_node/k": eval_knn_k,
            })

        wandb.log(wandb_payload)

    if is_distributed:
        dist.barrier()

    del eval_graph_dev
    del eval_patterns_dev
    core_model.train()
