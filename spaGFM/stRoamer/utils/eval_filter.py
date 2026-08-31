from __future__ import annotations

import logging

import torch


def is_regression_metric(metric: str) -> bool:
    return metric in {"rmse", "mae"}


def prepare_filtered_labels(
    dataset_name: str,
    y: torch.Tensor,
    min_label_ratio: float,
    min_label_count: int,
    logger: logging.Logger | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    active_logger = logger or logging.getLogger(__name__)

    if min_label_ratio <= 0:
        return None, y, y

    normalized_y = y
    if normalized_y.ndim == 2 and normalized_y.size(-1) == 1:
        normalized_y = normalized_y.reshape(-1)
    elif normalized_y.ndim != 1:
        active_logger.warning(
            "%s | skipping low-abundance filtering because label shape %s is not single-label",
            dataset_name,
            tuple(normalized_y.shape),
        )
        return None, normalized_y, normalized_y

    num_instances = normalized_y.numel()
    if num_instances == 0:
        raise ValueError(f"Dataset {dataset_name} has no labels to evaluate.")

    y_cpu = normalized_y.detach().cpu()
    unique_labels, counts = torch.unique(y_cpu, return_counts=True)
    label_ratios = counts.float() / float(num_instances)
    remove_label_mask = (label_ratios < min_label_ratio) & (counts < min_label_count)
    keep_label_mask = ~remove_label_mask

    if bool(keep_label_mask.all()):
        return None, normalized_y, normalized_y

    keep_labels = unique_labels[keep_label_mask]
    keep_instance_mask = torch.zeros(num_instances, dtype=torch.bool)
    for label in keep_labels.tolist():
        keep_instance_mask |= y_cpu == label

    filtered_y = normalized_y[keep_instance_mask]
    kept_instances = int(keep_instance_mask.sum().item())
    removed_instances = num_instances - kept_instances
    removed_labels = int((~keep_label_mask).sum().item())
    kept_labels = int(torch.unique(filtered_y.detach().cpu()).numel()) if kept_instances > 0 else 0

    if kept_instances == 0:
        raise ValueError(
            f"Low-abundance filtering removed every instance from {dataset_name}. "
            f"Reduce --min-label-ratio from {min_label_ratio} or --min-label-count from "
            f"{min_label_count}."
        )

    if kept_labels < 2:
        raise ValueError(
            f"Low-abundance filtering left fewer than 2 labels in {dataset_name}. "
            f"Reduce --min-label-ratio from {min_label_ratio} or --min-label-count from "
            f"{min_label_count}."
        )

    active_logger.info(
        "%s | filtered %d/%d instance(s) from %d low-abundance label(s) below ratio %.4f "
        "and count %d; kept %d instance(s) across %d label(s)",
        dataset_name,
        removed_instances,
        num_instances,
        removed_labels,
        min_label_ratio,
        min_label_count,
        kept_instances,
        kept_labels,
    )
    return keep_instance_mask, normalized_y, filtered_y


def filter_low_abundance_labels(
    dataset_name: str,
    embeddings: torch.Tensor,
    y: torch.Tensor,
    min_label_ratio: float,
    min_label_count: int,
    logger: logging.Logger | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_instance_mask, normalized_y, filtered_y = prepare_filtered_labels(
        dataset_name=dataset_name,
        y=y,
        min_label_ratio=min_label_ratio,
        min_label_count=min_label_count,
        logger=logger,
    )

    if embeddings.size(0) != normalized_y.size(0):
        raise ValueError(
            f"Embedding/label size mismatch for {dataset_name}: "
            f"{embeddings.size(0)} embeddings vs {normalized_y.size(0)} labels."
        )

    if keep_instance_mask is None:
        return embeddings, normalized_y

    filtered_embeddings = embeddings[keep_instance_mask]
    return filtered_embeddings, filtered_y
