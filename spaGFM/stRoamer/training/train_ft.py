from spaGFM.stRoamer.utils.helper import model_device
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import logging

logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    def __init__(self, y):
        self.y = y.detach().cpu()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return idx, self.y[idx]


def class_weights(graph):
    assigned = getattr(graph, "is_assigned", None)
    if assigned is None:
        assigned = torch.ones(graph.y.size(0), dtype=torch.bool)
    assigned = assigned.detach().cpu().bool()
    y = graph.y.detach().cpu().long()
    assigned_y = y[assigned]
    if assigned_y.numel() == 0:
        raise ValueError("No assigned labels are available for weighted fine-tuning.")

    unique, counts = torch.unique(assigned_y, return_counts=True)
    weight_by_label = {
        int(label.item()): 1.0 / float(count.item())
        for label, count in zip(unique, counts)
    }

    samples_weight = []
    for node_idx in range(y.shape[0]):
        if assigned[node_idx]:
            samples_weight.append(weight_by_label[int(y[node_idx].item())])
        else:
            samples_weight.append(0.0)
    return samples_weight


def _weighted_node_loader(graph, batch_size):
    dataset = GraphDataset(graph.y)
    sampler = WeightedRandomSampler(
        weights=class_weights(graph),
        num_samples=graph.y.size(0),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )


def _forward_logits(model, graph, patterns, cur_nodes):
    if model.ft_params.get("subgraph_pooling") == "attn":
        _, logits = model(graph, patterns[:, cur_nodes, :])
    else:
        logits = model(graph, patterns[:, cur_nodes, :])
    return logits


def _amp_settings(ft_params, device):
    amp_dtype = torch.float16 if ft_params.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
    use_amp = bool(ft_params.get("use_amp", False)) and (
        device.type == "cuda" or (device.type == "cpu" and amp_dtype == torch.bfloat16)
    )
    return use_amp, amp_dtype


def _train_ft_weighted_validation(
    model,
    train_graph,
    train_patterns,
    val_graph,
    val_patterns,
    optimizer,
    scheduler,
    loss_fn,
    eval_fn,
    scaler=None,
):
    device = model_device(model)
    use_amp, amp_dtype = _amp_settings(model.ft_params, device)
    use_scaler = use_amp and amp_dtype == torch.float16 and scaler is not None
    eval_fn_name = model.ft_params['task_eval_metric']
    bs = model.ft_params["ft_batch_size"]
    log_every = model.ft_params.get("log_every_n_batches", 50)
    total_loss = 0.0
    train_dataloader = _weighted_node_loader(train_graph, bs)
    val_dataloader = _weighted_node_loader(val_graph, bs)

    train_num_batches = len(train_dataloader)
    for batch_idx, (cur_nodes_idx, labels) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        cur_nodes_idx = cur_nodes_idx.to(device)
        labels = labels.to(device)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = _forward_logits(model, train_graph, train_patterns, cur_nodes_idx)
            loss = loss_fn(logits, labels)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        if log_every and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == train_num_batches):
            logger.info(f'Train batch {batch_idx + 1}/{train_num_batches}, Loss: {loss.item()}')

    avg_loss = total_loss / max(1, train_num_batches)

    eval_fn.reset()
    model.eval()
    val_num_batches = len(val_dataloader)
    with torch.no_grad():
        for batch_idx, (cur_nodes_idx, val_labels) in enumerate(val_dataloader):
            cur_nodes_idx = cur_nodes_idx.to(device)
            val_labels = val_labels.to(device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                val_logits = _forward_logits(model, val_graph, val_patterns, cur_nodes_idx)
            eval_fn.update(val_logits, val_labels)
            if log_every and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == val_num_batches):
                logger.info(
                    f'Val batch {batch_idx + 1}/{val_num_batches}, '
                    f'{eval_fn_name}: {eval_fn.compute().mean().item()}'
                )

    epoch_eval = float(eval_fn.compute().mean().item())
    return avg_loss, epoch_eval


def train_ft(
    model,
    graph,
    patterns,
    optimizer,
    scheduler,
    loss_fn,
    eval_fn,
    val_graph=None,
    val_patterns=None,
    scaler=None,
):
    if val_graph is not None and val_patterns is not None:
        return _train_ft_weighted_validation(
            model,
            graph,
            patterns,
            val_graph,
            val_patterns,
            optimizer,
            scheduler,
            loss_fn,
            eval_fn,
            scaler=scaler,
        )

    device = model_device(model)
    use_amp, amp_dtype = _amp_settings(model.ft_params, device)
    use_scaler = use_amp and amp_dtype == torch.float16 and scaler is not None
    eval_fn_name = model.ft_params['task_eval_metric']
    log_every = model.ft_params.get("log_every_n_batches", 50)
    
    bs = model.ft_params['ft_batch_size']
    total_loss = 0.0

    nodes = torch.randperm(graph.x.size(0), device=device)
    # nodes = torch.arange(graph.x.size(0))
    num_batches = (graph.x.size(0) + bs - 1) // bs

    # Metric should be aggregated across the epoch.
    eval_fn.reset()

    for i in range(num_batches):
        optimizer.zero_grad()
        cur_nodes = nodes[i * bs: (i + 1) * bs]

        labels = graph.y[cur_nodes]
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = _forward_logits(model, graph, patterns, cur_nodes)
            loss = loss_fn(logits, labels)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        eval_fn.update(logits, labels)

        if log_every and ((i + 1) % log_every == 0 or (i + 1) == num_batches):
            logger.info(
                f'Batch {i + 1}/{num_batches}, Loss: {loss.item()}; '
                f'{eval_fn_name}: {eval_fn.compute().mean().item()}'
            )
        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, num_batches)
    epoch_eval = float(eval_fn.compute().mean().item())
    return avg_loss, epoch_eval
