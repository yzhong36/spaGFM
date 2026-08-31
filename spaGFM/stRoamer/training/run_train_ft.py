from spaGFM.stRoamer.utils.helper import set_seed
from spaGFM.stRoamer.models.stRoamer_ft import stRoamer_ft
from spaGFM.stRoamer.training.train_ft import train_ft
from spaGFM.stRoamer.utils.rw_sampler import get_patterns
import logging
import torch
import torch.nn.functional as F
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torchmetrics import F1Score, MeanSquaredError
import os

logger = logging.getLogger(__name__)


def _is_better_metric(metric, best_metric, task_type):
    if best_metric is None:
        return True
    if task_type == "regression":
        return metric < best_metric
    return metric > best_metric


def _save_ft_checkpoint(save_file, epoch, model, optimizer):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "params": model.pretrain_params,
        "ft_params": model.ft_params,
    }, save_file)


def run_ft(model_file, graph=None, ft_params=None, train_graph=None, val_graph=None, logger_override=None):
    if ft_params is None:
        raise ValueError("ft_params must be provided.")
    if train_graph is None:
        train_graph = graph
    if train_graph is None:
        raise ValueError("A training graph must be provided.")

    run_logger = logger_override or logger
    set_seed(2025)
    
    epochs = ft_params['epochs']
    device=ft_params['device']
    lr = ft_params['lr']
    save_epoch = ft_params['save_epoch_wise_size']
    save_path = ft_params['save_path']

    model=stRoamer_ft(model_file=model_file,
                      device=device, ft_params=ft_params)
    model.ft_params["n_total_rw"] = ft_params.get("n_total_rw", model.pretrain_params["n_total_rw"])
    model.ft_params["walk_length"] = ft_params.get("walk_length", model.pretrain_params["walk_length"])
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    use_amp = bool(model.ft_params.get("use_amp", False))
    amp_dtype = model.ft_params.get("amp_dtype", "bfloat16")
    scaler = GradScaler(enabled=use_amp and amp_dtype == "float16") if use_amp else None
    if use_amp:
        run_logger.info(f"Mixed precision fine-tuning enabled with dtype: {amp_dtype}")

    output_dim = model.ft_params['output_dim']
    if model.ft_params['task_type'] == 'classification':
        loss_fn = F.cross_entropy
        eval_fn = F1Score(task="multiclass", num_classes=output_dim, average='macro').to(device)
        eval_fn_name = 'F1 Macro'
        model.ft_params['task_eval_metric'] = eval_fn_name
    elif model.ft_params['task_type'] == 'regression':
        loss_fn = F.mse_loss
        eval_fn = MeanSquaredError(num_outputs=output_dim).to(device)
        eval_fn_name = 'MSE'
        model.ft_params['task_eval_metric'] = eval_fn_name
    else:
        raise ValueError(f"Unknown task type: {model.ft_params['task_type']}")

    train_patterns = get_patterns(train_graph, model.ft_params, batch_size=8192)
    model.ft_params["train_random_walks"] = train_patterns
    train_patterns = train_patterns.to(device, non_blocking=True)

    val_patterns = None
    if val_graph is not None:
        val_patterns = get_patterns(val_graph, model.ft_params, batch_size=8192)
        model.ft_params["val_random_walks"] = val_patterns
        val_patterns = val_patterns.to(device, non_blocking=True)

    model.mlp_decoder.train()

    # Ensure graph data lives on the same device as the model.
    # This prevents slow CPU indexing and implicit CPU->GPU copies.
    train_graph = train_graph.to(device)
    if val_graph is not None:
        val_graph = val_graph.to(device)

    best_eval = None
    for epoch in range(epochs):
        avg_loss, epoch_eval = train_ft(
            model,
            train_graph,
            train_patterns,
            optimizer,
            None,
            loss_fn,
            eval_fn,
            val_graph=val_graph,
            val_patterns=val_patterns,
            scaler=scaler,
        )
        metric_prefix = "Val " if val_graph is not None else ""
        run_logger.info(f'Epoch {epoch + 1} done. Avg Loss: {avg_loss}, {metric_prefix}{eval_fn_name}: {epoch_eval}')

        if save_epoch and (epoch + 1) % save_epoch == 0:
            os.makedirs(save_path, exist_ok=True)
            save_file = f"{save_path}/epoch_{epoch + 1}.pt"
            run_logger.info(f"Saving model to {save_file}")
            _save_ft_checkpoint(save_file, epoch + 1, model, optimizer)

        if val_graph is not None and _is_better_metric(epoch_eval, best_eval, model.ft_params['task_type']):
            best_eval = epoch_eval
            os.makedirs(save_path, exist_ok=True)
            save_file = f"{save_path}/best_model.pt"
            run_logger.info(f"New best model with {eval_fn_name}: {best_eval}. Saving to {save_file}")
            _save_ft_checkpoint(save_file, epoch + 1, model, optimizer)
