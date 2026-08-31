import logging
import os
import os.path as osp

import torch
import torch.distributed as dist
import wandb
from torch.amp import autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from spaGFM.stRoamer.utils.helper import model_device
from spaGFM.stRoamer.utils.online_eval import build_online_eval_state, run_online_evaluation_step


logger = logging.getLogger(__name__)


def train(graph, model, optimizer, scheduler, params, scaler=None, eval_state=None):
    if params["mode"] not in {"pretrain", "adaptation"}:
        raise ValueError("Invalid mode for training; expected 'pretrain' or 'adaptation'")

    total_rw_count = params["n_total_rw"]
    walk_length = params["walk_length"]
    grad_clip = params["grad_clip"]
    ema_update_every = params["ema_update_every"]
    ema_alpha = params["ema_alpha"]
    grad_accum_steps = max(params.get("grad_accum_steps", 1), 1)
    amp_dtype = torch.float16 if params.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    use_amp = params.get("use_amp", False)
    use_scaler = use_amp and amp_dtype == torch.float16 and scaler is not None

    world_size = params["world_size"]
    current_epoch = params["current_epoch"]
    save_step_wise = params["save_step_wise"]
    save_step_wise_size = params["save_step_wise_size"]
    train_steps = params["train_steps"]

    save_path = params["save_path"]
    model.train()
    device = model_device(model)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    core_model = model.module if hasattr(model, "module") else model
    is_distributed = dist.is_available() and dist.is_initialized()

    if not isinstance(graph, DataLoader):
        raise ValueError("Invalid graph format")

    eval_every_steps = params.get("eval_every_steps")
    if eval_state is None:
        eval_state = build_online_eval_state(params, device, logger=logger)
    run_eval = eval_state is not None

    total_loss = 0.0
    stopped_for_train_steps = False
    optimizer.zero_grad()

    for i, subgraph in enumerate(graph):
        x_dev = subgraph.x.to(device, non_blocking=True)
        map_dev = subgraph.map_index.to(device, non_blocking=True)
        current_graph = Data(x=x_dev[map_dev])
        current_nodes = torch.arange(subgraph.map_index.numel(), device=device).view(
            total_rw_count, -1, walk_length + 1
        )

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, subgraph_emb = model(graph=current_graph, nodes=current_nodes)
            scaled_loss = loss / grad_accum_steps

        total_loss += loss.item()

        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        if i % ema_update_every == 0:
            core_model.ema_update(alpha=ema_alpha)

        log_loss = loss.detach()
        log_subgraph_emb_std = subgraph_emb.detach().float().std(dim=0)
        if is_distributed:
            log_loss = log_loss.detach().float().to(device)
            dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
            log_loss = log_loss.item()

            log_subgraph_emb_std = log_subgraph_emb_std.float().to(device)
            dist.all_reduce(log_subgraph_emb_std, op=dist.ReduceOp.AVG)
            log_subgraph_emb_std = log_subgraph_emb_std.mean().item()
        else:
            log_loss = float(loss.item())
            log_subgraph_emb_std = float(log_subgraph_emb_std.mean().item())

        if rank == 0:
            log_payload = {
                "training dynamics/step-wise train_loss": log_loss,
                "training dynamics/step-wise subgraph_emb_std": log_subgraph_emb_std,
            }
            try:
                log_payload["training dynamics/step-wise lr"] = scheduler.get_last_lr()[0]
            except Exception:
                pass
            wandb.log(log_payload)

        current_step = world_size * (i + 1) + (current_epoch - 1) * len(graph) * world_size

        if run_eval and current_step % eval_every_steps == 0:
            run_online_evaluation_step(
                core_model=core_model,
                params=params,
                device=device,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
                current_step=current_step,
                eval_state=eval_state,
                logger=logger,
            )

        params["current_step"] = current_step
        if save_step_wise and current_step % save_step_wise_size == 0 and rank == 0:
            checkpoint_dir = save_path
            checkpoint_path = osp.join(checkpoint_dir, f"step_{current_step}.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "step": current_step,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "params": params,
                },
                checkpoint_path,
            )
            logger.info("Model checkpoint saved at step %d to %s", current_step, checkpoint_path)

        stop_flag_local = 1 if current_step >= train_steps else 0
        if is_distributed:
            stop_tensor = torch.tensor(stop_flag_local, device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)
            stop_flag = stop_tensor.item() > 0
        else:
            stop_flag = bool(stop_flag_local)

        if stop_flag:
            stopped_for_train_steps = True
            if rank == 0:
                logger.info("Reached max steps, terminating training.")
            if is_distributed:
                dist.barrier()
            try:
                data_iter = getattr(graph, "_iterator", None)
                if data_iter is not None and hasattr(data_iter, "_shutdown_workers"):
                    data_iter._shutdown_workers()
            except Exception:
                pass
            break

    if not stopped_for_train_steps and (i + 1) % grad_accum_steps != 0:
        if use_scaler:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    total_loss /= (i + 1)
    return {"train": total_loss, "val": 0, "test": 0}
