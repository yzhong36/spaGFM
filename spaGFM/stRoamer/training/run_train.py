import logging
import os
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch_geometric.loader import DataLoader
import wandb
import time
from datetime import timedelta
import os.path as osp
from spaGFM.stRoamer.models.stRoamer import stRoamer
from spaGFM.stRoamer.training.train import train
from spaGFM.stRoamer.utils.helper import get_adamw_param_groups, get_model_parameters
from spaGFM.stRoamer.utils.online_eval import build_online_eval_state
from spaGFM.stRoamer.utils.scheduler import get_scheduler
from spaGFM.stRoamer.utils.helper import set_seed

logger = logging.getLogger(__name__)


def _load_checkpoint(checkpoint_path, device):
    if not checkpoint_path:
        raise ValueError("adaptation mode requires --pretrained_model_path")
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pretrained model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def _load_adaptation_model_inputs(checkpoint_path, device):
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if "model_state" not in checkpoint or "params" not in checkpoint:
        raise KeyError("Adaptation checkpoint must contain model_state and params")

    model_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state"].items()
    }
    model_params = checkpoint["params"]
    return model_state, model_params


def _valid_adaptation_override(value):
    return value is not None and value != {}


def run(rank: int, world_size: int, params, input_graph):
    set_seed(2025)

    # Initialize distributed process group and set device early
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(params.get('master_port', '12345'))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
        device_id=torch.device(f"cuda:{rank}")
)

    pretrained_model_state = None
    if params.get("mode") == "adaptation":
        pretrained_path = params.get("pretrained_model_path")
        current_params = dict(params)
        pretrained_model_state, checkpoint_params = _load_adaptation_model_inputs(
            pretrained_path,
            torch.device(f"cuda:{rank}"),
        )
        params.clear()
        params.update(checkpoint_params)
        params.update({
            key: value
            for key, value in current_params.items()
            if _valid_adaptation_override(value)
        })
        params["mode"] = "adaptation"

    # Initialize wandb per-process; enable only on rank 0
    if rank == 0:
        wandb.init(
            project=params['project_name'],
            name=params['run_name'],
            config=params,
            mode=params.get("wandb_mode", "online"),
        )
    else:
        wandb.init(mode="disabled")

    n_cpu = int(params.get('n_cpu', 0) or 0)
    if (
        bool(getattr(input_graph, 'on_the_fly', False))
        and hasattr(input_graph, 'warm_on_the_fly_graph_cache')
        and n_cpu == 0
    ):
        start_preload = time.time()
        preload_device = None
        if str(getattr(input_graph, 'on_the_fly_device', '')).startswith('cuda'):
            preload_device = torch.device(f"cuda:{rank}")
        n_preloaded = input_graph.warm_on_the_fly_graph_cache(rw_device=preload_device, shard_idx=rank)
        if rank == 0:
            logger.info(
                "Preloaded %d on-the-fly source graph(s) in %.2fs before DataLoader start",
                n_preloaded,
                time.time() - start_preload,
            )
    
    # Build DataLoader with DistributedSampler AFTER process group init
    sampler_shuffle = not bool(getattr(input_graph, 'on_the_fly', False))
    sampler = DistributedSampler(input_graph, num_replicas=world_size, rank=rank, shuffle=bool(sampler_shuffle))
    graph = DataLoader(
        input_graph,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['n_cpu'],
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True if params['n_cpu'] > 0 else False,
    )

    params['steps_per_epoch'] = len(graph)
    logger.info(f'Rank {rank}: Steps per epoch: {params["steps_per_epoch"]}')

    model = stRoamer(params).to(rank)
    if params["mode"] == "adaptation":
        model.load_state_dict(pretrained_model_state)
        if rank == 0:
            logger.info(
                "Loaded pretrained model for adaptation from %s using checkpoint architecture params",
                params.get("pretrained_model_path"),
            )

    model = DistributedDataParallel(model, device_ids=[rank])
    wrapped = model.module
    eval_state = build_online_eval_state(params, device=torch.device(f"cuda:{rank}"), logger=logger)

    num_params = get_model_parameters(wrapped)
    num_params_encoder1 = get_model_parameters(wrapped.rw_encoder) 
    num_params_encoder2 = get_model_parameters(wrapped.subgraph_encoder)
    num_params_decoder = get_model_parameters(wrapped.subgraph_decoder) + get_model_parameters(wrapped.linear_decoder)
    logger.info(f'The number of parameters: {num_params}M, RW_Encoder: {num_params_encoder1}M, Subgraph_Encoder: {num_params_encoder2}M, Subgraph_Decoder: {num_params_decoder}M')

     # Scale LR for multi-GPU (linear scaling rule) and gradient accumulation
    grad_accum = max(params.get('grad_accum_steps', 1), 1)
    params['world_size'] = world_size
    effective_lr = params['lr'] * world_size * grad_accum if params.get('scale_lr_by_world_size', True) else params['lr']
    if params.get('adamw_decay_all_params', False):
        param_groups = [
            {
                "params": [p for p in wrapped.parameters() if p.requires_grad],
                "weight_decay": params['weight_decay'],
            }
        ]
    else:
        param_groups = get_adamw_param_groups(wrapped, params['weight_decay'])

    if rank == 0:
        decay_param_count = sum(p.numel() for p in param_groups[0]["params"])
        no_decay_param_count = (
            sum(p.numel() for p in param_groups[1]["params"])
            if len(param_groups) > 1
            else 0
        )
        logger.info(
            "AdamW parameter groups: decay=%d params, no_decay=%d params, decay_all=%s",
            decay_param_count,
            no_decay_param_count,
            params.get('adamw_decay_all_params', False),
        )

    optimizer = torch.optim.AdamW(
        param_groups, lr=effective_lr,
        betas=(params['opt_beta1'], params['opt_beta2']), eps=params['opt_eps']
    )
    scheduler = get_scheduler(optimizer, params)
    
    # Initialize GradScaler for mixed precision training
    use_amp = params.get('use_amp', False)
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if use_amp and rank == 0:
        amp_dtype = params.get('amp_dtype', 'float16')
        logger.info(f'Mixed precision training enabled with dtype: {amp_dtype}')
    
    # Initialize current_step tracking
    params['current_step'] = 0
    
    training_times = []
    for epoch in range(1, params['epochs'] + 1):
        # Ensure each replica shuffles differently each epoch
        params['current_epoch'] = epoch
        graph.sampler.set_epoch(epoch)
        start_time = time.time()
        loss = train(graph, model, optimizer, scheduler=scheduler, params=params, scaler=scaler, eval_state=eval_state)
        # loss = {'train': 0, 'val': 0, 'test': 0}
        training_time = time.time() - start_time
        training_times.append(training_time)

        # Average losses across GPUs if needed
        if world_size > 1:
            loss_t = {}
            for k, v in loss.items():
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=rank)
                if not t.is_cuda:
                    t = t.to(rank)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                loss_t[k] = t
            # Convert back to Python floats for logging/printing
            loss = {k: float(v.item()) for k, v in loss_t.items()}

        if rank == 0:
            logger.info(
                f"Train loss: {loss['train']:.4f}, "
                f"Val loss: {loss['val']:.4f}, "
                f"Test loss: {loss['test']:.4f}"
            )
            wandb.log({
                "training dynamics/train_loss": loss['train'],
                "training dynamics/val_loss": loss['val'],
                "training dynamics/test_loss": loss['test'],
                "time/duration_training": training_time,
            })

        if rank == 0 and params['save_epoch_wise_size'] != 0 and epoch % params['save_epoch_wise_size'] == 0:
            # Construct checkpoint path
            checkpoint_dir = params['save_path']
            checkpoint_path = osp.join(checkpoint_dir, f"epoch_{epoch}.pt")
            
            # Create directory if needed
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model checkpoint
            # Save underlying module state dict for portability
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'params': params,
            }, checkpoint_path)
            logger.info(f'Model checkpoint saved at epoch {epoch} to {checkpoint_path}')
        
        if params['current_step'] >= params['train_steps']:
            break

    dist.destroy_process_group()

    wandb.finish()
        
