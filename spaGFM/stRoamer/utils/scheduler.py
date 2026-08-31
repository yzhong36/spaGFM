import math
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(optimizer, params):
    if params['scheduler'] == 'none':
        scheduler = None
    elif params['scheduler'] == 'inv_sqrt':
        scheduler = get_inverse_sqrt_scheduler(optimizer, params)
    elif params['scheduler'] == 'cosine':
        scheduler = get_cosine_with_warmup_scheduler(optimizer, params)
    elif params['scheduler'] == 'warmup_stable':
        scheduler = get_warmup_stable_decay_scheduler(optimizer, params)
    else:
        raise NotImplementedError("The scheduler is not implemented.")

    return scheduler

def get_inverse_sqrt_scheduler(optimizer, params):
    warmup_epochs = params['warmup_epochs']
    warmup_steps = warmup_epochs * params['steps_per_epoch']
    d_model = params['hidden_dim']

    def lr_lambda(step):
        if step == 0:
            return 0.0
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda)

def get_cosine_with_warmup_scheduler(optimizer, params):
    """Cosine scheduler with linear warmup.

    When `warmup_steps` is provided, epochs are ignored and the schedule is
    determined ONLY by `warmup_steps` and `train_steps` (or `total_steps`).
    If `warmup_steps` is absent, it falls back to epoch-based computation.
    """

    # If user supplies explicit step counts, ignore epoch definitions entirely.
    if params.get('warmup_steps') is not None:
        # Treat provided warmup_steps and train_steps as GLOBAL budgets across all ranks.
        world_size = max(params.get('world_size', 1), 1)
        warmup_steps_global = int(params.get('warmup_steps', 0))
        train_steps_global = int(params.get('train_steps', 0))
        # Convert to per-rank step counts so LambdaLR step index (local) maps consistently.
        warmup_steps = math.ceil(warmup_steps_global / world_size)
        num_training_steps = math.ceil(train_steps_global / world_size)
    else:
        # Legacy path: derive from epochs & steps_per_epoch (supports grad accumulation, global steps)
        warmup_epochs = params.get('warmup_epochs', 0)
        grad_accum = max(params.get('grad_accum_steps', 1), 1)
        steps_per_epoch_eff = math.ceil(params['steps_per_epoch'] / grad_accum)

        world_size = max(params.get('world_size', 1), 1)
        if params.get('warmup_in_global_steps', False):
            steps_per_epoch_used = steps_per_epoch_eff * world_size
        else:
            steps_per_epoch_used = steps_per_epoch_eff

        warmup_steps = warmup_epochs * steps_per_epoch_used
        num_training_steps = params['epochs'] * steps_per_epoch_used

    # Derive LR values from the optimizer; preserve ratios from params
    base_lr = optimizer.param_groups[0]['lr']
    # Avoid division by zero if params['lr'] is 0 in config
    denom = max(params.get('lr', base_lr), 1e-12)
    scale = base_lr / denom
    warmup_lr = params['warmup_lr'] * scale
    min_lr = params['min_lr'] * scale

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return (warmup_lr + (base_lr - warmup_lr) * (step / max(warmup_steps, 1))) / max(base_lr, 1e-12)
        else:
            # Cosine decay
            total_decay_steps = max(num_training_steps - warmup_steps, 1)
            decay_ratio = (step - warmup_steps) / total_decay_steps
            return (min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_ratio))) / max(base_lr, 1e-12)

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_stable_decay_scheduler(optimizer, params):
    """Scheduler: linear warmup -> stable plateau -> linear decay.

    Explicit step mode (preferred): Provide global `warmup_steps`, `stable_steps`, and `train_steps`.
    Epoch mode (fallback): Uses `warmup_epochs`, `stable_epochs`, `epochs`, and `steps_per_epoch` (+ grad accumulation and world_size flags) similar to cosine scheduler style.
    All provided *_steps are treated as GLOBAL and divided by world_size to obtain local steps.
    Decay is linear from base_lr down to min_lr after plateau.
    """

    world_size = max(params.get('world_size', 1), 1)

    if params.get('warmup_steps') is not None and params.get('train_steps') is not None:
        # Explicit step mode
        warmup_steps_global = int(params.get('warmup_steps', 0))
        stable_steps_global = int(params.get('stable_steps', 0))  # plateau duration after warmup
        train_steps_global = int(params.get('train_steps', 0))

        warmup_steps = math.ceil(warmup_steps_global / world_size)
        stable_steps = math.ceil(stable_steps_global / world_size)
        num_training_steps = math.ceil(train_steps_global / world_size)
    else:
        # Epoch-derived mode
        grad_accum = max(params.get('grad_accum_steps', 1), 1)
        steps_per_epoch_eff = math.ceil(params['steps_per_epoch'] / grad_accum)

        warmup_epochs = params.get('warmup_epochs', 0)
        stable_epochs = params.get('stable_epochs', 0)
        total_epochs = params['epochs']

        if params.get('warmup_in_global_steps', False):
            steps_per_epoch_used = steps_per_epoch_eff * world_size
        else:
            steps_per_epoch_used = steps_per_epoch_eff

        warmup_steps = warmup_epochs * steps_per_epoch_used
        stable_steps = stable_epochs * steps_per_epoch_used
        num_training_steps = total_epochs * steps_per_epoch_used

    # LR scaling similar to cosine scheduler
    base_lr = optimizer.param_groups[0]['lr']
    denom = max(params.get('lr', base_lr), 1e-12)
    scale = base_lr / denom
    warmup_lr = params.get('warmup_lr', 0.0) * scale
    min_lr = params.get('min_lr', 0.0) * scale

    plateau_start = warmup_steps
    decay_start = warmup_steps + stable_steps

    decay_steps = max(num_training_steps - decay_start, 1)

    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return (warmup_lr + (base_lr - warmup_lr) * (step / max(warmup_steps, 1))) / max(base_lr, 1e-12)
        # Plateau
        if step < decay_start:
            return 1.0  # keep at base lr
        # Linear decay
        progress = (step - decay_start) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        current_lr = base_lr - (base_lr - min_lr) * progress
        return current_lr / max(base_lr, 1e-12)

    return LambdaLR(optimizer, lr_lambda)