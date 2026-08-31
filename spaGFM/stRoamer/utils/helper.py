import os
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def model_device(model: nn.Module) -> torch.device:
    """Get the device of a model by checking its first parameter."""
    device = next(model.parameters()).device
    return device


def get_model_parameters(model: nn.Module) -> float:
    """Get the number of trainable parameters in millions."""
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(num / 1e6, 2)


def get_adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Build AdamW parameter groups with no decay on norms, biases, and special tokens."""
    decay_params = []
    no_decay_params = []
    no_decay_module_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.Embedding,
    )

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_bias = param_name.endswith("bias")
            is_norm_or_embedding = isinstance(module, no_decay_module_types)
            is_scalar_or_vector = param.ndim <= 1
            is_special_token = full_name.endswith("mask_token")

            if is_bias or is_norm_or_embedding or is_scalar_or_vector or is_special_token:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def set_seed(seed: int) -> Optional[torch.Generator]:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        
    Returns:
        A torch.Generator that can be used for DataLoader worker_init_fn
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Return a generator for DataLoader worker seeding
    g = torch.Generator()
    g.manual_seed(seed)
    return g
