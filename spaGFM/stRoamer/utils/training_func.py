from typing import Literal
import torch


def mask_feature(
        x: torch.Tensor,
        p: float = 0.5,
        mode: Literal['row', 'col', 'all'] = 'col',
        fill_value: float = 0.,
        training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask features by zeroing out rows, columns, or individual elements.
    
    Args:
        x: Input tensor of shape [N, D]
        p: Probability of masking (0 to 1)
        mode: 'row' masks entire rows, 'col' masks entire columns, 'all' masks individual elements
        fill_value: Value to fill masked positions
        training: If False, returns input unchanged
        
    Returns:
        Tuple of (masked tensor, boolean mask)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p})')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    if mode not in ('row', 'col', 'all'):
        raise ValueError(f"Mode must be 'row', 'col', or 'all', got {mode}")

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.rand_like(x) >= p

    x = x.masked_fill(~mask, fill_value)
    return x, mask


def mask_patterns(
        patterns: torch.Tensor,
        p: float = 0.5,
        mode: Literal['mask', 'random'] = 'mask',
        training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask patterns tensor by either zeroing or randomizing node indices.
    
    Args:
        patterns: Tensor of shape [h, n, k] containing node indices
        p: Probability of masking each position
        mode: 'mask' to mask with -1s, 'random' to replace with random node indices
        training: Whether in training mode
    
    Returns:
        Tuple of (masked patterns, mask boolean tensor)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 (got {p})')

    if not training or p == 0.0:
        return patterns, torch.ones_like(patterns, dtype=torch.bool)

    # Create mask of shape [h, n, k]
    mask = torch.rand_like(patterns.float()) >= p

    if mode == 'mask':
        # Mask positions with -1
        patterns = patterns.masked_fill(~mask, -1)
    elif mode == 'random':
        # Generate random node indices between 0 and n-1
        n = patterns.size(1)
        random_indices = torch.randint_like(patterns, 0, n)
        # Replace masked positions with random indices
        patterns = torch.where(mask, patterns, random_indices)
    else:
        raise ValueError(f"Mode must be 'mask' or 'random', got {mode}")

    return patterns, mask