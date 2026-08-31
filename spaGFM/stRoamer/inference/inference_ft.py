import torch
from typing import Any
from torch.amp import autocast
from torch_geometric.data import Data
import logging
from spaGFM.stRoamer.models.stRoamer_ft import stRoamer_ft
from spaGFM.stRoamer.utils.helper import model_device

logger = logging.getLogger(__name__)

def inference_ft(
    graph: Data,
    pattern: torch.Tensor,
    model_file: str,
    params: dict[str, Any],
    batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:

    # Input validation
    num_nodes_graph = graph.x.size(0)
    num_nodes_pattern = pattern.size(1)
    if num_nodes_graph != num_nodes_pattern:
        raise ValueError(
            f"Node count mismatch: graph has {num_nodes_graph} nodes, "
            f"but pattern has {num_nodes_pattern} nodes. "
            f"Ensure pattern was generated for this graph."
        )
    
    if pattern.dim() != 3:
        raise ValueError(
            f"Pattern must be 3D tensor [n_walks, num_nodes, walk_length+1], "
            f"got shape {pattern.shape}"
        )

    params['mode'] = 'inference'
    model=stRoamer_ft(model_file=model_file,
                      device=params['device'], ft_params=params)
    model.eval()
    logger.info(f'Using the following device during inference: {model_device(model)}')

    bs = batch_size
    nodes = torch.arange(graph.x.size(0))
    num_batches = (graph.x.size(0) + bs - 1) // bs
    device = params["device"] if isinstance(params["device"], torch.device) else torch.device(params["device"])
    amp_dtype = torch.float16 if params.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
    use_amp = bool(params.get("use_amp", False)) and (
        device.type == "cuda" or (device.type == "cpu" and amp_dtype == torch.bfloat16)
    )

    logits_output = []
    if model.ft_params.get("subgraph_pooling") == "attn":
        attn_weights_output = []
        with torch.no_grad():
            for i in range(num_batches):
                cur_nodes = nodes[i * bs: (i + 1) * bs]
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    attn_weights, logits = model(graph, pattern[:,cur_nodes,:])
                
                attn_weights_output.append(attn_weights)
                logits_output.append(logits)

        attn_weights_output = torch.cat(attn_weights_output, dim=0)
        logits_output = torch.cat(logits_output, dim=0)

        return attn_weights_output, logits_output
                    
    else:
        raise ValueError("Currently only attention-based subgraph pooling is supported for inference.")
