import logging
from typing import Any, OrderedDict
import torch
from torch.amp import autocast
from torch_geometric.data import Data
from tqdm.auto import tqdm

from spaGFM.stRoamer.models.stRoamer import stRoamer
from spaGFM.stRoamer.utils.helper import model_device

logger = logging.getLogger(__name__)


def inference(
    graph: Data,
    pattern: torch.Tensor,
    model_params: OrderedDict[str, torch.Tensor],
    params: dict[str, Any],
    batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference to extract node and subgraph embeddings.
    
    Args:
        graph: PyG Data object with node features in graph.x
        pattern: Random walk patterns of shape [n_walks, num_nodes, walk_length+1]
        model_params: Model state dict from checkpoint
        params: Model configuration parameters
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (node_embeddings, subgraph_embeddings)
        
    Raises:
        ValueError: If graph and pattern dimensions don't match
    """
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
    model = stRoamer(params=params)
    model = model.to(params['device'])

    model.load_state_dict(model_params)
    model.eval()
    logger.info(f'Using the following device during inference: {model_device(model)}')

    bs = batch_size
    nodes = torch.arange(graph.x.size(0))
    num_batches = (graph.x.size(0) + bs - 1) // bs
    amp_dtype = torch.float16 if params.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
    use_amp = bool(params.get("use_amp", False)) and (
        params["device"].type == "cuda" or (params["device"].type == "cpu" and amp_dtype == torch.bfloat16)
    )

    node_embeddings_list, subgraph_embeddings_list = [], []
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Embedding cells via spaGFM", unit="batch"):
            cur_nodes = nodes[i * bs: (i + 1) * bs]
            with autocast(device_type=params["device"].type, dtype=amp_dtype, enabled=use_amp):
                node_emb, subgraph_emb = model(graph, pattern[:,cur_nodes,:])
            node_embeddings_list.append(node_emb.detach().cpu())
            subgraph_embeddings_list.append(subgraph_emb.detach().cpu())

    node_embeddings = torch.cat(node_embeddings_list, dim=0)
    subgraph_embeddings = torch.cat(subgraph_embeddings_list, dim=0)

    return node_embeddings, subgraph_embeddings
