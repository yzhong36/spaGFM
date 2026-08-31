import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
from torch_geometric.data import Data
from tqdm import tqdm

from spaGFM.stRoamer.models.stRoamer_ft_tokenizer import stRoamer_ft_tokenizer
from spaGFM.stRoamer.utils.helper import model_device

logger = logging.getLogger(__name__)


def inference_attention(
    graph: Data,
    pattern: torch.Tensor,
    model_file: str,
    params: dict[str, Any],
    ft_params: dict[str, Any],
    batch_size: int = 1,
    num_attn_layers: int = 2,
) -> np.ndarray:

    # Input validation
    if pattern.dim() != 3:
        raise ValueError(
            f"Pattern must be 3D tensor [n_walks, num_nodes, walk_length+1], "
            f"got shape {pattern.shape}"
        )

    num_nodes_graph = graph.x.size(0)
    num_nodes_pattern = pattern.size(1)
    if num_nodes_graph != num_nodes_pattern:
        raise ValueError(
            f"Node count mismatch: graph has {num_nodes_graph} nodes, "
            f"but pattern has {num_nodes_pattern} nodes. "
            f"Ensure pattern was generated for this graph."
        )

    ft_params["mode"] = "inference"
    model = stRoamer_ft_tokenizer(
        model_file=model_file,
        device=ft_params["device"],
        ft_params=ft_params,
    )
    model.eval()
    logger.info(f"Using the following device during inference: {model_device(model)}")

    if model.ft_params.get("subgraph_pooling") != "attn":
        raise ValueError(
            "Currently only attention-based subgraph pooling is supported for inference."
        )

    if not 0 <= num_attn_layers < len(model.subgraph_encoder.encoder):
        raise ValueError(
            f"num_attn_layers must be in [0, {len(model.subgraph_encoder.encoder) - 1}], "
            f"got {num_attn_layers}."
        )

    # Attention extraction currently operates one node at a time.
    bs = batch_size
    nodes = torch.arange(graph.x.size(0))
    num_nodes = nodes.size(0)
    device = ft_params["device"] if isinstance(ft_params["device"], torch.device) else torch.device(ft_params["device"])
    amp_dtype = torch.float16 if ft_params.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
    use_amp = bool(ft_params.get("use_amp", False)) and (
        device.type == "cuda" or (device.type == "cpu" and amp_dtype == torch.bfloat16)
    )

    M = ft_params["n_total_rw"]
    N = num_nodes
    H = params["subgraph_encoder"]["n_heads"]

    count = 0
    cell_gene_correlations = np.zeros((N, H, M), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Embedding attention scores"):
            cur_nodes = nodes[start:start + bs]
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pattern_feature = model(graph, pattern[:, cur_nodes, :])

                x = pattern_feature

                # This mirrors Transformer_Encoder.forward. Keep in sync if encoder internals change.
                for layer, norm in zip(
                    model.subgraph_encoder.encoder,
                    model.subgraph_encoder.encoder_norm,
                ):
                    last_x = x
                    x = layer(norm(x))
                    x = last_x + x
                total_embs = x
                logger.debug("total_embs shape: %s", total_embs.size())

                # Send total_embs to the selected native-attn layer.
                attn_weight = model.subgraph_encoder.encoder[num_attn_layers].self_attn.in_proj_weight
                attn_bias = model.subgraph_encoder.encoder[num_attn_layers].self_attn.in_proj_bias
                qkv = F.linear(total_embs, attn_weight, attn_bias)
                logger.debug("qkv shape: %s", qkv.size())

                # Retrieve q and k from the native-attn wrapper.
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=H)
                q = qkv[:, :, 0, :, :]
                k = qkv[:, :, 1, :, :]

                # Implementation note: raw dot-product scores are left unscaled here
                # to preserve current behavior. PyTorch MultiheadAttention applies
                # sqrt(head_dim) scaling internally.
                attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
                logger.debug("attn_scores shape: %s", attn_scores.size())

                attn_scores = torch.softmax(attn_scores, dim=3)

            outputs = attn_scores.mean(dim=2).detach().cpu().numpy()  # B H M
            cell_gene_correlations[count: count + len(outputs)] = outputs
            count += len(outputs)

    return cell_gene_correlations
