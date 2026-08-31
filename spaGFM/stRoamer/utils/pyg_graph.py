import numpy as np
import torch
from torch_geometric.data import Data


def _graph_x_from_adata(adata, graph_dict, path="<adata>"):
    if "x" in graph_dict:
        return graph_dict["x"], graph_dict.get("x_key")

    if "x_key" not in graph_dict:
        raise KeyError(
            f"{path} adata.uns['pyG_graph'] must contain 'x' or 'x_key' "
            "to build Data.x"
        )

    x_key = graph_dict["x_key"]
    if not hasattr(adata, "obsm") or x_key not in adata.obsm:
        raise KeyError(
            f"{path} adata.obsm is missing x_key={x_key!r} required by "
            "adata.uns['pyG_graph']"
        )
    return adata.obsm[x_key], x_key


def data_from_adata_pyg_graph(adata, graph_dict, path="<adata>"):
    if "edge_index" not in graph_dict:
        raise KeyError(f"{path} adata.uns['pyG_graph'] must contain 'edge_index'")

    x, x_key = _graph_x_from_adata(adata, graph_dict, path=path)
    data = Data(
        x=torch.as_tensor(np.asarray(x), dtype=torch.float),
        edge_index=torch.as_tensor(graph_dict["edge_index"], dtype=torch.long),
    )
    if x_key is not None:
        data.x_key = x_key
    return data
