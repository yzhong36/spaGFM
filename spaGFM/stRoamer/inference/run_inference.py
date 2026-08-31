import glob
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import torch
from torch_geometric.data import Data

from spaGFM.stRoamer.utils.pyg_graph import data_from_adata_pyg_graph
from spaGFM.stRoamer.utils.rw_sampler import get_patterns
from spaGFM.stRoamer.inference.inference import inference
from spaGFM.stRoamer.utils.helper import set_seed


def inference_helper(graph, model, device, batch_size=256, use_amp=False, amp_dtype="bfloat16"):
    model_params = model['model_state']
    params = model['params']
    params['device'] = device
    params['use_amp'] = use_amp
    params['amp_dtype'] = amp_dtype
    
    set_seed(12345)
    patterns = get_patterns(graph, params, batch_size=8192)
    node_embeddings, subgraph_embeddings = inference(graph, patterns, model_params, params, batch_size=batch_size)
    return node_embeddings, subgraph_embeddings


def resolve_model_paths(model_pattern: str) -> List[str]:
    if os.path.isfile(model_pattern):
        return [model_pattern]

    model_list = sorted(glob.glob(model_pattern))
    if not model_list:
        raise FileNotFoundError(f"No model checkpoints matched: {model_pattern}")
    return model_list


def load_checkpoint(path: str, device: torch.device, weights_only: bool = True) -> dict:
    if weights_only:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            # Older torch versions may not support weights_only.
            return torch.load(path, map_location=device)
    return torch.load(path, map_location=device)


def to_graph_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def build_pyg_graph_from_h5ad_dict(
    graph_dict: dict,
    adata: Any = None,
    path: str = "<adata>",
) -> Data:
    return data_from_adata_pyg_graph(adata, graph_dict, path=path)


def graph_from_h5ad(adata: Any, path: str = "<adata>") -> Data:
    if "pyG_graph" not in adata.uns:
        raise KeyError(f"{path} is missing adata.uns['pyG_graph']")
    if not isinstance(adata.uns["pyG_graph"], dict):
        raise TypeError(
            f"{path} adata.uns['pyG_graph'] must be a dict with 'edge_index' "
            "and either 'x' or 'x_key'"
        )

    return build_pyg_graph_from_h5ad_dict(adata.uns["pyG_graph"], adata, path)


def load_h5ad_graph(path: str) -> Tuple[Any, Data]:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Loading .h5ad input requires the 'scanpy' package.") from exc

    adata = sc.read(path)
    return adata, graph_from_h5ad(adata, path)


def load_graph_inputs(graph_paths: Iterable[str], device: torch.device) -> List[Tuple[Any, str, Any]]:
    graph_inputs: List[Tuple[Any, str, Any]] = []
    for graph_path in graph_paths:
        if graph_path.lower().endswith(".h5ad"):
            adata, graph = load_h5ad_graph(graph_path)
            graph_inputs.append((graph, graph_path, adata))
            continue

        loaded = torch.load(graph_path, map_location=device, weights_only=False)
        for graph in to_graph_list(loaded):
            graph_inputs.append((graph, graph_path, None))
    if not graph_inputs:
        raise ValueError("No graphs loaded from graph paths")
    return graph_inputs


def attach_graph_embeddings(graph: Any, node_emb: torch.Tensor, subgraph_emb: torch.Tensor) -> Any:
    out_graph = graph.clone() if hasattr(graph, "clone") else graph
    out_graph.node_emb = node_emb
    out_graph.subgraph_emb = subgraph_emb
    return out_graph


def walk_length_from_model(model: dict) -> int:
    try:
        return int(model["params"]["walk_length"])
    except KeyError as exc:
        raise KeyError("Model params must contain walk_length to name AnnData embeddings") from exc


def attach_h5ad_embeddings(
    adata: Any,
    node_emb: torch.Tensor,
    subgraph_emb: torch.Tensor,
    walk_length: int,
) -> Any:
    node_emb_np = node_emb.detach().cpu().numpy()
    subgraph_emb_np = subgraph_emb.detach().cpu().numpy()
    if node_emb_np.shape[0] != adata.n_obs or subgraph_emb_np.shape[0] != adata.n_obs:
        raise ValueError(
            "Embedding row count must match adata.n_obs: "
            f"node_emb={node_emb_np.shape[0]}, subgraph_emb={subgraph_emb_np.shape[0]}, n_obs={adata.n_obs}"
        )
    adata.obsm[f"X_node_emb_wl_{walk_length}"] = node_emb_np
    adata.obsm[f"X_subgraph_emb_wl_{walk_length}"] = subgraph_emb_np
    return adata


def run_graph_inference(
    graph: Any,
    model: dict,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
) -> Any:
    node_emb, subgraph_emb = inference_helper(
        graph=graph,
        model=model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    return attach_graph_embeddings(graph, node_emb, subgraph_emb)


def run_h5ad_inference(
    adata: Any,
    graph: Data,
    model: dict,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
) -> Any:
    node_emb, subgraph_emb = inference_helper(
        graph=graph,
        model=model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    return attach_h5ad_embeddings(
        adata,
        node_emb,
        subgraph_emb,
        walk_length=walk_length_from_model(model),
    )


def run_inference(
    input_data: Any,
    model: dict,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
) -> Any:
    """Run inference for a .h5ad/.pt path or an in-memory AnnData/PyG object.

    .h5ad path or AnnData input returns AnnData with embeddings in .obsm.
    .pt path, PyG object, list/tuple of paths, or list/tuple of PyG objects
    returns graph object(s) with embedding attributes.
    """
    if isinstance(input_data, (str, os.PathLike)):
        input_path = str(input_data)
        if input_path.lower().endswith(".h5ad"):
            adata, graph = load_h5ad_graph(input_path)
            return run_h5ad_inference(
                adata=adata,
                graph=graph,
                model=model,
                device=device,
                batch_size=batch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        loaded = torch.load(input_path, map_location=device, weights_only=False)
        outputs = [
            run_graph_inference(
                graph=graph,
                model=model,
                device=device,
                batch_size=batch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            for graph in to_graph_list(loaded)
        ]
        return outputs if isinstance(loaded, (list, tuple)) else outputs[0]

    if hasattr(input_data, "uns") and hasattr(input_data, "obsm"):
        graph = graph_from_h5ad(input_data)
        return run_h5ad_inference(
            adata=input_data,
            graph=graph,
            model=model,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    if isinstance(input_data, (list, tuple)):
        return [
            run_inference(
                item,
                model=model,
                device=device,
                batch_size=batch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            if isinstance(item, (str, os.PathLike)) or (hasattr(item, "uns") and hasattr(item, "obsm"))
            else run_graph_inference(
                graph=item,
                model=model,
                device=device,
                batch_size=batch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            for item in input_data
        ]

    return run_graph_inference(
        graph=input_data,
        model=model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )


def safe_dataset_name(graph: Any, index: int) -> str:
    dataset = getattr(graph, "dataset", None)
    if dataset is None:
        dataset = f"graph_{index:04d}"
    return str(dataset).replace("/", "_").replace(" ", "_")


def output_path_for_result(
    graph: Any,
    graph_path: str,
    adata: Any,
    model_index: str,
    save_path: str,
    output_suffix: str,
    input_index: int,
) -> str:
    dataset_name = safe_dataset_name(graph, input_index)
    if adata is None:
        output_name = f"{model_index}_{dataset_name}_{output_suffix}"
    else:
        input_stem = Path(graph_path).stem.replace("/", "_").replace(" ", "_")
        output_name = f"{model_index}_{dataset_name}_{input_stem}.h5ad"
    return os.path.join(save_path, output_name)


def save_inference_result(result: Any, output_path: str) -> None:
    if hasattr(result, "write_h5ad"):
        result.write_h5ad(output_path)
    else:
        torch.save(result, output_path)
