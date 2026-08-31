#!/usr/bin/env python3
"""Convert AnnData files to PyG graph files for spaGFM fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial import Delaunay


from spaGFM.stRoamer.utils.graph_build import edge_index_from_delaunay, filter_edge, pyg_obj


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .h5ad files into .pt PyG graph files for fine-tuning."
    )
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="One or more .h5ad files or directories containing .h5ad files.",
    )
    parser.add_argument(
        "--label-key",
        required=True,
        help="adata.obs column containing labels.",
    )
    parser.add_argument(
        "--repr-key",
        required=True,
        help="adata.obsm key containing node features.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .pt outputs. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for .h5ad files.",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="adata.obsm key containing spatial coordinates (default: spatial).",
    )
    parser.add_argument(
        "--coord-keys",
        nargs=2,
        default=("x_pixel", "y_pixel"),
        metavar=("X_KEY", "Y_KEY"),
        help="adata.obs coordinate columns used if --spatial-key is unavailable.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Quantile threshold for Delaunay edge filtering (default: 0.99).",
    )
    parser.add_argument(
        "--unassigned-label",
        default="UNASSIGNED",
        help="Label value treated as unassigned for graph.is_assigned (default: UNASSIGNED).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )
    parser.add_argument(
        "--write-updated-h5ad",
        action="store_true",
        help="Write is_assigned and numeric label codes back to the input .h5ad.",
    )
    return parser.parse_args(args)


def iter_h5ad_paths(paths: Iterable[str], recursive: bool) -> list[Path]:
    h5ad_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.suffix == ".h5ad":
                h5ad_paths.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.h5ad" if recursive else "*.h5ad"
            h5ad_paths.extend(sorted(path.glob(pattern)))
            continue
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if not h5ad_paths:
        raise FileNotFoundError("No .h5ad files found.")
    return sorted(dict.fromkeys(h5ad_paths))


def spatial_coordinates(adata, spatial_key: str, coord_keys: tuple[str, str]) -> np.ndarray:
    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key])
    else:
        missing = [key for key in coord_keys if key not in adata.obs]
        if missing:
            raise KeyError(
                f"Missing adata.obsm['{spatial_key}'] and obs coordinate column(s): {missing}"
            )
        coords = adata.obs.loc[:, list(coord_keys)].to_numpy()

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must be a 2D array with at least two columns.")
    if coords.shape[0] != adata.n_obs:
        raise ValueError("Spatial coordinate row count must match adata.n_obs.")
    if coords.shape[0] < 3:
        raise ValueError("At least three observations are required for Delaunay graph construction.")
    return coords[:, :2].astype(float, copy=False)


def label_codes(labels: pd.Series) -> tuple[torch.Tensor, dict[int, str]]:
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(int)
        unique_values = sorted(int(value) for value in values.unique())
        if unique_values == list(range(len(unique_values))):
            mapping = {int(value): str(value) for value in unique_values}
            return torch.tensor(values.to_numpy(), dtype=torch.long), mapping

        value_to_code = {value: idx for idx, value in enumerate(unique_values)}
        codes = values.map(value_to_code)
        mapping = {idx: str(value) for value, idx in value_to_code.items()}
        return torch.tensor(codes.to_numpy(), dtype=torch.long), mapping

    categories = pd.Categorical(labels.astype(str))
    codes = torch.tensor(categories.codes, dtype=torch.long)
    mapping = {int(idx): str(label) for idx, label in enumerate(categories.categories)}
    return codes, mapping


def build_graph_from_adata(
    adata,
    label_key: str,
    repr_key: str,
    spatial_key: str,
    coord_keys: tuple[str, str],
    threshold: float,
    unassigned_label: str,
):
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs is missing label key: {label_key}")
    if repr_key not in adata.obsm:
        raise KeyError(f"adata.obsm is missing representation key: {repr_key}")

    features = np.asarray(adata.obsm[repr_key])
    if features.shape[0] != adata.n_obs:
        raise ValueError("Feature row count must match adata.n_obs.")

    coords = spatial_coordinates(adata, spatial_key, coord_keys)
    delaunay = filter_edge(Delaunay(coords), threshold=threshold)
    edge_index = edge_index_from_delaunay(delaunay.simplices)

    graph = pyg_obj(torch.tensor(features, dtype=torch.float), edge_index)
    graph.niche, graph.label_mapping = label_codes(adata.obs[label_key])
    graph.y = graph.niche
    graph.is_assigned = torch.tensor(
        (adata.obs[label_key].astype(str) != str(unassigned_label)).to_numpy(),
        dtype=torch.bool,
    )
    graph.source_label_key = label_key
    graph.source_repr_key = repr_key
    return graph


def output_path_for(input_path: Path, output_dir: str | None) -> Path:
    base_dir = Path(output_dir).expanduser() if output_dir else input_path.parent
    return base_dir / f"{input_path.stem}.pt"


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    output_dir = Path(parsed.output_dir).expanduser() if parsed.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for h5ad_path in iter_h5ad_paths(parsed.data_paths, parsed.recursive):
        out_path = output_path_for(h5ad_path, parsed.output_dir)
        if out_path.exists() and not parsed.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace: {out_path}")

        adata = sc.read_h5ad(h5ad_path)
        graph = build_graph_from_adata(
            adata=adata,
            label_key=parsed.label_key,
            repr_key=parsed.repr_key,
            spatial_key=parsed.spatial_key,
            coord_keys=tuple(parsed.coord_keys),
            threshold=parsed.threshold,
            unassigned_label=parsed.unassigned_label,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, out_path)

        if parsed.write_updated_h5ad:
            adata.obs["is_assigned"] = graph.is_assigned.detach().cpu().numpy()
            adata.obs[f"{parsed.label_key}_code"] = graph.niche.detach().cpu().numpy()
            adata.uns[f"{parsed.label_key}_mapping"] = graph.label_mapping
            adata.write_h5ad(h5ad_path)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
