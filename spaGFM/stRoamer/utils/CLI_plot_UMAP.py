#!/usr/bin/env python3
"""Plot UMAP from AnnData embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and plot UMAP coordinates from one or more .h5ad embedding files."
    )
    parser.add_argument(
        "figure_save_folder",
        help="Directory where UMAP figures and optional output AnnData are saved.",
    )
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="One or more .h5ad files or directories containing .h5ad files.",
    )
    parser.add_argument(
        "repr_key",
        help="adata.obsm key containing the representation to embed with UMAP.",
    )
    parser.add_argument(
        "--color-keys",
        nargs="+",
        default=("technology", "file", "condition"),
        help="adata.obs keys used for coloring plots.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively for .h5ad files.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional fraction of cells to sample from each file before UMAP.",
    )
    parser.add_argument(
        "--max-cells-per-file",
        type=int,
        default=None,
        help="Optional maximum number of cells sampled from each input file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and UMAP (default: 42).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="UMAP min_dist (default: 0.5).",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="UMAP distance metric (default: euclidean).",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dimensions used before UMAP. Set to 0 to disable (default: 50).",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=10000,
        help="Rows per batch for incremental PCA fit/transform (default: 10000).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="UMAP backend. gpu requires RAPIDS cudf/cuml; auto falls back to cpu.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI and saved DPI (default: 300).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for figure names. Defaults to repr_key.",
    )
    parser.add_argument(
        "--save-adata",
        default=None,
        help="Optional .h5ad path to save combined obs and X_umap.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving.",
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


def sampled_indices(n_obs: int, sample_fraction: float | None, max_cells: int | None, seed: int) -> np.ndarray:
    sample_n = n_obs
    if sample_fraction is not None:
        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError("--sample-fraction must be in (0, 1].")
        sample_n = min(sample_n, max(1, int(round(n_obs * sample_fraction))))
    if max_cells is not None:
        if max_cells < 1:
            raise ValueError("--max-cells-per-file must be >= 1.")
        sample_n = min(sample_n, max_cells)

    if sample_n >= n_obs:
        return np.arange(n_obs)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=sample_n, replace=False))


def read_embeddings(
    h5ad_paths: list[Path],
    repr_key: str,
    color_keys: Iterable[str],
    sample_fraction: float | None,
    max_cells_per_file: int | None,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = []
    obs_frames = []
    requested_keys = list(color_keys)

    for file_index, h5ad_path in enumerate(h5ad_paths):
        adata = sc.read_h5ad(h5ad_path)
        if repr_key not in adata.obsm:
            raise KeyError(f"{h5ad_path} is missing adata.obsm['{repr_key}']")

        indices = sampled_indices(
            adata.n_obs,
            sample_fraction=sample_fraction,
            max_cells=max_cells_per_file,
            seed=random_state + file_index,
        )
        embedding = np.asarray(adata.obsm[repr_key])[indices]
        if embedding.ndim != 2:
            raise ValueError(f"{h5ad_path} adata.obsm['{repr_key}'] must be 2D.")
        embeddings.append(embedding)

        obs = pd.DataFrame(index=[f"{h5ad_path.stem}_{idx}" for idx in indices])
        obs["source_file"] = h5ad_path.name
        for key in requested_keys:
            if key in adata.obs:
                obs[key] = adata.obs.iloc[indices][key].astype(str).to_numpy()
            elif key == "file":
                obs[key] = h5ad_path.name
            else:
                obs[key] = "NA"
        obs_frames.append(obs)

    return np.concatenate(embeddings, axis=0), pd.concat(obs_frames, axis=0)


def iter_batch_slices(n_rows: int, batch_size: int, min_batch_size: int) -> Iterable[slice]:
    start = 0
    while start < n_rows:
        end = min(start + batch_size, n_rows)
        if end < n_rows and n_rows - end < min_batch_size:
            end = n_rows
        yield slice(start, end)
        start = end


def run_pca(
    embedding: np.ndarray,
    n_components: int,
    batch_size: int,
) -> np.ndarray:
    if n_components < 0:
        raise ValueError("--pca-dim must be >= 0.")
    if n_components == 0:
        print("Skipping PCA before UMAP because --pca-dim is 0.")
        return embedding
    if batch_size < 1:
        raise ValueError("--pca-batch-size must be >= 1.")

    n_rows, n_features = embedding.shape
    if n_components >= n_features:
        print(
            f"Skipping PCA before UMAP because input has {n_features} dimensions "
            f"and --pca-dim is {n_components}."
        )
        return embedding
    if n_components > n_rows:
        raise ValueError(
            f"--pca-dim ({n_components}) cannot exceed the number of sampled cells ({n_rows})."
        )

    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError as exc:
        raise ImportError("PCA before UMAP requires scikit-learn.") from exc

    batch_size = max(batch_size, n_components)
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch_slice in iter_batch_slices(n_rows, batch_size, n_components):
        pca.partial_fit(embedding[batch_slice])

    reduced = np.empty((n_rows, n_components), dtype=np.float32)
    for batch_slice in iter_batch_slices(n_rows, batch_size, n_components):
        reduced[batch_slice] = pca.transform(embedding[batch_slice]).astype(np.float32, copy=False)

    print(f"Reduced embeddings from {n_features} to {n_components} dimensions with PCA before UMAP.")
    return reduced


def run_umap_cpu(
    embedding: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "CPU UMAP requires the 'umap-learn' package. It is listed in environment.yaml."
        ) from exc

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embedding)


def run_umap_gpu(
    embedding: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        import cudf
        from cuml.manifold import UMAP
    except ImportError as exc:
        raise ImportError(
            "GPU UMAP requires RAPIDS packages 'cudf' and 'cuml'. "
            "They are optional and are not currently listed in environment.yaml."
        ) from exc

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding_2d = reducer.fit_transform(cudf.DataFrame(embedding))
    return embedding_2d.to_numpy()


def run_umap(
    embedding: np.ndarray,
    backend: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[np.ndarray, str]:
    if backend == "gpu":
        return run_umap_gpu(embedding, n_neighbors, min_dist, metric, random_state), "gpu"
    if backend == "cpu":
        return run_umap_cpu(embedding, n_neighbors, min_dist, metric, random_state), "cpu"

    try:
        return run_umap_gpu(embedding, n_neighbors, min_dist, metric, random_state), "gpu"
    except ImportError as exc:
        print(f"{exc} Falling back to CPU UMAP.", file=sys.stderr)
        return run_umap_cpu(embedding, n_neighbors, min_dist, metric, random_state), "cpu"


def build_plot_adata(obs: pd.DataFrame, embedding_2d: np.ndarray) -> ad.AnnData:
    adata = ad.AnnData(X=np.empty((obs.shape[0], 0)), obs=obs)
    adata.obsm["X_umap"] = embedding_2d
    return adata


def plot_umaps(adata: ad.AnnData, color_keys: Iterable[str], repr_key: str, prefix: str, show: bool) -> None:
    for key in color_keys:
        sc.pl.umap(
            adata,
            color=key,
            title=f"{repr_key} colored by {key}",
            save=f"_{prefix}_colored_by_{key}.png",
            show=show,
        )


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    figure_dir = Path(parsed.figure_save_folder).expanduser()
    figure_dir.mkdir(parents=True, exist_ok=True)

    sc.settings.figdir = str(figure_dir)
    sc.set_figure_params(dpi=parsed.dpi, dpi_save=parsed.dpi)

    h5ad_paths = iter_h5ad_paths(parsed.data_paths, parsed.recursive)
    embedding, obs = read_embeddings(
        h5ad_paths=h5ad_paths,
        repr_key=parsed.repr_key,
        color_keys=parsed.color_keys,
        sample_fraction=parsed.sample_fraction,
        max_cells_per_file=parsed.max_cells_per_file,
        random_state=parsed.random_state,
    )

    embedding = run_pca(
        embedding=embedding,
        n_components=parsed.pca_dim,
        batch_size=parsed.pca_batch_size,
    )

    embedding_2d, backend_used = run_umap(
        embedding=embedding,
        backend=parsed.backend,
        n_neighbors=parsed.n_neighbors,
        min_dist=parsed.min_dist,
        metric=parsed.metric,
        random_state=parsed.random_state,
    )
    print(f"Computed UMAP for {embedding.shape[0]} cells using {backend_used} backend.")

    adata = build_plot_adata(obs, embedding_2d)
    prefix = parsed.output_prefix or parsed.repr_key
    plot_umaps(adata, parsed.color_keys, parsed.repr_key, prefix, parsed.show)

    if parsed.save_adata:
        save_path = Path(parsed.save_adata).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
