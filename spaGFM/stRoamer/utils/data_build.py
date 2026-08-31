import numpy as np
import torch
from scipy import sparse
from scipy.spatial import Delaunay
from spaGFM.stRoamer.utils.graph_build import filter_edge, edge_index_from_delaunay, pyg_obj
from spaGFM.stRoamer.utils.helper import set_seed

import scanpy as sc
import scgpt.tasks as scgpt_tasks
from concept import scConcept

def _standardize_spatial_adata(adata, coord_x, coord_y):
    # Check if the adata object has the required attributes
    if not hasattr(adata, "obsm") or "spatial" not in adata.obsm:
        if coord_x is None or coord_y is None:
            raise ValueError("coord_x and coord_y must be provided.")
        spatial_coords = np.column_stack([adata.obs[coord_x].values, adata.obs[coord_y].values])
        adata.obsm["spatial"] = spatial_coords
    
    return adata

def _copy_matrix(matrix):
    if sparse.issparse(matrix):
        return matrix.copy()
    return np.array(matrix, copy=True)


def _as_supported_expression_matrix(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.copy()
        if not sparse.isspmatrix_csr(matrix):
            matrix = sparse.csr_matrix(matrix)
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("Expression matrix must contain numeric values.")
        return matrix

    matrix = np.asarray(matrix)
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("Expression matrix must contain numeric values.")
    return np.array(matrix, copy=True)


def _matrix_values(matrix):
    return matrix.data if sparse.issparse(matrix) else np.asarray(matrix)


def _validate_finite_matrix(matrix):
    values = _matrix_values(matrix)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Expression matrix contains NaN or infinite values.")


def _validate_expression_matrix(matrix):
    _validate_finite_matrix(matrix)
    values = _matrix_values(matrix)
    if values.size and np.min(values) < -1e-8:
        raise ValueError("Expression matrix contains negative values.")


def _matrix_max(matrix):
    max_value = matrix.max()
    if sparse.issparse(max_value):
        max_value = max_value.toarray()
    return float(np.asarray(max_value).max())


def _expm1_matrix(matrix):
    matrix = _copy_matrix(matrix)
    if sparse.issparse(matrix):
        matrix.data = np.expm1(matrix.data)
        matrix.eliminate_zeros()
        return matrix
    return np.expm1(matrix)


def _scale_genes(matrix, scale, offset=None):
    if sparse.issparse(matrix):
        scaled = matrix.multiply(scale)
        if offset is None:
            return scaled.tocsr()
        return scaled.toarray() + offset

    scaled = np.asarray(matrix) * scale
    if offset is not None:
        scaled = scaled + offset
    return scaled


def _var_array(adata, key):
    if adata is None or key not in adata.var:
        return None
    values = adata.var[key].to_numpy(dtype=np.float64)
    if values.shape[0] != adata.n_vars or not np.all(np.isfinite(values)):
        return None
    return values


def _has_unit_gene_std(matrix):
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        gene_std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    else:
        gene_std = np.asarray(matrix).std(axis=0)

    gene_std = gene_std[np.isfinite(gene_std)]
    if gene_std.size == 0:
        return False
    return 0.8 <= float(np.median(gene_std)) <= 1.2


def _nonzero_values(matrix, max_values=100000):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        return values

    values = values[np.isfinite(values)]
    values = values[values != 0]
    if values.size > max_values:
        step = max(1, values.size // max_values)
        values = values[::step][:max_values]
    return values


def _is_count_like(matrix):
    _validate_finite_matrix(matrix)
    values = _matrix_values(matrix)
    if values.size and np.min(values) < -1e-8:
        return False

    values = _nonzero_values(matrix)
    if values.size == 0:
        return True
    return np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0)


def _is_log1p_like(matrix, adata=None):
    _validate_finite_matrix(matrix)
    values = _matrix_values(matrix)
    if values.size and np.min(values) < -1e-8:
        return False

    values = _nonzero_values(matrix)
    if values.size == 0:
        return False
    if _matrix_max(matrix) > 30.0:
        return False

    integer_like = np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0)
    return (not integer_like) or (adata is not None and "log1p" in adata.uns)


def _to_count_scale(matrix, adata=None):
    _validate_finite_matrix(matrix)

    if _is_count_like(matrix):
        return matrix

    std = _var_array(adata, "std")
    mean = _var_array(adata, "mean")

    if std is not None and np.all(std > 0) and _has_unit_gene_std(matrix):
        log_matrix = _scale_genes(matrix, std)
        if _is_log1p_like(log_matrix, adata=adata):
            return _expm1_matrix(log_matrix)

        if mean is not None:
            log_matrix = _scale_genes(matrix, std, offset=mean)
            if _is_log1p_like(log_matrix, adata=adata):
                return _expm1_matrix(log_matrix)

    if _is_log1p_like(matrix, adata=adata):
        return _expm1_matrix(matrix)

    if std is not None and np.all(std > 0):
        log_matrix = _scale_genes(matrix, std)
        if _is_log1p_like(log_matrix, adata=adata):
            return _expm1_matrix(log_matrix)

        if mean is not None:
            log_matrix = _scale_genes(matrix, std, offset=mean)
            if _is_log1p_like(log_matrix, adata=adata):
                return _expm1_matrix(log_matrix)

    raise ValueError(
        "Could not convert expression matrix to count scale. Provide raw counts "
        "with counts_layer, layers['counts'], or layers['raw_counts']; otherwise "
        "adata.X must look like counts, log1p, or scaled log1p data."
    )


def _set_preprocess_input(adata_copy, adata, counts_layer=None):
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(f"counts_layer={counts_layer!r} is not present in adata.layers.")
        adata_copy.X = _copy_matrix(adata.layers[counts_layer])
        return True

    elif "counts" in adata.layers:
        adata_copy.X = _copy_matrix(adata.layers["counts"])
        return True

    elif "raw_counts" in adata.layers:
        adata_copy.X = _copy_matrix(adata.layers["raw_counts"])
        return True

    return False


def _preprocess_adata(
    adata,
    min_counts=10,
    min_cells=5,
    counts_layer=None,
    n_top_genes=5000,
):
    """Filter on count-scale data, then normalize/log-transform ``adata.X``.

    ``counts_layer`` is used first when supplied. Otherwise the function tries
    ``layers["counts"]``, ``layers["raw_counts"]``, then ``adata.X``. If it must
    use ``adata.X``, it converts count-like, log1p, or scaled log1p input back
    to count scale before filtering.
    """
    adata_copy = adata.copy()
    input_is_counts = _set_preprocess_input(adata_copy, adata, counts_layer=counts_layer)

    adata_copy.X = _as_supported_expression_matrix(adata_copy.X)
    if input_is_counts:
        _validate_expression_matrix(adata_copy.X)
    else:
        adata_copy.X = _to_count_scale(adata_copy.X, adata=adata)
        _validate_expression_matrix(adata_copy.X)

    sc.pp.filter_cells(adata_copy, min_counts=min_counts)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells)

    if adata_copy.n_obs == 0:
        raise ValueError(f"No cells remain after filtering with min_counts={min_counts}.")
    if adata_copy.n_vars == 0:
        raise ValueError(f"No genes remain after filtering with min_cells={min_cells}.")

    adata_copy.layers["counts"] = _copy_matrix(adata_copy.X)
    adata_copy.uns.pop("log1p", None)
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)

    if n_top_genes is not None and adata_copy.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, flavor="seurat")
        adata_copy = adata_copy[:, adata_copy.var["highly_variable"]].copy()

    return adata_copy

def build_delaunay_edge_index(adata, threshold=0.99):
    spatial_coords = adata.obsm["spatial"]
    
    delaunay_obj = Delaunay(spatial_coords)
    delaunay_obj = filter_edge(delaunay_obj, threshold=threshold)
    edge_index = edge_index_from_delaunay(delaunay_obj.simplices)
    return edge_index


def build_delaunay_graph(adata, feature_key, threshold=0.99):
    edge_index = build_delaunay_edge_index(adata, threshold=threshold)

    pyg_data = pyg_obj(torch.tensor(adata.obsm[feature_key], dtype=torch.float), edge_index)
    
    return pyg_data

def _model_name_from_model_dir(model_dir):
    model_dir = str(model_dir).rstrip("/")
    leaf_name = model_dir.rsplit("/", 1)[-1]
    if "scConcept_" in leaf_name:
        return leaf_name.split("scConcept_", 1)[1]
    return leaf_name

def initial_pyG_graph(adata, model_dir, gene_col,
                      gene_col_case_sensitive=True,
                      batch_size = 512, 
                      coord_x=None, coord_y=None, 
                      threshold=0.99,
                      preprocess=False,
                      min_counts=10,
                      min_cells=5,
                      counts_layer=None,
                      n_top_genes=5000,
                      species=None):
    if preprocess:
        adata = _preprocess_adata(
            adata,
            min_counts=min_counts,
            min_cells=min_cells,
            counts_layer=counts_layer,
            n_top_genes=n_top_genes,
        )
    adata = _standardize_spatial_adata(adata, coord_x, coord_y)
    model_dir = str(model_dir)

    if gene_col not in adata.var.columns:
        if not gene_col_case_sensitive:
            adata.var_names = adata.var_names.str.upper()
        adata.var[gene_col] = adata.var_names
    elif not gene_col_case_sensitive:
        adata.var[gene_col] = adata.var[gene_col].str.upper()

    if 'scGPT' in model_dir:
        set_seed(12345)
        adata = scgpt_tasks.embed_data(
                                    adata,
                                    model_dir,
                                    gene_col=gene_col,
                                    batch_size=batch_size,
                                    use_fast_transformer=False,
                                    return_new_adata=False
                                    )
        feature_key = "X_scGPT"

    elif 'scConcept' in model_dir:
        resolved_species = species if species is not None else 'hsapiens'
        concept = scConcept(cache_dir=model_dir)
        try:
            concept.load_config_and_model(
                config=model_dir+'/config.yaml',
                model_path=model_dir+'/model.ckpt',
                gene_mappings_path=model_dir+'/gene_mappings'
                )
        except Exception as direct_error:
            fallback_model_name = _model_name_from_model_dir(model_dir)
            try:
                concept.load_config_and_model(model_name=fallback_model_name)
            except Exception as fallback_error:
                raise RuntimeError(
                    "Failed to load scConcept model from direct paths and "
                    f"from model_name={fallback_model_name!r}. Direct path error: {direct_error}"
                ) from fallback_error
            
        adata.var['gene_id'] = concept.map_gene_names_to_ids(
            species=resolved_species,
            gene_names=adata.var[gene_col].values.tolist()
        )

        set_seed(12345)
        result = concept.extract_embeddings(adata=adata, 
                                            gene_id_column='gene_id',
                                            batch_size=batch_size)
        adata.obsm["X_scConcept"] = result['cls_cell_emb']
        feature_key = "X_scConcept"

    else:
        raise ValueError(f"Unsupported model_dir: {model_dir}. Please use a directory containing 'scGPT' or 'scConcept'.")

    edge_index = build_delaunay_edge_index(adata, threshold=threshold)
    adata.uns["pyG_graph"] = {
        "edge_index": edge_index.detach().cpu().numpy(),
        "x_key": feature_key
        }
    
    return adata
