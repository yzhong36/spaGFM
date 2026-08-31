from spaGFM.stRoamer.utils.data_build import initial_pyG_graph
import gc
import logging
import torch
from pathlib import Path
import scanpy as sc
from spaGFM.stRoamer.utils.pyg_graph import data_from_adata_pyg_graph


logger = logging.getLogger(__name__)


def build_training_data(adata_dir, save_dir,
                        model_dir, gene_col,
                        gene_col_case_sensitive=True,
                        batch_size = 512, 
                        coord_x=None, coord_y=None, 
                        threshold=0.99,
                        preprocess=False,
                        min_counts=10,
                        min_cells=5,
                        counts_layer=None,
                        n_top_genes=5000,
                        resume=False,
                        species=None):

    adata_root = Path(adata_dir)
    if not adata_root.exists():
        raise FileNotFoundError(f"adata_dir does not exist: {adata_root}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    adata_files = sorted(p for p in adata_root.rglob("*.h5ad") if p.is_file())
    if not adata_files:
        raise FileNotFoundError(f"No .h5ad files found under: {adata_root}")

    output_names = set()
    for adata_file in adata_files:
        output_name = f"{adata_file.stem}.pt"
        if output_name in output_names:
            raise ValueError(
                "Duplicate output filename generated from input files: "
                f"{output_name}. Use unique parent folder names or flatten inputs less ambiguously."
            )
        output_names.add(output_name)

        output_file = save_path / output_name
        if resume and output_file.exists():
            logger.info("Skipping existing output: %s", output_file)
            continue

        logger.info("Processing AnnData file: %s", adata_file)
        adata = sc.read_h5ad(adata_file)
        adata = initial_pyG_graph(
            adata,
            model_dir=model_dir,
            gene_col=gene_col,
            gene_col_case_sensitive=gene_col_case_sensitive,
            species=species,
            batch_size=batch_size,
            coord_x=coord_x,
            coord_y=coord_y,
            threshold=threshold,
            preprocess=preprocess,
            min_counts=min_counts,
            min_cells=min_cells,
            counts_layer=counts_layer,
            n_top_genes=n_top_genes,
        )
        graph_dict = adata.uns["pyG_graph"]
        pyG_obj = data_from_adata_pyg_graph(adata, graph_dict, path=str(adata_file))

        torch.save(pyG_obj, output_file)
        del pyG_obj, graph_dict, adata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
