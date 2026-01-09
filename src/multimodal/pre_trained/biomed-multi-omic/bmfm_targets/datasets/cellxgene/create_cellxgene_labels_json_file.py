from pathlib import Path

from scanpy import read_h5ad

from bmfm_targets.datasets.datasets_utils import create_celltype_labels_json_file

adata_path = "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/h5ad/cellxgene.h5ad"
output_path = Path(__file__).parent / "cellxgene_labels.json"
label_column_name = "cell_type_ontology_term_id"
adata = read_h5ad(adata_path)
create_celltype_labels_json_file(adata, output_path, label_column_name)
