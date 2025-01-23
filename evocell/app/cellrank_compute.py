import os
import pandas as pd
import anndata as ad
import scanpy as sc
import cellrank as cr
import numpy as np
import igraph
from cellrank.estimators import GPCCA
from cellrank.kernels import PseudotimeKernel

# Read Files
counts_matrix = pd.read_csv("/home/oneai/evocell/kras/count_data.csv", index_col=0)
cell_metadata = pd.read_csv("/home/oneai/evocell/kras/cell_metadata.csv", dtype=str)

# Create AnnData Object
adata = ad.AnnData(X=counts_matrix.T, obs=cell_metadata.set_index("cell_id"))
# print(adata)
celltype_column = "CellCycle"
print(celltype_column)

# Normalize
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
# Dimensionality Reduction
sc.tl.pca(adata, random_state=0)
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
# Tsne
sc.tl.tsne(adata, n_pcs=30)

# Compute Pseudotime
sc.tl.diffmap(adata)

# Select cell_id as the root to compute pseudotime
# Also teh cellids that are in the selected path
cellid = "H358_A_AAACATACGCTTCC"
# print(np.flatnonzero(adata.obs['CellCycle'] == 'G1S')[0])
# print(adata.obs.index.get_loc(cellid))
adata.uns["iroot"] = adata.obs.index.get_loc(cellid)
# Calculate pseudotime
sc.tl.dpt(adata)
# Removing infinite values
adata_sub = adata[np.isfinite(adata.obs["dpt_pseudotime"])].copy()

# Add cluster
sc.tl.louvain(adata, key_added="clusters")

print("Finished Computing pseudotime")
print(adata_sub)

# Use the kernel to compute transition matrix
pk = PseudotimeKernel(adata_sub, time_key="dpt_pseudotime")
pk.compute_transition_matrix()

# Predict cell fates
g = GPCCA(pk).copy()
print(g)
# Extract celltypes
num_celltypes = list(adata_sub.obs[celltype_column].unique())

# Construct macrostate
print("Celltype Column: ", celltype_column)
# g.fit(n_states=10, cluster_key="CellCycle")
g.fit(n_states=10, cluster_key="clusters")


# g.fit(n_states=len(num_celltypes), cluster_key=celltype_column)
print("macostates have been computed")
print(adata_sub)
# Predict Terminal states
# g.predict_terminal_states(method="top_n", n_states=6)
# print("g has been computed")
# print(adata_sub)
# g.plot_macrostates(which="terminal")
