''' Disclaimer:
+ This script was adapted from the NBIS Tutorial Single-Cell RNA Seg Analysis
+ https://nbisweden.github.io/workshop-scRNAseq/labs/
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import scrublet as scr

# Set the paths to the raw data and the resulting figures
path_data = '1_QualityControl/Data/Input/scAtlas.h5ad'
sc.settings.figdir = '1_QualityControl/Figures'

# Ignore warnings
warnings.simplefilter(action='ignore', category=Warning)

# Set figure settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80)

# Read the .h5ad format into an AnnData object
adata = sc.read_h5ad(path_data)
adata.var_names_make_unique()
print(adata)

# Perform the Quality Control
gene_column = 'feature_name'

# Mitochondrial genes
adata.var['mt'] = adata.var[gene_column].str.startswith("MT-") 

# Ribosomal genes
adata.var['ribo'] = adata.var[gene_column].str.startswith(("RPS","RPL"))

# Hemoglobin genes
adata.var['hb'] = adata.var[gene_column].str.contains(("^HB[^(P|E|S)]"))

# Check the matrix
print(adata.var)

# Use the sc.pp.calculate_qc_metrics to calculate the QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], percent_top=None, log1p=False, inplace=True)

# Show the calculated QC metrics
print(adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb']].head())
print(adata.obs.columns)

# Plot the QC metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb'], jitter=0.4, groupby = 'cell_type', rotation= 90, multi_panel=True, save='_qc.png')
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', color='cell_type', save='_qc.png')

# Filtering
# Detection-based filtering: removal of genes with low expression
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Plot highest expressed genes before filtering
sc.pl.highest_expr_genes(adata, n_top=20, save='_qc_filtering.png')

# Mito/Ribo filtering
# Filter for a percentage of mitochondrial counts
adata = adata[adata.obs['pct_counts_mt'] < 6, :]

# Filter for a percentage of ribosomal counts > 0.05
adata = adata[adata.obs['pct_counts_ribo'] > 5, :]

# Print the remaining cells
print("Remaining cells %d"%adata.n_obs)

# Plot the filtered results
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt','pct_counts_ribo', 'pct_counts_hb'], jitter=0.4, groupby = 'cell_type', rotation = 90, multi_panel=True, save='_qc_filtered.png')

# Filter the genes
# Redefine the mito_genes since they were first calculated on the full object before removing low expressed genes.
mito_genes = adata.var[gene_column].str.startswith("MT-")
hb_genes = adata.var[gene_column].str.contains("^HB[^(P|E|S)]")

# Remove the genes
remove = mito_genes
remove = np.add(remove, hb_genes)
keep = np.invert(remove)

adata = adata[:, keep]

# Print the new adata
print(adata.n_obs, adata.n_vars)

# Predict doublets
# Split per batch into new objects
batches = adata.obs['cell_type'].cat.categories.tolist()
alldata = {}
for batch in batches:
    tmp = adata[adata.obs['cell_type'] == batch,]
    print(batch, ':', tmp.shape[0], ' cells')
    scrub = scr.Scrublet(tmp.raw.X)
    out = scrub.scrub_doublets(verbose=False, n_prin_comps = 10)
    alldata[batch] = pd.DataFrame({'doublet_score':out[0], 'predicted_doublets':out[1]}, index = tmp.obs.index)
    print(alldata[batch].predicted_doublets.sum(), ' predicted_doublets')

# Add the predictions to the adata object
scrub_pred = pd.concat(alldata.values())
adata.obs['doublet_scores'] = scrub_pred['doublet_score'] 
adata.obs['predicted_doublets'] = scrub_pred['predicted_doublets'] 

# Print the total number of predicted doublets
print(sum(adata.obs['predicted_doublets']))

# Add in column with singlet/doublet instead of True/False
#%matplotlib inline

adata.obs['doublet_info'] = adata.obs['predicted_doublets'].astype(str)
sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, groupby = 'doublet_info', rotation=45, save='_qc_doublet_info.png')

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['doublet_scores','doublet_info','cell_type'], save='_qc_umap.png')

# Revert back to the raw counts as the main matrix in adata
adata = adata.raw.to_adata() 

adata = adata[adata.obs['doublet_info'] == 'False', :]

adata.obs['cell_type'].value_counts()

# Save the new data
adata.write_h5ad('1_QualityControl/Data/Output/scanpy_qc_scAtlas.h5ad')