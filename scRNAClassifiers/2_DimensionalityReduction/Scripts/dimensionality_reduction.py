'''
Disclaimer:
+ This script was adapted from the NBIS Tutorial Single-Cell RNA Seg Analysis
+ https://nbisweden.github.io/workshop-scRNAseq/labs/
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import scrublet as scr

# Ignore warnings
warnings.simplefilter(action='ignore', category=Warning)

# Set the paths to the raw data and the resulting figures
path_file = '2_DimensionalityReduction/Data/Input/scanpy_qc_scAtlas.h5ad'
sc.settings.figdir = '2_DimensionalityReduction/Figures'
sc.settings.set_figure_params(dpi=80)
sc.settings.verbosity = 3

# Read in the data
adata = sc.read_h5ad(path_file)

# Subet the data so the dataset is smaller and easier to work with
adata = adata[adata.obs['author_first_cell_type'].isin(['Mononuclear Phagocytes','Plasmacytoid Dendritic Cells'])]

# Add the computed variables into the original dataset to perform the following steps
# Predict doublets + split per batch into new objects
batches = adata.obs['cell_type'].cat.categories.tolist()
alldata = {}
for batch in batches:
    tmp = adata[adata.obs['cell_type'] == batch,]
    print(batch, ':', tmp.shape[0], ' cells')
    scrub = scr.Scrublet(tmp.X)
    out = scrub.scrub_doublets(verbose=False, n_prin_comps = 10)
    alldata[batch] = pd.DataFrame({'doublet_score':out[0],'predicted_doublets':out[1]},index = tmp.obs.index)
    print(alldata[batch].predicted_doublets.sum(), ' predicted_doublets')

# Add the predictions to the adata object
scrub_pred = pd.concat(alldata.values())
adata.obs['doublet_scores'] = scrub_pred['doublet_score'] 
adata.obs['predicted_doublets'] = scrub_pred['predicted_doublets'] 

# Print the total number of predicted doublets
print('Predicted doublets sum:', sum(adata.obs['predicted_doublets']))

# Add in a column with singlet/doublet instead of True/Fals
#%matplotlib inline
adata.obs['doublet_info'] = adata.obs['predicted_doublets'].astype(str)

# Normalize to a depth 10 000
sc.pp.normalize_total(adata, target_sum=1e4)

# Log transform the data
sc.pp.log1p(adata)

# Store normalized counts in the raw slot + subset adata.X for variable genes, but want to keep all genes matrix as well
adata.raw = adata

# Feature Selection
#%matplotlib inline

# Compute variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print("Highly variable genes: %d"%sum(adata.var.highly_variable))

# Plot variable genes
sc.pl.highly_variable_genes(adata, save='_dr.png')

# Scale data, clip values exceeding standard deviation 10
sc.pp.scale(adata, max_value=10)

# PCA
sc.tl.pca(adata, svd_solver='arpack')

# Plot more PCS
sc.pl.pca(adata, color='author_first_cell_type', components = ['1,2' ,'3,4' ,'5,6' ,'7,8'], ncols=2, save='_dr_PCA.png')

#Plot loadings: only plots the positive axes genes from each PC
sc.pl.pca_loadings(adata, components=[1, 2, 3, 4, 5, 6, 7, 8], save='_dr_loadings_PCA.png')

# Quick notes:
# adata.obsm["X_pca"] is the embeddings
# adata.uns["pca"] is pc variance
# adata.varm['PCs'] is the loadings

genes = adata.var['feature_name']

for pc in [1,2,3,4]:
    g = adata.varm['PCs'][:,pc-1]
    o = np.argsort(g)
    sel = np.concatenate((o[:10],o[-10:])).tolist()
    emb = adata.obsm['X_pca'][:,pc-1]
    tempdata = adata[np.argsort(emb),]
    sc.pl.heatmap(tempdata, var_names = genes[sel].index.tolist(), groupby='predicted_doublets', swap_axes = True, use_raw=False, save='_dr_heatmap_PCA.png')

# Plot the variance ratio
sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 50, save='_dr_variance_PCA.png')

# tSNE
sc.tl.tsne(adata, n_pcs = 30)
sc.pl.tsne(adata, color='author_first_cell_type', save='_dr_tSNE.png')

# UMAP
sc.pp.neighbors(adata, n_pcs = 30, n_neighbors = 20)
sc.tl.umap(adata)
sc.pl.umap(adata, color='author_first_cell_type', save='_dr_UMAP.png')

# Run with 10 components
umap10 = sc.tl.umap(adata, n_components=10, copy=True)
fig, axs = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

sc.pl.umap(adata, color='author_first_cell_type',  title='UMAP',
           show=False, ax=axs[0], legend_loc=None, save='_dr_1_UMAP10.png')
sc.pl.umap(umap10, color='author_first_cell_type', title='UMAP10', show=False,
           ax=axs[1], components=['1, 2'], legend_loc=None, save='_dr_2_UMAP10.png')
sc.pl.umap(umap10, color='author_first_cell_type', title='UMAP10',
           show=False, ax=axs[2], components=['3, 4'], legend_loc=None, save='_dr_3_UMAP10.png')

# Plot the umap with neighbor edges
sc.pl.umap(adata, color='author_first_cell_type', title="UMAP", edges=True, save='_dr_UMAP_edges.png')

fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
sc.pl.pca(adata, color='author_first_cell_type', components=['1, 2'], ax=axs[0, 0], show=False, save='_dr_1_plot_comparison.png')
sc.pl.tsne(adata, color='author_first_cell_type', components=['1, 2'], ax=axs[0, 1], show=False, save='_dr_2_plot_comparison.png')
sc.pl.umap(adata, color='author_first_cell_type', components=['1, 2'], ax=axs[1, 0], show=False, save='_dr_3_plot_comparison.png')

# Genes of interest
# Convert the ENSEMBL IDs to the gene names
print(adata.var['feature_name'].head())

symbol_to_ensembl = dict(zip(adata.var['feature_name'], adata.var.index))

# List of the genes of interest
genes_of_interest = ['C1QB', 'C1QA', 'FCER1A', 'CD1C', 'FCN1', 'VCAN', 'CPA3', 'TPSAB1', 'ICAM1', 'CXCR2']

# Convert gene symbols to Ensembl IDs (only keep existing ones)
ensembl_ids = [symbol_to_ensembl[gene] for gene in genes_of_interest if gene in symbol_to_ensembl]

print('Mapped Ensembl IDs:', ensembl_ids)

# Genes of interest
sc.pl.umap(adata, color=ensembl_ids, save='_dr_gene_UMAP.png')
sc.pl.umap(adata, color=['author_first_cell_type', 'nCount_RNA', 'percent.mt'], ncols=3, use_raw=False, save='_dr_UMAP_RNA.png')

# Save the data
adata.write_h5ad('2_DimensionalityReduction/Data/Output/scanpy_dr_scAtlas.h5ad')