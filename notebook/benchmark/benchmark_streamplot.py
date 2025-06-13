import scanpy as sc
import scvelo as scv

import warnings
warnings.simplefilter('ignore')


methods=['celldancer', 'deepvelo', 'unitvelo', 'velovi', 'scv', 'gv']
datasets = ['fucci', 'scv_endocrinogenesis_day15','hematopoiesis_raw', 'dentategyrus_scv', 'organoid']
cluster_keys = {'fucci': 'cell_cycle_phase', 'scv_endocrinogenesis_day15': 'clusters', 'hematopoiesis_raw': 'cell_type', 'hgForebrainGlut': 'Clusters', 'dentategyrus_scv': 'clusters', 'organoid': 'cell_type'}

for dataset in datasets:
    print(dataset)
    for method in methods:
        ad = sc.read('benchmark/' + dataset + '_' + method + '.h5ad')

        if 'velocity_umap' not in ad.obsm.keys():
            scv.tl.velocity_graph(ad)
            scv.tl.velocity_embedding(ad, basis='umap')
        scv.pl.velocity_embedding_stream(ad, basis='umap', color=cluster_keys[dataset], save=f'figures/streamplot/{dataset}_{method}.png', dpi=300)