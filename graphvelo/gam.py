from typing import Optional
from pygam import LinearGAM, s

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from utils import flatten


def fit_gene_trend(
    adata, 
    genes, 
    tkey:str, 
    layer:str='Ms', 
    max_iter: int = 2000,
    grid_num: int = 200,
    **kwargs):
    """Fits smooth trends to sequential genomic data over time using Generalized Additive Models (GAM).

    This function fits a GAM to model the gene expression/velocity trajectories of selected genes based on a 
    continuous variable (e.g., time) stored in `adata.obs[tkey]`. The fitted trends are predicted at 
    evenly spaced time points and returned as an AnnData object.

    The function assumes that gene expression/velocity data is stored in a specified layer of the AnnData object 
    and that the provided genes exist in `adata.var_names`. Missing genes are reported and excluded from 
    the analysis.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The input AnnData object containing single-cell expression data. The object should have the gene 
        expression data in the specified layer and a continuous variable in `adata.obs[tkey]`.
    genes : list of str or str
        List of gene names or a single gene name whose dynamic pattern will be modeled. The function will 
        automatically filter out genes not present in `adata.var_names`.
    tkey : str
        The key in `adata.obs` that corresponds to the continuous variable (e.g., time, pseudotime) for 
        modeling gene trends.
    layer : str, optional, default 'Ms'
        The layer in the AnnData object that contains the gene data.
    max_iter : int, optional, default 2000
        The maximum number of iterations for the GAM fitting process.
    grid_num : int, optional, default 200
        The number of grid points at which to predict the gene trends.
    **kwargs
        Additional keyword arguments passed to the GAM model fitting, such as `n_splines` and `spline_order`.

    Returns
    -------
    :class:`~anndata.AnnData`
        An AnnData object containing the fitted trends for each gene at the specified grid points.
    """

    if tkey not in adata.obs.keys():
        raise ValueError(f'{tkey} not found in adata.obs')
    t = adata.obs[tkey]

    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)

    gam_kwargs = {
        'n_splines': 6,
        'spline_order': 3
    }
    gam_kwargs.update(kwargs)

    data = np.zeros((gn, grid_num)) # N_genes * N_grid

    for i, gene in tqdm(
        enumerate(genes),
        total=gn,
        desc="Fitting trends using GAM",):
        x = adata[:, gene].layers[layer]
        x = x.A.flatten() if sp.issparse(x) else x.flatten()

        ### GAM fitting
        term  =s(
            0,
            **gam_kwargs)
        gam = LinearGAM(term, max_iter=max_iter, verbose=False).fit(t, x)
        x_lins = np.linspace(t.min(), t.max(), grid_num)
        y_pred = gam.predict(x_lins)
        data[i] = y_pred
    
    gdata = AnnData(data)
    gdata.obs_names = pd.Index(genes)
    return gdata


def fit_response(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = 'M_sc',
    ykey: Optional[str] = 'jacobian',
    norm: bool = True,
    log: bool = False,
    grid_num: int = 200,
    **kwargs,
):
    """Fits response curves for gene pairs using GAM.

    This function fits GAMs to model the relationship between the expression of two genes (from a given gene pair) 
    across different conditions or time points, based on the Jacobian matrix or other relevant data stored in the 
    specified layers of the AnnData object. The fitted response curves are returned as a new AnnData object with 
    predictions for each gene pair.

    The function performs the following steps:
    - Validates the existence of required layers and genes in the AnnData object.
    - Extracts the gene expression values for each gene pair.
    - Optionally applies log-transformation and normalization.
    - Fits a GAM for each gene pair's response and predicts the response values across a specified grid.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The input AnnData object containing single-cell RNA-seq data. The object must include relevant layers 
        such as `xkey` and `ykey` (e.g., `M_sc` and `jacobian`).
    pairs_mat : :class:`~numpy.ndarray`
        A 2D array of gene pairs for which the response curves will be modeled. Each row should contain a pair 
        of genes (gene1 -> gene2).
    xkey : str, optional, default 'M_sc'
        The key in `adata.layers` representing the gene expression data for the first gene in each pair. This is 
        typically a matrix of gene expression values.
    ykey : str, optional, default 'jacobian'
        The key in `adata.layers` representing the Jacobian matrix or response data for the second gene in each pair.
    norm : bool, optional, default True
        Whether to normalize the response values (`ykey`). If True, the response values are scaled to their 
        maximum value.
    log : bool, optional, default False
        Whether to log-transform the input gene expression and response data. If True, a log-transformation 
        is applied to both `xkey` and `ykey` values, if they are non-negative.
    grid_num : int, optional, default 200
        The number of grid points at which to predict the response curves.
    **kwargs
        Additional keyword arguments passed to the GAM model fitting, such as `n_splines` and `spline_order`.

    Returns
    -------
    :class:`~anndata.AnnData`
        An AnnData object containing the fitted response curves for each gene pair, predicted at the specified 
        grid points. The object has the gene pairs as observation names.
    """
    try:
        from dynamo.vectorfield.utils import get_jacobian
    except ImportError:
        raise ImportError(
            "If you want to do jacobian analysis related to dynamo, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    
    all_genes_in_pair = np.unique(pairs_mat)
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )
    if not ykey.startswith("jacobian"):
        raise KeyError('The ykey should start with `jacobian`.')
    
    genes = []
    xy = pd.DataFrame()
    id = 0
    for _, gene_pairs in enumerate(pairs_mat):
        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]
        genes.append(gene_pairs[1])

        x = flatten(adata[:, gene_pairs[0]].layers[xkey])
        J_df = get_jacobian(
            adata,
            gene_pairs[0],
            gene_pairs[1],
        )
        jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
        y_ori = flatten(J_df[jkey])

        finite = np.isfinite(x + y_ori)
        nonzero = np.abs(x) + np.abs(y_ori) > 0
        valid_ids = np.logical_and(finite, nonzero)

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = x if sum(x < 0) else np.log(np.array(x) + 1), y_ori if sum(y_ori) < 0 else np.log(
                np.array(y_ori) + 1)

        if norm:
            y_ori = y_ori / y_ori.max()

        y = y_ori

        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)

        id = id + 1

    data = np.zeros((len(pairs_mat), grid_num)) # N_genes * N_grid

    for gene_idx, res_type in enumerate(xy.type.unique()):
        gene_pairs = res_type.split("->")
        xy_subset = xy[xy["type"]==res_type]
        x_val, y_val = xy_subset["x"], xy_subset["y"]

        gam_kwargs = {
            'n_splines': 12,
            'spline_order': 3
        }
        gam_kwargs.update(kwargs)
        term  =s(
            0,
            **gam_kwargs)
        gam = LinearGAM(term, max_iter=1000, verbose=False).fit(x_val, y_val)
        
        x_lins = np.linspace(x_val.min(), x_val.max(), grid_num)
        y_pred = gam.predict(x_lins)
        data[gene_idx] = y_pred
    
    adata = AnnData(data)
    adata.obs_names = pd.Index(genes)

    return adata