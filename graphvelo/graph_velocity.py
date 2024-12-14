import os
import logging
import warnings

import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

from tangent_space import corr_kernel, cos_corr, density_corrected_transition_matrix, _estimate_dt


def regression_phi(
    i: int, 
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 0.0,
    r: float = 1.0,
    loss_func: str = "linear",
    norm_dist: bool = False,
):
    """Performs optimization to find the coefficients for a given point based on 
    tangent space projection (TSP) loss.

    This function optimizes a weight vector `w` that minimizes a weighted combination of:
    - Reconstruction error between the predicted vector `v_` (based on the weight vector `w` and displacement vector)
      and the observed vector `v`.
    - Cosine regularization between the weight vector `w` and a target vector `c`.
    - L2-Regularization term to prevent overfitting.

    The optimization is performed using the `minimize` function, and the result is returned as a weight vector 
    that best represents the relationships between a point's neighborhood in the feature space.

    Parameters
    ----------
    i : int
        The index of the current point in the dataset `X` to optimize for.
    X : :class:`~numpy.ndarray`
        The feature matrix of shape (n_samples, n_features) containing the feature vectors for each data point.
    V : :class:`~numpy.ndarray`
        The vector field data, typically containing observed vectors for each point.
    C : :class:`~numpy.ndarray`
        A matrix containing target vectors `c` for each data point and its neighborhood.
    nbrs : list
        A list of indices representing the neighbors of each point in `X`.
    a : float, optional, default 1.0
        Weighting factor for the reconstruction error term in the objective function.
    b : float, optional, default 0.0
        Weighting factor for the cosine similarity term in the objective function. If `b == 0`, no cosine similarity is computed.
    r : float, optional, default 1.0
        Regularization strength for the weight vector `w` to prevent overfitting.
    loss_func : str, optional, default 'linear'
        The type of loss function to compute the reconstruction error. Can be 'linear' for direct error or 'log' for 
        the logarithm of the error.
    norm_dist : bool, optional, default False
        If True, normalizes the differences between the current point and its neighbors before using them in the optimization.

    Returns
    -------
    tuple
        A tuple containing:
        - The index `i` of the current point.
        - The optimized weight vector `w` as a result of the regression.
    """
    x, v, c, idx = X[i], V[i], C[i], nbrs[i]
    c = c[idx]

    # normalized differences
    D = X[idx] - x
    if norm_dist:
        dist = np.linalg.norm(D, axis=1)
        dist[dist == 0] = 1
        D /= dist[:, None]

    # co-optimization
    c_norm = np.linalg.norm(c)

    def func(w):
        v_ = w @ D

        # cosine similarity between w and c
        if b == 0:
            sim = 0
        else:
            cw = c_norm * np.linalg.norm(w)
            if cw > 0:
                sim = c.dot(w) / cw
            else:
                sim = 0

        # reconstruction error between v_ and v
        rec = v_ - v
        rec = rec.dot(rec)
        if loss_func is None or loss_func == "linear":
            rec = rec
        elif loss_func == "log":
            rec = np.log(rec)
        else:
            raise NotImplementedError(
                f"The function {loss_func} is not supported. Choose either `linear` or `log`."
            )

        # regularization
        reg = 0 if r == 0 else w.dot(w)

        ret = a * rec - b * sim + r * reg
        return ret

    def fjac(w):
        v_ = w @ D

        # reconstruction error
        jac_con = 2 * a * D @ (v_ - v)

        if loss_func is None or loss_func == "linear":
            jac_con = jac_con
        elif loss_func == "log":
            jac_con = jac_con / (v_ - v).dot(v_ - v)

        # cosine similarity
        w_norm = np.linalg.norm(w)
        if w_norm == 0 or b == 0:
            jac_sim = 0
        else:
            jac_sim = b * (c / (w_norm * c_norm) - w.dot(c) / (w_norm**3 * c_norm) * w)

        # regularization
        if r == 0:
            jac_reg = 0
        else:
            jac_reg = 2 * r * w

        return jac_con - jac_sim + jac_reg

    res = minimize(func, x0=C[i, idx], jac=fjac)
    return i, res["x"]

def tangent_space_projection(
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 0.0,
    r: float = 1.0,
    loss_func: str = "linear",
    n_jobs: int = None,
):
    """
    The function generates a graph based on the velocity data by minimizing the loss function:
                    L(w_i) = a |v_ - v|^2 - b cos(u, v_) + lambda * \sum_j |w_ij|^2
    where v_ = \sum_j w_ij*d_ij. The flow from i- to j-th node is returned as the edge matrix E[i, j],
    and E[i, j] = -E[j, i].

    Arguments
    ---------
        X: :class:`~numpy.ndarray`
            The coordinates of cells in the expression space.
        V: :class:`~numpy.ndarray`
            The velocity vectors in the expression space.
        C: :class:`~numpy.ndarray`
            The transition matrix of cells based on the correlation/cosine kernel.
        nbrs: list
            List of neighbor indices for each cell.
        a: float (default 1.0)
            The weight for preserving the velocity length.
        b: float (default 1.0)
            The weight for the cosine similarity.
        r: float (default 1.0)
            The weight for the regularization.
        n_jobs: `int` (default: available threads)
            Number of parallel jobs.

    Returns
    -------
        E: :class:`~numpy.ndarray`
            The coefficient matrix.
    """
    if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
            n_jobs > os.cpu_count()):
        n_jobs = os.cpu_count()
    if isinstance(n_jobs, int):
        logging.info(f'running {n_jobs} jobs in parallel')

    vgenes = np.ones(V.shape[-1], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]
    X = X[:, vgenes]

    E = np.zeros((X.shape[0], X.shape[0]))

    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(regression_phi)(
            i, 
            X,
            V,
            C,
            nbrs,
            a,
            b,
            r,
            loss_func,
        )
        for i in tqdm(
        range(X.shape[0]),
        total=X.shape[0],
        desc="Learning Phi in tangent space projection.",
    ))

    for i, res_x in res:
        E[i][nbrs[i]] = res_x
    
    return E


class GraphVelo():
    """Manifold-constrained velocity estimation class for analyzing gene expression dynamics in single-cell RNA-seq data.
    This class performs the following key tasks:
    1. Project velocity estimated from other packages to the tangent space to satisfied the manifold constrain.
    2. Transform velocity vectors across different representations.  

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix, which contains the gene expression and velocity data in its layers.
    xkey : str, optional, default 'Ms'
        The key in `adata.layers` that contains the gene expression data.
    vkey : str, optional, default 'velocity'
        The key in `adata.layers` that contains the velocity data.
    X_data : :class:`~numpy.ndarray`, optional, default None
        If provided, this will be used as the gene expression matrix. If not, `adata.layers[xkey]` is used.
    V_data : :class:`~numpy.ndarray`, optional, default None
        If provided, this will be used as the velocity matrix. If not, `adata.layers[vkey]` is used.
    gene_subset : list, optional, default None
        A list of genes to be used for analysis. If None, all genes are considered.
    approx : bool, default True
        If True, the velocity vectors are approximated using PCA; otherwise, the original vectors are used.
    n_pcs : int, default 30
        The number of principal components to use when approximating the velocity vectors.
    mo : bool, default False
        If True, uses the `WNN` (weighted nearest neighbors) graph from `adata.uns` instead of the default graph.

    Attributes
    ----------
    X : :class:`~numpy.ndarray`
        The gene expression data matrix after dimensionality reduction or from the original data.
    V : :class:`~numpy.ndarray`
        The velocity vectors after dimensionality reduction or from the original data.
    approx : bool
        A flag indicating whether the approximation method is used.
    nbrs_idx : list
        The indices of the nearest neighbors for each cell.
    T : :class:`~scipy.sparse.csr_matrix`
        The tangent space projection matrix obtained during the training process.

    Methods
    -------
    train(a=1, b=10, r=1, loss_func=None, transition_matrix=None, softmax_adjusted=False)
        Trains the model by computing the tangent space projection based on gene expression and velocity.
    project_velocity(X_embedding, T=None)
        Projects the velocity vectors onto a low-dimensional embedding of the data.
    """
    def __init__(
        self,
        adata, 
        xkey='Ms', 
        vkey='velocity', 
        X_data=None,
        V_data=None,
        gene_subset=None,
        approx=True,
        n_pcs=30,
        mo=False,):
        if X_data is not None and V_data is not None:
            X = np.array(X_data.A if sp.issparse(X_data)
            else X_data)
            V = np.array(V_data.A if sp.issparse(V_data)
            else V_data)
        else:
            X_org = np.array(
                adata.layers[xkey].A
                if sp.issparse(adata.layers[xkey])
                else adata.layers[xkey]
            )
            V_org = np.array(
                adata.layers[vkey].A
                if sp.issparse(adata.layers[vkey])
                else adata.layers[vkey]
            )
            
            subset = np.ones(adata.n_vars, bool)
            if gene_subset is not None:
                var_names_subset = adata.var_names.isin(gene_subset)
                subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
            X = X_org[:, subset]
            V = V_org[:, subset]

            nans = np.isnan(np.sum(V, axis=0))
            logging.info(f"{nans.sum()} genes are removed because of nan velocity values.")
            if np.any(nans):
                X = X[:, ~nans]
                V = V[:, ~nans]
        self.approx = False

        if "neighbors" not in adata.uns.keys():
            # Check construction knn in reduced space.
            if mo:
                if 'WNN' not in adata.uns.keys():
                    logging.error("`WNN` not in adata.uns")
                nbrs_idx = adata.uns['WNN']['indices']
            else:
                raise ValueError("Please run dyn.tl.neighbors first.")
        else:
            nbrs_idx = adata.uns["neighbors"]['indices']
        self.nbrs_idx = nbrs_idx
        
        if approx:
            self.approx = True
            dt = _estimate_dt(X, V, nbrs_idx)
            X_plus_V = X + V * dt 
            X_plus_V[X_plus_V < 0] = 0 
            X = np.log1p(X)
            X_plus_V = np.log1p(X_plus_V)
            pca = PCA(n_components=n_pcs, svd_solver='arpack', random_state=0)
            pca_fit = pca.fit(X)
            X_pca = pca_fit.transform(X)
            Y_pca = pca_fit.transform(X_plus_V)
            V_pca = (Y_pca - X_pca) / dt
            self.X = X_pca
            self.V = V_pca
        else: 
            self.X = X
            self.V = V

    def train(self, a=1, b=10, r=1, loss_func=None, transition_matrix=None, softmax_adjusted=False):
        """Trains the GraphVelo model by optimizing the TSP loss using gene expression and velocity data.

        Parameters
        ----------
        a : float, optional, default 1
            Weight for the reconstruction error term.
        b : float, optional, default 10
            Weight for the cosine similarity term.
        r : float, optional, default 1
            Regularization strength for the optimization.
        loss_func : str, optional, default None
            Type of loss function to use for reconstruction error. Can be 'linear' or 'log'.
        transition_matrix : :class:`~numpy.ndarray`, optional, default None
            Pre-computed transition matrix. If None, it is computed using `corr_kernel`.
        softmax_adjusted : bool, optional, default False
            Whether to apply softmax adjustment to the transition matrix.
        """
        if loss_func is None:
            loss_func = 'linear' if self.approx else 'log'
        if transition_matrix is None:
            P = corr_kernel(self.X, self.V, self.nbrs_idx, corr_func=cos_corr, softmax_adjusted=softmax_adjusted)
            P_dc = density_corrected_transition_matrix(P).A
        else:
            P_dc = transition_matrix
        T = tangent_space_projection(self.X, self.V, P_dc, self.nbrs_idx, a=a, b=b, r=r, loss_func=loss_func)
        self.T = sp.csr_matrix(T)
    
    def project_velocity(self, X_embedding, T=None) -> np.ndarray:
        """Projects the velocity vectors onto a low-dimensional embedding or different modalities.

        Parameters
        ----------
        X_embedding : :class:`~numpy.ndarray`
            The low-dimensional embedding of the gene expression data (e.g., PCA components).
        T : :class:`~scipy.sparse.csr_matrix`, optional, default None
            The tangent space projection matrix. If None, the matrix computed during training is used.

        Returns
        -------
        :class:`~numpy.ndarray`
            The projected velocity vectors, either as a sparse or dense matrix.
        """
        if T is None:
            T = self.T
        n = T.shape[0]
        delta_X = np.zeros((n, X_embedding.shape[1]))

        sparse_emb = False
        if sp.issparse(X_embedding):
            X_embedding = X_embedding.A
            sparse_emb = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in tqdm(
                range(n),
                total=n,
                desc="projecting velocity vector to low dimensional embedding",
            ):
                idx = T[i].indices
                diff_emb = X_embedding[idx] - X_embedding[i, None]
                if np.isnan(diff_emb).sum() != 0:
                    diff_emb[np.isnan(diff_emb)] = 0
                T_i = T[i].data
                delta_X[i] = T_i.dot(diff_emb)

        return sp.csr_matrix(delta_X) if sparse_emb else delta_X