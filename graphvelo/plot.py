from typing import Tuple, Optional, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import scipy.sparse as sp
from pygam import LinearGAM, s

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.patheffects as PathEffects
import seaborn as sns

from graphvelo.utils import flatten
from graphvelo.metrics import cross_boundary_correctness
from graphvelo.kernel_density_smooth import kde2d, kde2d_to_mean_and_sigma


def gene_score_histogram(
    adata,
    score_key: str,
    genes: Optional[List[str]] = None,
    bins: int = 100,
    quantile: Optional[float] = 0.95,
    extra_offset_fraction: float = 0.1,
    anno_min_diff_fraction: float = 0.05,
) -> plt.Figure:
    """
    Draw a histogram of gene scores with percentile line and annotations for specific genes.
    Adapted from Palantir (https://github.com/dpeerlab/Palantir/blob/master/src/palantir/plot.py#L1803)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    score_key : str
        The key in `adata.var` data frame for the gene score.
    genes : Optional[List[str]], default=None
        List of genes to be annotated. If None, no genes are annotated.
    bins : int, default=100
        The number of bins for the histogram.
    quantile : Optional[float], default=0.95
        Quantile line to draw on the histogram. If None, no line is drawn.
    extra_offset_fraction : float, default=0.1
        Fraction of max height to use as extra offset for annotation.
    anno_min_diff_fraction : float, default=0.05
        Fraction of the range of the scores to be used as minimum difference for annotation.

    Returns
    -------
    fig : matplotlib Figure
        Figure object with the histogram.

    Raises
    ------
    ValueError
        If input parameters are not as expected.
    """
    if score_key not in adata.var.columns:
        raise ValueError(f"Score key {score_key} not found in ad.var columns.")
    scores = adata.var[score_key]

    if genes is not None:
        if not all(gene in scores for gene in genes):
            raise ValueError("All genes must be present in the scores.")

    fig, ax = plt.subplots(figsize=(10, 6))
    n_markers = len(genes) if genes is not None else 0

    heights, bins, _ = ax.hist(scores, bins=bins, zorder=-n_markers - 2)

    if quantile is not None:
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be a float between 0 and 1.")
        ax.vlines(
            np.quantile(scores, quantile),
            0,
            np.max(heights),
            alpha=0.5,
            color="red",
            label=f"{quantile:.0%} percentile",
        )

    ax.legend()
    ax.set_xlabel(f"{score_key} score")
    ax.set_ylabel("# of genes")

    ax.spines[["right", "top"]].set_visible(False)
    plt.locator_params(axis="x", nbins=3)

    if genes is None:
        return fig

    previous_value = -np.inf
    extra_offset = extra_offset_fraction * np.max(heights)
    min_diff = anno_min_diff_fraction * (np.max(bins) - np.min(bins))
    marks = scores[genes].sort_values()
    ranks = scores.rank(ascending=False)
    for k, (highlight_gene, value) in enumerate(marks.items()):
        hl_rank = int(ranks[highlight_gene])
        i = np.searchsorted(bins, value)
        text_offset = -np.inf if value - previous_value > min_diff else previous_value
        previous_value = value
        height = heights[i - 1]
        text_offset = max(text_offset + extra_offset, height + 1.8 * extra_offset)
        txt = ax.annotate(
            f"{highlight_gene} #{hl_rank}",
            (value, height),
            (value, text_offset),
            arrowprops=dict(facecolor="black", width=1, alpha=0.5),
            rotation=90,
            horizontalalignment="center",
            zorder=-k,
        )
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground="w", alpha=0.8)]
        )

    return fig


def cbc_heatmap(
        adata,
        xkey:str='M_s',
        vkey:str='velocity_S',
        cluster_key:str='clusters', 
        basis:str='pca',
        neighbor_key:str='neighbors',
        vector:str='velocity', 
        corr_func='cosine',
        cmap = 'viridis',
        annot: bool = True,
        ):
    clusters = adata.obs[cluster_key].unique()
    rows, cols = clusters.to_list(), clusters.to_list()
    cluster_edges = []
    for i in clusters:
        for j in clusters:
            if i == j:
                continue
            else:
                cluster_edges.append((i, j))

    scores, _ = cross_boundary_correctness(
        adata, xkey=xkey, vkey=vkey, cluster_key=cluster_key, cluster_edges=cluster_edges,
        basis=basis, neighbor_key=neighbor_key, vector=vector, corr_func=corr_func, return_raw=False)

    df = pd.DataFrame(index=rows, columns=cols)
    for row in rows:
        for col in cols:
            if row == col:
                df.loc[row, col] = np.nan
            else:
                df.loc[row, col] = scores[(row, col)]
    df = df.apply(pd.to_numeric)
    sns.set_style("whitegrid")
    sns.heatmap(df, cmap=cmap, annot=annot)


def gene_trend(adata, 
               genes, 
               tkey:str, 
               layer:str='Ms', 
               n_x_grid: int = 100,
               max_iter: int = 2000,
               return_gam_result: bool = False,
               zero_indicator: bool = False,
               sharey: bool = False,
               hide_trend: bool = False,
               hide_cells: bool = False,
               hide_interval: bool = False,
               set_label: bool = True,
               same_plot: bool = False,
               color=None, cmap='plasma', pointsize=1, figsize=None, ncols=5, dpi=100, scatter_kwargs=None, **kwargs):
    """ Plot gene expression or velocity trends along trajectory using Generalized Addictive Model. """
    if tkey not in adata.obs.keys():
        raise ValueError(f'{tkey} not found in adata.obs')
    t = adata.obs[tkey]
    
    if same_plot:
        hide_cells = True
        logging.info('Setting `hide_cells` to True because of plotting trends in the same plot.')
    
    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        raise ValueError('genes not found in adata.var_names')
    if gn < ncols:
        ncols = gn

    cell_annot = None
    if color in adata.obs and is_numeric_dtype(adata.obs[color]):
        colors = adata.obs[color].values
    elif color in adata.obs and is_categorical_dtype(adata.obs[color])\
        and color+'_colors' in adata.uns.keys():
        cell_annot = adata.obs[color].cat.categories
        if isinstance(adata.uns[f'{color}_colors'], dict):
            colors = list(adata.uns[f'{color}_colors'].values())
        elif isinstance(adata.uns[f'{color}_colors'], list):
            colors = adata.uns[f'{color}_colors']
        elif isinstance(adata.uns[f'{color}_colors'], np.ndarray):
            colors = adata.uns[f'{color}_colors'].tolist()
        else:
            raise ValueError(f'Unsupported adata.uns[{color}_colors] object')
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    nrows = -(-gn // ncols)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(6*ncols, 4*(-(-gn // ncols)))
                            if figsize is None else figsize,
                            sharex=True, 
                            sharey=sharey,
                            tight_layout=True,
                            dpi=dpi)

    fig.patch.set_facecolor('white')
    axs = np.reshape(axs, (nrows, ncols))
    logging.info("Plotting trends")

    gam_kwargs = {
        'n_splines': 6,
        'spline_order': 3
    }
    gam_kwargs.update(kwargs)
    gam_results = np.zeros((len(genes), n_x_grid))

    cnt = 0
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
        tx = gam.generate_X_grid(term=0, n=n_x_grid)
        row = i // ncols
        col = i % ncols
        ax = axs[row, col] if not same_plot else axs

        if not hide_trend:
            ax.plot(tx[:, 0], gam.predict(tx))
            if not hide_interval:
                ci = gam.confidence_intervals(tx, width=0.95)
                lower_bound, upper_bound = ci[:, 0], ci[:, 1]
                ax.fill_between(tx[:, 0], lower_bound, upper_bound, color='#cabad7', alpha=0.5)
                ax.plot(tx[:, 0], gam.confidence_intervals(tx, width=0.95), c='#cabad7', ls='--')
        if not hide_cells:
            if cell_annot is not None:
                for j in range(len(cell_annot)):
                    filt = adata.obs[color] == cell_annot[j]
                    filt = np.ravel(filt)
                    ax.scatter(t[filt], x[filt], c=colors[j], s=pointsize, alpha=0.5)
            else:
                ax.scatter(t, x, c=colors, s=pointsize, alpha=0.5, cmap=cmap)
        if set_label:
            ax.set_ylabel(gene)

        if zero_indicator:
            ax.axhline(y=0, color='red', linestyle='--')

        if return_gam_result:
            gam_results[i] = gam.predict(tx)
        
        cnt += 1

    if set_label:
        fig.text(0.5, 0.01, tkey, ha='center') 
    for i in range(col+1, ncols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()
    
    if return_gam_result:
        return gam_results
    else:
        return


def response(
    adata,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = 'M_sc',
    ykey: Optional[str] = 'jacobian',
    hide_cells: bool = True,
    annot_key: str = 'celltype',
    downsampling: int = 3,
    log: bool = True,
    drop_zero_cells: bool = True,
    cell_idx: Optional[pd.Index] = None,
    perc: Optional[tuple] = None,
    grid_num: int = 25,
    kde_backend: Literal['fixbdw', 'scipy', 'statsmodels'] = 'statsmodels',
    integral_rule: Literal['trapz', 'simps'] = 'trapz',
    plot_integration_uncertainty: bool = False,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = None,
    curve_style: str = "c-",
    zero_indicator: bool = True,
    hide_mean: bool = False,
    hide_trend: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    show: bool = True,
    **kwargs,
    ):
    """ A modified plotting function of response curve.
     Adapted from dynamo package """
    from scipy import integrate
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    try:
        from dynamo.vectorfield.utils import get_jacobian
    except ImportError:
        raise ImportError(
            "If you want to show jacobian analysis in plotting function, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    
    if integral_rule == 'trapz':
        rule = integrate.cumulative_trapezoid
    elif integral_rule == 'simps': 
        rule = integrate.simpson
    else:
        raise ValueError("Integral rule only support `trapz` and `simps` implemented by scipy.") 

    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "response", ["#000000", "#000000", "#000000", "#800080", "#FF0000", "#FFFF00"]
        )
    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.0125, 0.0, 1, 1),
        "borderpad": 0,
    }

    if not hide_cells:
        if annot_key in adata.obs and is_categorical_dtype(adata.obs[annot_key]) \
                and annot_key+'_colors' in adata.uns.keys():
            cell_annot = adata.obs[annot_key].cat.categories
            if isinstance(adata.uns[f'{annot_key}_colors'], dict):
                colors = list(adata.uns[f'{annot_key}_colors'].values())
            else:
                colors = adata.uns[f'{annot_key}_colors']

    all_genes_in_pair = np.unique(pairs_mat)
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    flat_res = pd.DataFrame(columns=["x", "y", "den", "type"])
    xy = pd.DataFrame()
    # extract information from dynamo output in adata object
    id = 0
    for _, gene_pairs in enumerate(pairs_mat):
        f_ini_ind = (grid_num**2) * id

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        if xkey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
            x = flatten(J_df[jkey])
        else:
            x = flatten(adata[:, gene_pairs[0]].layers[xkey])

        if ykey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
            y_ori = flatten(J_df[jkey])
        else:
            y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])

        if drop_zero_cells:
            finite = np.isfinite(x + y_ori)
            nonzero = np.abs(x) + np.abs(y_ori) > 0
            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori)

        if cell_idx is not None:
            # subset cells for cell type-specific visualization
            subset_idx = np.zeros(adata.n_obs)
            idx = adata.obs_names.get_indexer(cell_idx)
            subset_idx[idx] = True
            valid_ids = np.logical_and(valid_ids, subset_idx)
        
        if perc is not None:
            # filter out outliers
            lb = np.percentile(x, perc[0])
            ub = np.percentile(x, perc[1])
            valid_ids = np.logical_and(valid_ids, np.logical_and(x>lb, x<ub))

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = x if sum(x < 0) else np.log(np.array(x) + 1), y_ori if sum(y_ori) < 0 else np.log(
                np.array(y_ori) + 1)

        y = y_ori

        # den_res[0, 0] is at the lower bottom; dens[1, 4]: is the 2nd on x-axis and 5th on y-axis
        x_meshgrid, y_meshgrid, den_res = kde2d(
            x, y, n=[grid_num, grid_num], lims=[min(x), max(x), min(y), max(y)], 
            backend=kde_backend
        )
        den_res = np.array(den_res)

        den_x = np.sum(den_res, axis=1)  # condition on each input x, sum over y

        for i in range(len(x_meshgrid)):
            tmp = den_res[i] / den_x[i]  # condition on each input x, normalize over y
            tmp = den_res[i]
            max_val = max(tmp)
            min_val = min(tmp)

            rescaled_val = (tmp - min_val) / (max_val - min_val)
            res_row = pd.DataFrame(
                {
                    "x": x_meshgrid[i],
                    "y": y_meshgrid,
                    "den": rescaled_val,
                    "type": gene_pair_name,
                },
                index=[i * len(x_meshgrid) + np.arange(len(y_meshgrid)) + f_ini_ind],
            )

            flat_res = pd.concat([flat_res, res_row])

        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)
        id = id + 1

    gene_pairs_num = len(flat_res.type.unique())

    # plot jacobian results and fitting curve
    n_col = -(-gene_pairs_num // n_row) if n_col is None else n_col

    if n_row * n_col < gene_pairs_num:
        raise ValueError("The number of row or column specified is less than the gene pairs")
    figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

    def scale_func(x, X, grid_num):
        return grid_num * (x - np.min(X)) / (np.max(X) - np.min(X))

    fit_integration = pd.DataFrame(columns=["x", "y", "mean", "ci_lower", "ci_upper", "regulators", "effectors"])

    for x, flat_res_type in enumerate(flat_res.type.unique()):
        gene_pairs = flat_res_type.split("->")

        flat_res_subset = flat_res[flat_res["type"] == flat_res_type]
        xy_subset = xy[xy["type"] == flat_res_type]

        x_val, y_val = flat_res_subset["x"], flat_res_subset["y"]

        i, j = x % n_row, x // n_row

        values = flat_res_subset["den"].values.reshape(grid_num, grid_num).T

        axins = inset_axes(axes[i, j], bbox_transform=axes[i, j].transAxes, **inset_dict)

        ext_lim = (min(x_val), max(x_val), min(y_val), max(y_val))
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            cmap=cmap)
        
        cb = fig.colorbar(im, cax=axins)
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=False)
        cb.update_ticks()

        # closest_x_ind = np.array([np.searchsorted(x_meshgrid, i) for i in xy_subset["x"].values])
        # closest_y_ind = np.array([np.searchsorted(y_meshgrid, i) for i in xy_subset["y"].values])
        # valid_ids = np.logical_and(closest_x_ind < grid_num, closest_y_ind < grid_num)
        # axes[i, j].scatter(closest_x_ind[valid_ids], closest_y_ind[valid_ids], color="gray", alpha=0.1, s=1)

        if xkey.startswith("jacobian"):
            axes[i, j].set_xlabel(r"$\partial f_{%s} / {\partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_xlabel(gene_pairs[0] + rf" (${xkey}$)")
        if ykey.startswith("jacobian"):
            axes[i, j].set_ylabel(r"$\partial f_{%s} / \partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
            axes[i, j].title.set_text(r"$\rho(\partial f_{%s} / \partial x_{%s})$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_ylabel(gene_pairs[1] + rf" (${ykey}$)")
            axes[i, j].title.set_text(rf"$\rho_{{{gene_pairs[1]}}}$ (${ykey}$)")

        xlabels = list(np.linspace(ext_lim[0], ext_lim[1], 5))
        ylabels = list(np.linspace(ext_lim[2], ext_lim[3], 5))

        # zero indicator
        if zero_indicator:
            axes[i, j].plot(
                scale_func([np.min(xlabels), np.max(xlabels)], xlabels, grid_num),
                scale_func(np.zeros(2), ylabels, grid_num),
                'w--',
                linewidth=2.0)

        # curve fiting using pygam
        gam_kwargs = {
            'n_splines': 6,
            'spline_order': 3
        }
        gam_kwargs.update(kwargs)
        if ykey.startswith("jacobian"):
            logging.info("Fitting response curve using GAM...")
            x_grid, y_mean, y_sigm = kde2d_to_mean_and_sigma(
                np.array(x_val), np.array(y_val), flat_res_subset["den"].values)
            y_mean[np.isnan(y_mean)] = 0
            term  =s(
                0,
                **gam_kwargs)
            # w = (1/y_sigm - (1/y_sigm).min()) / ((1/y_sigm).max()-(1/y_sigm).min())
            gam = LinearGAM(term, max_iter=1000, verbose=False).fit(x_grid, y_mean)
            if not hide_mean:
                axes[i, j].plot(
                    scale_func(x_grid, xlabels, grid_num), 
                    scale_func(y_mean, ylabels, grid_num), 
                    "c*"
                )
            if not hide_trend:
                axes[i, j].plot(
                    scale_func(x_grid, xlabels, grid_num), 
                    scale_func(gam.predict(x_grid), ylabels, grid_num), 
                    curve_style
                )

            # Integrate the fitted curve using the integration rule
            int_grid = np.linspace(x_grid.min(), x_grid.max(), 100)
            prediction = gam.predict(int_grid)
            confidence_intervals = gam.confidence_intervals(int_grid, width=0.95)
            integral = rule(prediction, int_grid, initial=0)
            mean_integral = integral
            ci_lower = np.zeros_like(integral)
            ci_upper = np.zeros_like(integral)

            # Perform Monte Carlo simulations to calculate confident interval
            if plot_integration_uncertainty: 
                logging.info('Simulation integral interval using Monte Carlo method...')
                n_simulations = 1000
                simulated_integrals = []
                for _ in range(n_simulations):
                    # Simulate the fitted values within their confidence intervals
                    simulated_values = np.random.uniform(confidence_intervals[:, 0], confidence_intervals[:, 1])
                    # Integrate the simulated values
                    simulated_integral = rule(simulated_values, int_grid, initial=0)
                    simulated_integrals.append(simulated_integral)
                simulated_integrals = np.array(simulated_integrals)
                mean_integral = np.mean(simulated_integrals, axis=0)
                ci_lower = np.percentile(simulated_integrals, 2.5, axis=0)
                ci_upper = np.percentile(simulated_integrals, 97.5, axis=0)

            tmp_integration = pd.DataFrame({
                "x": int_grid, "y": integral, "mean": mean_integral, "ci_lower": ci_lower, "ci_upper": ci_upper, 
                "regulators": gene_pairs[0], "effectors": gene_pairs[1]})
            fit_integration = pd.concat([fit_integration, tmp_integration])

        if not hide_cells:
            y_for_cells = y_meshgrid[-2]
            for type_i in range(len(cell_annot)):
                filt = adata[valid_ids].obs[annot_key] == cell_annot[type_i]
                filt = np.ravel(filt)
                tmp_x = xy_subset["x"].values[filt] # NOTE: xy already subset by valid_ids
                tmp_y = np.full_like(tmp_x, y_for_cells)
                axes[i, j].scatter(scale_func(tmp_x[::downsampling], xlabels, grid_num), tmp_y[::downsampling], s=12, color=colors[type_i], alpha=0.8)

        # set the x/y ticks
        inds = np.linspace(0, grid_num - 1, 5, endpoint=True)
        axes[i, j].set_xticks(inds)
        axes[i, j].set_yticks(inds)

        if ext_lim[1] < 1e-3:
            xlabels = ["{:.3e}".format(i) for i in xlabels]
        else:
            xlabels = [np.round(i, 2) for i in xlabels]
        if ext_lim[3] < 1e-3:
            ylabels = ["{:.3e}".format(i) for i in ylabels]
        else:
            ylabels = [np.round(i, 2) for i in ylabels]

        if ext_lim[1] < 1e-3:
            axes[i, j].set_xticklabels(xlabels, rotation=30, ha="right")
        else:
            axes[i, j].set_xticklabels(xlabels)

        axes[i, j].set_yticklabels(ylabels)

    plt.subplots_adjust(left=0.1, right=1, top=0.80, bottom=0.1, wspace=0.1)
    plt.tight_layout()
    if show:
        plt.show()

    else:
        return axes
