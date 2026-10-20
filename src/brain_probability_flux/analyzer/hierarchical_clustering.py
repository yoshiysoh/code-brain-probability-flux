import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import cluster
from sklearn.metrics.pairwise import cosine_similarity


def _similarity_matrix(data, matrix_type):
    """Similarity between the columns of `data`."""
    if matrix_type == "correlation":
        return np.corrcoef(data.T)
    if matrix_type == "cosine-similarity":
        return cosine_similarity(data.T)
    if matrix_type == "precomputed":
        return np.asarray(data)
    raise ValueError(
        f"Unknown matrix type: {matrix_type}. "
        "Supported types are: 'correlation', 'cosine-similarity', "
        "'precomputed'"
    )


def linkage_matrix(
    data,
    matrix_type="correlation",
    method="average",
    metric="euclidean",
    optimal_ordering=True,
    nan_to_zero=False,
):
    """Similarity matrix of the columns of `data` and its linkage matrix."""
    similarity = _similarity_matrix(data, matrix_type)
    if nan_to_zero:
        similarity = np.nan_to_num(similarity)
    Z = cluster.hierarchy.linkage(
        similarity, method, metric, optimal_ordering=optimal_ordering
    )
    return similarity, Z


def _set_link_colors(cmap):
    color_map = plt.get_cmap(cmap)
    colors = [mcolors.to_hex(color_map(i)) for i in range(color_map.N)]
    cluster.hierarchy.set_link_color_palette(colors)


def cluster_parcels(
    X_data,
    threshold=0.70,
    num_clusters=None,
    cmap="tab10",
    matrix_type="correlation",
):
    """Cluster index of each parcel (column of X_data).

    By default the clusters are the colored subtrees of the dendrogram cut
    at `threshold` times the largest merge distance, numbered as
    plot_dendrogram colors them. With `num_clusters`, the tree is instead
    cut into that many clusters (scipy's fcluster, "maxclust").
    """
    _, Z = linkage_matrix(X_data, matrix_type=matrix_type)
    if num_clusters is not None:
        return cluster.hierarchy.fcluster(Z, num_clusters, "maxclust") - 1
    _set_link_colors(cmap)
    dn = cluster.hierarchy.dendrogram(
        Z,
        color_threshold=threshold * max(Z[:, 2]),
        no_plot=True,
        above_threshold_color="k",
    )
    return dendrogram_to_cluster_inds(dn)


def _clustermap_layout(g, ax):
    """Shrink seaborn's outer whitespace to the constrained_layout pads.

    Leaves room for the heatmap's axis labels and moves the colorbar next
    to the column dendrogram so that its label does not overlap the
    row dendrogram.
    """
    _fw, _fh = g.figure.get_size_inches()
    _wp = plt.rcParams.get("figure.constrained_layout.w_pad", 0.025002)
    _hp = plt.rcParams.get("figure.constrained_layout.h_pad", 0.025002)
    _new_left = _wp / _fw
    g.figure.canvas.draw()
    renderer = g.figure.canvas.get_renderer()
    # Right: space for the y-axis label (seaborn puts it on the right).
    _bb_ylbl = ax.yaxis.label.get_window_extent(renderer)
    _ylbl_space = _bb_ylbl.x1 / g.figure.dpi - ax.get_position().x1 * _fw
    _new_right = 1.0 - (_ylbl_space + _wp) / _fw
    # Bottom: space for the x-axis label (below the heatmap).
    _bb_xlbl = ax.xaxis.label.get_window_extent(renderer)
    _xlbl_space = ax.get_position().y0 * _fh - _bb_xlbl.y0 / g.figure.dpi
    _new_bottom = (_hp + max(0.0, _xlbl_space)) / _fh
    # Top: h_pad above the column dendrogram.
    _new_top = 1.0 - _hp / _fh
    _cb0 = g.ax_cbar.get_position()
    _cb_top = _cb0.y0 + _cb0.height
    g.figure.subplots_adjust(
        left=_new_left, right=_new_right, bottom=_new_bottom, top=_new_top
    )
    _col_d_bot = g.ax_col_dendrogram.get_position().y0
    g.ax_cbar.set_position(
        [_new_left, _col_d_bot, _cb0.width, _cb_top - _col_d_bot]
    )
    g.ax_cbar.tick_params(labelsize=5)
    g.ax_cbar.yaxis.label.set_size(5)


def _seaborn_clustermap(data, **kwargs):
    """sns.clustermap with constrained_layout off (it calls tight_layout)."""
    use_cl = plt.rcParams["figure.constrained_layout.use"]
    plt.rcParams["figure.constrained_layout.use"] = False
    try:
        return sns.clustermap(data, rasterized=True, **kwargs)
    finally:
        plt.rcParams["figure.constrained_layout.use"] = use_cl


def _clear_heatmap_ticks(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_clustermap(
    data,
    matrix_type="correlation",
    method="average",
    metric="euclidean",
    optimal_ordering=True,
    ax_label=None,
    cbar_label="Correlation",
    return_linkage_matrices=False,
    nan_to_zero=False,
    figsize=None,
):
    """Clustered heat map of the similarity between the columns of data.

    Returns (seaborn ClusterGrid, heat-map axes[, linkage matrix]).
    """
    corr, Z = linkage_matrix(
        data, matrix_type, method, metric, optimal_ordering, nan_to_zero
    )
    g = _seaborn_clustermap(
        corr,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cc.cm.CET_D1A,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": cbar_label},
        figsize=figsize,
    )
    ax = g.ax_heatmap
    _clear_heatmap_ticks(ax, ax_label, ax_label)
    _clustermap_layout(g, ax)

    if return_linkage_matrices:
        return g, ax, Z
    else:
        return g, ax


def square_clustermap_height(data, width, num_passes=5, **kwargs):
    """Figure height at which plot_clustermap(data, figsize=(width,
    height), **kwargs) draws a square heatmap.

    Each pass trims the margins, which changes the heatmap's height
    fraction, so the height is refined iteratively; 5 passes converge to
    < 0.1 pt asymmetry.
    """
    height = width
    for _ in range(num_passes):
        g, ax = plot_clustermap(data, figsize=(width, height), **kwargs)
        heatmap = ax.get_position()
        fig_w, fig_h = g.figure.get_size_inches()
        heatmap_w = heatmap.width * fig_w
        heatmap_h = heatmap.height * fig_h
        plt.close(g.figure)
        if abs(heatmap_w - heatmap_h) < 1e-4:
            break
        height = height * (heatmap_w / heatmap_h)
    return height


def plot_dendrogram(
    Z,
    labels=None,
    no_labels=False,
    leaf_font_size=None,
    cmap="tab10",
    threshold=0.7,
    draw_threshold=True,
    draw_threshold_legend=True,
    figsize=(6.4, 6.4),
    dpi=None,
    return_dn=False,
):
    """
    Create dendrogram from linkage matrix.

    Parameters
    ----------
    Z : linkage matrix
        See scipy.cluster.hierarchy.linkage for details.
    labels : 1D array
        Name of elements.
    threshold : float
        Color threshold of dendrogram.
    draw_threshold : bool
        If True, the line of thredhold is drawn.
    figsize : 1D array
    dpi : int
    return_dn : bool
        If True, the dictionary of dendrogram is returned.


    Returns
    -------
    fig : matplotlib fig
    ax : matplotlib ax
    dn : dict
    """
    td = threshold * max(Z[:, 2])
    _set_link_colors(cmap)

    if labels is None:
        labels = np.arange(Z.shape[0] + 1) + 1

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    dn = cluster.hierarchy.dendrogram(
        Z,
        color_threshold=td,
        labels=labels,
        no_labels=no_labels,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        ax=ax,
        above_threshold_color="k",
    )
    ax.set_ylabel("Dissimilarity")

    if draw_threshold:
        ax.axhline(
            td,
            color="black",
            lw=plt.rcParams["axes.linewidth"],
            ls="--",
            label="Threshold",
        )
        if draw_threshold_legend:
            ax.legend(frameon=False, fontsize=5)
    if return_dn:
        return fig, ax, dn
    else:
        return fig, ax


def dendrogram_to_cluster_inds(dn):
    """Cluster index of each leaf, numbered as the dendrogram colors appear."""
    cluster_sorted_region = np.array(dn["leaves"])
    inds = np.argsort(cluster_sorted_region)
    cluster_color = np.array(dn["leaves_color_list"])

    # number the clusters by the order their colors first appear
    unique_colors = cluster_color[
        np.sort(np.unique(cluster_color, return_index=True)[1])
    ]
    color_to_id = {color: i for i, color in enumerate(unique_colors)}

    cluster_id = [color_to_id[color] for color in cluster_color]
    cluster_id = np.array(cluster_id)
    cluster_inds = cluster_id[inds]
    return cluster_inds


def plot_hierarchical_data(
    data,
    row_matrix_type="correlation",
    row_method="average",
    row_metric="euclidean",
    row_optimal_ordering=True,
    col_matrix_type="cosine-similarity",
    col_method="average",
    col_metric="euclidean",
    col_optimal_ordering=True,
    nan_to_zero=False,
    cmap=cc.cm.CET_D1A,
    vmin=None,
    vmax=None,
    ax_xlabel=None,
    ax_ylabel=None,
    cbar_label=None,
    figsize=(6.4, 6.4),
):
    """
    Create hierarchical data matrix.

    Parameters
    ----------
    data : 2D array
        Data matrix with Ising variables. Row index indicates state.
    method, metric : string
        See available option from scipy.cluster.hierarchy.linkage
    optimal_ordering : bool
        See available option from scipy.cluster.hierarchy.linkage

    Returns
    -------
    g : seaborn cluster grid
    ax : matplotlib ax


    """
    # rows of `data` are states, columns are spins
    _, row_link = linkage_matrix(
        data,
        row_matrix_type,
        row_method,
        row_metric,
        row_optimal_ordering,
        nan_to_zero,
    )
    _, col_link = linkage_matrix(
        data.T,
        col_matrix_type,
        col_method,
        col_metric,
        col_optimal_ordering,
        nan_to_zero,
    )
    g = _seaborn_clustermap(
        data,
        row_linkage=col_link,
        col_linkage=row_link,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
    )
    ax = g.ax_heatmap
    _clear_heatmap_ticks(ax, ax_xlabel, ax_ylabel)
    ax.collections[0].colorbar.set_label(cbar_label)
    _clustermap_layout(g, ax)
    return g, ax
