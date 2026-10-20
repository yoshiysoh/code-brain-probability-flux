"""Plots of interaction matrices and the networks they define."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_im(
    im,
    plot_cbar=True,
    return_cbar=False,
    fig=None,
    ax=None,
    figsize=None,
    dpi=None,
    vmin=None,
    vmax=None,
    cmap=None,
):
    """
    Plot function for interaction matrix
    """
    im = np.array(im)

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi, constrained_layout=True
        )
    if vmin is None and vmax is None:
        vmax = np.ceil(np.abs(im).max())
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin
    elif vmin is None:
        vmin = -vmax
    if cmap is None:
        cmap = plt.cm.RdYlGn
    cbar = ax.matshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    if plot_cbar:
        cb = fig.colorbar(cbar, label="Interaction", drawedges=False)
        cb.outline.set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.xaxis.set_label_position("top")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if return_cbar:
        return fig, ax, cbar
    else:
        return fig, ax


def plot_network(
    im,
    directed=None,
    pos="circular",
    node_color=None,
    plot_label=False,
    labels=None,
    cmap=plt.cm.binary,
    vmin=None,
    vmax=None,
    node_size=300,
    linewidths=0.1,
    edge_cmap=plt.cm.RdYlGn,
    arrowsize=10,
    edge_width_scale=1.0,
    arrow_rad=0.1,
    font_size=12,
    font_color="w",
    font_family="sans-serif",
    font_weight="normal",
    return_nodes=False,
    ax=None,
    figsize=None,
    dpi=None,
):
    """
    Network plot function for interaction matrix.

    Parameters
    ----------
    im : (N, N) array
        Interaction matrix, im[target, source].
    directed : bool or None
        If None, the graph is directed unless im is symmetric.
    pos : "circular" or dict
        Node positions.
    labels : 1D array
        Name of elements.
    node_size : float
        Size of node of network plot.
    arrow_size : float
        Size of arrow.
    arrow_width : float
        Width of arrow.
    arrow_rad : float
        Curvature of edges.

    Returns
    -------
    ax, or (fig, ax) if no ax was given; nodes is appended if return_nodes.

    """
    N = len(im)
    # NetworkX reads element (i, j) as an edge i -> j, so pass the transpose
    # of the interaction matrix (target, source).
    im = np.array(im)
    if directed is None:
        directed = not (im == im.T).all()
    create_using = nx.DiGraph() if directed else nx.Graph()
    G = nx.from_numpy_array(im.T, create_using=create_using)

    if pos == "circular":
        pos = nx.circular_layout(G)
    elif not isinstance(pos, dict):
        raise ValueError("pos should be a dict or 'circular'")

    if vmin is None and vmax is None:
        if node_color is None:
            node_color = np.ones(N)
            vabsmax = np.max(np.abs(node_color))
            vmin = -vabsmax
            vmax = vabsmax
        elif isinstance(node_color, (int, float)):
            node_color = np.ones(N) * node_color
            vabsmax = np.max(np.abs(node_color))
            vmin = -vabsmax
            vmax = vabsmax
        else:
            if len(np.unique(node_color)) == 2:
                if (np.unique(node_color) == np.array([-1.0, 1.0])).all():
                    vabsmax = 1
                    vmin = -1
                    vmax = 1
                    node_color = np.array(node_color)
                else:
                    vabsmax = np.max(np.abs(node_color))
                    vmin = -vabsmax
                    vmax = vabsmax
                    cmap = plt.cm.coolwarm
            else:
                vabsmax = np.max(np.abs(node_color))
                vmin = -vabsmax
                vmax = vabsmax
                cmap = plt.cm.coolwarm
    elif vmin is None:
        vmin = np.min(node_color)
    elif vmax is None:
        vmax = np.max(node_color)
    else:
        pass

    edge_weights = [
        edgedata["weight"] for _, _, edgedata in G.edges(data=True)
    ]
    edge_width = np.abs(edge_weights) * edge_width_scale
    edge_sign = np.sign(edge_weights)
    if edge_cmap is None:
        edge_color = "k"
    else:
        edge_color = edge_sign.copy()
    edge_style = edge_sign.copy().astype(str)
    edge_style = np.where(edge_style == "1.0", "-", edge_style)
    edge_style = np.where(edge_style == "0.0", "-", edge_style)
    edge_style = np.where(edge_style == "-1.0", "--", edge_style)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi, constrained_layout=True
        )
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        alpha=1.0,
        node_size=node_size,
        node_color=node_color,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        edgecolors="black",
        linewidths=linewidths,
    )
    # draw edges
    if directed:
        edges = nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=0.01,
            connectionstyle=f"arc3, rad={arrow_rad}",
            style=edge_style,
            width=edge_width,
            edge_color=edge_color,
            edge_cmap=edge_cmap,
            edge_vmin=-1,
            edge_vmax=1,
        )
        for edge in edges:
            edge.set_capstyle("butt")
        # draw arrow heads
        edge_arrows = nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrowsize,
            connectionstyle=f"arc3, rad={arrow_rad}",
            style="-",
            width=0,
            edge_color=edge_color,
            edge_cmap=edge_cmap,
            edge_vmin=-1,
            edge_vmax=1,
        )
        for edge, w in zip(edge_arrows, edge_width):
            edge.set_arrowstyle(
                f"-|>, head_length={0.4 * w}, head_width={0.2 * w}"
            )
    else:
        edges = nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            arrows=True,
            arrowsize=arrowsize,
            connectionstyle=f"arc3, rad={arrow_rad}",
            style=edge_style,
            width=edge_width,
            edge_color=edge_color,
            edge_cmap=edge_cmap,
            edge_vmin=-1,
            edge_vmax=1,
        )
    if plot_label:
        if labels is None:
            labels = {i: i + 1 for i in np.arange(len(im))}
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            labels=labels,
            font_size=font_size,
            font_color=font_color,
            font_family=font_family,
            font_weight=font_weight,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    returned = (ax,) if fig is None else (fig, ax)
    if return_nodes:
        returned += (nodes,)
    return returned if len(returned) > 1 else ax
