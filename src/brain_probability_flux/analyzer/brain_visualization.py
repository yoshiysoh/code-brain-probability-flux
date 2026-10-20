import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting


def _expose_env_libs_to_chrome() -> None:
    """Let kaleido's headless Chrome find NSS/NSPR installed in this env.

    Linux servers often lack the system libnss3/libnspr4 Chrome needs; the
    conda-forge copies live in <env>/lib, which the dynamic loader does not
    search for the Chrome subprocess unless listed in LD_LIBRARY_PATH.
    """
    if not sys.platform.startswith("linux"):
        return
    env_lib = str(Path(sys.prefix) / "lib")
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if env_lib not in paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [env_lib, *filter(None, paths)]
        )


_expose_env_libs_to_chrome()


def create_surf_plot(
    lh,
    atlas,
    cluster_inds,
    cluster_state=None,
    num_colors=10,
    base_color="white",
    cmap="tab10",
    view="lateral",
):
    """
    Creates a surface plot with a custom color scale and lighter color for inactive regions.

    Parameters:
    - lh: The left hemisphere surface data.
    - atlas: The parcellation atlas data.
    - cluster_inds: Indices for clustering.
    - cluster_state: Binary cluster_state array (e.g., [-1, -1, +1, -1, +1]) for lighter color.
    - num_colors: Number of colors in the cmap (default: 10).
    - base_color: The base color to map value 0 (default: 'white').
    - cmap: The cmap to use for colors (default: 'tab10').
    - view: The view for the plot (default: 'lateral').

    Returns:
    - A Plotly figure object.
    """
    # Load the cmap
    cmap_obj = plt.get_cmap(cmap, num_colors)

    def rgb_to_plotly_str(rgb_tuple):
        """Plotly 'rgb(r, g, b)' string of an RGB tuple in [0, 1]."""
        return f"rgb({int(rgb_tuple[0] * 255)}, {int(rgb_tuple[1] * 255)}, {int(rgb_tuple[2] * 255)})"

    def lighten_color(rgb_tuple, factor=0.5):
        """Blend a color with white by `amount`."""
        r, g, b = rgb_tuple
        r = r + (1.0 - r) * factor
        g = g + (1.0 - g) * factor
        b = b + (1.0 - b) * factor
        return (r, g, b)

    # Generate c_latlas based on atlas, set background to np.nan
    c_latlas = np.full_like(atlas, np.nan, dtype=float)
    for i, parcel in enumerate(atlas - 1):
        if parcel == -1:
            c_latlas[i] = np.nan
        else:
            c_latlas[i] = cluster_inds[parcel]

    # Create a surface plot figure using nilearn
    fig = plotting.plot_surf(
        lh,
        c_latlas,
        cmap=cmap_obj,
        colorbar=False,
        symmetric_cmap=False,
        view=view,
        engine="plotly",
    )
    plotly_fig = fig.figure

    # Prepare color list
    colors = [cmap_obj(i)[:3] for i in range(num_colors)]

    # Create vertex colors based on the custom colorscale and cluster_state
    vertex_colors = []
    for idx, val in enumerate(c_latlas):
        if np.isnan(val):
            vertex_colors.append(base_color)
        else:
            color = colors[min(int(val), num_colors - 1)]
            cluster_label = int(val)
            if (
                cluster_state is not None
                and cluster_state[cluster_label] == -1
            ):
                color = lighten_color(color, factor=0.5)
            vertex_colors.append(rgb_to_plotly_str(color))

    plotly_fig.data[0].update(vertexcolor=vertex_colors, showscale=False)

    plotly_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    return plotly_fig
