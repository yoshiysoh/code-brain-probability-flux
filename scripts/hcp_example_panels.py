"""Draw the panels that show one example recording.

These panels plot individual-level data, so they are not made from the
shipped results: they need the HCP recordings in data/HCP_data_task/.

figures/fig-01/f, g, h  binarization, state raster and transitions (Fig. 1)
figures/fig-s05/a-f     the static, dynamic and curve binarizations
figures/fig-s21/a-f     the same panels, repeated in fig. S21

Usage: python scripts/hcp_example_panels.py
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp

log = logging.getLogger(__name__)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
# The example: the first recording of the motor task
TASK = 7
# Row-2 panel size of Fig. 1 (inches)
ROW2_HEIGHT = 2.6
ROW2_WIDTH = (bpp.FIG_WIDTH * 72 - 2 * bpp.PANEL_GAP_PT) / 3 / 72
# Magnified window of Fig. 1h (s): every state box spans the full y-axis
# without overlapping (transitions at 54.57, 55.19 and 56.03 s).
T_START, T_END = 54.15, 56.65


def fig1_panels(recording, num_clusters, out_dir):
    t_trajectory = recording["t_trajectory"]
    fig, axs = plt.subplots(
        num_clusters,
        1,
        figsize=(ROW2_WIDTH, ROW2_HEIGHT),
        sharex=True,
        sharey=True,
    )
    bpp.plot_binarization(axs, recording)
    for j, ax in enumerate(axs):
        ax.axhline(
            0, c="k", ls="--", zorder=0, lw=plt.rcParams["axes.linewidth"]
        )
        ax.set_ylabel(f"Cluster {j + 1}", rotation=0, ha="right", va="center")
    ax.set_xlim(0, t_trajectory[-1])
    ax.set_xlabel("Time (s)")
    fig.supylabel("Z score of coarse-grained activity")
    bpp.save_panel(fig, out_dir, "f")

    fig, ax = plt.subplots(figsize=(ROW2_WIDTH, ROW2_HEIGHT))
    bpp.plot_state_raster(ax, recording)
    ax.set_xlim(0, t_trajectory[-1])
    ax.set_xlabel("Time (s)")
    bpp.save_panel(fig, out_dir, "g")

    fig, axs = plt.subplots(2, figsize=(ROW2_WIDTH, ROW2_HEIGHT), sharex=True)
    ax = axs[0]
    bpp.plot_state_raster(ax, recording)
    for t in t_trajectory:
        ax.axvline(t, lw=plt.rcParams["axes.linewidth"], c="k", ls="--")

    ax = axs[1]
    first = np.searchsorted(t_trajectory, T_START)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(-0.5, num_clusters - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(
        range(num_clusters),
        labels=[f"Cluster {j + 1}" for j in range(num_clusters)],
    )
    ax.set_xlabel("Time (s)")
    # Box width (data units) such that a 1 x N equal-aspect box fills the
    # axis height; computed after layout so the final axis size is used.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    box_w = (bbox.height / num_clusters) / bbox.width * (T_END - T_START)
    box_w_pt = bbox.height / num_clusters * 72 / fig.dpi

    arrow_y = 3
    edges = (
        [T_START] + [t for t in t_trajectory[first:] if t < T_END] + [T_END]
    )
    for k in range(len(edges) - 1):
        ax.annotate(
            "",
            [edges[k], arrow_y],
            [edges[k + 1], arrow_y],
            arrowprops={
                "facecolor": "black",
                "arrowstyle": "<-",
                "lw": 0.3,
                "shrinkA": 0,
                "shrinkB": (box_w_pt if k > 0 else 1),
            },
            transform=ax.transData,
            zorder=0,
        )
    for k, t in enumerate(edges[1:-1]):
        axin = ax.inset_axes(
            [t, -0.5, box_w, num_clusters], transform=ax.transData
        )
        state = recording["states"][first + k]
        color = np.vstack(np.arange(num_clusters)).astype(float)
        color[np.argwhere(state == -1)] = np.nan
        axin.pcolormesh(
            color[::-1],
            cmap=plt.cm.tab10,
            vmin=0,
            vmax=9,
            edgecolor="k",
            lw=0.25,
            zorder=1,
        )
        axin.set_aspect("equal")
        axin.set_xticks([])
        axin.set_yticks([])
        axin.spines["top"].set_visible(True)
        axin.spines["right"].set_visible(True)
    bpp.save_panel(fig, out_dir, "h")


def binarization_panels(pfa, num_clusters, out_dirs):
    """Panels a-f of figs. S5 and S21: binarization at the zeros of the z
    score's 0th, 1st and 2nd derivative."""
    names = [
        "Static transformation",
        "Dynamic transformation",
        "Curve transformation",
    ]
    with plt.rc_context({"font.size": 5.0}):
        for order, name in enumerate(names):
            pfa.zscore_to_trajectory(order)
            recording = bpp.example_recording(pfa, task_choice=TASK)
            t_end = recording["t_trajectory"][-1]

            fig, axs = plt.subplots(
                num_clusters,
                1,
                figsize=(bpp.FIG_WIDTH / 3, bpp.FIG_HEIGHT / 2.5),
                sharex=True,
                sharey=True,
            )
            bpp.plot_binarization(axs, recording)
            for j, ax in enumerate(axs):
                if order == 0:
                    ax.axhline(
                        0,
                        c="k",
                        ls="--",
                        zorder=0,
                        lw=plt.rcParams["axes.linewidth"],
                    )
                ax.set_ylabel(f"Cluster {j + 1}")
            ax.set_xlim(0, t_end)
            fig.suptitle(name)
            fig.supxlabel("Time (s)")
            fig.supylabel("Z score of coarse-grained activity")
            for out_dir in out_dirs:
                bpp.save_panel(fig, out_dir, "abc"[order])

            height = bpp.FIG_HEIGHT * (1 - 1 / 2.5 - 1 / 3) - 0.25
            fig, ax = plt.subplots(figsize=(bpp.FIG_WIDTH / 3, height))
            bpp.plot_state_raster(ax, recording)
            ax.set_xlim(0, t_end)
            ax.set_xlabel("Time (s)")
            for out_dir in out_dirs:
                bpp.save_panel(fig, out_dir, "def"[order])
            plt.close("all")


def main() -> None:
    pfa, cluster_inds = bpp.coarse_grained_pfa()
    num_clusters = len(np.unique(cluster_inds))
    recording = bpp.example_recording(pfa, task_choice=TASK)
    fig1_panels(recording, num_clusters, FIGURES / "fig-01")
    binarization_panels(
        pfa, num_clusters, [FIGURES / "fig-s05", FIGURES / "fig-s21"]
    )
    log.info("Wrote the example panels under %s", FIGURES)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
