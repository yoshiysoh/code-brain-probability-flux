"""Figure helpers shared by the notebooks and scripts.

Import as `import brain_probability_flux.plotting as bpp`. Holds the 8 HCP
task names, the coarse-grained PFA pipeline and the loaders of the
results in data/, the Arrhenius/Glauber model rates, hypercube
state-space helpers, task-color palettes, and the panel layouts that
several figures share.
"""

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance

from .analyzer import PFA
from .analyzer import hierarchical_clustering as hc
from .dina.dynamics import ising
from .dina.network import visualization as netvis
from .spin_states import (  # noqa: F401  (re-exported for the notebooks)
    create_state_vectors,
    generate_binary_codes,
    get_one_bit_difference_indices,
    get_one_bit_difference_indices_undirected,
)

# ── Constants ──────────────────────────────────────────────────────────────
# Canonical task names, matching analyzer.PFA().task_name_list exactly.
TASK_NAMES_FULL = [
    "Rest",
    "Emotion",
    "Working memory",
    "Social",
    "Language",
    "Relational",
    "Gambling",
    "Motor",
]
# Abbreviated form used for tick labels where space is tight.
TASK_NAMES_SHORT = [
    "Rest",
    "Emotion",
    "WM",
    "Social",
    "Language",
    "Relational",
    "Gambling",
    "Motor",
]
NUM_TASKS = len(TASK_NAMES_FULL)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AGGREGATES = "aggregates.h5"
# Panel sizes in inches: the full figure width (180 mm) and height (170 mm)
# that the panels were sized for.
FIG_WIDTH = 180 / 25.4
FIG_HEIGHT = 170 / 25.4
PANEL_FORMATS = ("pdf", "svg", "png")
# Gap between panels in the composite figures (pt)
PANEL_GAP_PT = 6.65048
# Yeo (2011) 7-network names in the Schaefer parcel labels -> abbreviations
YEO_ABBREVIATIONS = {
    "Default": "DMN",  # Default mode network
    "Cont": "FPN",  # Frontoparietal (control) network
    "Vis": "VIS",  # Visual network
    "SomMot": "SMN",  # Somatomotor network
    "DorsAttn": "DAN",  # Dorsal attention network
    "SalVentAttn": "VAN",  # Ventral attention (salience) network
    "Limbic": "LIM",  # Limbic network
}


# ── Coarse-grained PFA pipeline ────────────────────────────────────────────
def coarse_grained_pfa(threshold=0.70, num_clusters=None, order_derivative=0):
    """Load the HCP data, cluster the parcels, and binarize the cluster
    activity into trajectories (the setup shared by most figures).

    Returns (pfa, cluster_inds).
    """
    pfa = PFA()
    pfa.preprocessing()
    cluster_inds = hc.cluster_parcels(
        pfa.X_data, threshold=threshold, num_clusters=num_clusters
    )
    pfa.coarse_grain(cluster_inds, order_derivative)
    return pfa, cluster_inds


def spin_colors(num_spins, cmap=plt.cm.tab10):
    """One color per spin (cluster)."""
    return [cmap(i) for i in range(num_spins)]


def load_results(name, group="/", data_dir=None):
    """Datasets of one group of data/<name>.h5, as a dict of arrays."""
    path = Path(data_dir or DATA_DIR) / f"{name}.h5"
    with h5py.File(path, "r") as f:
        return {key: item[()] for key, item in f[group].items()}


def aggregated_pfa(setting="threshold_0.70", data_dir=None):
    """PFA restored from data/aggregates.h5 (group-level transition counts;
    no HCP recordings needed) for one coarse-graining setting.

    Returns (pfa, cluster_inds).
    """
    path = Path(data_dir or DATA_DIR) / AGGREGATES
    with h5py.File(path, "r") as f:
        group = f["settings"][setting]
        pfa = PFA.from_aggregates(group)
        cluster_inds = group["cluster_inds"][()]
    return pfa, cluster_inds


def parcel_correlation(data_dir=None):
    """Correlation matrix of the 100 parcels over all recordings."""
    path = Path(data_dir or DATA_DIR) / AGGREGATES
    with h5py.File(path, "r") as f:
        return f["parcel_correlation"][()]


def load_pc_grid(data_dir=None):
    """Flux on the grid of the first two PC scores (fig. S2): bin edges,
    bin centers, and one summary dict per task."""
    path = Path(data_dir or DATA_DIR) / AGGREGATES
    with h5py.File(path, "r") as f:
        group = f["pc_grid"]
        summaries = [
            {key: item[()] for key, item in group[f"task{k}"].items()}
            for k in range(NUM_TASKS)
        ]
        return group["bin_edges"][()], group["bin_pos"][()], summaries


def save_panel(fig, out_dir, name, formats=PANEL_FORMATS, bbox_inches="tight"):
    """Save `fig` as out_dir/name.<fmt> (PNG at 600 dpi on white)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"dpi": 600, "facecolor": "white"} if fmt == "png" else {}
        fig.savefig(
            out_dir / f"{name}.{fmt}", bbox_inches=bbox_inches, **kwargs
        )


def svg_width_pt(path):
    """Width (pt) of a saved SVG panel."""
    match = re.search(r'<svg[^>]*width="([0-9.]+)pt"', Path(path).read_text())
    return float(match.group(1))


def save_axes_panels(fig, axes_by_name, out_dir, formats=PANEL_FORMATS):
    """Save each axes of a multi-panel figure as its own file, cropped to
    the axes' tight bounding box (decorations included)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for name, ax in axes_by_name.items():
        axes = ax if isinstance(ax, (list, tuple)) else [ax]
        bbox = Bbox.union([a.get_tightbbox(renderer) for a in axes])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        save_panel(fig, out_dir, name, formats=formats, bbox_inches=bbox)


def flux_matrices_by_task(pfa):
    """Observed probability-flux matrix of each task, (tasks, NS, NS)."""
    matrices = []
    for k in range(NUM_TASKS):
        pfa.trajectories_to_flux(k, progress_message=False)
        matrices.append(pfa.flux_matrix.toarray())
    return np.array(matrices)


def observed_edge_flux(pfa):
    """Observed flux of each task on the directed hypercube edges.

    Returns (flux (tasks, edges), edge row indices, edge column indices).
    """
    num_spins = pfa.unique_states.shape[1]
    row, col = get_one_bit_difference_indices(num_spins)
    return flux_matrices_by_task(pfa)[:, row, col], row, col


def traffic_matrices_by_task(pfa):
    """Observed (symmetric) probability-traffic matrix of each task."""
    matrices = []
    for k in range(NUM_TASKS):
        pfa.trajectories_to_flux(k, progress_message=False)
        T = pfa.traffic_matrix.toarray()  # one triangle only
        matrices.append(T + T.T)
    return np.array(matrices)


def rate_matrix_from_transitions(transition_count, num_states):
    """Transition rates w[target, source] from rows (source, target, dwell
    time in source), as count / (total time * source probability)."""
    sources = transition_count[:, 0].astype(int)
    targets = transition_count[:, 1].astype(int)
    counts = np.zeros((num_states, num_states))
    np.add.at(counts, (targets, sources), 1)
    waiting = np.zeros(num_states)
    np.add.at(waiting, sources, transition_count[:, 2])
    total_time = waiting.sum()
    probability = np.where(waiting > 0, waiting / total_time, 1.0)
    rates = counts / (total_time * probability[np.newaxis, :])
    np.fill_diagonal(rates, 0.0)
    return rates


def model_traffic_matrix(beta_J, beta_h, A, weight_type="arrhenius"):
    """Traffic of the inferred model, scaled by the rate constant A as
    ising.im_to_flux does (ising.im_to_traffic has no rate_const)."""
    joint = ising.im_to_joint_transition(
        beta_J, 1.0, weight_type=weight_type, magnetic_field=beta_h
    )
    traffic = A * (joint + joint.T)
    np.fill_diagonal(traffic, 0.0)
    return traffic


def asymmetry_ratio(beta_J):
    """||J^(a)||_F / ||J^(s)||_F over the upper triangle; beta_J may be a
    stack (..., N, N)."""
    beta_J = np.asarray(beta_J)
    upper = np.triu_indices(beta_J.shape[-1], k=1)
    J_t = np.swapaxes(beta_J, -1, -2)
    sym = (0.5 * (beta_J + J_t))[..., upper[0], upper[1]]
    anti = (0.5 * (beta_J - J_t))[..., upper[0], upper[1]]
    return np.linalg.norm(anti, axis=-1) / np.linalg.norm(sym, axis=-1)


def interaction_panels(
    beta_J, beta_h, task_names, save_dir, vabs, labels, figsize
):
    """The nine coupling panels of fig-04 (and fig-s08), one figure each.

    In order: J, J^(s), J^(a) (|values| <= vabs); their relative change
    from rest; the network of J; the external input; and the network of J
    with nodes colored by the external input.
    """
    import colorcet as cc

    cmap = cc.cm.CET_D3_r
    sym = [get_symmetric(J) for J in beta_J]
    anti = [get_antisymmetric(J) for J in beta_J]
    num_clusters = beta_J[0].shape[0]
    grids = [
        (beta_J, "Interaction\n$J_{i, j}$", vabs, False),
        (sym, "Symmetric\n$J^{(s)}_{i, j}$", vabs, False),
        (anti, "Antisymmetric\n$J^{(a)}_{i, j}$", vabs, False),
        (
            [J / beta_J[0] - 1 for J in beta_J],
            "Relative change\nof interaction\n$J_{i, j}$",
            10,
            True,
        ),
        (
            [J / sym[0] - 1 for J in sym],
            "Relative change\nof symmetric\n$J^{(s)}_{i, j}$",
            10,
            True,
        ),
        (
            [J / anti[0] - 1 for J in anti],
            "Relative change\nof antisymmetric\n$J^{(a)}_{i, j}$",
            10,
            True,
        ),
    ]
    for label, (matrices, cbar_label, v, rasterized) in zip(labels, grids):
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        coupling_grid(
            fig, axs, matrices, -v, v, cmap, cbar_label, task_names, rasterized
        )
        save_panel(fig, save_dir, label, bbox_inches="tight")

    network_style = {
        "node_size": 25,
        "plot_label": True,
        "font_size": 5,
        "edge_width_scale": 0.8,
    }
    fig, axs = plt.subplots(3, 3, figsize=figsize)
    network_grid(
        axs,
        beta_J,
        [np.arange(num_clusters)] * len(beta_J),
        task_names,
        cmap=plt.cm.tab10,
        vmin=0,
        vmax=9,
        **network_style,
    )
    save_panel(fig, save_dir, labels[6], bbox_inches="tight")

    fig, axs = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
    external_input_grid(fig, axs, beta_h, task_names)
    save_panel(fig, save_dir, labels[7], bbox_inches="tight")

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    network_grid(
        axs,
        beta_J,
        beta_h,
        task_names,
        cmap=cmap,
        vmin=-0.15,
        vmax=0.15,
        font_color="k",
        **network_style,
    )
    save_panel(fig, save_dir, labels[8], bbox_inches="tight")


def _iqr_bars(ax, x, med, q25, q75, width):
    for xk, lo, hi in zip(x, q25, q75):
        ax.vlines(xk, lo, hi, colors="k", linewidth=0.4, zorder=5)
        ax.hlines(
            [lo, hi],
            xk - width / 2,
            xk + width / 2,
            colors="k",
            linewidth=0.4,
            zorder=5,
        )


def plot_rate_constant_comparison(
    axs,
    A_sd,
    A_sd_boot,
    loss_boot,
    r_boot,
    task_names,
    A_label,
    A_ylim=None,
    loss_legend_loc="best",
    r_ylim=(0, 1),
):
    """fig-s11/fig-s12 panels comparing spin-independent (SI) and
    spin-dependent (SD) rate constants.

    a: A_k per cluster and task (bars: point estimates; error bars: the
    bootstrap interquartile half-widths placed around the point estimate);
    b, c: bootstrap median and IQR of the flux loss and of Pearson r.
    loss_boot and r_boot are dicts {"SI": (tasks, boot), "SD": ...}.
    """
    num_tasks, num_spins = A_sd.shape
    gray_colors = plt.cm.Greys(np.linspace(0.2, 0.85, num_tasks))
    med = np.median(A_sd_boot, axis=2)
    q25 = np.percentile(A_sd_boot, 25, axis=2)
    q75 = np.percentile(A_sd_boot, 75, axis=2)

    ax = axs[0]
    x = np.arange(num_spins)
    offsets = np.linspace(-0.32, 0.32, num_tasks)
    bar_w = 0.08
    for t in range(num_tasks):
        ax.bar(
            x + offsets[t],
            A_sd[t],
            bar_w,
            color=gray_colors[t],
            label=task_names[t],
        )
        _iqr_bars(
            ax,
            x + offsets[t],
            A_sd[t],
            A_sd[t] - (med[t] - q25[t]),
            A_sd[t] + (q75[t] - med[t]),
            bar_w,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(k + 1) for k in range(num_spins)])
    if A_ylim is not None:
        ax.set_ylim(*A_ylim)
    ax.set_ylabel(A_label)
    ax.set_xlabel("Cluster")
    ax.legend(ncol=2, frameon=False)

    x = np.arange(num_tasks)
    bar_w = 0.4
    for ax, boot, ylabel in [
        (axs[1], loss_boot, "Loss"),
        (axs[2], r_boot, r"Pearson $r$ (flux reconstruction)"),
    ]:
        for key, offset, color, label in [
            ("SI", -0.2, "0.5", r"Spin-indep. $A$"),
            ("SD", +0.2, "0.75", r"Spin-dep. $A_k$"),
        ]:
            ax.bar(
                x + offset,
                np.median(boot[key], axis=1),
                bar_w,
                color=color,
                label=label,
            )
            _iqr_bars(
                ax,
                x + offset,
                None,
                np.percentile(boot[key], 25, axis=1),
                np.percentile(boot[key], 75, axis=1),
                bar_w,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=90)
        ax.set_ylabel(ylabel)
    axs[1].legend(frameon=False, loc=loss_legend_loc)
    axs[2].set_ylim(*r_ylim)
    axs[2].legend(frameon=False)


def print_rate_constant_table(A_sd, A_si, task_names):
    """LaTeX rows of A_k per task (and the spin-independent A)."""
    num_spins = A_sd.shape[1]
    header = " & ".join(
        ["Task"] + [f"$A_{k + 1}$" for k in range(num_spins)] + [r"$A$ (SI)"]
    )
    print(r"\begin{tabular}{l" + "r" * (num_spins + 1) + r"}")
    print(r"\hline")
    print(header + r" \\")
    print(r"\hline")
    for t in range(len(task_names)):
        values = " & ".join(f"{A_sd[t, k]:.3f}" for k in range(num_spins))
        print(f"{task_names[t]} & {values} & {A_si[t]:.3f}" + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")


# ── Binarization of one example recording ─────────────────────────────────
def example_recording(pfa, task_choice=7, example_id=0):
    """Z scores, spin-flip events and binarized states of one recording."""
    pfa.pick_task_data(task_choice)
    chosen_id = pfa.ID_unique_task[example_id]
    zscore = pfa.zscore_task[pfa.ID_data_task == chosen_id]
    events = pfa.event_step_time_spin[pfa.ID_data_event == chosen_id]
    in_recording = pfa.ID_data_trajectory == chosen_id
    states = pfa.trajectories[in_recording]
    return {
        "t": np.arange(zscore.shape[0]) * pfa.time_between_steps,
        "zscore": zscore,
        "events": events,
        "t_trajectory": np.hstack((0, np.cumsum(pfa.dts[in_recording]))),
        "states": np.vstack((states, states[-1])),
    }


def _event_times(recording, j):
    events = recording["events"]
    return events[events[:, 2] == j][:, 1]


def plot_binarization(axs, recording):
    """Per cluster: the z score (dots) and its binarized spin (lines), with
    open/filled markers at each flip."""
    t_traj = recording["t_trajectory"]
    states = recording["states"]

    def spin_at(time, j):
        """Spin value of state index mu at spin k."""
        return states[np.where(t_traj == time), j]

    for j, ax in enumerate(axs):
        ax.scatter(recording["t"], recording["zscore"][:, j], c="k", lw=0, s=1)
        event_time = _event_times(recording, j)
        for k in range(len(event_time) - 1):
            ising = spin_at(event_time[k], j)[0]
            ax.plot(event_time[k : k + 2], [ising, ising], c=f"C{j}", zorder=1)
        # initial and final segments
        ising = spin_at(event_time[0], j)[0]
        ax.plot([0.0, event_time[0]], [-ising, -ising], c=f"C{j}", zorder=1)
        ising = spin_at(event_time[-1], j)[0]
        ax.plot(
            [event_time[-1], t_traj[-1]], [ising, ising], c=f"C{j}", zorder=1
        )
        for time in event_time:
            ax.scatter(
                time,
                -spin_at(time, j),
                s=8,
                lw=0.4,
                edgecolors=f"C{j}",
                facecolor="w",
            )
            ax.scatter(time, spin_at(time, j), s=8, lw=0.4, c=f"C{j}")


def plot_state_raster(ax, recording):
    """Bars where each cluster's spin is +1, one row per cluster."""
    states = recording["states"]
    num_clusters = states.shape[1]
    for j in range(num_clusters):
        all_points = np.insert(_event_times(recording, j), 0, 0.0)
        all_points = np.append(all_points, recording["t_trajectory"][-1])
        start_duration = np.vstack([all_points[:-1], np.diff(all_points)]).T
        initial_ising = int((states[0, j] + 1) / 2)
        ax.broken_barh(
            start_duration[(initial_ising + 1) % 2 :: 2],
            (j - 0.5, 1.0),
            color=f"C{j}",
            lw=0,
        )
    ax.set_ylim(-0.5, num_clusters - 0.5)
    ax.set_yticks(
        range(num_clusters),
        labels=[f"Cluster {j + 1}" for j in range(num_clusters)],
    )
    ax.invert_yaxis()


def plot_task_clustering(
    flat,
    save_dir,
    edge_label,
    width,
    task_names,
    vmax=None,
    formats=PANEL_FORMATS,
):
    """Panels a-d of the task clustering by flux (or traffic) patterns.

    flat is (tasks, edges). a: task-task cosine-similarity clustermap;
    b: its dendrogram; c: edge-edge clustermap; d: the data matrix ordered
    by both linkages.
    """
    fh = hc.square_clustermap_height(
        flat.T,
        width,
        matrix_type="cosine-similarity",
        ax_label="Task",
        cbar_label="Cos. sim.",
    )
    g, _, Z = hc.plot_clustermap(
        flat.T,
        matrix_type="cosine-similarity",
        figsize=(width, fh),
        return_linkage_matrices=True,
        ax_label="Task",
        cbar_label="Cos. sim.",
    )
    save_panel(g.figure, save_dir, "a", formats=formats, bbox_inches="tight")

    fig, _ = hc.plot_dendrogram(
        Z,
        threshold=0,
        figsize=(width, fh),
        labels=task_names,
        leaf_font_size=7,
        draw_threshold=False,
    )
    save_panel(fig, save_dir, "b", formats=formats, bbox_inches="tight")

    g, _ = hc.plot_clustermap(
        flat,
        matrix_type="cosine-similarity",
        figsize=(width, fh),
        ax_label=edge_label,
        nan_to_zero=True,
        cbar_label="Cos. sim.",
    )
    save_panel(g.figure, save_dir, "c", formats=formats, bbox_inches="tight")

    g, _ = hc.plot_hierarchical_data(
        flat.T,
        row_matrix_type="cosine-similarity",
        nan_to_zero=True,
        ax_xlabel="Task",
        ax_ylabel=edge_label,
        cbar_label="",
        vmin=None if vmax is None else -vmax,
        vmax=vmax,
        figsize=(width, fh),
    )
    save_panel(g.figure, save_dir, "d", formats=formats, bbox_inches="tight")


# ── Brain parcels and surfaces ─────────────────────────────────────────────
def yeo_networks():
    """Yeo network index of each Schaefer parcel and the network names,
    in order of first appearance."""
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, verbose=0)
    labels = [str(label) for label in atlas.labels if label != "Background"]
    # labels such as "7Networks_LH_Vis_1"
    names = np.array(
        [YEO_ABBREVIATIONS[label.split("_")[2]] for label in labels]
    )
    unique = names[np.sort(np.unique(names, return_index=True)[1])]
    to_id = {name: i for i, name in enumerate(unique)}
    return np.array([to_id[name] for name in names]), unique


def network_profile(cluster_inds, network_ids, num_networks=7):
    """Number of parcels of each network in each cluster."""
    profile = np.zeros((len(np.unique(cluster_inds)), num_networks))
    for c, f in zip(cluster_inds, network_ids):
        profile[c, f] += 1
    return profile


def plot_network_profile(axs, profile, network_names):
    """One horizontal bar chart per cluster of its parcels' networks."""
    colors = [plt.colormaps["Dark2"](i) for i in range(8)]
    for i, ax in enumerate(axs):
        bars = ax.barh(np.arange(profile.shape[1]), profile[i])
        ax.set_yticks(np.arange(profile.shape[1]))
        for j, bar in enumerate(bars):
            bar.set_facecolor(colors[j])
        ax.set_yticklabels(network_names if i == 0 else [])
        ax.set_title(f"{i + 1}", c=f"C{i}")


SURFACE_VIEWS = ["lateral", "medial"]
# panel order of the 2 x 2 brain-view grids: (hemisphere, view)
SURFACE_GRID = [
    ("left", "lateral"),
    ("right", "medial"),
    ("left", "medial"),
    ("right", "lateral"),
]


def brain_surfaces():
    """fsLR 32k midthickness surfaces and the Schaefer-100 atlas.

    Returns (left surface, right surface, left atlas, right atlas).
    """
    from brainspace.datasets import load_parcellation
    from neuromaps.datasets import fetch_fslr

    lh, rh = fetch_fslr("32k")["midthickness"]
    latlas, ratlas = load_parcellation("schaefer", 100)
    return lh, rh, latlas, ratlas


def write_surface_views(surfaces, parcel_values, name, out_dirs, **kwargs):
    """Render the lateral and medial views of both hemispheres colored by
    parcel_values, as out_dir/<name><hemisphere>_<view>.{png,pdf}."""
    import plotly.io as pio

    from .analyzer.brain_visualization import create_surf_plot

    lh, rh, latlas, ratlas = surfaces
    for out_dir in out_dirs:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    for hemisphere, surface, atlas in [
        ("left", lh, latlas),
        ("right", rh, ratlas),
    ]:
        for view in SURFACE_VIEWS:
            fig = create_surf_plot(
                surface, atlas, parcel_values, view=view, **kwargs
            )
            for out_dir in out_dirs:
                for ext in ("png", "pdf"):
                    pio.write_image(
                        fig,
                        f"{out_dir}/{name}{hemisphere}_{view}.{ext}",
                        width=1.0 * 600,
                        height=0.75 * 600,
                        scale=1,
                    )


def show_surface_views(axs, image_dir, name):
    """Show the four views written by write_surface_views on a 2 x 2 grid."""
    from PIL import Image

    for i, (ax, (hemisphere, view)) in enumerate(zip(axs.flat, SURFACE_GRID)):
        path = f"{image_dir}/{name}{hemisphere}_{view}.png"
        ax.imshow(np.array(Image.open(path)))
        if i < 2:
            ax.set_xlim(60, 540)
            ax.set_ylim(430, 50)
        else:
            ax.set_xlim(-5)
            ax.set_ylim(450, -20)
        ax.axis("off")


# ── Task grids (3 x 3 axes, slot [0, 2] left for a legend/biplot) ─────────
def task_axes(axs):
    """The 8 task axes of a 3 x 3 grid, skipping axs[0, 2]."""
    return [axs.flat[k + (k > 1)] for k in range(NUM_TASKS)]


def plot_pca_by_task(
    pfa,
    axs,
    kind="flux",
    scale=40,
    color=None,
    x_component=0,
    y_component=1,
    arrows=None,
    titles=None,
):
    """Flux (or traffic) arrows of each task in WPCA space, the biplot in
    axs[0, 2], and shared PC axis labels.

    By default the observed arrows of each task are drawn. `arrows`, a
    per-task list of (source, target, values), draws model arrows instead.
    `scale` may be a per-task list; `titles` replaces the task names.
    """
    fig = axs.flat[0].figure
    scales = scale if np.ndim(scale) else [scale] * NUM_TASKS
    for k, ax in enumerate(task_axes(axs)):
        if arrows is None:
            pfa.trajectories_to_flux(k, progress_message=False)
        elif kind == "flux":
            pfa.flux_source, pfa.flux_target, pfa.flux = arrows[k]
        else:
            pfa.traffic_source, pfa.traffic_target, pfa.traffic = arrows[k]
        plot = (
            pfa.plot_hypercubic_flux_on_pca
            if kind == "flux"
            else pfa.plot_hypercubic_traffic_on_pca
        )
        plot(
            scale=scales[k],
            x_component=x_component,
            y_component=y_component,
            ax=ax,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(titles[k] if titles else pfa.task_name_list[k])
    ax = axs[0, 2]
    pfa.plot_biplot_vector(
        x_component=x_component, y_component=y_component, color=color, ax=ax
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    fig.supxlabel(f"PC{x_component + 1}")
    fig.supylabel(f"PC{y_component + 1}")


def color_cluster_ticks(ax, num_clusters, offset=0, x=True, y=True):
    """Number the cluster ticks 1..N in the cluster colors."""
    ticks = np.arange(num_clusters) + offset
    labels = np.arange(num_clusters) + 1
    for on, set_ticks, set_labels, get_labels in [
        (x, ax.set_xticks, ax.set_xticklabels, ax.get_xticklabels),
        (y, ax.set_yticks, ax.set_yticklabels, ax.get_yticklabels),
    ]:
        if not on:
            continue
        set_ticks(ticks)
        set_labels(labels)
        for j, label in enumerate(get_labels()):
            label.set_color(f"C{j}")


def coupling_grid(
    fig,
    axs,
    matrices,
    vmin,
    vmax,
    cmap,
    cbar_label,
    task_names,
    rasterized=False,
):
    """One coupling matrix per task, with a colorbar in axs[0, 2].

    rasterized=False draws with netvis.plot_im; rasterized=True draws a
    rasterized matshow styled the same way.
    """
    num_clusters = matrices[0].shape[0]
    for k, ax in enumerate(task_axes(axs)):
        if rasterized:
            image = ax.matshow(
                matrices[k], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
            )
        else:
            _, _, image = netvis.plot_im(
                matrices[k],
                vmin=vmin,
                vmax=vmax,
                plot_cbar=False,
                return_cbar=True,
                fig=fig,
                ax=ax,
                cmap=cmap,
            )
        color_cluster_ticks(ax, num_clusters)
        if rasterized:
            ax.tick_params(bottom=False)
            ax.xaxis.set_label_position("top")
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        ax.set_title(task_names[k])
        if k == 1:
            colorbar_image = image
    ax = axs[0, 2]
    ax.axis("off")
    cax = make_axes_locatable(ax).append_axes("left", size=0.02, pad=0)
    cb = fig.colorbar(colorbar_image, cax=cax, label=cbar_label)
    cb.outline.set_visible(False)


def external_input_grid(fig, axs, fields, task_names, xlabel="Cluster"):
    """Bar chart of each task's external input (one bar per cluster)."""
    for k, ax in enumerate(task_axes(axs)):
        bars = ax.bar(np.arange(len(fields[k])) + 1, fields[k])
        for j, bar in enumerate(bars):
            bar.set_facecolor(f"C{j}")
        ax.axhline(0.0, c="k", ls="--", lw=plt.rcParams["axes.linewidth"])
        ax.set_title(task_names[k])
        ax.set_xticks(np.arange(len(fields[k])) + 1)
        ax.set_xticklabels(np.arange(len(fields[k])) + 1)
        ax.set_ylim(-0.15, 0.15)
    axs[0, 2].axis("off")
    fig.supxlabel(xlabel)
    fig.supylabel("External input")


def network_grid(axs, couplings, node_colors, task_names, **kwargs):
    """Network of each task's couplings, nodes colored by node_colors[k]."""
    for k, ax in enumerate(task_axes(axs)):
        netvis.plot_network(
            couplings[k], ax=ax, node_color=node_colors[k], **kwargs
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(task_names[k])
    axs[0, 2].axis("off")


# ── Hypercube / state-space helpers ───────────────────────────────────────
def stationary_distribution(W: np.ndarray) -> np.ndarray:
    """Stationary distribution of a continuous-time rate matrix W."""
    vals, vecs = np.linalg.eig(W)
    idx = np.argmin(np.abs(vals.real))
    p = vecs[:, idx].real
    p = np.abs(p)
    return p / p.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats)."""
    return float(distance.jensenshannon(p, q, base=np.e) ** 2)


def get_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Symmetric part `(matrix + matrix.T) / 2`."""
    return 0.5 * (matrix + matrix.T)


def get_antisymmetric(matrix: np.ndarray) -> np.ndarray:
    """Antisymmetric part `(matrix - matrix.T) / 2`."""
    return 0.5 * (matrix - matrix.T)


def boltzmann_eq(
    beta_J: np.ndarray, beta_h: np.ndarray, sv: np.ndarray
) -> np.ndarray:
    """Equilibrium Boltzmann distribution using symmetric J^s = (J + J^T)/2."""
    J_sym = get_symmetric(beta_J)
    H = 0.5 * np.einsum("ni,ij,nj->n", sv, J_sym, sv) + sv @ beta_h
    p = np.exp(H - H.max())
    return p / p.sum()


def _build_W(beta_J, beta_h, sv, flip_rate):
    """Rate matrix W[target, source] of single-spin flips; flip_rate(s_k
    beta*g_k) gives the rate of flipping spin k."""
    num_spins = sv.shape[1]
    NS = sv.shape[0]
    W = np.zeros((NS, NS))
    for mu in range(NS):
        s_mu = sv[mu]
        for k in range(num_spins):
            nu = mu ^ (1 << (num_spins - 1 - k))
            g_k = beta_J[k, :] @ s_mu + beta_h[k]
            W[nu, mu] += flip_rate(s_mu[k] * g_k)
    np.fill_diagonal(W, -W.sum(axis=0))
    return W


def build_W_arrhenius(
    beta_J: np.ndarray, beta_h: np.ndarray, A: float, sv: np.ndarray
) -> np.ndarray:
    """Arrhenius rates A exp(-s_k beta*g_k)."""
    return _build_W(beta_J, beta_h, sv, lambda u: A * np.exp(-u))


def build_W_glauber(
    beta_J: np.ndarray, beta_h: np.ndarray, A: float, sv: np.ndarray
) -> np.ndarray:
    """Glauber rates A / (1 + exp(2 s_k beta*g_k))."""
    return _build_W(beta_J, beta_h, sv, lambda u: A / (1.0 + np.exp(2.0 * u)))


def model_flux_matrix(
    beta_J: np.ndarray,
    beta_h: np.ndarray,
    A: float | np.ndarray,
    weight_type: str = "arrhenius",
) -> np.ndarray:
    """Stationary probability flux J[target, source] of the kinetic Ising
    model.

    Parameters
    ----------
    beta_J, beta_h : couplings (N, N) and external input (N,)
    A : rate constant, a scalar or one per spin (N,)
    weight_type : "arrhenius" (A exp(-s_k beta*g_k)) or "glauber"
        (A / (1 + exp(2 s_k beta*g_k)))

    Returns
    -------
    (2^N, 2^N) antisymmetric flux, computed from the stationary solution
    of the full rate matrix.
    """
    num_spins = len(beta_h)
    sv = create_state_vectors(num_spins)
    A_vec = np.broadcast_to(np.asarray(A, dtype=float), (num_spins,))
    if weight_type == "arrhenius":
        flip_rate = np.exp
    elif weight_type == "glauber":

        def flip_rate(minus_u):
            return 1.0 / (1.0 + np.exp(np.clip(-2.0 * minus_u, -500, 500)))

    else:
        raise ValueError("weight_type must be 'arrhenius' or 'glauber'")
    num_states = len(sv)
    W = np.zeros((num_states, num_states))
    source = np.arange(num_states)
    for k in range(num_spins):
        target = source ^ (1 << (num_spins - 1 - k))
        u = sv[:, k] * (sv @ beta_J[k] + beta_h[k])
        W[target, source] += A_vec[k] * flip_rate(-u)
    np.fill_diagonal(W, -W.sum(axis=0))
    joint = W * stationary_distribution(W)[np.newaxis, :]
    np.fill_diagonal(joint, 0.0)
    return joint - joint.T


# ── Task-color palettes (two genuinely different conventions in use — see
# fig-s13 vs fig-s10; kept distinct rather than silently unified) ──────────
def task_colors_stepped(n: int = NUM_TASKS):
    """plt.cm.viridis(t/8) per task — fig-s01/s02/s13/s16's convention."""
    return [plt.cm.viridis(t / 8) for t in range(n)]


def task_colors_linspace(n: int = NUM_TASKS):
    """plt.cm.viridis(linspace(0,1,n)) — fig-s10's convention."""
    return plt.cm.viridis(np.linspace(0, 1, n))


# ── Misc colormaps ─────────────────────────────────────────────────────────
def pvalue_colormap():
    """Diverging Reds_r/Blues colormap for p-value heatmaps (fig-s16)."""
    r = plt.cm.Reds_r(np.linspace(0, 0.5, 256))
    b = plt.cm.Blues(np.linspace(0, 1, 256))
    c = np.vstack((r, b))
    return LinearSegmentedColormap.from_list("pvalue_map", c)


# ── Plot helpers ────────────────────────────────────────────────────────────
def state_indicator(
    ax,
    state,
    annot_coord,
    components,
    pc_x=0,
    pc_y=1,
    whiten=True,
    whiten_ratio=1.0,
    alpha_ratio=0.0,
):
    """Draw a state-glyph inset annotating a hypercube state in PCA space."""
    num_dims = len(state)
    state_coord = np.array([components[pc_x], components[pc_y]]) @ state
    ax.annotate(
        "",
        state_coord,
        annot_coord,
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 0.5},
        transform=ax.transData,
    )
    inset_width = 0.1
    inset_height = inset_width * num_dims
    axin = ax.inset_axes(
        [
            annot_coord[0] - inset_width / 2,
            annot_coord[1] - inset_height / 2,
            inset_width,
            inset_height,
        ],
        transform=ax.transData,
    )

    ax = axin
    if whiten:
        tab10_colors = np.array(plt.cm.tab10.colors)
        original_colors = tab10_colors[:num_dims]
        whiten_ratios = np.zeros_like(state).astype(float)
        whiten_ratios[np.argwhere(state == -1)] = whiten_ratio
        whiten_ratios = np.repeat([whiten_ratios], 3, axis=0).T
        blended_color = (
            1 - whiten_ratios
        ) * original_colors + whiten_ratios * np.ones_like(original_colors)
        blended_color = blended_color.reshape(num_dims, 1, 3)
        ax.pcolormesh(blended_color[::-1], edgecolor="k", lw=0.25)
    else:
        alpha = np.ones_like(state).astype(float)
        alpha[np.argwhere(state == -1)] = alpha_ratio
        ax.pcolormesh(
            np.vstack(np.arange(num_dims))[::-1],
            alpha=alpha[::-1],
            cmap=plt.cm.tab10,
            vmin=0,
            vmax=9,
            edgecolor="k",
            lw=0.25,
        )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
