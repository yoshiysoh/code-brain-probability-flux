"""Compute the group-level results that the figure notebooks plot.

Reads the HCP recordings (data/HCP_data_task/, not distributed) and writes
data/<YYYY-MM-DD>/aggregates.h5 with

parcel_correlation : (100, 100)
    Correlation between the parcels over all recordings (Fig. 1A, B).
settings/<name>/ : one group per coarse-graining
    cluster_inds : cluster of each parcel
    unique_states, probability : observed states and their probability
        over all tasks (used for the weighted PCA)
    task<k>/ : states, source, target, count (transition counts between
        state indices), waiting_time (dwell time in each state) and the
        total observation time (attribute total_time)
pc_grid/ : flux on the grid of the first two PC scores (fig-s02)
    bin_edges, bin_pos and, per task, the normalized histogram and the
    bootstrap mean flux and confidence ellipses

Usage: python scripts/compute_aggregates.py [--seed 42]
"""

import argparse
import logging

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp
from brain_probability_flux.analyzer import PFA
from brain_probability_flux.analyzer import (
    hierarchical_clustering as hc,
)
from brain_probability_flux.provenance import (
    run_dir,
    write_provenance,
)

log = logging.getLogger(__name__)

# name -> (dendrogram threshold, number of clusters, binarization order)
SETTINGS = {
    "threshold_0.70": (0.70, None, 0),  # 7 clusters, the main analysis
    "threshold_0.70_order1": (0.70, None, 1),  # fig-s05/fig-s21 dynamic
    "threshold_0.70_order2": (0.70, None, 2),  # fig-s05/fig-s21 curve
    "threshold_0.80": (0.80, None, 0),  # 4 clusters (fig-s04, fig-s20)
    "threshold_0.72": (0.72, None, 0),  # 5 clusters
    "threshold_0.71": (0.71, None, 0),  # 6 clusters
    "maxclust_4": (None, 4, 0),  # tree cut into 4 clusters (fig-s07, fig-s14)
    "maxclust_5": (None, 5, 0),
    "maxclust_6": (None, 6, 0),
}
NUM_GRID_BOOTSTRAP = 100


def write_pc_grid(pfa: PFA, group: h5py.Group, seed: int) -> None:
    """Flux on the PC1-PC2 grid of the parcel z scores, per task."""
    pfa.perform_PCA()
    pfa.digitize_PCA_data()
    group.create_dataset("bin_edges", data=pfa.bin_edges)
    group.create_dataset("bin_pos", data=pfa.bin_pos)
    seeds = np.random.SeedSequence(seed).spawn(pfa.num_tasks)
    for k in range(pfa.num_tasks):
        pfa.pick_task_data(k)
        pfa.create_transition_list()
        pfa.bootstrapping(seeds[k], NUM_GRID_BOOTSTRAP)
        pfa.calc_mean_vec_from_bootstrapping()
        pfa.ellipses_radii_angles()
        task = group.create_group(f"task{k}")
        for key, value in pfa.grid_summary().items():
            task.create_dataset(key, data=value)


def main(seed: int) -> None:
    out = run_dir() / bpp.AGGREGATES
    pfa = PFA()
    pfa.preprocessing()
    with h5py.File(out, "w-") as f:
        write_provenance(f, seed=seed, num_grid_bootstrap=NUM_GRID_BOOTSTRAP)
        f.create_dataset("parcel_correlation", data=np.corrcoef(pfa.X_data.T))
        log.info("PC grid")
        write_pc_grid(pfa, f.create_group("pc_grid"), seed)

        settings = f.create_group("settings")
        partitions = {
            name: hc.cluster_parcels(
                pfa.X_data, threshold=threshold or 0.70, num_clusters=n
            )
            for name, (threshold, n, _) in SETTINGS.items()
        }
        orders = {name: order for name, (_, _, order) in SETTINGS.items()}
        partitions["yeo7"], _ = bpp.yeo_networks()
        orders["yeo7"] = 0
        for name, cluster_inds in partitions.items():
            log.info("%s: %d clusters", name, len(np.unique(cluster_inds)))
            pfa.coarse_grain(cluster_inds, orders[name])
            group = settings.create_group(name)
            group.create_dataset("cluster_inds", data=cluster_inds)
            group.attrs["order_derivative"] = orders[name]
            pfa.save_aggregates(group)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
