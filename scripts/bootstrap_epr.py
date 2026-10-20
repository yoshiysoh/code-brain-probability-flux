"""Bootstrap the entropy production rate for 2 to 10 clusters.

Reads the HCP recordings (data/HCP_data_task/) and writes
data/<YYYY-MM-DD>/epr_bootstrap.h5 with

num_clusters : (9,) 2, ..., 10
thresholds : (9,) dendrogram threshold giving each number of clusters
eprs : (9, tasks, replicates) entropy production rate (nats/s)
frac_observed_transitions : (9, tasks, replicates)

Each replicate resamples the observed transitions of a task until their
dwell time reaches the observed time.

Usage: python scripts/bootstrap_epr.py [--seed 42]
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

# Dendrogram thresholds giving 2, 3, ..., 10 clusters
THRESHOLDS = [0.9, 0.88, 0.80, 0.72, 0.71, 0.70, 0.67, 0.66, 0.63]
NUM_REALIZATIONS = 100


def main(seed: int) -> None:
    pfa = PFA()
    pfa.preprocessing()
    seeds = np.random.SeedSequence(seed).spawn(len(THRESHOLDS))
    num_clusters, eprs, fracs = [], [], []
    for threshold, threshold_seed in zip(THRESHOLDS, seeds):
        cluster_inds = hc.cluster_parcels(pfa.X_data, threshold=threshold)
        num_clusters.append(len(np.unique(cluster_inds)))
        log.info("%d clusters", num_clusters[-1])
        pfa.coarse_grain(cluster_inds)
        task_eprs, task_fracs = [], []
        for k, task_seed in enumerate(threshold_seed.spawn(bpp.NUM_TASKS)):
            pfa.trajectories_to_flux(k, progress_message=False)
            pfa.hypercubic_bootstrapping(NUM_REALIZATIONS, seed=task_seed)
            task_eprs.append(pfa.bootstrapped["entropy_production_rates"])
            task_fracs.append(pfa.bootstrapped["fracs_observed_transitions"])
        eprs.append(task_eprs)
        fracs.append(task_fracs)

    out = run_dir() / "epr_bootstrap.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(f, seed=seed, num_realizations=NUM_REALIZATIONS)
        f.create_dataset("num_clusters", data=np.array(num_clusters))
        f.create_dataset("thresholds", data=np.array(THRESHOLDS))
        f.create_dataset("eprs", data=np.array(eprs))
        f.create_dataset("frac_observed_transitions", data=np.array(fracs))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
