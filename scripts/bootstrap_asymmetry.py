"""Bootstrap the asymmetry ratio of the inferred network for 4 to 6
clusters (fig. S13).

The tree is cut into K = 4, 5, 6 clusters; for each task the couplings are
inferred by least squares (Arrhenius), and the ratio of the Frobenius
norms of their antisymmetric and symmetric parts is bootstrapped over
resamples of the observed transitions (as many as observed, with
replacement).

Reads the HCP recordings (data/HCP_data_task/) and writes
data/<YYYY-MM-DD>/asymmetry_bootstrap.h5 with

num_clusters : (3,) 4, 5, 6
ratio : (3, tasks) point estimate
ratio_boot : (3, tasks, replicates)

Usage: python scripts/bootstrap_asymmetry.py [--seed 42]
"""

import argparse
import logging

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp
from brain_probability_flux.analyzer import (
    PFA,
    transition_rate_to_network_parameters_lstsq,
)
from brain_probability_flux.analyzer import (
    hierarchical_clustering as hc,
)
from brain_probability_flux.provenance import (
    run_dir,
    write_provenance,
)

log = logging.getLogger(__name__)

CLUSTER_SIZES = [4, 5, 6]
NUM_REALIZATIONS = 100


def asymmetry_ratio_of(transition_count, num_states):
    rates = bpp.rate_matrix_from_transitions(transition_count, num_states)
    beta_J, _, _, _ = transition_rate_to_network_parameters_lstsq(rates)
    return bpp.asymmetry_ratio(beta_J)


def main(seed: int) -> None:
    pfa = PFA()
    pfa.preprocessing()
    rng = np.random.default_rng(seed)
    ratio = np.zeros((len(CLUSTER_SIZES), bpp.NUM_TASKS))
    ratio_boot = np.zeros(
        (len(CLUSTER_SIZES), bpp.NUM_TASKS, NUM_REALIZATIONS)
    )
    for i, K in enumerate(CLUSTER_SIZES):
        pfa.coarse_grain(hc.cluster_parcels(pfa.X_data, num_clusters=K))
        for k in range(bpp.NUM_TASKS):
            pfa.trajectories_to_flux(k, progress_message=False)
            tc = pfa.transition_count
            m = len(tc)
            ratio[i, k] = asymmetry_ratio_of(tc, 2**K)
            ratio_boot[i, k] = [
                asymmetry_ratio_of(tc[rng.integers(0, m, size=m)], 2**K)
                for _ in range(NUM_REALIZATIONS)
            ]
        log.info("%d clusters: %s", K, np.round(ratio[i], 3))

    out = run_dir() / "asymmetry_bootstrap.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(f, seed=seed, num_realizations=NUM_REALIZATIONS)
        f.create_dataset("num_clusters", data=np.array(CLUSTER_SIZES))
        f.create_dataset("ratio", data=ratio)
        f.create_dataset("ratio_boot", data=ratio_boot)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
