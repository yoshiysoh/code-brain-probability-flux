"""Bootstrap the empirical state distribution of each task (fig-s09).

The binarized time points of each task are resampled with replacement
and the distribution over the 2^N states is recomputed.

Reads the HCP recordings (data/HCP_data_task/) and writes
data/<YYYY-MM-DD>/stationary_bootstrap.h5 with

probability_boot : (tasks, replicates, 2^N)

Usage: python scripts/bootstrap_stationary.py [--seed 42]
"""

import argparse
import logging

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp
from brain_probability_flux.provenance import (
    run_dir,
    write_provenance,
)

log = logging.getLogger(__name__)

NUM_REALIZATIONS = 200


def main(seed: int) -> None:
    pfa, _ = bpp.coarse_grained_pfa()
    num_spins = pfa.unique_states.shape[1]
    num_states = 2**num_spins
    powers = 2 ** np.arange(num_spins - 1, -1, -1)
    rng = np.random.default_rng(seed)
    probability_boot = np.zeros((bpp.NUM_TASKS, NUM_REALIZATIONS, num_states))
    for k in range(bpp.NUM_TASKS):
        pfa.trajectories_to_flux(k, progress_message=False)
        # +/-1 trajectory -> state index
        states = ((pfa.trajectories_task + 1) // 2).astype(int) @ powers
        n = len(states)
        for b in range(NUM_REALIZATIONS):
            counts = np.bincount(
                states[rng.integers(0, n, size=n)], minlength=num_states
            ).astype(float)
            probability_boot[k, b] = counts / counts.sum()

    out = run_dir() / "stationary_bootstrap.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(f, seed=seed, num_realizations=NUM_REALIZATIONS)
        f.create_dataset("probability_boot", data=probability_boot)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
