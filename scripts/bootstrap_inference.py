"""Bootstrap the coupling matrices inferred for each of the 8 tasks.

For each task and replicate, transitions are resampled from the observed
ones until their total dwell time matches the observed time, a rate
matrix is built from them, and the couplings are re-inferred by least
squares: the Arrhenius model (scalar A) or the Glauber model (A fixed at
twice the 99th-percentile rate; the problem is then exactly linear, so
least squares matches the gradient descent of scripts/infer_networks.py).

Reads the HCP recordings (data/HCP_data_task/) and writes
data/<YYYY-MM-DD>/inference_bootstrap.h5 with groups arrhenius and
glauber, each holding beta_J (tasks, replicates, N, N).

Usage: python scripts/bootstrap_inference.py [--seed 42]
"""

import argparse
import logging

import h5py
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp
from brain_probability_flux.analyzer import (
    transition_rate_to_network_parameters_lstsq,
    transition_rate_to_network_parameters_lstsq_glauber,
)
from brain_probability_flux.provenance import (
    run_dir,
    write_provenance,
)

N_REALIZATIONS = 100
MODELS = {
    "arrhenius": transition_rate_to_network_parameters_lstsq,
    "glauber": transition_rate_to_network_parameters_lstsq_glauber,
}

log = logging.getLogger(__name__)


def resample_transitions(
    transition_count: np.ndarray,
    time_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw transitions with replacement until their dwell time reaches
    time_threshold."""
    n_orig = len(transition_count)
    n_draw = int(n_orig * 1.5)
    while True:
        idxs = rng.integers(0, n_orig, size=n_draw)
        cumtimes = np.cumsum(transition_count[idxs, 2])
        if cumtimes[-1] >= time_threshold:
            cutoff = int(np.searchsorted(cumtimes, time_threshold)) + 1
            return transition_count[idxs[:cutoff]]
        n_draw = int(n_draw * 1.5)


def bootstrap_infer(
    infer,
    transition_count: np.ndarray,
    time_threshold: float,
    num_states: int,
    n_realizations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrapped beta_J matrices, shape (n_realizations, K, K)."""
    results = []
    for _ in tqdm(range(n_realizations), leave=False, desc="Bootstrap"):
        sampled = resample_transitions(transition_count, time_threshold, rng)
        sources = sampled[:, 0].astype(int)
        targets = sampled[:, 1].astype(int)
        joint = np.zeros((num_states, num_states))
        waiting = np.zeros(num_states)
        np.add.at(joint, (sources, targets), 1)
        np.add.at(waiting, sources, sampled[:, 2])
        # rate[i, j]: rate of j -> i, as in PFA.transition_rate_matrix
        rates = joint.T / (waiting + 1e-8)
        try:
            beta_J, _, _, _ = infer(rates)
        except np.linalg.LinAlgError as e:
            log.warning("Bootstrap rep failed: %s", e)
            continue
        results.append(beta_J)
    return np.array(results)


def main(seed: int) -> None:
    pfa, _ = bpp.coarse_grained_pfa()
    out = run_dir() / "inference_bootstrap.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(f, seed=seed, num_realizations=N_REALIZATIONS)
        for model, infer in MODELS.items():
            # Each model restarts the random stream from the seed.
            rng = np.random.default_rng(seed)
            beta_J_boot = []
            for k in tqdm(range(bpp.NUM_TASKS), desc=model):
                pfa.trajectories_to_flux(k, progress_message=False)
                beta_J_boot.append(
                    bootstrap_infer(
                        infer,
                        pfa.transition_count,
                        pfa.dts_task.sum(),
                        pfa.num_unique_states,
                        N_REALIZATIONS,
                        rng,
                    )
                )
            f.create_group(model).create_dataset(
                "beta_J", data=np.array(beta_J_boot)
            )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
