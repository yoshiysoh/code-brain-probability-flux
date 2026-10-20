"""Infer the Ising network of each task from its transition rates.

Reads data/aggregates.h5 (the 7-cluster coarse-graining) and writes
data/<YYYY-MM-DD>/inferred_networks.h5 with one group per model:

arrhenius          least squares, one rate constant A per task (Fig. 4)
arrhenius_spindep  least squares, one A_k per spin (fig. S11)
glauber            gradient descent, A = twice the 99th-percentile rate
                   (figs. S8, S9, S12, S24)
glauber_spindep    least squares, one A_k per spin (fig. S12)
shared_network     gradient descent, one Arrhenius network for all tasks
                   with a task-specific external input (figs. S14, S15)

Each group holds beta_J (tasks, N, N), beta_h (tasks, N), rate_const and
the final loss or least-squares residuals.

Usage: python scripts/infer_networks.py [--data-dir data]
"""

import argparse
import logging
import os
from pathlib import Path

# JAX runs on the CPU unless JAX_PLATFORMS is set, so that the results do
# not depend on the GPU model.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import numpy as np

import brain_probability_flux.plotting as bpp
from brain_probability_flux.analyzer import (
    transition_rate_to_network_parameters_gradient,
    transition_rate_to_network_parameters_gradient_same_network,
    transition_rate_to_network_parameters_lstsq,
    transition_rate_to_network_parameters_lstsq_glauber_spindep,
    transition_rate_to_network_parameters_lstsq_spindep,
)
from brain_probability_flux.provenance import run_dir, write_provenance

log = logging.getLogger(__name__)

LEARNING_RATE = 1e-3
NUM_STEPS = 40000


def write_model(group: h5py.Group, results: list, last: str) -> None:
    """Stack per-task (beta_J, beta_h, rate_const, last) tuples."""
    for i, key in enumerate(["beta_J", "beta_h", "rate_const", last]):
        group.create_dataset(key, data=np.array([r[i] for r in results]))


def main(data_dir: Path) -> None:
    pfa, _ = bpp.aggregated_pfa(data_dir=data_dir)
    rates, rates_per_recording = [], []
    for k in range(bpp.NUM_TASKS):
        pfa.trajectories_to_flux(k)
        rates.append(pfa.transition_rate_matrix())
        # The shared-network fit normalizes by the length of one recording.
        rates_per_recording.append(
            pfa.transition_rate_matrix(
                total_time=pfa.time_between_steps * pfa.num_time_steps
            )
        )

    out = run_dir() / "inferred_networks.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(
            f,
            learning_rate=LEARNING_RATE,
            num_steps=NUM_STEPS,
            input=str(Path(data_dir) / bpp.AGGREGATES),
        )
        lstsq = transition_rate_to_network_parameters_lstsq
        write_model(
            f.create_group("arrhenius"), [lstsq(r) for r in rates], "residuals"
        )
        spindep = transition_rate_to_network_parameters_lstsq_spindep
        write_model(
            f.create_group("arrhenius_spindep"),
            [spindep(r) for r in rates],
            "residuals",
        )
        glauber = []
        for k, rate in enumerate(rates):
            log.info("Glauber gradient descent, task %d", k)
            beta_J, beta_h, rate_const, loss = (
                transition_rate_to_network_parameters_gradient(
                    rate,
                    learning_rate=LEARNING_RATE,
                    num_steps=NUM_STEPS,
                    rate_model="glauber",
                )
            )
            glauber.append(
                (beta_J, beta_h, float(rate_const), float(loss[-1]))
            )
        write_model(f.create_group("glauber"), glauber, "final_loss")
        glauber_spindep = (
            transition_rate_to_network_parameters_lstsq_glauber_spindep
        )
        write_model(
            f.create_group("glauber_spindep"),
            [glauber_spindep(r) for r in rates],
            "residuals",
        )
        log.info("Shared-network gradient descent")
        beta_J, beta_h, rate_const, loss = (
            transition_rate_to_network_parameters_gradient_same_network(
                rates_per_recording,
                learning_rate=LEARNING_RATE,
                num_steps=NUM_STEPS,
            )
        )
        group = f.create_group("shared_network")
        group.create_dataset("beta_J", data=np.asarray(beta_J))
        group.create_dataset("beta_h", data=np.asarray(beta_h))
        group.create_dataset("rate_const", data=np.asarray(rate_const))
        group.create_dataset("final_loss", data=float(loss[-1]))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=bpp.DATA_DIR)
    main(parser.parse_args().data_dir)
