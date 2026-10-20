"""Bootstrap the spin-independent and spin-dependent rate-constant fits.

For each task and replicate, the observed transitions are resampled with
replacement (as many as observed), the rate matrix is rebuilt, and the
couplings are re-inferred with a spin-independent rate constant A (SI) and
a spin-dependent A_k (SD); the stationary flux of each fitted model is
compared with the observed flux.

arrhenius : least squares (fig. S11)
glauber   : gradient descent (5000 steps, learning rate 0.01) with A fixed
            at twice the 99th percentile of the resampled rates (SI), and
            A_k scaled from the point estimate by the same factor (SD)
            (fig. S12)

Reads the HCP recordings (data/HCP_data_task/) and data/inferred_networks.h5
and writes data/<YYYY-MM-DD>/rate_constant_bootstrap.h5 with groups
arrhenius and glauber, each holding A_sd (tasks, N, replicates) and, for
SI and SD, r_<model> and loss_<model> (tasks, replicates): the Pearson
correlation and the squared error between the fitted and observed flux.

Usage: python scripts/bootstrap_rate_constants.py [--seed 42]
"""

import argparse
import logging
import os
from pathlib import Path

# JAX runs on the CPU unless JAX_PLATFORMS is set, so that the results do
# not depend on the GPU model.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")

import brain_probability_flux.plotting as bpp
from brain_probability_flux.analyzer import (
    transition_rate_to_network_parameters_lstsq,
    transition_rate_to_network_parameters_lstsq_spindep,
)
from brain_probability_flux.provenance import (
    run_dir,
    write_provenance,
)

log = logging.getLogger(__name__)

NUM_REALIZATIONS = 100
GLAUBER_LEARNING_RATE = 0.01
GLAUBER_STEPS = 5000


def resample(transition_count, rng):
    """Draw as many transitions as observed, with replacement."""
    m = len(transition_count)
    return transition_count[rng.integers(0, m, size=m)]


def arrhenius_bootstrap(pfa, flux_flat, edges, rng):
    num_spins = pfa.unique_states.shape[1]
    num_states = 2**num_spins
    shape = (bpp.NUM_TASKS, NUM_REALIZATIONS)
    A_sd = np.zeros((bpp.NUM_TASKS, num_spins, NUM_REALIZATIONS))
    r = {"SI": np.zeros(shape), "SD": np.zeros(shape)}
    loss = {"SI": np.zeros(shape), "SD": np.zeros(shape)}
    for k in range(bpp.NUM_TASKS):
        pfa.trajectories_to_flux(k, progress_message=False)
        for b in range(NUM_REALIZATIONS):
            rates = bpp.rate_matrix_from_transitions(
                resample(pfa.transition_count, rng), num_states
            )
            bJ_si, bh_si, A_si_b, _ = (
                transition_rate_to_network_parameters_lstsq(rates)
            )
            bJ_sd, bh_sd, A_sd_b, _ = (
                transition_rate_to_network_parameters_lstsq_spindep(rates)
            )
            A_sd[k, :, b] = A_sd_b
            model_flux = {
                "SI": bpp.model_flux_matrix(bJ_si, bh_si, float(A_si_b)),
                "SD": bpp.model_flux_matrix(bJ_sd, bh_sd, A_sd_b),
            }
            for key, flux in model_flux.items():
                fitted = flux[edges]
                r[key][k, b] = stats.pearsonr(flux_flat[k], fitted).statistic
                loss[key][k, b] = float(np.sum((flux_flat[k] - fitted) ** 2))
    return A_sd, r, loss


def glauber_descent_function(num_spins):
    """Jitted gradient descent of the linearized Glauber loss, with the
    diagonal of J held at zero."""
    num_states = 2**num_spins
    state_vectors = bpp.create_state_vectors(num_spins)
    source = np.repeat(np.arange(num_states), num_spins)
    spin = np.tile(np.arange(num_spins), num_states)
    flip = jnp.asarray(state_vectors[source, spin])
    source_vectors = jnp.asarray(state_vectors[source])
    spin_j = jnp.asarray(spin)
    diag = jnp.arange(num_spins)

    def loss_function(bJ_flat, bh, rates, A_k):
        bJ = bJ_flat.reshape(num_spins, num_spins).at[diag, diag].set(0.0)
        A_t = A_k[spin_j]
        feasible = (rates > 0.0) & (rates < A_t)
        n_feasible = jnp.maximum(feasible.sum().astype(float), 1.0)
        log_arg = jnp.where(feasible, A_t / rates - 1.0, 1.0)
        reference = flip / 2.0 * jnp.log(log_arg)
        prediction = (source_vectors * bJ[spin_j]).sum(axis=1) + bh[spin_j]
        squared = jnp.where(feasible, jnp.square(prediction - reference), 0.0)
        return squared.sum() / n_feasible

    value_and_grad = jax.value_and_grad(loss_function, argnums=(0, 1))

    @jax.jit
    def descent(rates, A_k):
        def step(_, carry):
            bJ_f, bh = carry
            _, (gJ, gh) = value_and_grad(bJ_f, bh, rates, A_k)
            return (
                bJ_f - GLAUBER_LEARNING_RATE * gJ,
                bh - GLAUBER_LEARNING_RATE * gh,
            )

        init = (jnp.zeros(num_spins * num_spins), jnp.zeros(num_spins))
        bJ_f, bh = jax.lax.fori_loop(0, GLAUBER_STEPS, step, init)
        bJ = bJ_f.reshape(num_spins, num_spins)
        return bJ.at[diag, diag].set(0.0), bh

    target = source ^ (1 << (num_spins - 1 - spin))
    return descent, source, target


def glauber_bootstrap(pfa, flux_flat, edges, A_si, A_sd_point, rng):
    num_spins = pfa.unique_states.shape[1]
    num_states = 2**num_spins
    descent, source, target = glauber_descent_function(num_spins)
    shape = (bpp.NUM_TASKS, NUM_REALIZATIONS)
    A_sd = np.zeros((bpp.NUM_TASKS, num_spins, NUM_REALIZATIONS))
    r = {"SI": np.zeros(shape), "SD": np.zeros(shape)}
    loss = {"SI": np.zeros(shape), "SD": np.zeros(shape)}
    for k in range(bpp.NUM_TASKS):
        pfa.trajectories_to_flux(k, progress_message=False)
        for b in range(NUM_REALIZATIONS):
            rate_matrix = bpp.rate_matrix_from_transitions(
                resample(pfa.transition_count, rng), num_states
            )
            rates_np = rate_matrix[target, source]
            rates = jnp.asarray(rates_np, dtype=jnp.float32)
            positive = rates_np[rates_np > 0]
            A_si_b = (
                2.0 * np.percentile(positive, 99)
                if len(positive) > 0
                else float(A_si[k])
            )
            A_sd_b = A_sd_point[k] * (
                A_si_b / float(A_si[k]) if A_si[k] > 0 else 1.0
            )
            A_sd[k, :, b] = A_sd_b
            for key, A in [("SI", A_si_b), ("SD", A_sd_b)]:
                A_k = jnp.broadcast_to(
                    jnp.asarray(A, dtype=jnp.float32), (num_spins,)
                )
                bJ, bh = descent(rates, A_k)
                fitted = bpp.model_flux_matrix(
                    np.array(bJ, dtype=float),
                    np.array(bh, dtype=float),
                    A,
                    weight_type="glauber",
                )[edges]
                r[key][k, b] = stats.pearsonr(flux_flat[k], fitted).statistic
                loss[key][k, b] = float(np.sum((flux_flat[k] - fitted) ** 2))
    return A_sd, r, loss


def write_group(group, A_sd, r, loss):
    group.create_dataset("A_sd", data=A_sd)
    for key in ["SI", "SD"]:
        group.create_dataset(f"r_{key}", data=r[key])
        group.create_dataset(f"loss_{key}", data=loss[key])


def main(seed: int, data_dir: Path) -> None:
    glauber = bpp.load_results("inferred_networks", "glauber", data_dir)
    glauber_sd = bpp.load_results(
        "inferred_networks", "glauber_spindep", data_dir
    )
    pfa, _ = bpp.coarse_grained_pfa()
    flux_flat, row, col = bpp.observed_edge_flux(pfa)
    edges = (row, col)

    out = run_dir() / "rate_constant_bootstrap.h5"
    with h5py.File(out, "w-") as f:
        write_provenance(
            f,
            seed=seed,
            num_realizations=NUM_REALIZATIONS,
            glauber_learning_rate=GLAUBER_LEARNING_RATE,
            glauber_steps=GLAUBER_STEPS,
            input=str(Path(data_dir) / "inferred_networks.h5"),
        )
        log.info("Arrhenius")
        rng = np.random.default_rng(seed)
        write_group(
            f.create_group("arrhenius"),
            *arrhenius_bootstrap(pfa, flux_flat, edges, rng),
        )
        log.info("Glauber")
        rng = np.random.default_rng(seed)
        write_group(
            f.create_group("glauber"),
            *glauber_bootstrap(
                pfa,
                flux_flat,
                edges,
                glauber["rate_const"],
                glauber_sd["rate_const"],
                rng,
            ),
        )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=bpp.DATA_DIR)
    args = parser.parse_args()
    main(args.seed, args.data_dir)
