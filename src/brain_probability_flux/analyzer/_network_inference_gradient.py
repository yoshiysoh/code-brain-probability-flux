"""Gradient-descent inference of Ising network parameters (JAX).

`transition_rate[mu, nu]` is the observed rate of the single-spin flip
nu -> mu. States are indexed as in `brain_probability_flux.spin_states`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.notebook import trange

from ..spin_states import create_state_vectors
from ._network_inference_lstsq import _num_spins, create_transition_pairs


def _flip_data(transition_rate, state_vectors):
    """Per observed flip: flipped spin, its value, the rate, the source."""
    pairs, spins = create_transition_pairs(transition_rate)
    pairs = jnp.asarray(pairs)
    spins = jnp.asarray(spins)
    flipping_spins = state_vectors[pairs[:, 0], spins]
    rates = transition_rate[pairs[:, 1], pairs[:, 0]]
    source_states = state_vectors[pairs[:, 0], :]
    return spins, flipping_spins, rates, source_states


def _local_fields(beta_J, beta_h, spins, source_states):
    """beta*g_k(s) = sum_j beta*J_kj s_j + beta*h_k for each flip."""
    fields = jnp.einsum("nj, nj -> n", beta_J[spins], source_states)
    return fields + beta_h[spins]


def _all_finite(grads):
    return all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(grads))


def _descend(loss_function, variables, learning_rate, num_steps, verbose):
    """Plain gradient descent; stops early at a non-finite gradient."""
    value_grad_function = jax.jit(jax.value_and_grad(loss_function))
    loss_history = []
    for _ in trange(num_steps, disable=not verbose):
        loss, grads = value_grad_function(variables)
        if not _all_finite(grads):
            print("NaN or infinity")
            break
        loss_history.append(loss)
        variables = jax.tree_util.tree_map(
            lambda v, g: v - learning_rate * g, variables, grads
        )
    return variables, np.array(loss_history)


def _zero_diagonal(beta_J):
    return np.array(beta_J.at[jnp.diag_indices_from(beta_J)].set(0.0))


def glauber_loss_function(transition_rate, rate_const=None):
    """Mean squared error of the linearized Glauber rate equation.

    The Glauber rate is w = A / (1 + exp(2 s_k beta g_k)). A is fixed (by
    default at twice the 99th-percentile rate) because the loss is flat
    along (A -> inf, g -> inf). Flips with w >= A, which involve states of
    near-zero probability, are excluded from the loss.

    Returns (loss_function, rate_const, num_spins).
    """
    num_spins = _num_spins(transition_rate)
    state_vectors = jnp.asarray(create_state_vectors(num_spins))
    spins, flipping_spins, rates, source_states = _flip_data(
        transition_rate, state_vectors
    )
    if rate_const is None:
        rate_const = float(np.percentile(rates, 99)) * 2.0
    rate_const_jnp = jnp.asarray(rate_const, dtype=jnp.float32)
    feasible = jnp.asarray(rates < rate_const)
    num_feasible = int(feasible.sum())

    def loss_function(variables):
        """Mean squared error of the linearized rates."""
        fields = _local_fields(
            variables["beta_interaction_matrix"],
            variables["beta_external_field"],
            spins,
            source_states,
        )
        log_arg = jnp.where(feasible, rate_const_jnp / rates - 1.0, 1.0)
        reference = flipping_spins / 2.0 * jnp.log(log_arg)
        sq_err = jnp.square(fields - reference)
        return jnp.where(feasible, sq_err, 0.0).sum() / num_feasible

    return loss_function, rate_const, num_spins


def transition_rate_to_network_parameters_gradient(
    transition_rate,
    learning_rate=1e-3,
    num_steps=40000,
    rate_model="glauber",
    verbose=True,
):
    """Fit the Glauber model by gradient descent from zero couplings.

    Returns (beta_J, beta_h, A, loss_history).
    """
    if rate_model != "glauber":
        raise ValueError(f"Unknown rate model: {rate_model}")
    print("If the GPU is correctly recognized '[cuda(id=0)]' will be printed.")
    print(jax.devices())
    loss_function, rate_const, num_spins = glauber_loss_function(
        transition_rate
    )
    variables = {
        "beta_interaction_matrix": jnp.zeros((num_spins, num_spins)),
        "beta_external_field": jnp.zeros(num_spins),
    }
    variables, loss_history = _descend(
        loss_function, variables, learning_rate, num_steps, verbose
    )
    return (
        _zero_diagonal(variables["beta_interaction_matrix"]),
        np.array(variables["beta_external_field"]),
        rate_const,
        loss_history,
    )


def transition_rate_to_network_parameters_gradient_same_network(
    transition_rates, learning_rate=1e-3, num_steps=40000
):
    """Fit one Arrhenius network shared by all tasks.

    The coupling matrix and the rate constant A are common to all tasks;
    each task has its own external field. The loss is the sum over tasks
    of the mean squared error of the log-linearized Arrhenius rates.

    Returns (beta_J, beta_h per task, A, loss_history).
    """
    print("If the GPU is correctly recognized '[cuda(id=0)]' will be printed.")
    print(jax.devices())
    num_spins = _num_spins(transition_rates[0])
    state_vectors = jnp.asarray(create_state_vectors(num_spins))
    tasks = [_flip_data(rate, state_vectors) for rate in transition_rates]

    def loss_function(variables):
        """Mean squared error of the linearized rates."""
        beta_J = variables["beta_interaction_matrix"]
        rate_const = variables["rate_const"]
        loss = 0.0
        for beta_h, (spins, flipping, rates, sources) in zip(
            variables["beta_external_fields"], tasks
        ):
            fields = _local_fields(beta_J, beta_h, spins, sources)
            reference = -flipping * jnp.log(rates / rate_const)
            loss += jnp.square(fields - reference).mean()
        return loss

    variables = {
        "beta_interaction_matrix": jnp.ones((num_spins, num_spins)),
        "beta_external_fields": [jnp.ones(num_spins) for _ in tasks],
        "rate_const": jnp.asarray(1.0, dtype=jnp.float32),
    }
    variables, loss_history = _descend(
        loss_function, variables, learning_rate, num_steps, verbose=True
    )
    return (
        _zero_diagonal(variables["beta_interaction_matrix"]),
        np.array(variables["beta_external_fields"]),
        variables["rate_const"],
        loss_history,
    )
