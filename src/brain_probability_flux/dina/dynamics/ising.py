"""
Ising-model transition rates, stationary distributions, and fluxes
"""

import numpy as np
from numba import njit


################################
# Commons
################################
# spin flip weight function from each input energy
@njit(cache=True, nogil=False)
def _arrhenius_weight(beta, energy1d, freq=1, barrier=0):
    weight = freq * np.exp(beta * (energy1d - barrier))
    return weight


@njit(cache=True, nogil=False)
def _glauber_weight(beta, energy1d, freq=1, barrier=0):
    weight = 1.0 / (1.0 + np.exp(-beta * 2 * energy1d))
    return weight


def temperature_to_beta(T):
    """Inverse temperature 1/T (0 where T = 0)."""
    with np.errstate(divide="ignore"):
        beta = np.divide(1.0, T)
    beta = np.nan_to_num(beta)
    return beta


################################
# Master equation solver
################################
@njit(cache=True, nogil=False)
def generate_all_ising_state(Ndims):
    """(2**Ndims, Ndims) +/-1 spin vectors, MSB-first (as spin_states)."""
    states = np.zeros((2**Ndims, Ndims))
    for i in range(2**Ndims):
        for k in range(Ndims):
            states[i, k] = 2.0 * ((i >> (Ndims - k - 1)) & 1) - 1.0
    return states


@njit(cache=True, nogil=False)
def _all_states_to_all_energy(all_states, im, magnetic_field):
    energy = np.zeros((len(all_states), len(im)))
    for i in range(len(all_states)):
        state = all_states[i]
        e = -state * (im @ state) - state * magnetic_field
        energy[i] = e
    return energy


@njit(cache=True, nogil=False)
def _all_states_to_transition_rate(beta, all_states, energy, _weight_func):
    num_states = len(all_states)
    transition_rate = np.zeros((num_states, num_states))
    for i in range(transition_rate.shape[0]):
        for j in range(transition_rate.shape[1]):
            if i == j:
                continue
            else:
                change = np.argwhere(all_states[i] * all_states[j] == -1)
                if len(change) != 1:
                    continue
                else:
                    transition_rate[i, j] = _weight_func(
                        beta, energy[j, change[0, 0]], freq=1, barrier=0
                    )

    for i in range(num_states):
        transition_rate[i, i] = -np.sum(transition_rate[:, i])
    return transition_rate


def im_to_transition_rate(im, T, weight_type="arrhenius", magnetic_field=None):
    """Master-equation rate matrix W[target, source] of the Ising model."""
    if weight_type == "arrhenius":
        _weight_func = _arrhenius_weight
    elif weight_type == "glauber":
        _weight_func = _glauber_weight
    else:
        raise ValueError("weight_type must be 'arrhenius' or 'glauber'")

    ## generate all state
    im = np.array(im).astype(float)
    Ndims = len(im)
    all_states = generate_all_ising_state(Ndims)
    if magnetic_field is None:
        magnetic_field = 0.0
    magnetic_field = np.broadcast_to(
        np.asarray(magnetic_field, dtype=np.float64), (Ndims,)
    ).copy()

    energy = _all_states_to_all_energy(all_states, im, magnetic_field)

    beta = temperature_to_beta(T)
    transition_rate = _all_states_to_transition_rate(
        beta, all_states, energy, _weight_func
    )

    return transition_rate


################################
# Solving master equation as eigenvale problem
################################


# Not jitted: Numba cannot type np.linalg.eig's complex return values.
def transition_rate_to_eig(transition_rate):
    """Eigenvalues and normalized eigenvectors of W, slowest first."""
    values, vectors = np.linalg.eig(transition_rate)
    vectors = np.abs(vectors)
    vectors = vectors / vectors.sum(axis=0)
    vectors = vectors.T
    sorted_index = np.argsort(values.real)[::-1]
    values = values[sorted_index]
    vectors = vectors[sorted_index]
    return values, vectors


def im_to_joint_transition(
    im, T, weight_type="arrhenius", magnetic_field=None
):
    """
    Interaction matrix to master equation term $w_{ij}P_{i}$.
    """
    transition_rate = im_to_transition_rate(im, T, weight_type, magnetic_field)
    _values, vectors = transition_rate_to_eig(transition_rate)
    joint_transition = transition_rate * vectors[0]
    return joint_transition


def im_to_flux(
    im, T, weight_type="arrhenius", magnetic_field=None, rate_const=1.0
):
    """Stationary probability flux matrix of the Ising model."""
    joint_transition = im_to_joint_transition(
        im, T, weight_type, magnetic_field
    )
    flux = joint_transition - joint_transition.T
    np.fill_diagonal(flux, 0.0)
    return rate_const * flux


def im_to_traffic(im, T, weight_type="arrhenius", magnetic_field=None):
    """Stationary probability traffic matrix of the Ising model."""
    joint_transition = im_to_joint_transition(
        im, T, weight_type, magnetic_field
    )
    traffic = joint_transition + joint_transition.T
    np.fill_diagonal(traffic, 0.0)
    return traffic


@njit(cache=True, nogil=False)
def _nonzero_indexes_to_list(matrix, all_states, nonzero_indexes, num_spins):
    num_rates = len(nonzero_indexes[0])
    source_list = np.zeros((num_rates, num_spins))
    target_list = np.zeros((num_rates, num_spins))
    rate_list = np.zeros(num_rates)
    for i in range(num_rates):
        s = all_states[nonzero_indexes[1][i]]
        t = all_states[nonzero_indexes[0][i]]
        r = matrix[nonzero_indexes[0][i], nonzero_indexes[1][i]]
        source_list[i] = s
        target_list[i] = t
        rate_list[i] = r
    return source_list, target_list, rate_list


@njit(cache=True, nogil=False)
def flux_to_flux_list(flux):
    """(source states, target states, flux) of the positive fluxes."""
    num_spins = np.log2(len(flux))
    num_spins = np.int64(num_spins)
    all_states = generate_all_ising_state(num_spins)
    nonzero_indexes = flux.nonzero()
    num_nonzero = len(nonzero_indexes[0])
    mask = np.zeros(num_nonzero, dtype=np.bool_)
    for n in range(num_nonzero):
        i = nonzero_indexes[0][n]
        j = nonzero_indexes[1][n]
        if flux[i, j] > 0.0:
            mask[n] = True
    nonzero_indexes = (nonzero_indexes[0][mask], nonzero_indexes[1][mask])
    return _nonzero_indexes_to_list(
        flux, all_states, nonzero_indexes, num_spins
    )


@njit(cache=True, nogil=False)
def traffic_to_traffic_list(traffic):
    """(source states, target states, traffic) of each nonzero traffic."""
    num_spins = np.log2(len(traffic))
    num_spins = np.int64(num_spins)
    all_states = generate_all_ising_state(num_spins)
    triu_traffic = np.triu(traffic, 1)
    nonzero_indexes = triu_traffic.nonzero()
    return _nonzero_indexes_to_list(
        traffic, all_states, nonzero_indexes, num_spins
    )
