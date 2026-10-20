"""Least-squares inference of Ising network parameters from transition rates.

`transition_rate[mu, nu]` is the observed rate of the single-spin flip
nu -> mu. States are indexed as in `brain_probability_flux.spin_states`.
"""

import numpy as np

from ..spin_states import create_state_vectors, locate_bit


def _num_spins(transition_rate: np.ndarray) -> int:
    return int(np.round(np.log2(len(transition_rate))))


def create_transition_pairs(
    transition_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed single-spin flips as (source, target) pairs and flipped spin.

    Ordered by source state, then by spin.
    """
    num_spins = _num_spins(transition_rate)
    sources = np.arange(len(transition_rate))[:, None]
    masks = 1 << (num_spins - 1 - np.arange(num_spins))[None, :]
    targets = sources ^ masks
    observed = transition_rate[targets, sources] > 0
    source_idx, spins = np.nonzero(observed)
    pairs = np.stack([source_idx, targets[source_idx, spins]], axis=1)
    return pairs, spins


def least_square_inference(transition_rate: np.ndarray):
    """Infer beta*J, beta*h and the Arrhenius rate constant A jointly.

    Each observed flip nu -> mu gives one linear equation
    2 ln w(mu|nu) = 2 ln A + sum_ij C_ij(mu, nu) x_ij.

    Returns
    -------
    beta_J : (N, N) ndarray, zero diagonal
    beta_h : (N,) ndarray
    rate_const : float
    residuals : float
    """
    num_spins = _num_spins(transition_rate)
    state_vectors = create_state_vectors(num_spins)
    pairs, _ = create_transition_pairs(transition_rate)
    source = state_vectors[pairs[:, 0]]
    target = state_vectors[pairs[:, 1]]
    b = 2 * np.log(transition_rate[pairs[:, 1], pairs[:, 0]])

    diff = target - source
    coeff = diff[:, :, None] * source[:, None, :]
    diag = np.arange(num_spins)
    coeff[:, diag, diag] = diff
    C = np.ones((len(pairs), num_spins**2 + 1))
    C[:, :-1] = coeff.reshape(len(pairs), -1)

    x, residuals, _, _ = np.linalg.lstsq(C, b, rcond=None)

    beta_J = x[:-1].reshape(num_spins, num_spins).copy()
    beta_h = np.diag(beta_J).copy()
    np.fill_diagonal(beta_J, 0.0)
    return beta_J, beta_h, float(np.exp(x[-1] / 2)), residuals[0]


def _per_spin_least_squares(
    transition_rate: np.ndarray,
    model: str,
    rate_const: float | np.ndarray | None = None,
):
    """Fit beta*J[k], beta*h[k] from spin k's flips alone, for each spin k.

    model="arrhenius": w = A_k exp(-s_k g_k), with eta_k = 2 ln A_k fitted
    as an intercept; returns eta as the third element.
    model="glauber": w = A_k / (1 + exp(2 s_k g_k)) with A_k fixed at
    `rate_const` (scalar, or one value per spin); flips with w >= A_k are
    dropped. Returns A as the third element.
    """
    num_spins = _num_spins(transition_rate)
    state_vectors = create_state_vectors(num_spins)
    nu = np.arange(len(transition_rate))
    rate_consts = np.broadcast_to(
        0.0 if rate_const is None else rate_const, (num_spins,)
    )

    beta_J = np.zeros((num_spins, num_spins))
    beta_h = np.zeros(num_spins)
    eta = np.zeros(num_spins)
    total_residuals = 0.0

    for k in range(num_spins):
        w = transition_rate[nu ^ (1 << locate_bit(num_spins, k)), nu]
        if model == "arrhenius":
            rows = w > 0
            s = state_vectors[rows]
            s_k = s[:, k]
            features = np.empty((len(s), num_spins + 1))
            features[:, 0] = 1.0
            features[:, 1:] = -2.0 * s_k[:, None] * s
            features[:, 1 + k] = -2.0 * s_k
            b = 2.0 * np.log(w[rows])
        else:
            rows = (w > 0) & (w < rate_consts[k])
            s = state_vectors[rows]
            features = s.copy()
            features[:, k] = 1.0
            b = s[:, k] / 2.0 * np.log(rate_consts[k] / w[rows] - 1.0)

        x, res, _, _ = np.linalg.lstsq(features, b, rcond=None)
        if model == "arrhenius":
            eta[k], x = x[0], x[1:]
        beta_h[k] = x[k]
        beta_J[k] = x
        beta_J[k, k] = 0.0
        if len(res) > 0:
            total_residuals += float(res[0])

    third = eta if model == "arrhenius" else np.array(rate_consts)
    return beta_J, beta_h, third, total_residuals


def _glauber_rate_const(
    transition_rate, spin_dependent, percentile=99, percentile_multiplier=2.0
):
    """Fixed Glauber rate constant: a multiple of a high percentile of the
    observed rates, which keeps rare-state outliers from setting A."""
    if not spin_dependent:
        nonzero = transition_rate[transition_rate > 0]
        return percentile_multiplier * float(
            np.percentile(nonzero, percentile)
        )
    num_spins = _num_spins(transition_rate)
    nu = np.arange(len(transition_rate))
    rate_const = np.zeros(num_spins)
    for k in range(num_spins):
        w = transition_rate[nu ^ (1 << locate_bit(num_spins, k)), nu]
        rate_const[k] = percentile_multiplier * float(
            np.percentile(w[w > 0], percentile)
        )
    return rate_const


def transition_rate_to_network_parameters_lstsq(transition_rate):
    """Arrhenius model with one rate constant A.

    Returns (beta_J, beta_h, A, residuals).
    """
    return least_square_inference(transition_rate)


def transition_rate_to_network_parameters_lstsq_spindep(transition_rate):
    """Arrhenius model with a rate constant A_k per spin.

    A_k cancels in w(mu^(k)|mu) / w(mu|mu^(k)), so the fit decouples into
    one least-squares problem per spin. Returns (beta_J, beta_h, A_k,
    residuals).
    """
    beta_J, beta_h, eta, residuals = _per_spin_least_squares(
        transition_rate, "arrhenius"
    )
    return beta_J, beta_h, np.exp(eta / 2), residuals


def transition_rate_to_network_parameters_lstsq_glauber(
    transition_rate, rate_const=None
):
    """Glauber model with one fixed rate constant A.

    A defaults to twice the 99th percentile of the observed rates, matching
    the gradient-descent inference. Returns (beta_J, beta_h, A, residuals).
    """
    if rate_const is None:
        rate_const = _glauber_rate_const(transition_rate, spin_dependent=False)
    beta_J, beta_h, _, residuals = _per_spin_least_squares(
        transition_rate, "glauber", rate_const
    )
    return beta_J, beta_h, rate_const, residuals


def transition_rate_to_network_parameters_lstsq_glauber_spindep(
    transition_rate,
):
    """Glauber model with a fixed rate constant A_k per spin.

    A_k is twice the 99th percentile of spin k's own observed flip rates.
    Returns (beta_J, beta_h, A_k, residuals).
    """
    rate_const = _glauber_rate_const(transition_rate, spin_dependent=True)
    return _per_spin_least_squares(transition_rate, "glauber", rate_const)
