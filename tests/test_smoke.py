"""Fast checks of the pipeline on synthetic data (no HCP recordings)."""

import h5py
import numpy as np
import pytest

from brain_probability_flux.analyzer import (
    PFA,
    transition_rate_to_network_parameters_lstsq,
    transition_rate_to_network_parameters_lstsq_glauber,
)
from brain_probability_flux.analyzer._probability_flux_analyzer import (
    _entropy_production_rate,
)
from brain_probability_flux.dina.dynamics import ising
from brain_probability_flux.spin_states import (
    create_state_vectors,
    get_one_bit_difference_indices,
)

NUM_SPINS = 3


@pytest.fixture
def couplings():
    rng = np.random.default_rng(42)
    beta_J = rng.normal(0.0, 0.5, (NUM_SPINS, NUM_SPINS))
    np.fill_diagonal(beta_J, 0.0)
    beta_h = rng.normal(0.0, 0.3, NUM_SPINS)
    return beta_J, beta_h


def model_rates(beta_J, beta_h, rate_const, weight_type="arrhenius"):
    """Off-diagonal rates w[target, source] of the Ising model."""
    rates = rate_const * ising.im_to_transition_rate(
        beta_J, 1.0, weight_type=weight_type, magnetic_field=beta_h
    )
    np.fill_diagonal(rates, 0.0)
    return rates


def test_lstsq_recovers_arrhenius_parameters(couplings):
    beta_J, beta_h = couplings
    rates = model_rates(beta_J, beta_h, rate_const=0.7)
    fit_J, fit_h, fit_A, _ = transition_rate_to_network_parameters_lstsq(rates)
    np.testing.assert_allclose(fit_J, beta_J, atol=1e-10)
    np.testing.assert_allclose(fit_h, beta_h, atol=1e-10)
    assert fit_A == pytest.approx(0.7)


def test_lstsq_glauber_returns_shapes(couplings):
    beta_J, beta_h = couplings
    rates = model_rates(beta_J, beta_h, 1.0, weight_type="glauber")
    fit_J, fit_h, _, _ = transition_rate_to_network_parameters_lstsq_glauber(
        rates
    )
    assert fit_J.shape == (NUM_SPINS, NUM_SPINS)
    assert fit_h.shape == (NUM_SPINS,)


def test_model_flux_is_antisymmetric_and_stationary(couplings):
    beta_J, beta_h = couplings
    flux = ising.im_to_flux(beta_J, 1.0, magnetic_field=beta_h)
    np.testing.assert_allclose(flux, -flux.T, atol=1e-12)
    # Stationarity: the net flux into each state vanishes.
    np.testing.assert_allclose(flux.sum(axis=1), 0.0, atol=1e-12)


def test_entropy_production_vanishes_at_detailed_balance():
    counts = np.array([[0.0, 3.0, 1.0], [3.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    assert _entropy_production_rate(counts, 10.0) == pytest.approx(0.0)
    counts[0, 1] = 5.0
    assert _entropy_production_rate(counts, 10.0) > 0.0


def test_aggregates_round_trip(tmp_path):
    """A PFA restored from synthetic counts gives an antisymmetric flux
    on the hypercube edges and a normalized probability."""
    rng = np.random.default_rng(42)
    states = create_state_vectors(NUM_SPINS)
    num_states = len(states)
    source, target = get_one_bit_difference_indices(NUM_SPINS)
    path = tmp_path / "aggregates.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("setting")
        group.create_dataset("unique_states", data=states.astype(np.int8))
        group.create_dataset("probability", data=np.full(num_states, 0.125))
        for k in range(8):
            task = group.create_group(f"task{k}")
            task.create_dataset("states", data=states.astype(np.int8))
            task.create_dataset("source", data=source)
            task.create_dataset("target", data=target)
            task.create_dataset(
                "count", data=rng.integers(1, 20, len(source)).astype(float)
            )
            task.create_dataset("waiting_time", data=rng.random(num_states))
            task.attrs["total_time"] = 100.0
    with h5py.File(path, "r") as f:
        pfa = PFA.from_aggregates(f["setting"])
    pfa.perform_WPCA()
    for k in range(8):
        pfa.trajectories_to_flux(k)
        flux = pfa.flux_matrix.toarray()
        np.testing.assert_allclose(flux, -flux.T)
        assert pfa.entropy_production_rate >= 0.0
        assert pfa.transition_rate_matrix().shape == (num_states, num_states)
    assert pfa.wpca.components_.shape == (NUM_SPINS, NUM_SPINS)
