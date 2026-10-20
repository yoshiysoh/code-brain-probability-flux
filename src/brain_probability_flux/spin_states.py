"""Hypercube state-space helpers shared by the analysis and figure code.

A state of N spins is indexed by an integer whose binary digits are the
spins, most significant bit first: spin k is bit N - k - 1, and a bit
value of 1 means s_k = +1.
"""

import numpy as np


def locate_bit(num_spins: int, k: int) -> int:
    """Bit position of spin `k` in the integer state index."""
    return num_spins - k - 1


def generate_binary_codes(num_spins: int) -> list[str]:
    """Zero-padded binary strings for 0..2**num_spins-1."""
    return [bin(i)[2:].zfill(num_spins) for i in range(2**num_spins)]


def create_state_vectors(num_spins: int) -> np.ndarray:
    """(2**num_spins, num_spins) array of +/-1 spin vectors, MSB-first."""
    states = np.arange(2**num_spins)[:, None]
    bits = num_spins - 1 - np.arange(num_spins)[None, :]
    return ((states >> bits) & 1) * 2.0 - 1.0


def binary_codes_to_ising(codes: list[str]) -> np.ndarray:
    """Convert binary strings ("0101") to +/-1 spin vectors."""
    if len(codes) == 0:
        return np.zeros((0,))
    return np.array([list(map(int, code)) for code in codes]) * 2 - 1


def _one_bit_mask(num_spins: int) -> np.ndarray:
    states = np.arange(2**num_spins)
    xor = states[:, None] ^ states[None, :]
    return np.bitwise_count(xor) == 1


def get_one_bit_difference_indices(
    num_spins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Directed (row, col) indices of all one-bit-flip hypercube transitions.

    Row-major (i, j) order: the order of the edge fluxes used throughout
    (observed_edge_flux, the model comparisons).
    """
    return np.nonzero(_one_bit_mask(num_spins))


def get_one_bit_difference_indices_undirected(
    num_spins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same adjacency as get_one_bit_difference_indices, i < j only."""
    return np.nonzero(np.triu(_one_bit_mask(num_spins)))
