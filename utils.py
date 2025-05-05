import numpy as np
import random
from typing import Optional


def sbm(n_nodes: int, community_matrix: np.ndarray, assignments: np.ndarray, seed: int):
    """
    Returns SBM sample with the given communities

    community_matrix: np.ndarray with shape (n_communities, n_communities)
    assignments: np.ndarray with shape (n_nodes,)

    """
    rng = np.random.default_rng(seed=seed)

    network = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            network[i,j] = rng.random() < community_matrix[assignments[i],assignments[j]]

    return network

def evolve_communities_markov(
    assignments: np.ndarray,
    transition_matrix: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:

    """
    Evolves a vector of community assignments one step forward according to a Markov chain.

    Each element in the assignments vector represents an independent entity
    whose state (community) transitions according to the shared transition_matrix.

    Parameters:
    -----------
    assignments : np.ndarray
        A 1D NumPy array of integers representing the current community assignment
        for each entity (node/person). Shape (n_entities,).
        Values should be 0-based indices corresponding to the states/communities.
    transition_matrix : np.ndarray
        A k x k NumPy array where transition_matrix[i, j] is the probability
        of transitioning from community/state i to community/state j in one step.
        k is the number of communities/states.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    np.ndarray
        A 1D NumPy array of the same shape as current_communities, containing
        the community assignments after one step of Markov evolution.
    """
    n_entities = len(assignments)
    if n_entities == 0:
        return np.array([], dtype=int) # Handle empty input

    num_communities = transition_matrix.shape[0]

    # --- Input Validation ---
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be a square matrix (k x k).")
    if not np.allclose(transition_matrix.sum(axis=1), 1.0):
        rows_sum_not_one = np.where(~np.isclose(transition_matrix.sum(axis=1), 1.0))[0]
        raise ValueError(f"Rows of transition_matrix must sum to 1. Problematic rows: {rows_sum_not_one}")
    if not np.all((assignments >= 0) & (assignments < num_communities)):
         invalid_indices = np.where(~((assignments >= 0) & (assignments < num_communities)))[0]
         raise ValueError(f"Input 'assignments' contains invalid state indices (must be 0 to k-1={num_communities-1}). Invalid values at indices: {invalid_indices}")
    # --- End Validation ---

    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Initialize array for the next states
    next_communities = np.zeros(n_entities, dtype=int)

    # Iterate through each entity and evolve its state independently
    for i in range(n_entities):
        current_state = assignments[i]
        # Get the probabilities for transitioning from the current state
        probabilities = transition_matrix[current_state, :]
        # Sample the next state based on these probabilities
        next_communities[i] = rng.choice(num_communities, p=probabilities)

    return next_communities


def simple_rw_transition(m: int, dtype=float) -> np.ndarray:
    """
    Transition matrix for a 1‑D simple random walk with reflecting
    boundaries on the state set {0,…,m‑1}.

    Parameters
    ----------
    m : int
        Number of integer states (m ≥ 2).
    dtype : data‑type, optional
        NumPy floating type for the matrix (default: float64).

    Returns
    -------
    P : ndarray of shape (m, m)
        Stochastic matrix whose (i,j) entry is P[X_{k+1}=j | X_k=i].
    """
    P = np.zeros((m, m), dtype=dtype)

    idx = np.arange(1, m - 1)
    P[idx, idx - 1] = 0.5
    P[idx, idx + 1] = 0.5

    P[0, 1]       = 1.0       
    P[m - 1, m-2] = 1.0       
    return P
