import numpy as np
import random
from utils import sbm, evolve_communities_markov
import matplotlib.pyplot as plt
from inference.hiersbm import run_hbsm_inference
from inference.dpsbm import run_dpsbm_inference
from inference.pisces import run_pisces_inference
from typing import Optional

class EvolvingCommunityModel:

    def __init__(self, n_nodes: int, n_layers: int, num_communities:int, community_matrix: np.ndarray, initial_distribution: np.ndarray):
        
        """
        n_nodes: int
            Number of nodes
        n_layers: int
            Number of layers
        num_communities: int
            Number of communities
        community_matrix: np.ndarray with shape (num_communities, num_communities)
            Matrix of community assignments

        initial_distribution: np.ndarray with shape (num_communities,)
            Initial distribution
        """
        
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.num_communities = num_communities
        self.community_matrix = community_matrix
        self.initial_distribution = initial_distribution
        self.dynamicNetwork = None
        self.true_community_assignments = None 
        self.mcmc_sample_history = None 
        self.learned_trajectories = None 
        self.learned_community_matrix = None 
        self.inference_method = None 
         
    def generate_contracting_network(self):
        """
        Generates a dynamic network based on the Contracting Network Model (CNM) definition.
        (Definition from user prompt, aiming to replicate Def. of CNM for Fig 3a HBSM paper)

        Uses self.n_nodes, self.n_layers, and self.num_communities (as 'm' from the CNM definition).
        
        Sets:
        -----
        self.true_community_assignments : np.ndarray
            Shape (self.n_nodes, self.n_layers). Community assignments z_t. Values in [0, self.num_communities-1].
        self.dynamicNetwork : np.ndarray
            Shape (self.n_nodes, self.n_nodes, self.n_layers). Adjacency matrices A_t.
        
        The connectivity matrix C for the SBM is C_ij = exp(-|i - j|).
        Note: This method uses its own internal random number generator state.
        """
        n = self.n_nodes
        T = self.n_layers
        m = self.num_communities # Using self.num_communities as 'm' from the CNM definition

        if not isinstance(m, int) or m <= 0:
            raise ValueError(f"self.num_communities (m) must be a positive integer, got {m}")
        if n <=0 or T <=0:
            raise ValueError(f"self.n_nodes (n) and self.n_layers (T) must be positive, got n={n}, T={T}")
        if m < n:
            # This warning is important if self.num_communities is small (e.g. K_true for a Markov model)
            # but is used as 'm' which the CNM definition suggests should be m >> n.
            print(f"Warning: self.num_communities (m={m}) is not >> self.n_nodes (n={n}) as suggested by the CNM definition.")

        rng = np.random.default_rng() 
        increments = rng.normal(loc=0.0, scale=3.0, size=T)
        B_t_values = np.cumsum(increments) # B_t_values will have shape (T,)

        self.true_community_assignments = np.zeros((n, T), dtype=int)
        for t_idx in range(T):
            abs_Bt = np.abs(B_t_values[t_idx])
            for i_node_idx in range(n): 

                node_factor_i = i_node_idx + 1 
                community_val = np.floor(node_factor_i/2 * abs_Bt)

                clamped_community = int(np.minimum(community_val, m - 1))
                clamped_community = max(0, clamped_community) # Ensure non-negative
                self.true_community_assignments[i_node_idx, t_idx] = clamped_community

        C_sbm = np.zeros((m, m))
        for i_comm_idx in range(m):
            for j_comm_idx in range(m):
                C_sbm[i_comm_idx, j_comm_idx] = np.exp(-np.abs(i_comm_idx - j_comm_idx))

        # 4. Sample each adjacency matrix A_t
        self.dynamicNetwork = np.zeros((n, n, T), dtype=int)
        for t_idx in range(T):
            current_comms_z_t = self.true_community_assignments[:, t_idx] # Shape (n,)
            for u_node_idx in range(n):
                for v_node_idx in range(u_node_idx + 1, n): # Undirected graph, no self-loops
                    comm_u = current_comms_z_t[u_node_idx]
                    comm_v = current_comms_z_t[v_node_idx]
                    
                    if not (0 <= comm_u < m and 0 <= comm_v < m):
                        print(f"Warning: Invalid community index at t={t_idx} for node u={u_node_idx} (comm {comm_u}) or v={v_node_idx} (comm {comm_v}). Max index should be {m-1}. Skipping.")
                        continue 

                    p_edge = C_sbm[comm_u, comm_v]
                    if rng.random() < p_edge:
                        self.dynamicNetwork[u_node_idx, v_node_idx, t_idx] = 1
                        self.dynamicNetwork[v_node_idx, u_node_idx, t_idx] = 1
    
    def generate_markov_network(self, transition_matrix: np.ndarray, seed: Optional[int] = None):
        """
        Generate a Markov network with the given transition matrix.

        transition_matrix: np.ndarray with shape (num_communities, num_communities)
            Transition matrix
        seed : int, optional
            Random seed for reproducible network generation.
        """
        # Initialize the random number generator for this method
        rng = np.random.default_rng(seed)

        # Generate initial community assignments using the local rng
        next_assignment = rng.choice(
            self.num_communities,
            size=self.n_nodes,
            p=self.initial_distribution
        )

        networks = []
        community_assignments = []
        # Use a consistent seed for each step if a main seed is provided
        step_seed = seed
        for i in range(self.n_layers):
            # Generate SBM network for this layer using the step_seed
            network_i = sbm(self.n_nodes, self.community_matrix, next_assignment, seed=step_seed)
            networks.append(network_i)
            community_assignments.append(next_assignment)
            # Evolve communities for the next layer using the step_seed
            next_assignment = evolve_communities_markov(next_assignment, transition_matrix, seed=step_seed)
            # Increment step_seed deterministically if a seed was given
            if step_seed is not None:
                step_seed += 1

        self.dynamicNetwork = np.stack(networks, axis=-1).astype(networks[0].dtype if networks else int)
        self.true_community_assignments = np.stack(community_assignments, axis=-1).astype(community_assignments[0].dtype if community_assignments else int)

    def learn_community_dynamics(self, method: str, **kwargs):
        """
            Learn the community dynamics using the specified method.

            method: str
                Method to use for learning the community dynamics.
                Methods supported:
                    - 'hbsm'
                    - 'dpsbm'
                    - 'pisces'
                    - 'spectral'
            **kwargs: dict
                Method-specific hyperparameters (e.g., niter, K, alpha, verb, etc.)
                Commonly expected: 'niter' (number of iterations)
        """
        print(f"  learn_community_dynamics called for method '{method}' with kwargs: {kwargs}")

        if method == 'hbsm':
            hbsm_kwargs = kwargs.copy()
            hbsm_kwargs.setdefault('Kcap', self.num_communities + 5)
            hbsm_kwargs.setdefault('Gcap', self.num_communities + 5)
            print(f"  Calling run_hbsm_inference with: {hbsm_kwargs}")
            run_hbsm_inference(self, **hbsm_kwargs)

        elif method == 'dpsbm':
            run_dpsbm_inference(self, **kwargs)

        elif method == 'pisces':
            if 'K' not in kwargs:
                print("Warning: 'K' not found in kwargs for PISCES, defaulting to self.num_communities")
                kwargs['K'] = self.num_communities if self.num_communities < self.n_nodes else self.n_nodes - 2  
            print(f"  Calling run_pisces_inference with: {kwargs}")
            run_pisces_inference(self, **kwargs)

        elif method == 'spectral':
            print("Spectral method not implemented yet.")
            pass

        else:
            raise ValueError(f"Method {method} not supported.")

if __name__ == "__main__":

    rng = np.random.default_rng(seed=40)

    ## PARAMETERS FOR MARKOV MODEL
    n_nodes = 10
    n_layers = 20
    num_communities = 100  # Changed from 10 to match transition matrix dimensions
    community_matrix = rng.normal(0, 1, (num_communities, num_communities))
    initial_distribution = rng.dirichlet(np.ones(num_communities), size = 1)[0]
    transition_matrix = rng.dirichlet(np.ones(num_communities), size = num_communities)

    ## INITIALIZE MODEL
    model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)

    ## GENERATE NETWORKS
    model.generate_markov_network(transition_matrix)
