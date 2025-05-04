import numpy as np
import random
from utils import sbm, evolve_communities_markov
import matplotlib.pyplot as plt
from inference.hiersbm import run_hbsm_inference
from inference.dpsbm import run_dpsbm_inference
from inference.pisces import run_pisces_inference

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
        self.dynamicNetwork = None # Eventually (n_nodes, n_nodes, n_layers)
        self.community_assignments = None # (n_nodes, n_layers)
        self.learned_trajectories = None # (n_nodes, n_layers, iters)
        self.learned_community_matrix = None # (n_nodes, n_nodes)
        self.inference_method = None # str
         
    def generate_markov_network(self, transition_matrix: np.ndarray):
        """
        Generate a Markov network with the given transition matrix.

        transition_matrix: np.ndarray with shape (num_communities, num_communities)
            Transition matrix
        """
        # Generate community assignments
        next_assignment = np.random.choice(
            self.num_communities,
            size=self.n_nodes,
            p=self.initial_distribution
        )
        
        networks = []
        community_assignments = []
        for i in range(self.n_layers):

            network_i = sbm(self.n_nodes, self.community_matrix, next_assignment)
            networks.append(network_i) 
            community_assignments.append(next_assignment)
            next_assignment = evolve_communities_markov(next_assignment, transition_matrix)

        self.dynamicNetwork = np.stack(networks, axis=-1).astype(networks[0].dtype if networks else int)
        self.community_assignments = np.stack(community_assignments, axis=-1).astype(community_assignments[0].dtype if community_assignments else int)

    def learn_community_dynamics(self, method: str, num_iters: int):
        """
            Learn the community dynamics using the specified method.

            method: str
                Method to use for learning the community dynamics. 
                Methods supported:
                    - 'hbsm'
                    - 'dpsbm'
                    - 'pisces'
                    - 'spectral'
        """

        if method == 'hbsm':
            kcap = self.num_communities + 5 # Example default
            gcap = self.num_communities + 5 # Example default
            run_hbsm_inference(self, niter=num_iters, Kcap=kcap, Gcap=gcap)
        elif method == 'dpsbm':
            run_dpsbm_inference(self, niter = num_iters)
        elif method == 'pisces':
            run_pisces_inference(self, K=self.num_communities, niter=num_iters, verb = True)
            
        else:
            raise ValueError(f"Method {method} not supported.")
        

if __name__ == "__main__":

    rng = np.random.default_rng(seed=42)

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

    
