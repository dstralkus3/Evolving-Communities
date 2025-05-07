import numpy as np
from model import EvolvingCommunityModel
from visualize import Visualizer
from eval import Evaluation
import matplotlib.pyplot as plt

def figure_3a_test():

    # TESTING SYNTHETIC EXPERIMENT FROM hbsm.pdf section 4 figure 3a

    transition_probs = np.linspace(0.0, 1.0, 50)
    results = {}

    for tau in transition_probs:

        # Parameters
        n_nodes = 200
        n_layers = 10
        num_communities = 3

        # Community matrix
        community_matrix = np.array([[0.9, 0.75, 0.5],
                                    [0.75, 0.6, 0.25],
                                    [0.5, 0.25, 0.1]])
        
        # Trasition + initial distribution
        initial_distribution = np.array([0.40, 0.35, 0.25])                                           
        transition_matrix  = (1 - tau) * np.eye(3) + tau * np.ones((3, 1)) * initial_distribution

        # INITIALIZE MODELS
        hbsm_model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)
        dpsbm_model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)
        pisces_model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)

        # Generate same network for all models
        hbsm_model.generate_markov_network(transition_matrix, seed = 1)
        dpsbm_model.generate_markov_network(transition_matrix, seed = 1)
        pisces_model.generate_markov_network(transition_matrix, seed = 1)

        # Learn community dynamics
        hbsm_model.learn_community_dynamics(method = 'hbsm', K = num_communities, niter = 100)
        dpsbm_model.learn_community_dynamics(method = 'dpsbm', K = num_communities, niter = 100)
        pisces_model.learn_community_dynamics(method = 'pisces', K = num_communities, niter = 100)
        
        # EVALUATE MODELS
        evaluator = Evaluation([hbsm_model, dpsbm_model, pisces_model])
        agg_nmi = evaluator.aggregate_nmi()
        results[tau] = agg_nmi

    # PLOT RESULTS
    method_results = lambda method: [results[tau][method] for tau in transition_probs]

    fig, ax = plt.subplots(figsize=(6, 4)) 
    for method in ['hbsm', 'dpsbm', 'pisces']:
        ax.plot(transition_probs, method_results(method), label = f'{method}')
    ax.legend()
    ax.set_xlabel('Transition Probability')
    ax.set_ylabel('Aggregate NMI')
    ax.set_title('Figure 3A ')
    plt.show()

if __name__ == "__main__":

    # Test figure 3a
    figure_3a_test()


