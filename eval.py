import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from model import EvolvingCommunityModel
from utils import locality_sbm, simple_rw_transition
class Evaluation:
    """
    Evaluation of dynamic community detection using NMI metrics.
    """
    def __init__(self, models):
        # Accept either a single model or list of models
        if isinstance(models, EvolvingCommunityModel):
            self.models = [models]
        elif isinstance(models, (list, tuple)) and all(isinstance(m, EvolvingCommunityModel) for m in models):
            self.models = list(models)
        else:
            raise ValueError("models must be an EvolvingCommunityModel instance or a list of them")
        # Derive display names from each model
        self.names = [getattr(m, 'inference_method', None) or f"Model{i+1}" for i,m in enumerate(self.models)]

    def aggregate_nmi(self):
        """
        Compute aggregate NMI between true and final learned assignments for each model.
        Returns a dict mapping each model name to its NMI score (or None if not learned).
        """
        results = {}
        for name, m in zip(self.names, self.models):
            traj = m.learned_trajectories
            if traj is None:
                results[name] = None
            else:
                true = m.community_assignments
                pred = traj[:, :, -1]
                results[name] = normalized_mutual_info_score(true.flatten(), pred.flatten())
        return results

    def slice_nmi(self):
        """
        Compute slice-wise NMI per layer and mean for each model.
        Returns a dict mapping each model name to a dict with 'slice' and 'mean' (or None if not learned).
        """
        results = {}
        for name, m in zip(self.names, self.models):
            traj = m.learned_trajectories
            if traj is None:
                results[name] = None
            else:
                true = m.community_assignments
                pred = traj[:, :, -1]
                nmis = [normalized_mutual_info_score(true[:, t], pred[:, t]) for t in range(true.shape[1])]
                results[name] = {
                    'slice': np.array(nmis),
                    'mean': float(np.mean(nmis))
                }
        return results


if __name__ == "__main__":
    import time
    # Set random seed for reproducibility
    seed = 42
    rng = np.random.default_rng(seed=seed)
    # Fixed parameters
    n_nodes = 100
    n_layers = 20
    # List of community counts to sweep
    K_values = [5, 10, 15, 20, 25, 30]
    methods = ['hbsm', 'dpsbm', 'pisces']
    # Store aggregate NMI for each method across K
    agg_results = {m: [] for m in methods}

    for K in K_values:
        # Generate generative model parameters for this K
        community_matrix = locality_sbm(K, beta=0.05)
        transition_matrix = simple_rw_transition(K)
        initial_distribution = np.ones(K) / K

        # Run each inference method
        models = []
        for method in methods:
            model = EvolvingCommunityModel(
                n_nodes, n_layers, K, community_matrix, initial_distribution
            )
            model.generate_markov_network(transition_matrix, seed=seed)
        
            if method == 'pisces':
                model.learn_community_dynamics(method=method, niter=100, K=K)
            else:
                model.learn_community_dynamics(method=method, niter=100)
            models.append(model)

        # Evaluate
        evaluation = Evaluation(models)
        agg_scores = evaluation.aggregate_nmi()
        for m in methods:
            agg_results[m].append(agg_scores.get(m))
        # slight pause
        time.sleep(0.5)

    # Plot aggregate NMI vs K for each method
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ax.plot(K_values, agg_results[m], marker='o', label=m)
    ax.set_xlabel('Number of communities (K)')
    ax.set_ylabel('Aggregate NMI')
    ax.set_title('Aggregate NMI vs Number of Communities')
    ax.legend()
    plt.tight_layout()
    # display longer
    plt.show(block=False)
    plt.pause(100)
    


