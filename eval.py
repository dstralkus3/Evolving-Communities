import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from model import EvolvingCommunityModel
from utils import locality_sbm, simple_rw_transition

class Evaluation:
    """
    Evaluation of dynamic community detection using NMI/AMI metrics.
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
        Compute aggregate AMI between true and final learned assignments for each model.
        Returns a dict mapping each model name to its AMI score (or np.nan if not learned/error).
        """
        results = {}
        for name, m in zip(self.names, self.models):
            learned_trajectories_dict = getattr(m, 'learned_trajectories', None)
            method_name = getattr(m, 'inference_method', None)

            if not isinstance(learned_trajectories_dict, dict) or method_name is None or method_name not in learned_trajectories_dict:
                results[name] = np.nan
                print(f"Warning: Learned trajectory for method '{name}' (key: '{method_name}') not found in aggregate_nmi. Score set to NaN.")
                continue

            pred = learned_trajectories_dict.get(method_name)
            true = getattr(m, 'true_community_assignments', None)

            if true is None or pred is None:
                results[name] = np.nan
                print(f"Warning: True or predicted labels are None for method '{name}' in aggregate_nmi. Score set to NaN.")
                continue

            if not isinstance(true, np.ndarray) or not isinstance(pred, np.ndarray) or true.ndim != 2 or pred.ndim != 2:
                print(f"Warning: True or predicted labels are not 2D numpy arrays for method '{name}'. Shapes: True {true.shape if hasattr(true, 'shape') else type(true)}, Pred {pred.shape if hasattr(pred, 'shape') else type(pred)}. Score set to NaN.")
                results[name] = np.nan
                continue

            if true.shape != pred.shape:
                print(f"Warning: Shape mismatch for AMI. True: {true.shape}, Pred: {pred.shape} for method '{name}'. Score set to NaN.")
                results[name] = np.nan
                continue
            try:
                # Using ravel() to flatten layer-wise labels for a single aggregate score
                score = adjusted_mutual_info_score(true.ravel(), pred.ravel(), average_method='arithmetic')
                results[name] = score
            except Exception as e:
                print(f"Error calculating AMI for {name}: {e}. Score set to NaN.")
                results[name] = np.nan
        return results

    def slice_nmi(self):
        """
        Compute slice-wise AMI per layer and mean for each model.
        Returns a dict mapping each model name to a dict with 'slice' and 'mean' (or np.nan if error).
        """
        results = {}
        for name, m in zip(self.names, self.models):
            learned_trajectories_dict = getattr(m, 'learned_trajectories', None)
            method_name = getattr(m, 'inference_method', None)

            if not isinstance(learned_trajectories_dict, dict) or method_name is None or method_name not in learned_trajectories_dict:
                results[name] = {'slice': None, 'mean': np.nan}
                print(f"Warning: Learned trajectory for method '{name}' (key: '{method_name}') not found in slice_nmi.")
                continue

            pred = learned_trajectories_dict.get(method_name)
            true = getattr(m, 'true_community_assignments', None)

            if true is None or pred is None:
                results[name] = {'slice': None, 'mean': np.nan}
                print(f"Warning: True or predicted labels are None for method '{name}' in slice_nmi.")
                continue

            if not isinstance(true, np.ndarray) or not isinstance(pred, np.ndarray) or true.ndim != 2 or pred.ndim != 2:
                print(f"Warning: True or predicted labels are not 2D numpy arrays for method '{name}' in slice_nmi. Shapes: True {true.shape if hasattr(true, 'shape') else type(true)}, Pred {pred.shape if hasattr(pred, 'shape') else type(pred)}.)")
                results[name] = {'slice': None, 'mean': np.nan}
                continue

            if true.shape != pred.shape:
                print(f"Warning: Shape mismatch for AMI. True: {true.shape}, Pred: {pred.shape} for method '{name}'.")
                results[name] = {'slice': None, 'mean': np.nan}
                continue

            nmis = []
            try:
                for t in range(true.shape[1]):
                    nmis.append(adjusted_mutual_info_score(true[:, t], pred[:, t], average_method='arithmetic'))
                results[name] = {
                    'slice': np.array(nmis) if nmis else None,
                    'mean': float(np.mean(nmis)) if nmis else np.nan
                }
            except Exception as e:
                print(f"Error calculating slice-wise AMI for {name}: {e}.")
                results[name] = {'slice': None, 'mean': np.nan}
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
    K_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    methods = ['hbsm', 'dpsbm', 'pisces']
    # Store aggregate NMI for each method across K
    agg_results = {m: [] for m in methods}

    # Inference kwargs (can be moved to top if preferred)
    NITER_EVAL = 50 # Using a smaller niter for quicker eval script testing
    HBSM_KWARGS_EVAL = dict(niter=NITER_EVAL, Kcap=max(K_values) + 5, Gcap=max(K_values) + 5, r_get_map_labels_burnin=NITER_EVAL//2, r_get_map_labels_consecutive=True)
    DPSBM_KWARGS_EVAL = dict(niter=NITER_EVAL, Zcap=max(K_values) + 5, r_get_map_labels_burnin=NITER_EVAL//2, r_get_map_labels_consecutive=True)
    # PisCES K will be set to current K in loop
    PISCES_KWARGS_EVAL = dict(niter=NITER_EVAL, alpha=0.1, tol=1e-6, verb=False, kmeans_n_init=10)

    METHOD_KWARGS_EVAL = {
        'hbsm': HBSM_KWARGS_EVAL,
        'dpsbm': DPSBM_KWARGS_EVAL,
        'pisces': PISCES_KWARGS_EVAL,
    }

    for K_val in K_values:
        print(f"\n--- Running evaluation for K = {K_val} ---")
        community_matrix = locality_sbm(K_val, beta=0.05)
        transition_matrix = simple_rw_transition(K_val)
        initial_distribution = np.ones(K_val) / K_val

        models_for_k = []
        for method_name in methods:
            print(f"  Processing method: {method_name} for K={K_val}")
            model_instance = EvolvingCommunityModel(
                n_nodes, n_layers, K_val, community_matrix, initial_distribution
            )
            model_instance.generate_markov_network(transition_matrix, seed=seed)

            current_method_kwargs = METHOD_KWARGS_EVAL.get(method_name, {}).copy()
            if method_name == 'pisces':
                current_method_kwargs['K'] = K_val

            try:
                model_instance.learn_community_dynamics(method=method_name, **current_method_kwargs)
                models_for_k.append(model_instance)
            except Exception as e_learn:
                print(f"    Error during learn_community_dynamics for {method_name} with K={K_val}: {e_learn}")
                # Optionally append a dummy/failed model or skip
                # For now, skipping if learning fails, so it won't be in Evaluation

        if not models_for_k:
            print(f"  No models successfully processed for K={K_val}. Skipping evaluation for this K.")
            for m_key in methods:
                agg_results[m_key].append(np.nan) # Record NaN if no model for this K
            continue

        evaluation_instance = Evaluation(models_for_k)
        current_agg_scores = evaluation_instance.aggregate_nmi() # Using AMI now

        for method_key in methods: # Iterate through all expected methods
            # The name in current_agg_scores might be Model1, Model2 etc. if inference_method not set
            # This part needs refinement if names in Evaluation class don't match method_key directly.
            # For now, assuming they align or we take the order.
            # Let's try to find the score by the method name used in this loop.
            score_found = False
            for eval_name, score_val in current_agg_scores.items():
                # This relies on m.inference_method being set correctly to 'hbsm', 'dpsbm', 'pisces'
                if eval_name == method_key:
                    agg_results[method_key].append(score_val)
                    score_found = True
                    break
            if not score_found:
                agg_results[method_key].append(np.nan) # If a method didn't run or had no score

        time.sleep(0.1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_label in methods:
        ax.plot(K_values, agg_results[method_label], marker='o', label=method_label.upper())
    ax.set_xlabel('Number of True Communities (K)')
    ax.set_ylabel('Aggregate AMI') # Changed NMI to AMI
    ax.set_title('Model Performance vs. Number of Communities')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("eval_k_vs_ami.png")
    print("\nEvaluation plot saved to eval_k_vs_ami.png")
    plt.show()
    


