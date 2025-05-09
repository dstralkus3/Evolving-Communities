import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score
from model import EvolvingCommunityModel
from utils import sbm  # ensure utils.sbm is importable
from visualize import Visualizer # Added import

"""
fig3a.py
---------
Replicates the Markov Community Model (MCM) experiment from
HBSM paper Figure 3(a).

For each value of τ in [0,1] we:
  • build a dynamic SBM with parameters
        n=200, m=3, T=5,
        community matrix B (same for all layers),
        initial distribution π,
        Markov kernel P = (1-τ) I + τ 1 πᵀ
  • run three inference methods (HBSM, DP-SBM, PisCES)
  • compute aggregate NMI (ANMI) between true and learned labels
  • plot ANMI versus τ for every method

The script can be executed directly:
    python3 fig3a.py
"""

# ---------------- Experiment parameters ----------------
N_NODES = 200
N_LAYERS = 5
NUM_COMMUNITIES = 3  # m

# Community connectivity matrix B (η) – symmetric
COMMUNITY_MATRIX = np.array([
    [0.9, 0.75, 0.5],
    [0.75, 0.6, 0.25],
    [0.5, 0.25, 0.1]
])

# Initial distribution π
PI_INIT = np.array([0.40, 0.35, 0.25])

# τ grid
TAU_VALUES = np.linspace(0.0, 1.0, 11)  # 0.0,0.1,...,1.0
TAU_FOR_VISUALIZATION = 0.5 # Choose a specific tau for detailed visualization

# Inference settings
NITER = 100  # Gibbs iterations for HBSM/DP-SBM, iterations for PisCES

METHODS = ['hbsm', 'dpsbm', 'pisces']

HBSM_KWARGS = dict(niter=NITER, Kcap=NUM_COMMUNITIES, Gcap=NUM_COMMUNITIES, r_get_map_labels_burnin=NITER//2, r_get_map_labels_consecutive=True)
DPSBM_KWARGS = dict(niter=NITER, Zcap=NUM_COMMUNITIES, r_get_map_labels_burnin=NITER//2, r_get_map_labels_consecutive=True)
PISCES_KWARGS = dict(K=NUM_COMMUNITIES, niter=NITER, alpha=0.1, tol=1e-6, verb=False,
                     kmeans_n_init=10)
METHOD_KWARGS = {
    'hbsm': HBSM_KWARGS,
    'dpsbm': DPSBM_KWARGS,
    'pisces': PISCES_KWARGS,
}

# --------------------------------------------------------

def aggregate_nmi(true_lbls: np.ndarray, learned_lbls: np.ndarray) -> float:
    """Compute layer-wise AMI and take the mean (Aggregate NMI as in paper)."""
    if true_lbls is None or learned_lbls is None:
        return np.nan
    if true_lbls.shape != learned_lbls.shape:
        print(f"Warning: Shape mismatch for NMI. True: {true_lbls.shape}, Learned: {learned_lbls.shape}")
        return np.nan
        
    N, T = true_lbls.shape
    scores = []
    for t in range(T):
        scores.append(
            adjusted_mutual_info_score(true_lbls[:, t], learned_lbls[:, t])
        )
    return float(np.mean(scores)) if scores else 0.0


def run_single_tau(tau: float, rng_seed: int = 1234, for_visualization: bool = False):
    """Generate network at a given τ, run inference methods, return ANMI dict and optionally model."""
    print(f"\n--- Running for τ = {tau:.2f} ---")
    P = (1 - tau) * np.eye(NUM_COMMUNITIES) + tau * np.outer(np.ones(NUM_COMMUNITIES), PI_INIT)

    model = EvolvingCommunityModel(
        n_nodes=N_NODES,
        n_layers=N_LAYERS,
        num_communities=NUM_COMMUNITIES,
        community_matrix=COMMUNITY_MATRIX,
        initial_distribution=PI_INIT,
    )

    model.generate_markov_network(P, seed=rng_seed)

    anmi_results = {}
    for method in METHODS:
        print(f"  Applying method: {method}")
        current_method_kwargs = METHOD_KWARGS.get(method, {}).copy()
        
        # For visualization, ensure verbosity for PisCES is False to avoid excessive R output during plot generation
        if for_visualization and method == 'pisces':
            current_method_kwargs['verb'] = False
            
        model.learn_community_dynamics(method=method, **current_method_kwargs)

        learned_labels_dict = getattr(model, 'learned_trajectories', None)
        true_labels = getattr(model, 'true_community_assignments', None)
        
        retrieved_learned_labels = None
        if isinstance(learned_labels_dict, dict) and method in learned_labels_dict:
            retrieved_learned_labels = learned_labels_dict[method]
        
        if true_labels is not None and retrieved_learned_labels is not None:
            anmi = aggregate_nmi(true_labels, retrieved_learned_labels)
            anmi_results[method] = anmi
            print(f"    τ={tau:.2f}, Method={method}, ANMI={anmi:.3f}")
        else:
            print(f"    τ={tau:.2f}, Method={method}, Learned labels not found or shape mismatch. ANMI set to NaN.")
            anmi_results[method] = np.nan

    if for_visualization:
        return anmi_results, model
    return anmi_results


def main():
    all_results = {method: [] for method in METHODS}
    model_for_visualization = None
    visualization_tau_found = False

    # Make sure TAU_FOR_VISUALIZATION is one of the values we test
    # To avoid floating point comparison issues, check if it's close to any TAU_VALUES
    actual_tau_for_viz = -1
    for t_val in TAU_VALUES:
        if np.isclose(t_val, TAU_FOR_VISUALIZATION):
            actual_tau_for_viz = t_val
            visualization_tau_found = True
            break
    if not visualization_tau_found:
        print(f"Warning: TAU_FOR_VISUALIZATION ({TAU_FOR_VISUALIZATION}) not in TAU_VALUES. Visualization will be skipped.")


    for i, tau_val in enumerate(TAU_VALUES):
        # Use a different seed for each tau run to get variability, but consistent for that tau across script runs
        # Or use the same seed if you want less variability in network generation across taus.
        # For now, let's use a seed that varies with tau's index to ensure some network difference.
        current_seed = 1234 + i 
        
        if np.isclose(tau_val, actual_tau_for_viz) and visualization_tau_found:
            # This run is for the specific tau we want to visualize later
            # We need to run it again *after* the loop if we want clean ANMI plot first
            # Or, we can store this specific model. Let's opt to re-run cleanly for visualization.
            pass # We'll re-run for this tau specifically for visualization later
        
        result_tau = run_single_tau(float(tau_val), rng_seed=current_seed, for_visualization=False)
        for method in METHODS:
            all_results[method].append(result_tau.get(method, np.nan))

    # Plotting ANMI results
    plt.figure(figsize=(10, 7))
    for method in METHODS:
        plt.plot(TAU_VALUES, all_results[method], marker='o', linestyle='-', label=method.upper())
    plt.title(f'Aggregate NMI vs τ (N={N_NODES}, T={N_LAYERS}, M={NUM_COMMUNITIES})')
    plt.xlabel('τ (Community Re-randomization Probability)')
    plt.ylabel('Aggregate NMI')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("fig3a_anmi_vs_tau.png")
    print("\nANMI vs τ plot saved to fig3a_anmi_vs_tau.png")
    plt.show()

    # --- Visualization for a specific tau ---
    if visualization_tau_found:
        print(f"\n--- Generating data for visualization at τ = {actual_tau_for_viz:.2f} ---")
        # Re-run for the specific tau to get the model object with all inference data
        # Use a consistent seed for this specific visualization run
        _, model_for_visualization = run_single_tau(float(actual_tau_for_viz), rng_seed=4242, for_visualization=True)

        if model_for_visualization:
            print("\n--- Visualizing Learned Trajectories --- ")
            visualizer = Visualizer(model_for_visualization)
            visualizer.visualize_learned_trajectories() # Should create its own figure and plt.show()

            print("\n--- Visualizing Temporal Network (True Communities) --- ")
            visualizer.display_temporal_network() # Should create its own figure and plt.show()

            if 'hbsm' in METHODS and hasattr(model_for_visualization, 'learned_community_matrix') and model_for_visualization.learned_community_matrix is not None:
                print("\n--- Visualizing Learned Community Matrix (HBSM) --- ")
                plt.figure() # Create a new figure for this plot
                ax_cm = plt.gca()
                visualizer.plot_community_matrix(learned=True, ax=ax_cm)
                plt.suptitle(f"Learned Community Matrix (HBSM) for τ={actual_tau_for_viz:.2f}")
                plt.show()
        else:
            print(f"Could not obtain model for visualization at τ = {actual_tau_for_viz:.2f}.")
    else:
        print("Skipping detailed visualization as TAU_FOR_VISUALIZATION was not found in TAU_VALUES.")

if __name__ == '__main__':
    main()
