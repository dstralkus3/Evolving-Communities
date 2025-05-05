import numpy as np
import matplotlib.pyplot as plt
from model import EvolvingCommunityModel
from utils import simple_rw_transition, match_labels # Assuming match_labels is in utils.py
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix as calculate_confusion_matrix
from scipy.optimize import linear_sum_assignment
import time

# Function to calculate NMI for the final iteration after matching
def calculate_final_nmi(true_assignments, learned_final_assignments, K):
    """
    Matches labels of final learned assignments to true assignments and calculates NMI.

    Args:
        true_assignments (np.ndarray): Shape (N, T).
        learned_final_assignments (np.ndarray): Shape (N, T).
        K (int): Number of communities.

    Returns:
        float or None: Final NMI score, or None if calculation fails.
    """
    if true_assignments is None or learned_final_assignments is None or learned_final_assignments.size == 0 or K <= 0:
        print("Warning: Invalid input to calculate_final_nmi.")
        return None

    N, T = true_assignments.shape
    N_l, T_l = learned_final_assignments.shape

    if N != N_l or T != T_l:
        print(f"Warning: Shape mismatch between true ({N},{T}) and learned ({N_l},{T_l}) assignments.")
        return None

    true_flat = true_assignments.ravel()
    learned_flat = learned_final_assignments.ravel()

    try:
        # Compute the confusion matrix: rows=true, cols=learned
        labels_range = np.arange(K)
        # Handle potential edge case where learned labels don't cover full range
        unique_learned = np.unique(learned_flat)
        if not np.all(np.isin(unique_learned, labels_range)):
             print(f"Warning: Learned labels {unique_learned} outside expected range {labels_range}. Clipping.")
             learned_flat = np.clip(learned_flat, 0, K-1)

        cm = calculate_confusion_matrix(true_flat, learned_flat, labels=labels_range)

        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cm.max() - cm)

        # Create the permutation map
        permutation_map = np.zeros(K, dtype=int)
        permutation_map[col_ind] = row_ind

        # Apply the permutation map
        matched_learned_flat = permutation_map[np.clip(learned_flat, 0, K-1)] # Clip again just in case

        # Calculate NMI on matched final labels
        score = normalized_mutual_info_score(true_flat, matched_learned_flat, average_method='arithmetic')
        return score
    except Exception as e:
        print(f"Error during label matching or NMI calculation for K={K}: {e}")
        return None


if __name__ == "__main__":
    master_seed = 42
    rng = np.random.default_rng(seed=master_seed)

    # --- Fixed Parameters ---
    n_nodes = 50  # Ensure N is larger than max K
    n_layers = 30 # Shorter T for faster runs per K
    num_inference_iters = 50 # Fixed iterations for inference

    # --- K Values to Test ---
    K_values = [2, 3, 4, 5, 8, 10, 15, 20] # Example range, ensure K < N
    K_values = [k for k in K_values if k < n_nodes] # Filter K > N

    print(f"Evaluation Setup: N={n_nodes}, T={n_layers}, Iters={num_inference_iters}")
    print(f"Testing K values: {K_values}")
    print("-------------------------------------------------")

    # --- Store Results ---
    methods_to_run = ['pisces', 'hbsm', 'dpsbm']
    final_nmi_results = {method: [] for method in methods_to_run}
    inference_times = {method: [] for method in methods_to_run} # Store avg time per K?

    # --- Loop Over K ---
    for k_idx, K in enumerate(K_values):
        print(f"--- Running for K = {K} ({k_idx+1}/{len(K_values)}) ---")
        current_seed = master_seed + k_idx # Use a different seed for each K run

        # --- Setup for current K ---
        p_in = 0.6
        p_out = 0.2
        community_matrix = np.full((K, K), p_out)
        np.fill_diagonal(community_matrix, p_in)
        p_stay = 0.9
        transition_matrix = simple_rw_transition(K, p_stay=p_stay)
        initial_distribution = np.ones(K) / K

        # --- Initialize Model & Generate Network ---
        print(f"  Initializing Model (K={K})...")
        model = EvolvingCommunityModel(n_nodes, n_layers, K, community_matrix, initial_distribution)
        print(f"  Generating Network (K={K})...")
        model.generate_markov_network(transition_matrix, seed=current_seed)
        true_community_assignments = model.community_assignments.copy()
        print("  Network Generation Complete.")

        # --- Loop over methods for this K ---
        for method in methods_to_run:
            print(f"    Running Method: {method.upper()}...")
            start_time = time.time()
            final_nmi = None # Reset final NMI for this method/K run

            try:
                # --- Run Inference ---
                model.learned_trajectories = None # Reset
                model.inference_method = None

                kwargs = {'niter': num_inference_iters}
                # Pass current K value to methods that need it
                if method == 'pisces':
                    kwargs['K'] = K
                    kwargs['verb'] = False
                elif method == 'hbsm':
                    # Pass Kcap/Gcap based on current K
                    kwargs['Kcap'] = K + 5
                    kwargs['Gcap'] = K + 5
                elif method == 'dpsbm':
                     # Assuming dpsbm might also benefit from knowing K or caps?
                     # If it infers K, maybe don't pass it. Adjust as needed.
                     pass # Pass only niter for now

                model.learn_community_dynamics(method=method, **kwargs)
                method_time = time.time() - start_time
                print(f"      Inference Time: {method_time:.2f}s")
                inference_times[method].append(method_time) # Store time

                # --- Evaluate Final Iteration ---
                if model.learned_trajectories is not None and model.learned_trajectories.size > 0:
                    learned_final = model.learned_trajectories[:, :, -1] # Get last iteration
                    print(f"      Matching labels and calculating final NMI...")
                    final_nmi = calculate_final_nmi(true_community_assignments, learned_final, K)
                    if final_nmi is not None:
                        print(f"      Final NMI ({method.upper()}): {final_nmi:.4f}")
                    else:
                         print(f"      Final NMI calculation failed.")
                else:
                    print(f"      {method.upper()} did not produce learned trajectories.")

            except Exception as e:
                 print(f"      Error running/evaluating {method.upper()} for K={K}: {e}")

            # Store the calculated final NMI (or NaN if failed)
            final_nmi_results[method].append(final_nmi if final_nmi is not None else np.nan)

        print("-------------------------------------------------")


    # --- Plotting Final NMI vs. K Results ---
    print("Generating Final NMI vs. K plot...")
    plt.figure(figsize=(10, 6))

    for method, scores in final_nmi_results.items():
        # Filter out NaN values for plotting if any method failed for some K
        valid_k = [K_values[i] for i, score in enumerate(scores) if not np.isnan(score)]
        valid_scores = [score for score in scores if not np.isnan(score)]
        if valid_scores:
            plt.plot(valid_k, valid_scores, marker='o', linestyle='-', label=f"{method}")
        else:
             print(f"No valid NMI scores to plot for method: {method}")

    plt.title('Final NMI Score vs. Number of Communities (K)')
    plt.xlabel('Number of Communities (K)')
    plt.ylabel('Final NMI Score (vs True Labels, Matched)')
    plt.xticks(K_values) # Ensure ticks are at the tested K values
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0, top=1.05) # NMI is bounded [0, 1]
    plt.tight_layout()
    print("Displaying plot...")
    plt.show()

    print("Evaluation script finished.") 