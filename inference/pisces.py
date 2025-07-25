import numpy as np
import time
import sys
import os

# Add parent directory to path to import model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Scikit-learn Check ---
SKLEARN_LOADED = False
KMeans = None
try:
    from sklearn.cluster import KMeans
    SKLEARN_LOADED = True
    print("scikit-learn loaded successfully.")
except ImportError:
    print("Warning: scikit-learn not found. PISCES inference requires it for K-Means clustering.")
    print("Please install it: pip install scikit-learn")

# --- rpy2 setup ---
RPY2_LOADED = False
R_Matrix = None
R_RSpectra = None
R_nett = None 
R_competing_methods = None 
R_robjects = None
R_numpy2ri = None
R_base = None 

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    R_robjects = robjects
    R_numpy2ri = numpy2ri
    R_Matrix = importr('Matrix')
    R_RSpectra = importr('RSpectra')
    R_base = importr('base')
    try:
        # Attempt to import 'nett' if it's installed as a package
        R_nett = importr('nett')
        print("R package 'nett' loaded successfully.")
    except Exception as nett_err:
        print(f"Could not load R package 'nett': {nett_err}. Will attempt sourcing.")
        R_nett = None # Ensure it's None if import fails

    # Activate numpy<->R conversion
    R_numpy2ri.activate()

    # --- Source the R script containing the 'pisces' function ---
    # Construct the absolute path to the R script relative to this Python script
    competing_methods_r_path = os.path.join(parent_dir, 'hbsm', 'hsbm_package', 'R', 'competing_methods.R')
    if os.path.exists(competing_methods_r_path):
        try:
            print(f"Sourcing R script: {competing_methods_r_path}")
            R_base.source(competing_methods_r_path)
            # Check if pisces function is now in the global environment
            if 'pisces' in R_robjects.globalenv:
                 R_pisces_func = R_robjects.globalenv['pisces']
                 RPY2_LOADED = True # Consider loaded only if source is successful
                 print("rpy2 setup and R script sourcing complete for pisces.py.")
            else:
                print(f"Error: 'pisces' function not found in R global environment after sourcing {competing_methods_r_path}")
                print("PISCES inference will not be available.")

        except Exception as source_err:
            print(f"Error sourcing R script {competing_methods_r_path}: {source_err}")
            print("PISCES inference will not be available.")
    else:
        print(f"Error: R script not found at {competing_methods_r_path}")
        print("PISCES inference will not be available.")

except ImportError as e:
    print(f"Error importing rpy2 or R packages for pisces.py: {e}")
    print("Please ensure rpy2 is installed (pip install rpy2) and the R packages 'Matrix', 'RSpectra', and potentially 'nett' are installed.")
except Exception as e: # Catch other R interface errors
    print(f"Error during rpy2 setup or R sourcing: {e}")

# --- End rpy2/sklearn setup ---


def run_pisces_inference(model, **kwargs):
    """
    Runs PISCES inference using the R function sourced from competing_methods.R via rpy2
    to get the history of spectral embeddings, then performs K-Means in Python.

    Populates model.learned_trajectories with assignment history (N, T, Iters).
    Populates model.learned_community_matrix with None.

    Parameters:
    -----------
    model : EvolvingCommunityModel
        The model instance containing the dynamic network.
    **kwargs : dict
        Hyperparameters for the R 'pisces' function (e.g., K, alpha, niter, tol, verb)
        and K-Means (e.g., kmeans_n_init, kmeans_random_state).
        'K' (number of communities) is required.
    """
    if not RPY2_LOADED:
        print("Cannot run PISCES inference because rpy2 setup or R sourcing failed.")
        model.learned_trajectories = None
        model.learned_community_matrix = None
        return

    if not SKLEARN_LOADED:
        print("Cannot run PISCES inference because scikit-learn failed to load.")
        model.learned_trajectories = None
        model.learned_community_matrix = None
        return

    if model.dynamicNetwork is None:
        raise ValueError("Model dynamic network not generated. Call model.generate_markov_network() first.")

    if 'K' not in kwargs:
         raise ValueError("Missing required hyperparameter 'K' (number of communities) for PISCES.")

    print("\n--- Running PISCES Inference (R spectral + Python K-Means) ---")
    start_time = time.time()

    # --- Prepare Input for R ---
    T = model.n_layers
    N = model.n_nodes
    A_tensor = model.dynamicNetwork # Shape (N, N, T)

    # Convert to list of numpy arrays (ensure float for R)
    A_list_np = [A_tensor[:, :, t].astype(np.float64) for t in range(T)]

    # Convert to R list of sparse matrices
    try:
        r_A_list = R_robjects.ListVector({
            str(i+1): R_Matrix.Matrix(A, sparse=True)
            for i, A in enumerate(A_list_np)
        })
        print(f"Converted {T} network layers to R list of sparse matrices.")
    except Exception as e:
         print(f"Error converting numpy arrays to R sparse matrices: {e}")
         # Ensure learned_trajectories is a dict before trying to assign None to a key
         if not hasattr(model, 'learned_trajectories') or model.learned_trajectories is None:
            model.learned_trajectories = {}
         model.learned_trajectories['PISCES'] = None # Default key if shared_kmeans_init is not specified
         model.learned_trajectories['PISCES-SH'] = None 
         model.learned_community_matrix = None
         return

    # Determine method label for storage
    method_name_key = 'PISCES-SH' if kwargs.get('shared_kmeans_init', False) else 'PISCES'
    model.inference_method = method_name_key

    # --- Set Hyperparameters ---
    # Number of nodes
    N = model.n_nodes
    K_req = int(kwargs['K'])
    if K_req >= N:
        K = N - 1
        print(f"Warning: requested K={K_req} ≥ n_nodes={N}, setting K to {K}.")
    elif K_req < 1:
        raise ValueError(f"Invalid K={K_req}; must be between 1 and n_nodes-1={N-1}.")
    else:
        K = K_req
    alpha = float(kwargs.get('alpha', 0.1))
    niter = int(kwargs.get('niter', 50)) # Max iterations in R
    tol = float(kwargs.get('tol', 1e-6))
    verb = bool(kwargs.get('verb', False))
    # For Python KMeans
    kmeans_n_init = int(kwargs.get('kmeans_n_init', 10))
    kmeans_random_state = kwargs.get('kmeans_random_state', None) # Can be int or None
    if kmeans_random_state is not None:
        kmeans_random_state = int(kmeans_random_state)

    # --- Call R Function to get V history ---
    # Initialize dictionaries if they don't exist
    if not hasattr(model, 'mcmc_sample_history') or model.mcmc_sample_history is None:
        model.mcmc_sample_history = {}
    if not hasattr(model, 'learned_trajectories') or model.learned_trajectories is None:
        model.learned_trajectories = {}

    model.mcmc_sample_history[method_name_key] = None # Reset for this run
    model.learned_trajectories[method_name_key] = None # Reset for this run
    model.learned_community_matrix = None # PISCES doesn't return this
    V_history_r = None

    # Call the R 'pisces' function, retrying with smaller K upon eigen failures
    RRuntimeError = None
    try:
        from rpy2.rinterface_lib.embedded import RRuntimeError
    except ImportError:
        pass
    V_history_r = None
    current_K = K
    while current_K > 0:
        try:
            print(f"Calling R 'pisces' with K={current_K}, alpha={alpha}, niter={niter}...")
            V_history_r = R_pisces_func(
                A=r_A_list,
                K=R_robjects.IntVector([current_K]),
                alpha=R_robjects.FloatVector([alpha]),
                niter=R_robjects.IntVector([niter]),
                tol=R_robjects.FloatVector([tol]),
                verb=R_robjects.BoolVector([verb])
            )
            # If call succeeds, break out
            K = current_K
            break
        except Exception as e:
            err_msg = str(e)
            print(f"PISCES R call failed with K={current_K}: {err_msg}")
            current_K -= 1
            if current_K <= 0:
                raise RuntimeError(f"PISCES failed for all K values down to 1.")
            print(f"Retrying PISCES with reduced K={current_K}...")

    # --- Process V_history and perform K-Means in Python ---
    print("Extracting V history from R list and performing K-Means...")
    if V_history_r is not None and V_history_r is not R_robjects.NULL and len(V_history_r) > 0:
        # V_history_r is an R list (iterations) of lists (layers) of matrices (embeddings)
        num_iters_actual = len(V_history_r)
        print(f"  R function ran for {num_iters_actual} iterations.")

        # Check structure of the first iteration's result
        first_iter_layers = V_history_r[0]
        if len(first_iter_layers) != T:
             raise ValueError(f"Mismatch in number of layers returned by R ({len(first_iter_layers)}) and expected ({T})")

        # Initialize storage for assignments history
        learned_assignments_history = np.zeros((N, T, num_iters_actual), dtype=int)

        # Loop through iterations returned by R
        for iter_idx in range(num_iters_actual):
            V_iter_r = V_history_r[iter_idx] # R list of matrices for this iteration
            # Loop through layers
            for layer_idx in range(T):
                V_layer_r = V_iter_r[layer_idx] # R matrix
                # Convert R matrix to numpy array
                try:
                    V_layer_np = np.array(V_layer_r)
                    if V_layer_np.shape[0] != N or V_layer_np.shape[1] != K:
                         raise ValueError(f"Unexpected shape {V_layer_np.shape} for V matrix at iter {iter_idx+1}, layer {layer_idx+1}. Expected ({N}, {K})")
                except Exception as convert_err:
                     print(f"  Error converting V matrix to numpy at iter {iter_idx+1}, layer {layer_idx+1}: {convert_err}")
                     # Handle error - maybe skip this iter/layer or stop?
                     # For now, let's fill with a placeholder like -1 and continue, but log error
                     learned_assignments_history[:, layer_idx, iter_idx] = -1
                     continue # Skip k-means for this layer/iter

                # Perform K-Means
                try:
                    kmeans = KMeans(n_clusters=K, n_init=kmeans_n_init, random_state=kmeans_random_state)
                    labels = kmeans.fit_predict(V_layer_np)
                    learned_assignments_history[:, layer_idx, iter_idx] = labels # 0-based labels
                except Exception as kmeans_err:
                    print(f"  Error during K-Means at iter {iter_idx+1}, layer {layer_idx+1}: {kmeans_err}")
                    learned_assignments_history[:, layer_idx, iter_idx] = -1 # Error placeholder

        # Store the full 3D history of assignments (N, T, Iterations)
        model.mcmc_sample_history[method_name_key] = learned_assignments_history
        print(f"  Stored PISCES full assignment history (N, T, Iterations): {learned_assignments_history.shape} into model.mcmc_sample_history['{method_name_key}']")

        # Store the assignments from the last iteration as the primary learned trajectory
        if num_iters_actual > 0:
            final_learned_trajectory = learned_assignments_history[:, :, -1] # Shape (N, T)
            model.learned_trajectories[method_name_key] = final_learned_trajectory
            print(f"  Stored PISCES final trajectory (last iteration, N, T): {final_learned_trajectory.shape} into model.learned_trajectories['{method_name_key}']")
        else:
            print("  PISCES ran for 0 iterations, no final trajectory to store.")
            model.learned_trajectories[method_name_key] = None # Ensure it's None if no iterations

    else:
        print("  Warning: PISCES R function did not return a valid V history.")

    end_time = time.time()
    print(f"--- PISCES Inference finished in {end_time - start_time:.2f}s ---")


