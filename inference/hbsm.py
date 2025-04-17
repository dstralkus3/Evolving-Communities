import numpy as np
import time
import sys
import os

# Add parent directory to path to import model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- rpy2 setup ---
RPY2_LOADED = False
R_Matrix = None
R_hsbm = None
R_robjects = None
R_numpy2ri = None
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    R_robjects = robjects
    R_numpy2ri = numpy2ri
    R_Matrix = importr('Matrix')
    R_hsbm = importr('hsbm') # Assumes R package 'hsbm' is installed
    R_numpy2ri.activate()
    RPY2_LOADED = True
    print("rpy2 and R packages (Matrix, hsbm) loaded successfully for hbsm.py.")
except ImportError as e:
    print(f"Error importing rpy2 or R packages for hbsm.py: {e}")
    print("Please ensure rpy2 is installed (pip install rpy2) and the R packages 'Matrix' and 'hsbm' are installed in your R environment.")
except Exception as e: # Catch other R interface errors
    print(f"Error loading R packages (Matrix, hsbm) via rpy2: {e}")
# --- End rpy2 setup ---

def run_hbsm_inference(model, **kwargs):
    """
    Runs HBSM inference using the R 'hsbm' package via rpy2.
    Populates model.learned_trajectories with assignment history (N, T, Iters).
    Populates model.learned_community_matrix with None (as iterative history isn't available).
    Prints info about the final learned eta matrix.

    Parameters:
    -----------
    model : EvolvingCommunityModel
        The model instance containing the dynamic network.
    **kwargs : dict
        Hyperparameters for hsbm::fit_hsbm (e.g., niter, Kcap, Gcap, verb, beta0, gam0).
    """
    if not RPY2_LOADED:
        print("Cannot run HBSM inference because rpy2 or R packages failed to load.")
        model.learned_trajectories = None
        model.learned_community_matrix = None
        return

    if model.dynamicNetwork is None:
        raise ValueError("Model dynamic network not generated. Call model.generate_markov_network() first.")

    print("\n--- Running HBSM Inference (via R) ---")
    start_time = time.time()
    print(type(model))

    # --- Prepare Input for R ---
    T = model.n_layers
    N = model.n_nodes
    K_true = model.num_communities
    A_tensor = model.dynamicNetwork # Shape (N, N, T)
    # Convert to list of numpy arrays
    A_list_np = [A_tensor[:, :, t].astype(np.float64) for t in range(T)] # Ensure float for R
    # Convert to R list of sparse matrices (using activated numpy2ri)
    try:
        r_A_list = R_robjects.ListVector({
            str(i+1): R_Matrix.Matrix(A, sparse=True)
            for i, A in enumerate(A_list_np)
        })
    except Exception as e:
         print(f"Error converting numpy arrays to R sparse matrices: {e}")
         model.learned_trajectories = None
         model.learned_community_matrix = None
         return
    
    model.inference_method = 'hbsm'
    # --- Set Hyperparameters ---
    niter = int(kwargs.get('niter', 50))
    kcap = int(kwargs.get('Kcap', K_true + 5))
    gcap = int(kwargs.get('Gcap', K_true + 5))
    verb = bool(kwargs.get('verb', False))
    beta0 = float(kwargs.get('beta0', 0.1))
    gam0 = float(kwargs.get('gam0', 0.5))
    # Add others like alpha_eta, beta_eta if needed
    alpha_eta = float(kwargs.get('alpha_eta', 1.0))
    beta_eta = float(kwargs.get('beta_eta', 1.0))

    # --- Call R Function ---
    model.learned_trajectories = None # Reset
    model.learned_community_matrix = None # Reset
    final_eta = None
    print(f"Calling hsbm::fit_hsbm with niter={niter}, Kcap={kcap}, Gcap={gcap}...")
    try:
        # Ensure R boolean and integer types are used where appropriate
        hbsm_result_r = R_hsbm.fit_hsbm(
            r_A_list,
            beta0=R_robjects.FloatVector([beta0]),
            gam0=R_robjects.FloatVector([gam0]),
            alpha_eta=R_robjects.FloatVector([alpha_eta]),
            beta_eta=R_robjects.FloatVector([beta_eta]),
            niter=R_robjects.IntVector([niter]),
            Kcap=R_robjects.IntVector([kcap]),
            Gcap=R_robjects.IntVector([gcap]),
            verb=R_robjects.BoolVector([verb]),
            # Add other params like rand_init, seq_g_update if needed
            rand_init=R_robjects.BoolVector([kwargs.get('rand_init', True)]),
            seq_g_update=R_robjects.BoolVector([kwargs.get('seq_g_update', True)])
        )

        # --- Extract Trajectory (zb_list) ---
        print("Extracting results from R object...")
        zb_list_r = hbsm_result_r.rx2('zb_list')
        if zb_list_r is not R_robjects.NULL:
            # R list: iterations -> layers -> nodes (1-based)
            num_iters_r = len(zb_list_r)
            num_layers_r = len(zb_list_r[0]) # Assumes >= 1 iter
            num_nodes_r = len(zb_list_r[0][0]) # Assumes >= 1 layer
            print(f"  Raw zb_list dimensions (iters, layers, nodes): ({num_iters_r}, {num_layers_r}, {num_nodes_r})")

            # Convert to numpy array (iters, layers, nodes)
            trajectory_raw = np.zeros((num_iters_r, num_layers_r, num_nodes_r), dtype=int)
            for i in range(num_iters_r):
                for l in range(num_layers_r):
                    trajectory_raw[i, l, :] = np.array(zb_list_r[i][l], dtype=int) - 1 # Convert to 0-based

            # Transpose to: (nodes, layers, iterations)
            model.learned_trajectories = trajectory_raw.transpose(2, 1, 0)
            print(f"  Stored HBSM trajectory with shape: {model.learned_trajectories.shape}")
        else:
            print("  Warning: HBSM R result did not contain 'zb_list'. Trajectories not stored.")

        # --- Extract Final Community Matrix (eta) ---
        final_eta_r = hbsm_result_r.rx2('eta')
        if final_eta_r is not R_robjects.NULL:
             final_eta = np.array(final_eta_r)
             print(f"  Extracted FINAL HBSM community matrix (eta) with shape: {final_eta.shape}")
             model.learned_community_matrix = final_eta
        else:
             print("  Warning: HBSM R result did not return final 'eta'.")
    
    except Exception as e:
        print(f"  Error during R hsbm::fit_hsbm execution or result extraction: {e}")
        import traceback
        traceback.print_exc()

    end_time = time.time()
    print(f"--- HBSM Inference finished in {end_time - start_time:.2f}s ---")
